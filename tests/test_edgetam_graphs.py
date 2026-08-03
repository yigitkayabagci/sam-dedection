"""Numerical parity tests for the TensorRT export wrappers.

These run on CPU with random weights at a shrunken resolution, which is enough
to prove the *rewrites* are faithful: the fixed-slot padded memory bank, the
additive attention mask, the real-arithmetic RoPE, and the specialised SAM
head all have to reproduce stock EdgeTAM bit-for-bit (up to float noise)
regardless of tensor sizes.

What they deliberately do not cover: TensorRT itself, fp16 accuracy, and CUDA
graphs. Use `tools/check_trt_parity.py` on the Orin for those.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sam2", reason="EdgeTAM not installed (scripts/setup_edgetam.sh)")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.edgetam_graphs import (  # noqa: E402
    MemoryAttentionGraph,
    MemoryEncoderGraph,
    MemoryLayout,
    SamHeadGraph,
    apply_rope_real,
    build_memory_mask,
    rope_tables,
)

# A miniature EdgeTAM: same module graph, 1/4 the spatial extent, fewer memory
# slots and latents. Everything the wrappers rewrite is size-independent, and
# this keeps a full CPU forward pass under a second.
SMALL_OVERRIDES = [
    "++model.image_size=256",
    "++model.memory_attention.layer.cross_attention.q_sizes=[16,16]",
    "++model.memory_attention.layer.cross_attention.k_sizes=[4,4]",
    "++model.spatial_perceiver.num_latents=8",
    "++model.spatial_perceiver.num_latents_2d=16",
    "++model.num_maskmem=3",
]


@pytest.fixture(scope="module")
def model():
    from sam2.build_sam import build_sam2_video_predictor

    torch.manual_seed(0)
    m = build_sam2_video_predictor(
        "configs/edgetam.yaml", None, device="cpu", hydra_overrides_extra=SMALL_OVERRIDES
    )
    m.eval()
    return m


@pytest.fixture(scope="module")
def layout(model):
    return MemoryLayout.from_model(model, ptr_tokens=40)


def _feat_side(model):
    return model.image_size // model.backbone_stride


# ---------------------------------------------------------------- RoPE


def test_rope_real_matches_complex_self_attention(model):
    """`apply_rope_real` reproduces `apply_rotary_enc` (self-attention form)."""
    from sam2.modeling.position_encoding import apply_rotary_enc

    attn = model.memory_attention.layers[0].self_attn
    side = _feat_side(model)
    tokens = side * side
    head_dim = attn.internal_dim // attn.num_heads
    freqs = attn.compute_cis(end_x=side, end_y=side)

    torch.manual_seed(1)
    q = torch.randn(2, attn.num_heads, tokens, head_dim)
    k = torch.randn(2, attn.num_heads, tokens, head_dim)

    ref_q, ref_k = apply_rotary_enc(q.clone(), k.clone(), freqs_cis=freqs, repeat_freqs_k=False)
    cos, sin = rope_tables(freqs)
    assert torch.allclose(apply_rope_real(q, cos, sin), ref_q, atol=1e-6)
    assert torch.allclose(apply_rope_real(k, cos, sin), ref_k, atol=1e-6)


def test_rope_real_matches_complex_cross_attention(model, layout):
    """The tiled/split key rotation reproduces `apply_rotary_enc_v2`."""
    from sam2.modeling.position_encoding import apply_rotary_enc_v2

    attn = model.memory_attention.layers[0].cross_attn_image
    head_dim = attn.internal_dim // attn.num_heads
    slots = layout.num_slots

    torch.manual_seed(2)
    k = torch.randn(2, attn.num_heads, layout.spatial_tokens, head_dim)
    ref = apply_rotary_enc_v2(k.clone(), attn.freqs_cis_k, repeat_freqs=slots)

    from tools.edgetam_graphs import _CrossAttentionGraph

    graph = _CrossAttentionGraph(attn, layout, _feat_side(model) ** 2)
    assert torch.allclose(graph._rope_spatial(k), ref, atol=1e-6)


# ------------------------------------------------- memory attention parity


def _reference_memory_attention(model, curr, curr_pos, mems, mem_poss, ptrs, ptr_pos):
    """Stock EdgeTAM call: variable-length memory, sequence-first tensors."""
    parts = list(mems) + ([ptrs] if ptrs is not None else [])
    pos_parts = list(mem_poss) + ([ptr_pos] if ptr_pos is not None else [])
    memory = torch.cat(parts, dim=0)
    memory_pos = torch.cat(pos_parts, dim=0)
    out = model.memory_attention(
        curr=[curr],
        curr_pos=[curr_pos],
        memory=memory,
        memory_pos=memory_pos,
        num_obj_ptr_tokens=0 if ptrs is None else ptrs.shape[0],
        num_spatial_mem=len(mems),
    )
    batch, dim = out.shape[1], out.shape[2]
    side = int(round(math.sqrt(out.shape[0])))
    return out.permute(1, 2, 0).reshape(batch, dim, side, side)


@pytest.mark.parametrize("num_mem,num_ptrs", [(1, 1), (2, 5), (3, 8)])
def test_padded_memory_matches_variable_length(model, layout, num_mem, num_ptrs):
    """The headline claim: padding to fixed slots + masking changes nothing.

    Runs stock `MemoryAttention` over a memory bank of `num_mem` slots and
    `num_ptrs` pointers, then runs `MemoryAttentionGraph` over the same content
    padded out to full capacity with garbage in the unused positions. The
    additive mask has to make the garbage invisible.
    """
    torch.manual_seed(3 + num_mem)
    batch = 2
    side = _feat_side(model)
    tokens = side * side
    dim = model.hidden_dim
    mem_dim = model.mem_dim
    per_ptr = dim // mem_dim
    ptr_tokens = num_ptrs * per_ptr
    assert ptr_tokens <= layout.ptr_tokens

    curr = torch.randn(tokens, batch, dim)
    curr_pos = torch.randn(tokens, batch, dim)
    mems = [torch.randn(layout.tokens_per_slot, batch, mem_dim) for _ in range(num_mem)]
    mem_poss = [torch.randn(layout.tokens_per_slot, batch, mem_dim) for _ in range(num_mem)]
    ptrs = torch.randn(ptr_tokens, batch, mem_dim)
    ptr_pos = torch.randn(ptr_tokens, batch, mem_dim)

    with torch.no_grad():
        expected = _reference_memory_attention(
            model, curr, curr_pos, mems, mem_poss, ptrs, ptr_pos
        )

    # Fill the padded buffers with large garbage so a missing mask cannot pass.
    memory = torch.full((batch, layout.total_tokens, mem_dim), 7.5)
    memory_pos = torch.full((batch, layout.total_tokens, mem_dim), -3.25)
    used = num_mem * layout.tokens_per_slot
    memory[:, :used] = torch.cat(mems, dim=0).transpose(0, 1)
    memory_pos[:, :used] = torch.cat(mem_poss, dim=0).transpose(0, 1)
    end = layout.spatial_tokens + ptr_tokens
    memory[:, layout.spatial_tokens : end] = ptrs.transpose(0, 1)
    memory_pos[:, layout.spatial_tokens : end] = ptr_pos.transpose(0, 1)
    mask = build_memory_mask(layout, num_mem, ptr_tokens, batch=batch)

    graph = MemoryAttentionGraph(model, layout).eval()
    # The graph takes BCHW straight off the image encoder, not token sequences.
    to_bchw = lambda x: x.permute(1, 2, 0).reshape(batch, dim, side, side)  # noqa: E731
    with torch.no_grad():
        got = graph(to_bchw(curr), to_bchw(curr_pos), memory, memory_pos, mask)

    assert got.shape == expected.shape
    torch.testing.assert_close(got, expected, atol=2e-4, rtol=2e-4)


def test_memory_mask_layout(layout):
    mask = build_memory_mask(layout, num_spatial_mem=2, num_ptr_tokens=8, batch=1)
    live = (mask == 0.0)[0, 0, 0]
    assert live[: 2 * layout.tokens_per_slot].all()
    assert not live[2 * layout.tokens_per_slot : layout.spatial_tokens].any()
    assert live[layout.spatial_tokens : layout.spatial_tokens + 8].all()
    assert not live[layout.spatial_tokens + 8 :].any()


# ------------------------------------------------- memory encoder parity


def test_memory_encoder_graph_matches(model):
    torch.manual_seed(4)
    batch = 2
    side = _feat_side(model)
    pix_feat = torch.randn(batch, model.hidden_dim, side, side)
    mask_for_mem = torch.randn(batch, 1, model.image_size, model.image_size)

    with torch.no_grad():
        out = model.memory_encoder(pix_feat, mask_for_mem, skip_mask_sigmoid=True)
        ref_feats, ref_pos = model.spatial_perceiver(
            out["vision_features"], out["vision_pos_enc"][0]
        )
        got_feats, got_pos = MemoryEncoderGraph(model).eval()(pix_feat, mask_for_mem)

    torch.testing.assert_close(got_feats, ref_feats)
    torch.testing.assert_close(got_pos, ref_pos)


def test_memory_encoder_output_matches_layout(model, layout):
    torch.manual_seed(5)
    side = _feat_side(model)
    with torch.no_grad():
        feats, pos = MemoryEncoderGraph(model).eval()(
            torch.randn(1, model.hidden_dim, side, side),
            torch.randn(1, 1, model.image_size, model.image_size),
        )
    assert feats.shape == (1, layout.tokens_per_slot, model.mem_dim)
    assert pos.shape == (1, layout.tokens_per_slot, model.mem_dim)


# ------------------------------------------------------- SAM head parity


def _sam_head_inputs(model, batch=2, seed=6):
    torch.manual_seed(seed)
    side = _feat_side(model)
    dim = model.hidden_dim
    return (
        torch.randn(batch, dim, side, side),
        torch.randn(batch, dim // 8, side * 4, side * 4),
        torch.randn(batch, dim // 4, side * 2, side * 2),
    )


def test_sam_head_graph_matches(model):
    pix_feat, hi0, hi1 = _sam_head_inputs(model)

    with torch.no_grad():
        ref = model._forward_sam_heads(
            backbone_features=pix_feat,
            point_inputs=None,
            mask_inputs=None,
            high_res_features=[hi0, hi1],
            multimask_output=True,
        )
        got = SamHeadGraph(model).eval()(pix_feat, hi0, hi1)

    (
        ref_low_multi,
        _ref_high_multi,
        ref_ious,
        ref_low,
        ref_high,
        ref_ptr,
        ref_score,
    ) = ref
    low_multi, ious, low, high, ptr, score = got

    torch.testing.assert_close(low_multi, ref_low_multi)
    torch.testing.assert_close(ious, ref_ious)
    torch.testing.assert_close(low, ref_low)
    torch.testing.assert_close(high, ref_high)
    torch.testing.assert_close(ptr, ref_ptr)
    torch.testing.assert_close(score, ref_score)


def test_sam_head_graph_matches_when_object_absent(model):
    """The occluded branch (object_score_logits <= 0) takes a different path."""
    pix_feat, hi0, hi1 = _sam_head_inputs(model, seed=7)
    graph = SamHeadGraph(model).eval()
    # Force every object score negative so `no_obj_ptr` and NO_OBJ_SCORE kick in.
    with torch.no_grad():
        model.sam_mask_decoder.pred_obj_score_head.layers[-1].bias.fill_(-50.0)
        ref = model._forward_sam_heads(
            backbone_features=pix_feat,
            point_inputs=None,
            mask_inputs=None,
            high_res_features=[hi0, hi1],
            multimask_output=True,
        )
        got = graph(pix_feat, hi0, hi1)
        model.sam_mask_decoder.pred_obj_score_head.layers[-1].bias.fill_(0.0)

    assert (ref[6] <= 0).all()
    torch.testing.assert_close(got[2], ref[3])  # low_res_masks
    torch.testing.assert_close(got[4], ref[5])  # obj_ptr


# -------------------------------------------------- why the rewrite exists


def test_stock_memory_attention_cannot_be_exported(model, layout):
    """The stock module does not export to ONNX at all.

    This is the whole reason `MemoryAttentionGraph` exists, so it is worth
    pinning: `apply_rotary_enc*` multiplies by a complex `freqs_cis` table, and
    ONNX has no complex tensor type. If a future torch gains complex export
    support this test starts failing, which is the signal to reconsider the
    rewrite -- not a regression.
    """
    import tempfile

    import torch.nn as nn

    slot = model.spatial_perceiver.num_latents + model.spatial_perceiver.num_latents_2d
    tokens = _feat_side(model) ** 2
    slots, ptr_tokens = 2, 8

    class Stock(nn.Module):
        def __init__(self, mem_attn):
            super().__init__()
            self.mem_attn = mem_attn

        def forward(self, curr, curr_pos, memory, memory_pos):
            return self.mem_attn(
                curr=curr, curr_pos=curr_pos, memory=memory, memory_pos=memory_pos,
                num_obj_ptr_tokens=ptr_tokens, num_spatial_mem=slots,
            )

    args = (
        torch.randn(tokens, 1, model.hidden_dim),
        torch.randn(tokens, 1, model.hidden_dim),
        torch.randn(slots * slot + ptr_tokens, 1, model.mem_dim),
        torch.randn(slots * slot + ptr_tokens, 1, model.mem_dim),
    )
    with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
        with pytest.raises(Exception) as excinfo:
            torch.onnx.export(
                Stock(model.memory_attention).eval(), args, f.name, opset_version=18,
                input_names=["curr", "curr_pos", "memory", "memory_pos"],
                output_names=["out"],
            )
    assert "complex" in str(excinfo.value).lower()


def test_stock_memory_attention_uses_complex_tensors(model):
    """Name the ops that block the export, so the diagnosis stays visible."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class RecordOps(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.ops: set[str] = set()

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.ops.add(str(getattr(func, "_overloadpacket", func)))
            return func(*args, **(kwargs or {}))

    slot = model.spatial_perceiver.num_latents + model.spatial_perceiver.num_latents_2d
    tokens = _feat_side(model) ** 2
    recorder = RecordOps()
    with torch.no_grad(), recorder:
        model.memory_attention(
            curr=torch.randn(tokens, 1, model.hidden_dim),
            curr_pos=torch.randn(tokens, 1, model.hidden_dim),
            memory=torch.randn(2 * slot + 8, 1, model.mem_dim),
            memory_pos=torch.randn(2 * slot + 8, 1, model.mem_dim),
            num_obj_ptr_tokens=8,
            num_spatial_mem=2,
        )
    assert "aten.view_as_complex" in recorder.ops
    assert "aten.view_as_real" in recorder.ops


def test_memory_key_length_varies_across_frames(model, layout):
    """The key sequence length a CUDA graph would have to fix is not fixed.

    Reproduces the growth pattern `tools/analyze_cost.py --trace-frames` shows
    on the real model: one more spatial memory per frame until the bank is
    full, then one more pointer per frame until that saturates too.
    """
    lengths = []
    for spatial in range(1, layout.num_slots + 1):
        for ptr in (0, 4, 8):
            lengths.append(spatial * layout.tokens_per_slot + ptr)
    assert len(set(lengths)) > 1
    assert max(lengths) <= layout.total_tokens, (
        "the padded buffer must be able to hold every length the tracker can produce"
    )


# ------------------------------------------------------------ ONNX export


@pytest.fixture(scope="module")
def exported(model, layout, tmp_path_factory):
    """Export every graph once; the tests below reuse the artifacts."""
    pytest.importorskip("onnx")
    from tools import export_edgetam_onnx as exporter

    outdir = tmp_path_factory.mktemp("onnx")
    plans = {
        "image_encoder": exporter._plan_image_encoder(model, 1, True),
        "memory_attention": exporter._plan_memory_attention(model, 1, layout),
        "memory_encoder": exporter._plan_memory_encoder(model, 1),
        "sam_head": exporter._plan_sam_head(model, 1),
    }
    paths = {
        name: exporter.export_plan(plan, outdir, opset=18, dynamic_batch=True, verify=False)
        for name, plan in plans.items()
    }
    return outdir, paths


@pytest.mark.parametrize("name", list(__import__("tools.export_edgetam_onnx", fromlist=["MODULES"]).MODULES))
def test_onnx_export_writes_spec(exported, name):
    import json

    _, paths = exported
    spec = json.loads(paths[name].with_suffix(".spec.json").read_text())
    assert spec["module"] == name
    assert spec["dynamic_batch"] is True
    assert spec["inputs"] and spec["outputs"]
    assert all(io["batch_axis"] == 0 for io in spec["inputs"] + spec["outputs"])


@pytest.mark.parametrize("batch", [1, 3])
def test_onnx_matches_pytorch_at_any_batch(model, layout, exported, batch):
    """Traced at batch=1; the graphs must stay correct at other batch sizes.

    Multi-object tracking runs EdgeTAM with batch == object count, so a graph
    that silently baked in batch=1 would break the moment a second target is
    selected -- and would do it by producing wrong numbers, not by erroring.
    """
    ort = pytest.importorskip("onnxruntime")
    import numpy as np

    from tools import export_edgetam_onnx as exporter

    _, paths = exported
    builders = {
        "image_encoder": lambda b: exporter._plan_image_encoder(model, b, True),
        "memory_attention": lambda b: exporter._plan_memory_attention(model, b, layout),
        "memory_encoder": lambda b: exporter._plan_memory_encoder(model, b),
        "sam_head": lambda b: exporter._plan_sam_head(model, b),
    }
    for name, build in builders.items():
        plan = build(batch)
        args = tuple(plan.inputs[n] for n in plan.inputs)
        with torch.no_grad():
            ref = plan.module(*args)
        ref = (ref,) if isinstance(ref, torch.Tensor) else ref
        sess = ort.InferenceSession(str(paths[name]), providers=["CPUExecutionProvider"])
        got = sess.run(None, {n: t.numpy() for n, t in zip(plan.inputs, args)})
        for expected, actual in zip(ref, got):
            assert actual.shape[0] == batch, f"{name}: batch axis was baked in"
            np.testing.assert_allclose(
                expected.numpy(), actual, atol=1e-4, rtol=1e-4, err_msg=name
            )


@pytest.mark.parametrize("name", list(__import__("tools.export_edgetam_onnx", fromlist=["MODULES"]).MODULES))
def test_onnx_has_no_trt_hostile_ops(exported, name):
    """No graph may contain an op TensorRT's parser rejects.

    Exporting is the cheap step; building an engine on the Orin is the slow
    one, and a parse failure there costs minutes and reads as a TensorRT
    internal error rather than as an export bug. Two of these were real:
    `repeat_interleave` lowering to OneHot -> Tile in the mask decoder, and
    `Tensor.squeeze(dim)` lowering to an If whose branches differ in rank.
    """
    import onnx

    from tools.export_edgetam_onnx import TRT_HOSTILE_OPS, _traces_to_onehot

    _, paths = exported
    graph = onnx.load(str(paths[name])).graph
    producer = {out: n for n in graph.node for out in n.output}

    offenders = [
        f"{n.op_type} ({n.name})"
        for n in graph.node
        if n.op_type in TRT_HOSTILE_OPS
        or (
            n.op_type == "Tile"
            and len(n.input) > 1
            and _traces_to_onehot(n.input[1], producer)
        )
    ]
    assert not offenders, f"{name} would fail to parse in TensorRT: {offenders}"


def test_hostile_op_scan_detects_repeat_interleave():
    """The scan must actually fire -- otherwise the test above proves nothing.

    Builds the exact pattern that broke the sam_head build: OneHot feeding the
    repeats input of a Tile.
    """
    onnx = pytest.importorskip("onnx")

    from tools.export_edgetam_onnx import _traces_to_onehot

    nodes = [
        onnx.helper.make_node("OneHot", ["idx", "depth", "values"], ["oh"], name="oh"),
        onnx.helper.make_node("Cast", ["oh"], ["reps"], to=onnx.TensorProto.INT64, name="c"),
        onnx.helper.make_node("Tile", ["x", "reps"], ["y"], name="tile"),
    ]
    producer = {out: n for n in nodes for out in n.output}
    assert _traces_to_onehot("reps", producer)
    assert not _traces_to_onehot("x", producer)
