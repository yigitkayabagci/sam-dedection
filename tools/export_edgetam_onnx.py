#!/usr/bin/env python3
"""Export EdgeTAM's per-frame modules to ONNX for TensorRT.

Four graphs cover everything that runs on every tracked frame:

    image_encoder     RepViT trunk + FPN neck (+ the two high-res 1x1 convs)
    memory_attention  2-layer self/cross attention over the memory bank
    memory_encoder    memory encoder + 2D spatial Perceiver
    sam_head          SAM mask decoder, specialised to the propagation path

Everything else EdgeTAM does per frame is bookkeeping in Python.

The wrapper modules (and, importantly, why they are safe rewrites of the
PyTorch originals) live in `tools/edgetam_graphs.py`.

Alongside each .onnx this writes a `<name>.spec.json` describing the input and
output shapes and which axis is batch. `tools/build_trt_engines.py` reads those
to assemble the trtexec optimisation profiles, so shapes are never duplicated
by hand.

Usage:
    python tools/export_edgetam_onnx.py --module all \\
        --checkpoint third_party/EdgeTAM/checkpoints/edgetam.pt \\
        --outdir models/

Then, on the same machine that will run inference:
    python tools/build_trt_engines.py --outdir models/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.edgetam_graphs import (  # noqa: E402
    ImageEncoderGraph,
    MemoryAttentionGraph,
    MemoryEncoderGraph,
    MemoryLayout,
    SamHeadGraph,
)

MODULES = ("image_encoder", "memory_attention", "memory_encoder", "sam_head")


class ExportPlan:
    """One module's wrapper, dummy inputs, and IO names."""

    def __init__(
        self,
        name: str,
        module: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        output_names: Sequence[str],
        extra: dict | None = None,
    ) -> None:
        self.name = name
        self.module = module.eval()
        self.inputs = inputs
        self.output_names = list(output_names)
        self.extra = extra or {}


# --------------------------------------------------------------------------
# Plans
# --------------------------------------------------------------------------


def _plan_image_encoder(model, batch: int, fuse_high_res: bool) -> ExportPlan:
    size = model.image_size
    graph = ImageEncoderGraph(model, fuse_high_res_proj=fuse_high_res)
    return ExportPlan(
        "image_encoder",
        graph,
        {"image": torch.randn(batch, 3, size, size)},
        ["backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2"],
        {"fused_high_res_proj": graph.fuse},
    )


def _plan_memory_attention(model, batch: int, layout: MemoryLayout) -> ExportPlan:
    side = model.image_size // model.backbone_stride
    dim, mem_dim = model.hidden_dim, model.mem_dim
    tokens = layout.total_tokens
    return ExportPlan(
        "memory_attention",
        MemoryAttentionGraph(model, layout),
        {
            "curr": torch.randn(batch, dim, side, side),
            "curr_pos": torch.randn(batch, dim, side, side),
            "memory": torch.randn(batch, tokens, mem_dim),
            "memory_pos": torch.randn(batch, tokens, mem_dim),
            "memory_mask": torch.zeros(batch, 1, 1, tokens),
        },
        ["pix_feat"],
        {"layout": layout.__dict__.copy()},
    )


def _plan_memory_encoder(model, batch: int) -> ExportPlan:
    side = model.image_size // model.backbone_stride
    return ExportPlan(
        "memory_encoder",
        MemoryEncoderGraph(model),
        {
            "pix_feat": torch.randn(batch, model.hidden_dim, side, side),
            # `_encode_new_memory` already applied sigmoid/scale/bias, so the
            # realistic input range is [bias, scale + bias].
            "mask_for_mem": torch.rand(batch, 1, model.image_size, model.image_size)
            * model.sigmoid_scale_for_mem_enc
            + model.sigmoid_bias_for_mem_enc,
        },
        ["maskmem", "maskmem_pos"],
    )


def _plan_sam_head(model, batch: int) -> ExportPlan:
    side = model.image_size // model.backbone_stride
    dim = model.hidden_dim
    return ExportPlan(
        "sam_head",
        SamHeadGraph(model),
        {
            "pix_feat": torch.randn(batch, dim, side, side),
            "high_res_0": torch.randn(batch, dim // 8, side * 4, side * 4),
            "high_res_1": torch.randn(batch, dim // 4, side * 2, side * 2),
        },
        [
            "low_res_multimasks",
            "ious",
            "low_res_masks",
            "high_res_masks",
            "obj_ptr",
            "object_score_logits",
        ],
    )


PLAN_BUILDERS: dict[str, Callable] = {
    "image_encoder": _plan_image_encoder,
    "memory_attention": _plan_memory_attention,
    "memory_encoder": _plan_memory_encoder,
    "sam_head": _plan_sam_head,
}


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


def export_plan(
    plan: ExportPlan, outdir: Path, opset: int, dynamic_batch: bool, verify: bool
) -> Path:
    onnx_path = outdir / f"edgetam_{plan.name}.onnx"
    input_names = list(plan.inputs)
    args = tuple(plan.inputs[n] for n in input_names)

    with torch.no_grad():
        sample = plan.module(*args)
    if isinstance(sample, torch.Tensor):
        sample = (sample,)
    if len(sample) != len(plan.output_names):
        raise RuntimeError(
            f"{plan.name}: wrapper returned {len(sample)} tensors but "
            f"{len(plan.output_names)} output names were declared."
        )

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {n: {0: "batch"} for n in input_names + plan.output_names}
        # `ious` is [B, M] and `object_score_logits` is [B, 1]; both still have
        # batch on axis 0, so the blanket rule above is correct for every
        # tensor these graphs move.

    torch.onnx.export(
        plan.module,
        args,
        str(onnx_path),
        input_names=input_names,
        output_names=plan.output_names,
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )

    spec = {
        "module": plan.name,
        "onnx": onnx_path.name,
        "engine": f"edgetam_{plan.name}.engine",
        "opset": opset,
        "dynamic_batch": dynamic_batch,
        "batch": int(args[0].shape[0]),
        "inputs": [
            {"name": n, "shape": list(plan.inputs[n].shape), "batch_axis": 0}
            for n in input_names
        ],
        "outputs": [
            {"name": n, "shape": list(t.shape), "batch_axis": 0}
            for n, t in zip(plan.output_names, sample)
        ],
        **plan.extra,
    }
    spec_path = onnx_path.with_suffix(".spec.json")
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")

    size_mb = sum(p.stat().st_size for p in _artifacts(onnx_path)) / 1e6
    print(f"  wrote {onnx_path.name} ({size_mb:.1f} MB) + {spec_path.name}")
    for name, tensor in zip(plan.output_names, sample):
        print(f"    out {name:<20} {tuple(tensor.shape)} {tensor.dtype}")

    if verify:
        _verify(onnx_path, plan, args, sample)
    return onnx_path


def _artifacts(onnx_path: Path) -> list[Path]:
    """The .onnx plus any external-data blobs torch may have written next to it."""
    found = [onnx_path]
    for extra in onnx_path.parent.glob(onnx_path.name + ".*"):
        found.append(extra)
    return found


def _verify(onnx_path: Path, plan: ExportPlan, args, sample) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("    WARN: onnxruntime not installed; skipping verification.")
        return
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feed = {n: t.numpy() for n, t in zip(plan.inputs, args)}
    got = sess.run(None, feed)
    worst = max(
        float((ref.detach().numpy().astype("float64") - out.astype("float64")).__abs__().max())
        for ref, out in zip(sample, got)
    )
    status = "ok" if worst <= 1e-3 else "FAIL"
    print(f"    verify: max |PyTorch - onnxruntime| = {worst:.2e}  [{status}]")
    if worst > 1e-3:
        raise SystemExit(f"{plan.name}: ONNX parity check failed ({worst:.2e} > 1e-3).")


def build_model(checkpoint: str | None, model_cfg: str):
    from sam2.build_sam import build_sam2_video_predictor

    model = build_sam2_video_predictor(model_cfg, checkpoint, device="cpu")
    model.eval()
    return model


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--module",
        default="all",
        choices=("all",) + MODULES,
        help="Which graph to export (default: all).",
    )
    p.add_argument("--checkpoint", default="third_party/EdgeTAM/checkpoints/edgetam.pt")
    p.add_argument(
        "--model-cfg",
        default="configs/edgetam.yaml",
        help="Hydra config name resolved by the sam2 package.",
    )
    p.add_argument("--outdir", default="models")
    p.add_argument("--batch", type=int, default=1, help="Batch size baked into the trace.")
    p.add_argument(
        "--static-batch",
        action="store_true",
        help="Pin the batch dimension. Slightly faster engines, but the tracker "
        "then falls back to PyTorch whenever the object count differs.",
    )
    p.add_argument(
        "--max-ptr-tokens",
        type=int,
        default=None,
        help="Object-pointer token capacity in the memory bank (default: "
        "4 * (max_obj_ptrs_in_encoder + 8) = 96 for EdgeTAM). Raise it if you "
        "prompt on many frames; frames that overflow fall back to PyTorch.",
    )
    p.add_argument(
        "--no-fuse-high-res-proj",
        action="store_true",
        help="Keep the mask decoder's conv_s0/conv_s1 out of the image encoder "
        "engine. Costs ~40 MB of extra engine output per frame; only useful if "
        "you consume the unprojected FPN levels yourself.",
    )
    p.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset. 18 is the lowest torch's exporter emits natively; "
        "TensorRT 10 supports >= 11.",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="Run each graph through onnxruntime (CPU) and assert parity with PyTorch.",
    )
    args = p.parse_args(argv)

    checkpoint = args.checkpoint or None
    if checkpoint and not Path(checkpoint).exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f">> Building EdgeTAM from {checkpoint or '<random weights>'}")
    model = build_model(checkpoint, args.model_cfg)
    layout = MemoryLayout.from_model(model, ptr_tokens=args.max_ptr_tokens)
    print(f">> Memory layout: {layout}")
    print(
        f"   memory sequence = {layout.num_slots} x {layout.tokens_per_slot} spatial "
        f"+ {layout.ptr_tokens} pointer = {layout.total_tokens} tokens"
    )

    wanted = MODULES if args.module == "all" else (args.module,)
    for name in wanted:
        print(f">> Exporting {name}")
        if name == "image_encoder":
            plan = _plan_image_encoder(model, args.batch, not args.no_fuse_high_res_proj)
        elif name == "memory_attention":
            plan = _plan_memory_attention(model, args.batch, layout)
        elif name == "memory_encoder":
            plan = _plan_memory_encoder(model, args.batch)
        else:
            plan = _plan_sam_head(model, args.batch)
        export_plan(plan, outdir, args.opset, not args.static_batch, args.verify)

    print(f">> Done. Next: python tools/build_trt_engines.py --outdir {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
