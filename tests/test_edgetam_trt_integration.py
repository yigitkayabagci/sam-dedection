"""End-to-end test of the TensorRT-patched tracker, without TensorRT.

The wrappers in `tools/edgetam_graphs.py` are proven correct in isolation by
`test_edgetam_graphs.py`. What is left to get wrong is the *integration*:
padding the memory bank into fixed slots in the right order, building the mask
that matches it, keeping the object-pointer region where the RoPE split expects
it, and copying out the outputs that outlive a frame instead of handing back a
buffer the next frame overwrites.

So this substitutes `reference_engines.py`, which runs the PyTorch
wrapper behind the exact `TRTEngine` surface the tracker uses -- including
persistent output buffers that get overwritten on every call, which is what
turns a missing `clone` from an invisible aliasing bug into a failing
assertion here.

The reference is stock `EdgeTAMTracker` on identical weights and frames: the
patched tracker has to produce the same masks, frame for frame.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sam2", reason="EdgeTAM not installed (scripts/setup_edgetam.sh)")
cv2 = pytest.importorskip("cv2")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prompts import BoxPrompt, PromptSet  # noqa: E402
from src.trackers.edgetam_tracker import EdgeTAMTracker  # noqa: E402
from src.trackers.edgetam_trt_tracker import EdgeTAMTRTTracker  # noqa: E402
from tests.reference_engines import build_reference_engines  # noqa: E402

# Same miniature EdgeTAM as test_edgetam_graphs, so a full propagation run
# stays inside a few seconds on CPU.
SMALL_OVERRIDES = [
    "++model.image_size=256",
    "++model.memory_attention.layer.cross_attention.q_sizes=[16,16]",
    "++model.memory_attention.layer.cross_attention.k_sizes=[4,4]",
    "++model.spatial_perceiver.num_latents=8",
    "++model.spatial_perceiver.num_latents_2d=16",
    "++model.num_maskmem=3",
]
PTR_TOKENS = 40


def build_model():
    from sam2.build_sam import build_sam2_video_predictor

    # Same seed every call: the two trackers must compare on identical weights.
    torch.manual_seed(0)
    model = build_sam2_video_predictor(
        "configs/edgetam.yaml", None, device="cpu", hydra_overrides_extra=SMALL_OVERRIDES
    )
    return model.eval()


# --------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def frames_dir(tmp_path_factory):
    """A short synthetic clip: two blobs drifting across a textured background."""
    out = tmp_path_factory.mktemp("frames")
    rng = np.random.default_rng(0)
    background = rng.integers(0, 60, size=(120, 160, 3), dtype=np.uint8)
    for idx in range(10):
        frame = background.copy()
        cv2.rectangle(frame, (20 + 4 * idx, 30), (60 + 4 * idx, 70), (240, 40, 40), -1)
        cv2.circle(frame, (120 - 3 * idx, 80), 14, (40, 240, 90), -1)
        cv2.imwrite(str(out / f"{idx:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return out


def run_tracker(tracker, frames_dir, prompts):
    tracker.prepare(frames_dir)
    tracker.set_prompts(prompts)
    masks = [(r.frame_idx, {k: v.copy() for k, v in r.masks.items()}) for r in tracker.propagate()]
    tracker.reset()
    return masks


def make_trt_tracker(model, batch, **kwargs):
    tracker = EdgeTAMTRTTracker(
        model_cfg="configs/edgetam.yaml",
        checkpoint="unused",
        device="cpu",
        precision="float32",
        strict=True,  # any fallback becomes an error, so the engines really run
        **kwargs,
    )
    tracker._predictor = model
    # build_model() is seeded, so this snapshot carries identical weights. It
    # must be a separate instance: a real engine is frozen at export time, and
    # sharing live modules would have a graph call back into its patched self.
    tracker._engines = build_reference_engines(
        build_model(), batch=batch, ptr_tokens=PTR_TOKENS
    )
    return tracker


BOXES = {
    1: (20.0, 30.0, 60.0, 70.0),
    2: (106.0, 66.0, 134.0, 94.0),
}


def _prompts(num_objects):
    return PromptSet(
        boxes=[
            BoxPrompt(obj_id=obj_id, frame_idx=0, xyxy=BOXES[obj_id])
            for obj_id in list(BOXES)[:num_objects]
        ]
    )


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


@pytest.mark.parametrize("num_objects", [1, 2])
def test_patched_tracker_matches_pytorch(frames_dir, num_objects):
    """The whole patched pipeline reproduces stock EdgeTAM, frame for frame.

    Two objects matters as its own case: EdgeTAM broadcasts one frame's
    features across objects with `expand()`, so the tracker feeds the engines
    stride-0 views and the memory bank carries a real batch dimension.
    """
    prompts = _prompts(num_objects)

    # `_ensure_predictor` is a no-op once `_predictor` is set, so this uses the
    # injected model rather than trying to load a checkpoint.
    reference_tracker = EdgeTAMTracker(
        model_cfg="configs/edgetam.yaml",
        checkpoint="unused",
        device="cpu",
        precision="float32",
    )
    reference_tracker._predictor = build_model()
    reference = run_tracker(reference_tracker, frames_dir, prompts)

    patched = run_tracker(make_trt_tracker(build_model(), num_objects), frames_dir, prompts)

    assert len(patched) == len(reference) == 10
    for (ref_idx, ref_masks), (got_idx, got_masks) in zip(reference, patched):
        assert ref_idx == got_idx
        assert sorted(ref_masks) == sorted(got_masks) == sorted(list(BOXES)[:num_objects])
        for obj_id, ref_mask in ref_masks.items():
            got_mask = got_masks[obj_id]
            disagreement = int(np.logical_xor(ref_mask, got_mask).sum())
            assert disagreement == 0, (
                f"frame {ref_idx} object {obj_id}: {disagreement} pixels differ "
                f"from stock EdgeTAM"
            )


def test_patched_tracker_survives_a_second_video(frames_dir):
    """reset() then prepare() again must not leak state through the buffers."""
    tracker = make_trt_tracker(build_model(), batch=1)
    prompts = _prompts(1)
    first = run_tracker(tracker, frames_dir, prompts)
    second = run_tracker(tracker, frames_dir, prompts)
    for (_, a), (_, b) in zip(first, second):
        for obj_id in a:
            np.testing.assert_array_equal(a[obj_id], b[obj_id])


def test_memory_attention_falls_back_when_pointers_overflow(frames_dir):
    """A pointer count past the reserved capacity must not silently corrupt.

    With strict=False it drops to PyTorch; with strict=True it raises. Either
    way it must never pad past the region the graph's RoPE split expects.
    """
    model = build_model()
    tracker = make_trt_tracker(model, batch=1)
    tracker.prepare(frames_dir)
    tracker.set_prompts(_prompts(1))

    from src.trackers._trt_layout import MemoryLayout

    layout = MemoryLayout.from_model_and_engine(
        model, tracker._engines["memory_attention"].engine_shape("memory")[1]
    )
    forward = model.memory_attention.forward
    curr = torch.randn(
        (model.image_size // model.backbone_stride) ** 2, 1, model.hidden_dim
    )
    too_many = layout.ptr_tokens + 4
    kwargs = dict(
        curr=[curr],
        curr_pos=[torch.randn_like(curr)],
        memory=torch.randn(layout.tokens_per_slot + too_many, 1, model.mem_dim),
        memory_pos=torch.randn(layout.tokens_per_slot + too_many, 1, model.mem_dim),
        num_obj_ptr_tokens=too_many,
        num_spatial_mem=1,
    )
    with pytest.raises(RuntimeError, match="object-pointer tokens"):
        forward(**kwargs)

    tracker.strict = False
    with torch.no_grad():
        out = forward(**kwargs)
    assert out.shape == curr.shape  # PyTorch fallback, sequence-first as usual


def test_fill_hole_policy_disables_when_extension_missing(frames_dir):
    model = build_model()
    model.fill_hole_area = 8
    tracker = make_trt_tracker(model, batch=1)
    tracker._apply_fill_hole_policy()
    from src.trackers.edgetam_trt_tracker import _connected_components_available

    expected = 8 if _connected_components_available() else 0
    assert model.fill_hole_area == expected

    explicit = make_trt_tracker(build_model(), batch=1, fill_hole_area=4)
    explicit._apply_fill_hole_policy()
    assert explicit._predictor.fill_hole_area == 4
