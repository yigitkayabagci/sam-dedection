#!/usr/bin/env python3
"""How much accuracy does reduced precision actually cost EdgeTAM?

Tracks a clip at full fp32, then re-tracks it with reduced-precision
activations, and reports mask IoU against the fp32 run -- per module, and per
frame. Runs on CPU with the real checkpoint; no GPU or TensorRT required, so
the answer does not have to wait for an engine build.

Why simulate rather than use `torch.autocast`
---------------------------------------------
On an Orin, a TensorRT fp16 layer reads and writes fp16 but accumulates its
matmuls in fp32 -- that is what the tensor cores do. Simulating that by
rounding each layer's *output* to the target dtype while accumulating in fp32
is therefore a closer model of the deployed engine than PyTorch's autocast,
which also lowers the accumulation. It is also ~90x faster than fp16 autocast
on a CPU, which has no half-precision SIMD to fall back on.

It is a pessimistic model in one direction: it rounds normalisation and
activation outputs too, which TensorRT often keeps in fp32. An fp16 result
that looks clean here is therefore safe, not optimistic.

INT8 is modelled the same way, with symmetric per-tensor activation
quantisation -- what TensorRT does for activations by default. Weights are
left alone (TensorRT quantises those per-channel, which is far more forgiving),
so these numbers are a *lower bound* on INT8 damage.

Usage:
    python tools/analyze_precision.py --frames 30
    python tools/analyze_precision.py --frames 30 --json outputs/precision.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prompts import BoxPrompt, PromptSet  # noqa: E402
from src.trackers.edgetam_tracker import EdgeTAMTracker  # noqa: E402

# The four blocks the TensorRT build splits EdgeTAM into. `sam_head` covers the
# prompt encoder and mask decoder; the rest map one-to-one onto engines.
MODULE_ATTRS = {
    "image_encoder": ("image_encoder",),
    "memory_attention": ("memory_attention",),
    "memory_encoder": ("memory_encoder", "spatial_perceiver"),
    "sam_head": ("sam_prompt_encoder", "sam_mask_decoder", "obj_ptr_proj"),
}


# --------------------------------------------------------------------------
# Precision simulation
# --------------------------------------------------------------------------


def round_to(dtype):
    def apply(t: torch.Tensor) -> torch.Tensor:
        return t.to(dtype).to(t.dtype)

    return apply


def fake_quant_int8(t: torch.Tensor) -> torch.Tensor:
    """Symmetric per-tensor int8, the activation scheme TensorRT defaults to."""
    scale = t.abs().amax()
    if scale == 0:
        return t
    step = scale / 127.0
    return torch.clamp(torch.round(t / step), -127, 127) * step


def _map_tensors(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value) if value.is_floating_point() else value
    if isinstance(value, (list, tuple)):
        mapped = [_map_tensors(v, fn) for v in value]
        return type(value)(mapped) if not isinstance(value, tuple) else tuple(mapped)
    if isinstance(value, dict):
        return {k: _map_tensors(v, fn) for k, v in value.items()}
    return value


class PrecisionSimulation:
    """Round the output of every leaf layer inside the chosen modules.

    Hooks leaf `nn.Module`s rather than intercepting every ATen op: layer
    boundaries are where a quantised engine actually changes precision, and
    hooking there sidesteps the aliasing hazards of rewriting the output of
    view and in-place operations.
    """

    def __init__(self, model, modules, transform):
        self.transform = transform
        self.handles = []
        self.roots = []
        for name in modules:
            for attr in MODULE_ATTRS[name]:
                root = getattr(model, attr, None)
                if root is not None:
                    self.roots.append(root)

    def __enter__(self):
        def hook(_module, _inputs, output):
            return _map_tensors(output, self.transform)

        for root in self.roots:
            for module in root.modules():
                if not list(module.children()):  # leaf
                    self.handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False


# --------------------------------------------------------------------------
# Clip + tracking
# --------------------------------------------------------------------------


def write_clip(outdir: Path, frames: int, size: int = 640):
    import cv2

    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    background = cv2.GaussianBlur(
        rng.integers(25, 95, size=(size, size, 3), dtype=np.uint8), (0, 0), 3
    )
    radius = size // 10
    for t in range(frames):
        frame = background.copy()
        phase = t / max(frames - 1, 1)
        cx = int(size * (0.25 + 0.45 * phase))
        cy = int(size * (0.5 + 0.18 * np.sin(2 * np.pi * phase)))
        cv2.circle(frame, (cx, cy), radius, (225, 70, 60), -1)
        cv2.circle(frame, (cx, cy), radius // 2, (250, 250, 250), 3)
        cv2.imwrite(str(outdir / f"{t:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cx0, cy0 = int(size * 0.25), int(size * 0.5)
    return PromptSet(
        boxes=[
            BoxPrompt(1, 0, (cx0 - radius, cy0 - radius, cx0 + radius, cy0 + radius))
        ]
    )


def track(model, frames_dir, prompts, simulation=None):
    tracker = EdgeTAMTracker(
        model_cfg="configs/edgetam.yaml", checkpoint="unused",
        device="cpu", precision="float32",
    )
    tracker._predictor = model
    masks = []
    context = simulation if simulation is not None else _null()
    with context:
        tracker.prepare(frames_dir)
        tracker.set_prompts(prompts)
        for result in tracker.propagate():
            masks.append({k: v.copy() for k, v in result.masks.items()})
    tracker.reset()
    return masks


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0  # both empty: perfect agreement, not undefined
    return float(np.logical_and(a, b).sum() / union)


def compare(reference, candidate):
    per_frame = []
    for ref_masks, got_masks in zip(reference, candidate):
        scores = [iou(ref_masks[k], got_masks[k]) for k in sorted(ref_masks)]
        per_frame.append(sum(scores) / len(scores))
    return per_frame


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", default="third_party/EdgeTAM/checkpoints/edgetam.pt")
    p.add_argument("--model-cfg", default="configs/edgetam.yaml")
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--clip-size", type=int, default=640)
    p.add_argument("--json", default=None)
    p.add_argument("--skip-int8", action="store_true")
    args = p.parse_args(argv)

    from sam2.build_sam import build_sam2_video_predictor

    if not Path(args.checkpoint).exists():
        raise SystemExit(
            f"{args.checkpoint} not found. Precision results with random weights "
            "would be meaningless -- run scripts/setup_edgetam.sh first."
        )
    model = build_sam2_video_predictor(args.model_cfg, args.checkpoint, device="cpu").eval()

    tmp = Path(tempfile.mkdtemp(prefix="precision_"))
    try:
        prompts = write_clip(tmp, args.frames, args.clip_size)
        all_modules = tuple(MODULE_ATTRS)

        print(f"tracking {args.frames} frames at fp32 (reference)...")
        reference = track(model, tmp, prompts)

        runs = [
            ("bf16  (all modules)", all_modules, round_to(torch.bfloat16)),
            ("fp16  (all modules)", all_modules, round_to(torch.float16)),
        ]
        if not args.skip_int8:
            runs.append(("int8  (all modules)", all_modules, fake_quant_int8))
            runs += [
                (f"int8  ({name} only)", (name,), fake_quant_int8)
                for name in all_modules
            ]

        results = {}
        for label, modules, transform in runs:
            print(f"tracking {args.frames} frames with {label}...")
            with PrecisionSimulation(model, modules, transform) as sim:
                masks = track(model, tmp, prompts, simulation=sim)
            results[label] = compare(reference, masks)

        print(f"\n{'configuration':<26}{'mean IoU':>10}{'worst':>9}{'last':>9}"
              f"{'frames <0.99':>14}")
        print("-" * 68)
        for label, per_frame in results.items():
            arr = np.array(per_frame)
            print(f"{label:<26}{arr.mean():>10.4f}{arr.min():>9.4f}{arr[-1]:>9.4f}"
                  f"{int((arr < 0.99).sum()):>10} /{len(arr):>3}")
        print("-" * 68)
        print("IoU is against the fp32 run, so 1.0000 means pixel-identical masks.")

        # Drift over time separates a one-off rounding error from one that the
        # memory bank feeds back into itself.
        print(f"\nIoU vs frame index (every 5th frame)")
        idx = list(range(0, len(reference), 5))
        header = "".join(f"{i + 1:>8}" for i in idx)
        print(f"{'configuration':<26}{header}")
        for label, per_frame in results.items():
            row = "".join(f"{per_frame[i]:>8.4f}" for i in idx)
            print(f"{label:<26}{row}")

        if args.json:
            out = Path(args.json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(
                {"frames": args.frames, "per_frame_iou": results}, indent=2) + "\n")
            print(f"\nwrote {out}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
