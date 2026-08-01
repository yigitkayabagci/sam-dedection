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
    """Symmetric per-tensor int8 scaled to the largest value present.

    Per-tensor is not a simplification: TensorRT quantises *activations* per
    tensor (per-channel applies to weights), so this is the scheme a real INT8
    engine uses. Scaling to absmax is the naive choice — see
    `fake_quant_int8_percentile` for what calibration buys.
    """
    scale = t.abs().amax()
    if scale == 0:
        return t
    step = scale / 127.0
    return torch.clamp(torch.round(t / step), -127, 127) * step


# torch.quantile's input-size ceiling; tensors above this are strided down.
_QUANTILE_LIMIT = 1_000_000


def fake_quant_int8_percentile(percentile: float = 0.999):
    """Symmetric per-tensor int8 with the scale clipped to a percentile.

    Stands in for TensorRT's entropy/percentile calibration: outliers are
    clipped rather than allowed to stretch the scale and starve the bulk of
    the distribution of resolution. If a module survives this but not absmax,
    its problem is calibration; if it survives neither, its problem is
    structural.
    """

    def apply(t: torch.Tensor) -> torch.Tensor:
        flat = t.detach().abs().flatten()
        if flat.numel() == 0:
            return t
        # torch.quantile caps out around 16M elements, so large tensors have to
        # be subsampled. Stride rather than sample randomly: a random subsample
        # makes the whole run non-deterministic, and a quantile estimated from
        # a different draw each frame is a moving target on top of the effect
        # being measured.
        if flat.numel() > _QUANTILE_LIMIT:
            flat = flat[:: (flat.numel() // _QUANTILE_LIMIT) + 1]
        scale = torch.quantile(flat.float(), percentile)
        if scale == 0:
            scale = flat.max()
        if scale == 0:
            return t
        step = scale / 127.0
        return torch.clamp(torch.round(t / step), -127, 127) * step

    return apply


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
    """Round the output of layers inside the chosen modules.

    Hooks `nn.Module`s rather than intercepting every ATen op: layer boundaries
    are where a quantised engine actually changes precision, and hooking there
    sidesteps the aliasing hazards of rewriting the output of view and in-place
    operations.

    Where to place the rounding is a real modelling choice, so both are offered:

    `compute`  only convolutions and linear layers. This is the faithful model
               of a TensorRT INT8 network: those are the layers with INT8
               kernels, batch norm is folded into their weights, and
               activations fuse into them — one quantisation point per
               fused block, which is also where QAT toolkits insert Q/DQ.

    `leaf`     every leaf module, normalisations and activations included.
               Deliberately pessimistic: a conv/BN/GELU block gets quantised
               three times instead of once.

    Reporting both is the point. A conclusion that survives the faithful model
    and the pessimistic one does not depend on where the rounding was put.
    """

    COMPUTE_LAYERS = (torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Linear,
                      torch.nn.ConvTranspose2d)

    def __init__(self, model, modules, transform, placement: str = "leaf"):
        if placement not in ("leaf", "compute"):
            raise ValueError(f"placement must be 'leaf' or 'compute', got {placement}")
        self.transform = transform
        self.placement = placement
        self.handles = []
        self.roots = []
        for name in modules:
            for attr in MODULE_ATTRS[name]:
                root = getattr(model, attr, None)
                if root is not None:
                    self.roots.append(root)

    def _selected(self, module) -> bool:
        if self.placement == "compute":
            return isinstance(module, self.COMPUTE_LAYERS)
        return not list(module.children())

    def __enter__(self):
        def hook(_module, _inputs, output):
            return _map_tensors(output, self.transform)

        for root in self.roots:
            for module in root.modules():
                if self._selected(module):
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


def track(model, frames_dir, prompts, simulation=None, engines=None):
    """Track a clip, optionally through the TensorRT integration path.

    `engines` routes the four hot modules through `tools/reference_engines.py`
    -- the same wrapper graphs the ONNX is exported from, behind the same
    runtime API the real engines use. That isolates the *rewrite* from
    TensorRT and from fp16: if masks match here, anything that goes wrong on
    the Orin is in the engine build or the precision, not in this code.
    """
    if engines is not None:
        from src.trackers.edgetam_trt_tracker import EdgeTAMTRTTracker

        tracker = EdgeTAMTRTTracker(
            model_cfg="configs/edgetam.yaml", checkpoint="unused",
            device="cpu", precision="float32", strict=True,
        )
        tracker._predictor = model
        tracker._engines = engines
    else:
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
    p.add_argument("--only", default=None,
                   help="Run only configurations whose label contains this string "
                        "(the fp32 reference always runs).")
    p.add_argument("--int8-percentile", type=float, default=0.999,
                   help="Clipping percentile for the calibrated INT8 variant.")
    p.add_argument("--skip-precision", action="store_true",
                   help="Only verify the graph rewrite, skip the precision sweep.")
    p.add_argument("--verify-graphs", action="store_true",
                   help="Also track through tools/reference_engines.py at fp32, "
                        "which checks the TensorRT rewrite itself against stock "
                        "EdgeTAM on this model at this resolution.")
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

        results = {}

        if args.verify_graphs:
            from sam2.build_sam import build_sam2_video_predictor as _build
            from tools.reference_engines import build_reference_engines

            print("tracking through the TensorRT graph rewrite at fp32...")
            # A separate instance: a real engine is a frozen export-time
            # snapshot, so the graphs must not share the modules being patched.
            snapshot = _build(args.model_cfg, args.checkpoint, device="cpu").eval()
            engines = build_reference_engines(snapshot, batch=1)
            masks = track(model, tmp, prompts, engines=engines)
            results["graph rewrite (fp32)"] = compare(reference, masks)

        runs = [] if args.skip_precision else [
            ("bf16  (all modules)", all_modules, round_to(torch.bfloat16), "leaf"),
            ("fp16  (all modules)", all_modules, round_to(torch.float16), "leaf"),
        ]
        if not args.skip_int8 and not args.skip_precision:
            runs.append(("int8  (all modules)", all_modules, fake_quant_int8, "leaf"))
            runs += [
                (f"int8  ({name} only)", (name,), fake_quant_int8, "leaf")
                for name in all_modules
            ]
            # Does calibration rescue whatever absmax broke, or is it structural?
            calib = fake_quant_int8_percentile(args.int8_percentile)
            runs.append(("int8+calib  (all modules)", all_modules, calib, "leaf"))
            runs += [
                (f"int8+calib  ({name} only)", (name,), calib, "leaf")
                for name in all_modules
            ]
            # And does the conclusion survive the faithful fusion model?
            runs.append(("int8 fused  (all modules)", all_modules, fake_quant_int8, "compute"))
            runs += [
                (f"int8 fused  ({name} only)", (name,), fake_quant_int8, "compute")
                for name in all_modules
            ]

        if args.only:
            runs = [r for r in runs if args.only in r[0]]
            if not runs:
                raise SystemExit(f"--only {args.only!r} matched no configuration.")

        for label, modules, transform, placement in runs:
            print(f"tracking {args.frames} frames with {label} [{placement}]...")
            with PrecisionSimulation(model, modules, transform, placement) as sim:
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
