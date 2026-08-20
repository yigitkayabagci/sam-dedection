#!/usr/bin/env python3
"""INT8 for EdgeTAM: capture real calibration data, then quantise per module.

The headline finding this tool is built around is already measured and written
down in `docs/tensorrt_fp16.md`: **INT8 damage is not uniform across EdgeTAM,
and the right calibrator is not the same for every module.**

| module | share of FLOPs | absmax (TensorRT's default) | percentile-clipped |
|---|---:|---:|---:|
| memory attention | **61.9 %** | 0.9998 | 0.9998 |
| memory encoder | 8.8 % | **0.9999** | 0.9397 |
| SAM head | 2.5 % | 0.9993 | 0.9990 |
| image encoder | 26.8 % | **0.8170** | 0.9936 |

Read the two bold entries together and the whole strategy follows:

* The **image encoder collapses under min/max** -- a handful of outliers stretch
  the scale until the bulk of the distribution has no resolution left -- and is
  recovered by clipping. It gets **entropy** calibration.
* The **memory encoder is the opposite**: clipping *hurts* it, badly, and the
  shape of the failure says why. It is the write port of the recurrent loop, so
  a clipping bias does not scatter like rounding does -- it accumulates in the
  memory bank and is read back for the next seven frames. It gets **max**.
* The **memory attention is 62 % of the frame's arithmetic and barely notices
  INT8 either way.** That is the best ratio in the model, and it is the reason
  this is worth doing at all.

So the default plan quantises every module, with a per-module calibrator chosen
from that table, and each engine can still be dropped back to fp16 on its own
because the tracker treats every engine as independent and optional.

Three steps, runnable separately:

    capture   track a real clip with recording engines, writing one .npz of
              real inputs per module. Needs no TensorRT and no INT8 engine --
              which matters, because calibration has to happen before one
              exists.
    quantize  NVIDIA Model Optimizer PTQ on each ONNX graph -> Q/DQ ONNX.
    plan      print the plan and the trtexec commands, change nothing.

Usage:
    python tools/quantize_edgetam.py plan
    python tools/quantize_edgetam.py capture --outdir models512/ --calib outputs/calib/
    python tools/quantize_edgetam.py quantize --outdir models512/ --calib outputs/calib/
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULES = ("image_encoder", "memory_attention", "memory_encoder", "sam_head")


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModulePlan:
    """What to do with one module, and the measurement that says so."""

    name: str
    quantize: bool
    calibration: str  # "entropy" | "max"
    flops_share: float
    reason: str

    def row(self) -> str:
        what = f"INT8 / {self.calibration}" if self.quantize else "fp16 (kept)"
        return f"| `{self.name}` | {self.flops_share:.1%} | {what} | {self.reason} |"


DEFAULT_PLAN = {
    "memory_attention": ModulePlan(
        "memory_attention", True, "max", 0.619,
        "62% of the frame's FLOPs and 0.9998 IoU under the harshest simulated "
        "INT8. The best ratio in the model."),
    "memory_encoder": ModulePlan(
        "memory_encoder", True, "max", 0.088,
        "0.9999 with absmax but 0.9397 when clipped: it writes the memory bank, "
        "so a clipping bias accumulates instead of scattering. Never entropy."),
    "sam_head": ModulePlan(
        "sam_head", True, "max", 0.025,
        "0.9993, and small enough that the choice barely matters."),
    "image_encoder": ModulePlan(
        "image_encoder", True, "entropy", 0.268,
        "min/max collapses it to 0.8170 -- outliers stretch the scale until the "
        "bulk of the distribution has no resolution. Clipping recovers 0.9936."),
}


def plan_table(plan: dict[str, ModulePlan]) -> str:
    lines = ["| module | share of FLOPs | precision | why |", "|---|---:|---|---|"]
    lines += [plan[name].row() for name in MODULES if name in plan]
    covered = sum(p.flops_share for p in plan.values() if p.quantize)
    lines += ["", f"INT8 covers **{covered:.1%}** of the per-frame arithmetic."]
    return "\n".join(lines)


def apply_overrides(plan: dict[str, ModulePlan], keep_fp16: list[str],
                    calibration: dict[str, str]) -> dict[str, ModulePlan]:
    """Drop modules back to fp16, or change a calibrator, from the CLI.

    This is the knob the accuracy gate turns: a module that fails its check
    goes back to fp16 on its own, and the other three keep their speedup.
    """
    from dataclasses import replace

    out = dict(plan)
    for name in keep_fp16:
        if name not in out:
            raise KeyError(f"unknown module {name!r}; pick from {MODULES}")
        out[name] = replace(out[name], quantize=False)
    for name, method in calibration.items():
        if method not in ("entropy", "max"):
            raise ValueError(f"{name}: calibration must be 'entropy' or 'max'")
        out[name] = replace(out[name], calibration=method)
    return out


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------


def capture(args) -> int:
    """Track a clip through PyTorch stand-ins for the engines, keeping inputs.

    The stand-ins are `tests/reference_engines.py` -- the same wrapper graphs
    the ONNX is exported from, behind the same runtime API. So the recorded
    tensors are shaped and packed exactly as the real engine receives them,
    with no TensorRT anywhere in the loop.
    """
    from sam2.build_sam import build_sam2_video_predictor

    from src.prompts import BoxPrompt, PromptSet
    from src.trackers._hydra_overrides import image_size_overrides
    from src.trackers.calibration import wrap_engines
    from src.trackers.edgetam_trt_tracker import EdgeTAMTRTTracker
    from src.training.antiuav import list_sequences
    from tests.reference_engines import build_reference_engines

    overrides = image_size_overrides(args.image_size)
    build = lambda: build_sam2_video_predictor(  # noqa: E731
        args.model_cfg, args.checkpoint, device=args.device,
        hydra_overrides_extra=overrides).eval()

    tracker = EdgeTAMTRTTracker(
        model_cfg=args.model_cfg, checkpoint=args.checkpoint, device=args.device,
        precision="float32", strict=True, image_size=args.image_size,
    )
    tracker._predictor = build()
    # A real engine is a frozen export-time snapshot, so the graphs must not
    # share live modules with the predictor they are patched into.
    tracker._engines = build_reference_engines(build(), batch=1)
    stores = wrap_engines(tracker._engines, args.limit, args.stride)
    tracker._patch()

    calib = Path(args.calib)
    sequences = list_sequences(args.data, args.split)[:args.sequences]
    if not sequences:
        raise SystemExit(f"no sequences under {args.data}/{args.split}")

    for sequence in sequences:
        visible = sequence.labels.visible_indices()
        start = int(visible[0])
        box = sequence.labels.boxes[start]
        print(f">> {sequence.name}: {min(args.frames, len(sequence))} frames")
        tracker.prepare(sequence.frames[0].parent)
        tracker.set_prompts(PromptSet(boxes=[
            BoxPrompt(1, start, tuple(float(v) for v in box))
        ]))
        for index, _ in enumerate(tracker.propagate()):
            if index >= args.frames:
                break
        tracker.reset()
        if all(store.full for store in stores.values()):
            print(">> every store is full; stopping early")
            break

    calib.mkdir(parents=True, exist_ok=True)
    for name, store in stores.items():
        store.save(calib / f"{name}.npz")
        print(f"{name:<18} {store.summary()}")
    print(f"\nwrote {len(stores)} calibration sets -> {calib}")
    return 0


# --------------------------------------------------------------------------
# Quantise
# --------------------------------------------------------------------------


def _calibration_arrays(path: Path, spec: dict) -> dict[str, np.ndarray]:
    """Load a capture and check it against the graph it will calibrate."""
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    expected = {io["name"]: tuple(io["shape"]) for io in spec["inputs"]}

    missing = set(expected) - set(arrays)
    if missing:
        raise ValueError(f"{path}: no calibration data for input(s) {sorted(missing)}")
    for name, shape in expected.items():
        axis = next(io["batch_axis"] for io in spec["inputs"] if io["name"] == name)
        got = list(arrays[name].shape)
        want = list(shape)
        got[axis] = want[axis] = -1
        if got != want:
            raise ValueError(
                f"{path}: {name} was captured at {arrays[name].shape} but the graph "
                f"expects {shape}. The capture ran at a different image size."
            )
    return {name: arrays[name] for name in expected}


def quantize_module(onnx_path: Path, calib_path: Path, plan: ModulePlan,
                    out_path: Path, dry_run: bool = False) -> None:
    """One graph through Model Optimizer's ONNX PTQ, producing a Q/DQ graph.

    Everything version-sensitive about `nvidia-modelopt` is in this function
    and nowhere else, so a change in its API costs one edit here rather than a
    rewrite. `--dry-run` exercises the whole pipeline without it installed,
    including the shape checks, which is where most mistakes actually are.
    """
    spec = json.loads(onnx_path.with_suffix(".spec.json").read_text())
    arrays = _calibration_arrays(calib_path, spec)
    print(f"  {plan.name}: {plan.calibration} calibration over "
          f"{min(len(a) for a in arrays.values())} samples")
    if dry_run:
        print(f"  (dry run) would write {out_path.name}")
        return

    from modelopt.onnx.quantization import quantize

    quantize(
        onnx_path=str(onnx_path),
        quantize_mode="int8",
        calibration_data=arrays,
        calibration_method=plan.calibration,
        output_path=str(out_path),
    )


def write_spec(source: Path, destination: Path, precision: str) -> None:
    """Copy a graph's spec next to it, recording the precision to build it at.

    Carrying the choice in the spec is what lets a *mixed* set build in one
    command. The alternative -- a global `--precision int8` -- would put the
    modules held back at fp16 into a strongly-typed build, where "no Q/DQ
    nodes" means fp32, not fp16, and they would quietly get slower.
    """
    spec = json.loads(source.read_text())
    spec["precision"] = precision
    destination.write_text(json.dumps(spec, indent=2) + "\n")


def run_quantize(args) -> int:
    plan = apply_overrides(DEFAULT_PLAN, args.fp16,
                           dict(pair.split("=", 1) for pair in args.calibration))
    outdir, calib = Path(args.outdir), Path(args.calib)
    target = Path(args.qdq_outdir)
    target.mkdir(parents=True, exist_ok=True)
    print(plan_table(plan) + "\n")

    quantised, kept = [], []
    for name in MODULES:
        source = outdir / f"edgetam_{name}.onnx"
        if not source.exists():
            raise SystemExit(f"{source} not found — run export_edgetam_onnx.py first.")
        destination = target / source.name
        module = plan.get(name)

        if module is None or not module.quantize:
            # Copied rather than left behind, so the output directory is a
            # complete engine set that builds and runs on its own.
            shutil.copy(source, destination)
            write_spec(source.with_suffix(".spec.json"),
                       destination.with_suffix(".spec.json"), "fp16")
            kept.append(name)
            continue

        calib_path = calib / f"{name}.npz"
        if not calib_path.exists():
            raise SystemExit(f"{calib_path} not found — run `capture` first.")
        quantize_module(source, calib_path, module, destination, args.dry_run)
        write_spec(source.with_suffix(".spec.json"),
                   destination.with_suffix(".spec.json"), "int8")
        quantised.append(name)

    (target / "quantization_plan.json").write_text(json.dumps(
        {name: {"precision": "int8" if plan[name].quantize else "fp16",
                "calibration": plan[name].calibration,
                "flops_share": plan[name].flops_share,
                "reason": plan[name].reason}
         for name in MODULES}, indent=2) + "\n")

    print(f"\nINT8: {', '.join(quantised) or 'none'}")
    print(f"fp16: {', '.join(kept) or 'none'}")
    print(f"-> {target}")
    print("\nBuild on the Orin, then gate each module against fp16:\n")
    print(f"  python tools/build_trt_engines.py --outdir {target} "
          "--precision auto --max-batch 4")
    print(f"  python tools/check_trt_parity.py  --outdir {target} "
          f"--image-size {args.image_size}")
    print("  python tools/eval_antiuav.py --data <anti-uav410> --split val \\\n"
          "      --tracker edgetam_trt --config configs/edgetam_trt_int8_512.yaml")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)

    show = sub.add_parser("plan", help="Print the per-module plan and stop.")
    show.add_argument("--fp16", nargs="*", default=[], metavar="MODULE")
    show.add_argument("--calibration", nargs="*", default=[], metavar="MODULE=METHOD")

    cap = sub.add_parser("capture", help="Record real engine inputs from a tracking run.")
    cap.add_argument("--data", required=True, help="Anti-UAV410 root.")
    cap.add_argument("--split", default="train")
    cap.add_argument("--calib", default="outputs/calib")
    cap.add_argument("--checkpoint", default="checkpoints/edgetam_lora_512.pt")
    cap.add_argument("--model-cfg", default="configs/edgetam.yaml")
    cap.add_argument("--image-size", type=int, default=512)
    cap.add_argument("--device", default="cpu")
    cap.add_argument("--sequences", type=int, default=8)
    cap.add_argument("--frames", type=int, default=64,
                     help="Frames per sequence. Enough to fill the memory bank "
                          "and then some -- the first ~16 frames are a transient.")
    cap.add_argument("--limit", type=int, default=512,
                     help="Samples per tensor. NVIDIA suggests >=500 for vision models.")
    cap.add_argument("--stride", type=int, default=2,
                     help="Sample every Nth call rather than the first N frames.")

    quant = sub.add_parser("quantize", help="Model Optimizer PTQ on each ONNX graph.")
    quant.add_argument("--outdir", default="models512", help="Where the .onnx graphs are.")
    quant.add_argument("--qdq-outdir", default="models512_int8",
                       help="Where the quantised set goes. A complete engine set: "
                            "modules held at fp16 are copied in unchanged, so one "
                            "build command produces the whole mixed configuration.")
    quant.add_argument("--calib", default="outputs/calib")
    quant.add_argument("--image-size", type=int, default=512)
    quant.add_argument("--fp16", nargs="*", default=[], metavar="MODULE",
                       help="Leave these modules at fp16 (use after a failed gate).")
    quant.add_argument("--calibration", nargs="*", default=[], metavar="MODULE=METHOD",
                       help="Override a calibrator, e.g. image_encoder=max.")
    quant.add_argument("--dry-run", action="store_true",
                       help="Check shapes and print the plan without calling modelopt.")

    args = p.parse_args(argv)
    if args.command == "plan":
        print(plan_table(apply_overrides(
            DEFAULT_PLAN, args.fp16,
            dict(pair.split("=", 1) for pair in args.calibration))))
        return 0
    return capture(args) if args.command == "capture" else run_quantize(args)


if __name__ == "__main__":
    raise SystemExit(main())
