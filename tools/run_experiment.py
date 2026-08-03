#!/usr/bin/env python3
"""Run the whole measurement suite once and collect it in one folder.

Four separate tools, each with its own flags, produced files scattered across
`outputs/` with no record of which run they came from. This runs all of them
against one set of engines, writes every artifact into a single directory, and
writes a SUMMARY.md pulling the headline numbers together.

The four steps answer four different questions, and the summary keeps them
labelled so they are not confused with each other:

  1. parity    Does each TensorRT engine reproduce its PyTorch module?
  2. accuracy  Over a whole clip, with memory feedback, do the masks change?
  3. speed     Model-only ms/frame, PyTorch vs TensorRT, plus per-module split.
  4. video     End-to-end app ms/frame (model + frame decode + mask overlay),
               with a watchable .mp4.

Steps 3 and 4 report different numbers on purpose, and both belong in a report:
step 3 is what TensorRT changed, step 4 is what a user of the app experiences.
Neither one alone is "the" frame time.

Usage:
    python tools/run_experiment.py --outdir results/1024
    python tools/run_experiment.py --outdir results/512 --engines models512/ \\
        --config configs/edgetam_trt_512.yaml
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PY = sys.executable


def run_step(name: str, cmd: list[str], log: Path) -> tuple[bool, str]:
    """Run one tool, tee its output to `log`, return (ok, captured text)."""
    print(f"\n{'=' * 70}\n>> {name}\n{'=' * 70}")
    print("   " + " ".join(str(c) for c in cmd))
    started = time.time()
    proc = subprocess.run(
        [str(c) for c in cmd], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    elapsed = time.time() - started
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(proc.stdout)
    print(proc.stdout)
    ok = proc.returncode == 0
    print(f"   [{'ok' if ok else f'FAILED (exit {proc.returncode})'}] {elapsed:.0f}s "
          f"-> {log.name}")
    return ok, proc.stdout


def find(pattern: str, text: str, group: int = 1) -> str | None:
    m = re.search(pattern, text)
    return m.group(group).strip() if m else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--outdir", required=True, help="Directory to collect results into.")
    p.add_argument("--engines", default="models", help="Directory holding the .engine files.")
    p.add_argument("--config", default="configs/edgetam_trt.yaml",
                   help="TensorRT backend YAML.")
    p.add_argument("--baseline-config", default="configs/edgetam.yaml",
                   help="Stock PyTorch backend YAML, for the speed comparison. "
                        "Ignored with --trt-only.")
    p.add_argument("--trt-only", action="store_true",
                   help="Measure the TensorRT backend alone: skip the stock PyTorch "
                        "run in the speed step. Halves the runtime, and is the right "
                        "choice when you already have the PyTorch number and are "
                        "comparing two TensorRT configurations against each other.")
    p.add_argument("--reference-config", default=None,
                   help="What the accuracy step compares against (default: "
                        "--baseline-config). Point this at another TensorRT YAML to "
                        "ask a different question: with configs/edgetam_trt.yaml as "
                        "the reference and a 512 config as --config, the mask IoU "
                        "answers 'what did dropping to 512 cost?' rather than 'did "
                        "TensorRT change anything?'. Both models emit masks at the "
                        "source frame resolution, so the two are directly comparable.")
    p.add_argument("--frames", type=int, default=500)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--radius", type=int, default=12,
                   help="Target radius for the video clip, px (default 12: a small, "
                        "hard-to-track object).")
    p.add_argument("--size", default="1280x720", help="Source frame size, WxH.")
    p.add_argument("--image-size", type=int, default=None,
                   help="Model input resolution, for the parity step's PyTorch "
                        "reference. Must match how the engines were exported. The "
                        "tracking steps read it from the YAMLs instead.")
    p.add_argument("--skip", default="", help="Comma-separated steps to skip: "
                                              "parity,accuracy,speed,video")
    args = p.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    results: dict[str, str] = {}
    status: dict[str, bool] = {}

    # A reference that is itself a TensorRT config means the accuracy step is
    # comparing two engine sets, so it has to run the edgetam_trt backend on
    # both sides rather than stock PyTorch on the reference side.
    reference_config = args.reference_config or args.baseline_config
    reference_is_trt = "trt" in Path(reference_config).stem
    reference_tracker = "edgetam_trt" if reference_is_trt else "edgetam"
    accuracy_question = (
        f"what did this config cost against `{Path(reference_config).name}`?"
        if reference_is_trt
        else "did TensorRT change the masks?"
    )

    # ---------------------------------------------------------------- parity
    if "parity" not in skip:
        cmd = [PY, "tools/check_trt_parity.py", "--outdir", args.engines]
        if args.image_size:
            cmd += ["--image-size", args.image_size]
        ok, out = run_step(
            "1/4  parity — does each engine reproduce its module?",
            cmd,
            outdir / "01_parity.txt",
        )
        status["parity"] = ok
        results["parity_total"] = find(
            r"total\s+([\d.]+ ms ->\s+[\d.]+ ms\s+\([\d.]+x\))", out
        ) or "not found"

    # -------------------------------------------------------------- accuracy
    if "accuracy" not in skip:
        cmd = [PY, "tools/compare_backends.py", "--frames", args.frames,
               "--offload-video",
               "--candidate-config", args.config,
               "--reference", reference_tracker,
               "--reference-config", reference_config,
               "--json", outdir / "02_accuracy.json"]
        if not reference_is_trt:
            # fp32 PyTorch is the strictest ground truth available. Against a
            # TensorRT reference there is nothing to force -- the engines are
            # built at whatever precision they were built at.
            cmd += ["--reference-precision", "float32"]
        ok, out = run_step(
            f"2/4  accuracy — {accuracy_question}",
            cmd,
            outdir / "02_accuracy.txt",
        )
        status["accuracy"] = ok
        results["mean_iou"] = find(r"mean IoU\s+([\d.]+)", out) or "not found"
        results["min_iou"] = find(r"min\s+IoU\s+([\d.]+)", out) or "not found"
        results["below"] = find(r"below [\d.]+\s+(\d+ of \d+)", out) or "not found"
        results["trend"] = find(r"trend\s+([-+\d.]+ IoU per 1000 frames)", out) or "n/a"

    # ----------------------------------------------------------------- speed
    if "speed" not in skip:
        # Same --radius as the video step: without it these two measure the
        # same model on different clips, and the numbers stop being comparable.
        cmd = [PY, "tools/benchmark_tracking.py", "--frames", args.frames,
               "--warmup", args.warmup, "--size", args.size, "--offload-video",
               "--radius", args.radius,
               "--config", args.config,
               "--fps-chart", outdir / "03_speed.png"]
        if args.trt_only:
            cmd += ["--tracker", "edgetam_trt"]
            headline = "3/4  speed — model-only ms/frame, TensorRT"
        else:
            cmd += ["--baseline-config", args.baseline_config]
            headline = "3/4  speed — model-only ms/frame, PyTorch vs TensorRT"
        ok, out = run_step(headline, cmd, outdir / "03_speed.txt")
        status["speed"] = ok

        # With one backend there is no comparison block to parse, so fall back
        # to the per-run summary line, which is always printed:
        #   propagate             26.40 ms/frame   ->   37.88 FPS
        m = re.search(r"propagate\s+([\d.]+) ms/frame\s+->\s+([\d.]+) FPS", out)
        fallback = f"{m.group(1)} ms   {m.group(2)} FPS" if m else "not found"

        if args.trt_only:
            results["speed_torch"] = "not measured (--trt-only)"
            results["speed_trt"] = fallback
        else:
            # `\s+\(` after the name is what separates the two rows: the
            # TensorRT row reads "edgetam_trt  (", so the underscore blocks
            # this pattern and only the stock row can match.
            results["speed_torch"] = find(
                r"\bedgetam\s+\([^)]*\)\s+([\d.]+ ms\s+[\d.]+ FPS)", out
            ) or "not found"
            results["speed_trt"] = find(
                r"edgetam_trt\s+\([^)]*\)\s+([\d.]+ ms\s+[\d.]+ FPS(?:\s+\([\d.]+x\))?)",
                out,
            ) or fallback

    # ----------------------------------------------------------------- video
    if "video" not in skip:
        clip = outdir / "clip"
        ok, _ = run_step(
            "4/4a  building the video clip",
            [PY, "tools/make_synthetic_video.py", "--frames", args.frames,
             "--size", args.size, "--radius", args.radius, "--outdir", clip],
            outdir / "04_clip.txt",
        )
        if ok:
            ok, out = run_step(
                "4/4b  video — end-to-end app ms/frame, with a watchable mp4",
                [PY, "cli.py", "--tracker", "edgetam_trt", "--config", args.config,
                 "--frames-dir", clip, "--frame-pattern", "*.jpg", "--fps", "30",
                 "--output", outdir / "04_video.mp4",
                 "--prompt", "file", "--prompt-file", clip / "prompts.json",
                 "--offload-video",
                 "--fps-warmup", args.warmup,
                 "--fps-chart", outdir / "04_video_latency.png",
                 "--stage-chart", outdir / "04_video_stages.png"],
                outdir / "04_video.txt",
            )
            results["video_fps"] = find(
                r"avg ([\d.]+ FPS over \d+ frames)", out
            ) or find(r"tracked \d+ frames in [\d.]+s \(([\d.]+ FPS) all\)", out) or "not found"
        status["video"] = ok

    # --------------------------------------------------------------- summary
    lines = [
        f"# Experiment results — {outdir.name}",
        "",
        f"engines: `{args.engines}`  ·  TensorRT config: `{args.config}`",
        "",
        f"accuracy reference: `{reference_config}` ({reference_tracker})"
        + ("  ·  PyTorch baseline: not measured (`--trt-only`)" if args.trt_only
           else f"  ·  PyTorch baseline: `{args.baseline_config}`"),
        "",
        f"{args.frames} frames at {args.size}  ·  {args.warmup} warm-up excluded"
        + (f"  ·  model input {args.image_size}x{args.image_size}" if args.image_size else ""),
        "",
        "## The two frame times, and why they differ",
        "",
        "| | what it measures | number |",
        "|---|---|---|",
        f"| **tracking only (step 3)** | the four engines + EdgeTAM's bookkeeping, "
        f"nothing else | **{results.get('speed_trt', '-')}** |",
        f"| demo video (step 4) | the above **plus** re-reading each frame, drawing "
        f"the mask on it, and encoding an mp4 | {results.get('video_fps', '-')} |",
        "",
        "**Quote step 3 as the frame budget.** Re-reading the frame, drawing the "
        "overlay and encoding an mp4 exist to produce something watchable; a "
        "real-time deployment sends masks to a consumer and does none of them. "
        "Step 4 is the demo, and its extra cost scales with output resolution, not "
        "with the model.",
        "",
        "`python cli.py --no-video ...` runs step 4's path with those three removed, "
        "if you want the same tool to report the real-time number.",
        "",
        "## Results",
        "",
        "| step | question | result |",
        "|---|---|---|",
        f"| 1 parity | does each engine reproduce its module? | {results.get('parity_total', '-')} |",
        f"| 2 accuracy | {accuracy_question} | mean IoU "
        f"{results.get('mean_iou', '-')}, min {results.get('min_iou', '-')}, "
        f"below threshold: {results.get('below', '-')} |",
        f"| 2 accuracy | does error accumulate? | trend {results.get('trend', '-')} |",
        f"| 3 speed | stock PyTorch | {results.get('speed_torch', '-')} |",
        f"| 3 speed | TensorRT fp16 + CUDA graphs | {results.get('speed_trt', '-')} |",
        f"| 4 video | end-to-end | {results.get('video_fps', '-')} |",
        "",
        "## Files",
        "",
        "| file | what |",
        "|---|---|",
        "| `01_parity.txt` | per-module engine-vs-PyTorch error and speed |",
        f"| `02_accuracy.txt` / `.json` | per-frame mask IoU against "
        f"`{Path(reference_config).name}` |",
        "| `03_speed.txt` | FPS, p50/p90/p99, per-module ms, both backends |",
        "| `03_speed_edgetam.png` / `_edgetam_trt.png` | per-frame latency, one per backend |",
        "| `04_video.mp4` | the tracked clip, watchable |",
        "| `04_video_latency.png` | per-frame latency, end to end |",
        "| `04_video_stages.png` | decode / inference / render histograms |",
        "",
    ]
    failed = [k for k, v in status.items() if not v]
    if failed:
        lines += [f"> **Steps that failed: {', '.join(failed)}** — see their .txt log.", ""]

    (outdir / "SUMMARY.md").write_text("\n".join(lines))

    print(f"\n{'=' * 70}\n>> Everything is in {outdir}/\n{'=' * 70}")
    print((outdir / "SUMMARY.md").read_text())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
