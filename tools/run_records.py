#!/usr/bin/env python3
"""Track every recorded frame folder several ways and collect the comparison.

`frames/<record>/` holds one recording as an image sequence. This runs each of
them through several input configurations and writes the results into
`frame_output/<record>/<mode>/`. The prompt is picked once per record and
reused by every mode, which is the whole value of running them from here
rather than one `cli.py` line at a time: a target selected separately per
command gives each mode a different box, and then nothing is comparable.

The TensorRT set, which needs engines built at that resolution:

  full1024   whole frame resized to 1024x1024   the reference configuration
  crop1024   centred 1024x1024 window           native pixels, cropped scene
  full768    whole frame resized to 768x768     56% of the tokens, whole scene
  crop768    centred 768x768 window             native pixels, cropped scene
  full512    whole frame resized to 512x512     4x fewer tokens, whole scene
  crop512    centred 512x512 window             native pixels, cropped scene

These six carry the STOCK checkpoint, because full512 is the other half of the
"PyTorch against TensorRT" comparison and both halves have to hold the same
weights. For a deployment number, use the modes built on the adapted weights:

  lora512_trt, lora512_trt_crop                  engines from models512_lora/
  lora512_trt_samurai, ..._samurai_crop          the same, plus the memory gate
  lora512_trt_int8, lora512_trt_int8_crop        the calibrated INT8 engine set

The crop modes are what a camera that already centres and focuses on its
target can offer: the object is in the middle, so the outer frame is mostly
background being paid for twice -- once in the resize, once in the model.
Cropping to exactly the model input removes the resize step entirely, so it is
a speed experiment as much as an accuracy one.

Note the crop is clamped to the frame, so a source shorter than the crop is
cropped on one axis and still resized on the other: at 1280x720, `crop1024`
takes 1024x720 and the model still stretches it vertically. The summary
reports the window each run actually used.

These are not equivalent inputs and the comparison is not only FPS: 512 sees
the whole scene at half the detail, a crop sees part of the scene at full
detail. If the target leaves the centre window, the crop modes lose it -- that
is the trade being measured, and the mp4 is there to watch it happen.

768 is the midpoint, and it only earns its place on sources that carry more
than 512x512 of real detail -- on a 640x512 thermal frame it upsamples and pays
2.25x the 512 token count for interpolated pixels. See configs/edgetam_768.yaml.
Each mode needs its own engine set; `--modes` runs the subset you have built.

And the PyTorch set at 512, which needs no engines and no export -- these are
the modes that answer "did the thermal adaptation help on my own recording",
on any CUDA machine, before anything is built for the device:

  stock512              stock EdgeTAM                     the floor
  finetune512           the partial fine-tune             docs/lora_vs_finetune.md
  lora512               the LoRA checkpoint               the 512 thermal default
  lora512_crop          the same, native 512 window       no resize at all
  lora512_samurai       the same + SAMURAI memory gate    docs/samurai.md
  lora512_samurai_crop  both of the above

`--modes` takes mode names or one of the groups, which exist so a comparison
holds exactly one variable: `trt` (the default six), `torch` (all six above),
`weights` (stock / fine-tune / LoRA), `samurai` (the gate on and off, same
weights, same input), `crop` (resize against native pixels), `trt512` (the 512
latency ladder on the device, on the adapted weights).

Prompts are picked once per record, on the full frame, and reused by every
mode (crop modes shift them automatically). They come from, in order:
`<record>/prompts.json`, a previous pick saved under the output folder,
--box, or an interactive window.

Usage:
    python tools/run_records.py --records frames --out frame_output
    python tools/run_records.py --records frames --out frame_output \\
        --prompt point --multi
    python tools/run_records.py --records frames --out frame_output \\
        --modes crop512 --box 700,300,830,430

    # SAMURAI on and off over one recording, on the adapted weights, off one
    # selection. No engines needed.
    python tools/run_records.py --records frames/record_050 --out frame_output \\
        --modes samurai --box 700,300,830,430

    # Same records, half the inferences: results land in <mode>_skip2/ next to
    # the baseline, off the same saved prompt, so the two are comparable.
    python tools/run_records.py --records frames --out frame_output \\
        --modes full1024 --frame-skip
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import read_first_frame_dir  # noqa: E402
from src.prompts import BoxPrompt, PromptSet  # noqa: E402
from src.prompts.file_source import save_prompts  # noqa: E402
from tools.run_experiment import PY, find, run_step  # noqa: E402

class Mode(NamedTuple):
    """One input configuration: which backend, which weights, which window.

    The tracker travels with the mode rather than being a flag, because it is
    not free to choose: a TensorRT config without engines is a silent fallback
    to PyTorch on stock weights, which is the one comparison nobody wants to
    make by accident.
    """

    tracker: str
    config: str
    crop: int | None
    model: str      # for the summary table
    window: str     # for the summary table


MODES = {
    # TensorRT, the Orin deployment path. Each of these needs an engine set
    # built at its resolution first (tools/build_trt_engines.py).
    "full1024": Mode("edgetam_trt", "configs/edgetam_trt.yaml", None,
                     "1024 TRT (stock)", "whole frame, resized"),
    "crop1024": Mode("edgetam_trt", "configs/edgetam_trt.yaml", 1024,
                     "1024 TRT (stock)", "centred 1024x1024 window"),
    "full768": Mode("edgetam_trt", "configs/edgetam_trt_768.yaml", None,
                    "768 TRT (stock)", "whole frame, resized"),
    "crop768": Mode("edgetam_trt", "configs/edgetam_trt_768.yaml", 768,
                    "768 TRT (stock)", "centred 768x768 window"),
    "full512": Mode("edgetam_trt", "configs/edgetam_trt_512.yaml", None,
                    "512 TRT (stock)", "whole frame, resized"),
    "crop512": Mode("edgetam_trt", "configs/edgetam_trt_512.yaml", 512,
                    "512 TRT (stock)", "centred 512x512 window"),
    # The same six, on the adapted weights: engines built from
    # checkpoints/edgetam_lora_512.pt into models512_lora/, so nothing here
    # ever mixes adapted engines with a stock scaffold. These are the modes to
    # quote a deployment frame time from; the stock six above answer "what did
    # TensorRT change", which is a different question.
    "lora512_trt": Mode("edgetam_trt", "configs/edgetam_trt_lora_512.yaml", None,
                        "512 TRT LoRA", "whole frame, resized"),
    "lora512_trt_crop": Mode("edgetam_trt", "configs/edgetam_trt_lora_512.yaml", 512,
                             "512 TRT LoRA", "centred 512x512 window"),
    "lora512_trt_samurai": Mode("edgetam_trt", "configs/edgetam_trt_samurai_512.yaml",
                                None, "512 TRT LoRA + SAMURAI", "whole frame, resized"),
    "lora512_trt_samurai_crop": Mode("edgetam_trt", "configs/edgetam_trt_samurai_512.yaml",
                                     512, "512 TRT LoRA + SAMURAI",
                                     "centred 512x512 window"),
    "lora512_trt_int8": Mode("edgetam_trt", "configs/edgetam_trt_int8_512.yaml", None,
                             "512 TRT LoRA INT8", "whole frame, resized"),
    "lora512_trt_int8_crop": Mode("edgetam_trt", "configs/edgetam_trt_int8_512.yaml", 512,
                                  "512 TRT LoRA INT8", "centred 512x512 window"),

    # PyTorch at 512. No engines, no export, runs on any CUDA machine -- which
    # is what makes these the modes to answer "did the thermal adaptation help
    # on my own recording" with, before anything is built for the device.
    "stock512": Mode("edgetam", "configs/edgetam_512.yaml", None,
                     "512 stock", "whole frame, resized"),
    "finetune512": Mode("edgetam", "configs/edgetam_512_thermal.yaml", None,
                        "512 fine-tune", "whole frame, resized"),
    "lora512": Mode("edgetam", "configs/edgetam_512_lora.yaml", None,
                    "512 LoRA", "whole frame, resized"),
    "lora512_crop": Mode("edgetam", "configs/edgetam_512_lora.yaml", 512,
                         "512 LoRA", "centred 512x512 window"),
    "lora512_samurai": Mode("edgetam", "configs/edgetam_samurai_512.yaml", None,
                            "512 LoRA + SAMURAI", "whole frame, resized"),
    "lora512_samurai_crop": Mode("edgetam", "configs/edgetam_samurai_512.yaml", 512,
                                 "512 LoRA + SAMURAI", "centred 512x512 window"),
}

# `--modes` still defaults to the TensorRT six this tool was written for, so an
# Orin run does not silently grow. The named groups below are the comparisons
# worth running as a unit; each holds exactly one variable.
DEFAULT_MODES = ("full1024", "crop1024", "full768", "crop768", "full512", "crop512")
GROUPS = {
    "trt": DEFAULT_MODES,
    "torch": ("stock512", "finetune512", "lora512", "lora512_crop",
              "lora512_samurai", "lora512_samurai_crop"),
    # Did the adaptation help? Three sets of weights, one prompt, one input.
    "weights": ("stock512", "finetune512", "lora512"),
    # Does SAMURAI help? Same weights, same input, memory gate on and off.
    "samurai": ("lora512", "lora512_samurai"),
    # Does the resize cost anything? Same weights, whole frame against a
    # native-pixel window.
    "crop": ("lora512", "lora512_crop"),
    # The 512 latency ladder on the device, all on the adapted weights:
    # engines, native window, and the memory gate on each. One engine set,
    # one checkpoint, so the only things that vary are the window and the gate.
    "trt512": ("lora512_trt", "lora512_trt_crop",
               "lora512_trt_samurai", "lora512_trt_samurai_crop"),
}


def expand_modes(spec: str) -> list[str]:
    """Mode names and group names, in the order given, without repeats."""
    modes: list[str] = []
    for token in (t.strip() for t in spec.split(",")):
        if not token:
            continue
        if token in GROUPS:
            names = GROUPS[token]
        elif token in MODES:
            names = (token,)
        else:
            raise SystemExit(
                f"Unknown mode or group {token!r}. Modes: {', '.join(MODES)}. "
                f"Groups: {', '.join(GROUPS)}.")
        modes += [n for n in names if n not in modes]
    if not modes:
        raise SystemExit("--modes selected nothing.")
    return modes


def resolve_prompts(record: Path, outdir: Path, args) -> Path | None:
    """One selection per record, in full-frame coordinates, reused by every mode.

    Picking interactively needs a display, so every non-interactive source is
    tried first and the pick is saved -- a re-run, or a second mode, never asks
    again.
    """
    own = record / "prompts.json"
    if own.exists():
        return own
    saved = outdir / "prompts.json"
    if saved.exists() and not args.repick:
        print(f">> {record.name}: reusing {saved}")
        return saved
    if args.box:
        xyxy = tuple(float(v) for v in args.box.split(","))
        return save_prompts(PromptSet(boxes=[BoxPrompt(1, 0, xyxy)]), saved)

    from src.prompts import interactive

    first = read_first_frame_dir(record, args.pattern)
    print(f">> {record.name}: select the target ({args.pattern} frame 0)")
    if args.prompt == "box":
        picked = interactive.pick_boxes(first) if args.multi else interactive.pick_box(first)
    else:
        picked = (interactive.pick_points_multi(first) if args.multi
                  else interactive.pick_points(first))
    return save_prompts(picked, saved)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--records", default="frames",
                   help="Directory of record folders, each an image sequence.")
    p.add_argument("--out", default="frame_output", help="Where results go.")
    p.add_argument("--modes", default=",".join(DEFAULT_MODES),
                   help=f"Comma-separated modes ({', '.join(MODES)}) or groups "
                        f"({', '.join(GROUPS)}). Default: the TensorRT set.")
    p.add_argument("--pattern", default="*.tif*", help="Frame glob inside a record.")
    p.add_argument("--fps", type=float, default=30.0,
                   help="Playback fps for the output mp4 (a sequence has none).")
    p.add_argument("--warmup", type=int, default=20,
                   help="Frames excluded from every reported statistic.")
    p.add_argument("--frame-skip", "--frameskip", dest="frame_skip", type=int,
                   nargs="?", const=2, default=1, metavar="N",
                   help="Infer one frame in N and hold that mask over the rest "
                        "(bare flag = 2). The clip still comes out at full length; "
                        "what halves is the inferences behind it, so each one gets N "
                        "frame periods instead of one. Results go to <mode>_skipN/ so "
                        "a baseline run is never overwritten, and the saved prompt is "
                        "shared with it -- same target, same frames, one variable.")
    p.add_argument("--prompt", choices=("box", "point"), default="box",
                   help="How to select the target interactively, when no "
                        "prompts.json and no --box is available. Needs a display.")
    p.add_argument("--multi", action="store_true",
                   help="Select several targets: multiple boxes, or points with "
                        "'n' to start the next object.")
    p.add_argument("--box", default=None,
                   help="x1,y1,x2,y2 in full-frame coordinates, applied to every "
                        "record. Skips the interactive window entirely.")
    p.add_argument("--repick", action="store_true",
                   help="Select again, replacing a pick saved by an earlier run.")
    p.add_argument("--pick-only", action="store_true",
                   help="Select each record's target, save it, and stop without "
                        "tracking. Use this when you want to run the modes "
                        "yourself, one command at a time, against one selection "
                        "-- picking separately per command would give each mode a "
                        "different box and make them incomparable.")
    p.add_argument("--no-video", action="store_true",
                   help="Measure only. The overlay and the mp4 are already "
                        "outside the reported frame budget, but they still run "
                        "on the same CPU and memory bus; drop them for a "
                        "measurement with nothing else competing.")
    args = p.parse_args(argv)

    root = Path(args.records)
    if not root.is_dir():
        raise SystemExit(f"{root} is not a directory.")
    # A folder holding frames is one record; a folder holding folders is a set
    # of them -- so --records can name the whole set or a single clip.
    records = [root] if list(root.glob(args.pattern)) else \
        sorted(d for d in root.iterdir() if d.is_dir())
    if not records:
        raise SystemExit(f"No frames matching {args.pattern!r} in {root}/, and no "
                         "record folders inside it either.")
    modes = expand_modes(args.modes)

    out_root = Path(args.out)
    rows: dict[str, dict[str, dict]] = {}
    failed: list[str] = []

    for record in records:
        rows[record.name] = {}
        try:
            prompts = resolve_prompts(record, out_root / record.name, args)
        except Exception as exc:  # no display, nothing selected, unreadable frame
            print(f"!! {record.name}: no prompts ({exc}); skipped. Pass --box to "
                  "run without a display.")
            failed += [f"{record.name}/{m}" for m in modes]
            continue

        if args.pick_only:
            print(f">> {record.name}: {prompts}")
            continue

        for mode in modes:
            spec = MODES[mode]
            # A skipped run is a different measurement of the same mode, not a
            # replacement for it, so it gets its own folder next to the baseline.
            outdir = out_root / record.name / (mode if args.frame_skip == 1
                                               else f"{mode}_skip{args.frame_skip}")
            outdir.mkdir(parents=True, exist_ok=True)

            cmd = [PY, "cli.py", "--tracker", spec.tracker, "--config", spec.config,
                   "--frames-dir", record, "--frame-pattern", args.pattern,
                   "--fps", args.fps,
                   "--prompt", "file", "--prompt-file", prompts,
                   "--offload-video", "--fps-warmup", args.warmup,
                   "--fps-chart", outdir / "latency.png",
                   "--stage-chart", outdir / "stages.png"]
            cmd += ["--no-video"] if args.no_video else ["--output", outdir / "tracked.mp4"]
            if spec.crop:
                cmd += ["--center-crop", spec.crop]
            if args.frame_skip > 1:
                cmd += ["--frame-skip", args.frame_skip]

            ok, text = run_step(f"{record.name} — {mode}", cmd, outdir / "run.txt")
            if not ok:
                failed.append(f"{record.name}/{mode}")
            stages = re.search(
                r"median per frame: pre ([\d.]+) \+ inference ([\d.]+) \+ post ([\d.]+) ms",
                text,
            )
            rows[record.name][mode] = {
                "fps": find(r"avg ([\d.]+) FPS over", text) or "-",
                # Under a skip, FPS counts inferences; this is the frame rate
                # they cover, which is what has to clear the source's.
                "source_fps": find(r"\(([\d.]+) source FPS\)", text) or "-",
                "stages": "  /  ".join(stages.groups()) if stages else "-",
                "demo": find(r"overlay \+ mp4 encoding: ([\d.]+) ms/frame", text) or "-",
                # What the crop actually came out as: clamped to the frame, so
                # a source shorter than the crop still gets resized on that axis.
                "input": find(r"centre crop (\d+x\d+) at", text) or "whole frame",
            }

    if args.pick_only:
        return 1 if failed else 0

    # --------------------------------------------------------------- summary
    skip = args.frame_skip
    title = f"# Recorded clips — {len(records)} record(s), {len(modes)} mode(s)"
    lines = [
        title + (f", frame skip {skip}" if skip > 1 else ""),
        "",
        "| mode | model | input |",
        "|---|---|---|",
        *(f"| `{m}` | {MODES[m].model} | {MODES[m].window} |" for m in modes),
        "",
        f"`{args.warmup}` warm-up frames excluded from every number below. FPS is "
        "the real-time budget: per-frame decode + resize, the model, and masks "
        "back to source resolution. Drawing the overlay and encoding the mp4 are "
        "excluded from it and reported separately.",
        "",
    ]
    if skip > 1:
        lines += [
            f"Frame skip {skip}: one frame in {skip} is inferred and its mask held "
            f"over the rest, so the clip still comes out at full length on 1/{skip} "
            "of the inferences. FPS below counts inferences; source FPS is the frame "
            f"rate they cover. At {args.fps:g} fps in, each inference has "
            f"{1000.0 * skip / args.fps:.1f} ms instead of {1000.0 / args.fps:.1f} ms "
            "-- that threshold, not the average, is what this run is testing, so read "
            "the tail off `latency.png` rather than the median.",
            "",
        ]
    columns = ["mode", "model input from", "FPS"]
    if skip > 1:
        columns.append("source FPS")
    columns += ["median ms: pre / inference / post", "overlay + mp4 (excluded)"]
    for name, per_mode in rows.items():
        lines += [
            f"## {name}",
            "",
            "| " + " | ".join(columns) + " |",
            "|" + "---|" * len(columns),
        ]
        for mode in modes:
            r = per_mode.get(mode)
            if r is None:
                cells = ["-", "did not run"] + ["-"] * (len(columns) - 3)
            else:
                cells = [r["input"], r["fps"]]
                if skip > 1:
                    cells.append(r["source_fps"])
                cells += [r["stages"], f"{r['demo']} ms"]
            lines.append(f"| `{mode}` | " + " | ".join(cells) + " |")
        suffix = "" if skip == 1 else f"_skip{skip}"
        lines += ["", f"Videos and charts: `{name}/<mode>{suffix}/`", ""]

    if failed:
        lines += [f"> **Did not complete: {', '.join(failed)}** — see their `run.txt`.", ""]

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "SUMMARY.md").write_text("\n".join(lines))
    print(f"\n{'=' * 70}\n>> Everything is in {out_root}/\n{'=' * 70}")
    print((out_root / "SUMMARY.md").read_text())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
