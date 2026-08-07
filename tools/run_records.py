#!/usr/bin/env python3
"""Track every recorded frame folder three ways and collect the comparison.

`frames/<record>/` holds one recording as an image sequence. This runs each of
them through the TensorRT backend in four input configurations and writes the
results into `frame_output/<record>/<mode>/`:

  full1024   whole frame resized to 1024x1024   the reference configuration
  crop1024   centred 1024x1024 window           native pixels, cropped scene
  full512    whole frame resized to 512x512     4x fewer tokens, whole scene
  crop512    centred 512x512 window             native pixels, cropped scene

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
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import read_first_frame_dir  # noqa: E402
from src.prompts import BoxPrompt, PromptSet  # noqa: E402
from src.prompts.file_source import save_prompts  # noqa: E402
from tools.run_experiment import PY, find, run_step  # noqa: E402

# mode -> (backend YAML, centre crop size or None)
MODES = {
    "full1024": ("configs/edgetam_trt.yaml", None),
    "crop1024": ("configs/edgetam_trt.yaml", 1024),
    "full512": ("configs/edgetam_trt_512.yaml", None),
    "crop512": ("configs/edgetam_trt_512.yaml", 512),
}


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
    p.add_argument("--modes", default=",".join(MODES),
                   help=f"Comma-separated subset of {', '.join(MODES)}.")
    p.add_argument("--pattern", default="*.tif*", help="Frame glob inside a record.")
    p.add_argument("--fps", type=float, default=30.0,
                   help="Playback fps for the output mp4 (a sequence has none).")
    p.add_argument("--warmup", type=int, default=20,
                   help="Frames excluded from every reported statistic.")
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
    p.add_argument("--realtime-fps", type=float, default=None,
                   help="Feed each run from a source at this rate and drop the "
                        "frames that arrive while the model is busy, instead of "
                        "letting the clip wait. Adds a deadline verdict (1/fps "
                        "per frame) and a drop count to the summary. Off by "
                        "default: every frame is tracked, which is the right "
                        "measurement for comparing modes against each other.")
    p.add_argument("--deadline-ms", type=float, default=None,
                   help="Per-frame ceiling every mode is judged against, e.g. 35. "
                        "Adds a column counting the frames that went over it. "
                        "Independent of --realtime-fps, and the more direct way "
                        "to ask 'does this mode ever exceed X ms'.")
    p.add_argument("--frame-stride", type=int, default=None,
                   help="Track every Nth frame, skipping the rest entirely -- fixed, "
                        "clock-free frame skipping, as opposed to --realtime-fps's "
                        "adapt-to-measured-latency queue. Mutually exclusive with "
                        "--realtime-fps; pick one.")
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
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in MODES:
            raise SystemExit(f"Unknown mode {m!r}; pick from {', '.join(MODES)}")
    if args.frame_stride is not None and args.realtime_fps is not None:
        raise SystemExit("--frame-stride and --realtime-fps are two different "
                         "frame-skip mechanisms; pick one.")

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
            config, crop = MODES[mode]
            outdir = out_root / record.name / mode
            outdir.mkdir(parents=True, exist_ok=True)

            cmd = [PY, "cli.py", "--tracker", "edgetam_trt", "--config", config,
                   "--frames-dir", record, "--frame-pattern", args.pattern,
                   "--fps", args.fps,
                   "--prompt", "file", "--prompt-file", prompts,
                   "--offload-video", "--fps-warmup", args.warmup,
                   "--fps-chart", outdir / "latency.png",
                   "--stage-chart", outdir / "stages.png"]
            cmd += ["--no-video"] if args.no_video else ["--output", outdir / "tracked.mp4"]
            if crop:
                cmd += ["--center-crop", crop]
            if args.realtime_fps:
                cmd += ["--realtime-fps", args.realtime_fps]
            if args.deadline_ms:
                cmd += ["--deadline-ms", args.deadline_ms]
            if args.frame_stride:
                cmd += ["--frame-stride", args.frame_stride]

            ok, text = run_step(f"{record.name} — {mode}", cmd, outdir / "run.txt")
            if not ok:
                failed.append(f"{record.name}/{mode}")
            stages = re.search(
                r"median per frame: pre ([\d.]+) \+ inference ([\d.]+) \+ post ([\d.]+) ms",
                text,
            )
            # Same idea, two mechanisms: --realtime-fps drops a measured count
            # ("dropped N (%) as stale"); --frame-stride skips a fixed, known
            # count ("tracked T of M (%)"). Whichever ran, fold it into one
            # "skipped" figure so the table has a single column for it.
            skipped = find(r"dropped (\d+ \([\d.]+%\)) as stale", text)
            if skipped is None:
                stride = re.search(r"fixed stride \d+: tracked (\d+) of (\d+) "
                                   r"frames \(([\d.]+)%\)", text)
                if stride:
                    tracked, total, pct = int(stride[1]), int(stride[2]), float(stride[3])
                    skipped = f"{total - tracked} ({100.0 - pct:.1f}%)"
            rows[record.name][mode] = {
                "fps": find(r"avg ([\d.]+) FPS over", text) or "-",
                "stages": "  /  ".join(stages.groups()) if stages else "-",
                "demo": find(r"overlay \+ mp4 encoding: ([\d.]+) ms/frame", text) or "-",
                # What the crop actually came out as: clamped to the frame, so
                # a source shorter than the crop still gets resized on that axis.
                "input": find(r"centre crop (\d+x\d+) at", text) or "whole frame",
                # The tail, not the average: a mode holds a frame rate only if
                # its slowest frames do. p99 and max are the two that decide it.
                "p99": find(r"p99 ([\d.]+) ·", text) or "-",
                "max": find(r"max ([\d.]+) ms", text) or "-",
                "deadline": find(r"ms (?:ceiling|[\d.]+ FPS slot): "
                                 r"(met|\d+ over \([\d.]+%\))", text) or "-",
                "skipped": skipped or "-",
            }

    if args.pick_only:
        return 1 if failed else 0

    # --------------------------------------------------------------- summary
    described = {
        "full1024": "| `full1024` | 1024x1024 | whole frame, resized |",
        "crop1024": "| `crop1024` | 1024x1024 | centred 1024x1024 window |",
        "full512": "| `full512` | 512x512 | whole frame, resized |",
        "crop512": "| `crop512` | 512x512 | centred 512x512 window |",
    }
    lines = [
        f"# Recorded clips — {len(records)} record(s), {len(modes)} mode(s)",
        "",
        "| mode | model | input |",
        "|---|---|---|",
        *(described[m] for m in modes),
        "",
        f"`{args.warmup}` warm-up frames excluded from every number below. FPS is "
        "the real-time budget: per-frame decode + resize, the model, and masks "
        "back to source resolution. Drawing the overlay and encoding the mp4 are "
        "excluded from it and reported separately.",
        "",
        "Read `p99` and `max` before `FPS`. Holding a frame rate is decided by "
        "the slowest frames, not the average one: at 30 FPS every frame has a "
        "33.3 ms slot, and a mode averaging 30 FPS with a 60 ms worst frame does "
        "not hold it.",
        "",
    ]
    skipping = args.realtime_fps is not None or args.frame_stride is not None
    ceiling = args.deadline_ms or (1000.0 / args.realtime_fps if args.realtime_fps else None)
    skipped_label = "skipped (fixed stride)" if args.frame_stride else "dropped (stale)"
    for name, per_mode in rows.items():
        head = ["mode", "model input from", "FPS", "median ms: pre / inference / post",
                "p99 ms", "max ms"]
        if ceiling:
            head.append(f"over {ceiling:.1f} ms")
        if skipping:
            head.append(skipped_label)
        head.append("overlay + mp4 (excluded)")
        lines += [
            f"## {name}",
            "",
            "| " + " | ".join(head) + " |",
            "|" + "---|" * len(head),
        ]
        for mode in modes:
            r = per_mode.get(mode)
            if r is None:
                lines.append(f"| `{mode}` | did not run " + "| - " * (len(head) - 2) + "|")
                continue
            cells = [f"`{mode}`", r["input"], r["fps"], r["stages"], r["p99"], r["max"]]
            if ceiling:
                cells.append(r["deadline"])
            if skipping:
                cells.append(r["skipped"])
            cells.append(f"{r['demo']} ms")
            lines.append("| " + " | ".join(cells) + " |")
        lines += ["", f"Videos and charts: `{name}/<mode>/`", ""]

    if failed:
        lines += [f"> **Did not complete: {', '.join(failed)}** — see their `run.txt`.", ""]

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "SUMMARY.md").write_text("\n".join(lines))
    print(f"\n{'=' * 70}\n>> Everything is in {out_root}/\n{'=' * 70}")
    print((out_root / "SUMMARY.md").read_text())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
