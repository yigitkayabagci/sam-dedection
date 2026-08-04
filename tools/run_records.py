#!/usr/bin/env python3
"""Track every recorded frame folder three ways and collect the comparison.

`frames/<record>/` holds one recording as an image sequence. This runs each of
them through the TensorRT backend in three input configurations and writes the
results into `frame_output/<record>/<mode>/`:

  full1024   whole frame resized to 1024x1024   the reference configuration
  full512    whole frame resized to 512x512     4x fewer tokens, whole scene
  crop512    centred 512x512 window, no resize  native pixels, cropped scene

`crop512` is the configuration a camera that already centres and focuses on
its target can offer: the object is in the middle, so the outer frame is
mostly background being paid for twice -- once in the resize, once in the
model. Cropping to exactly the model input removes the resize step entirely,
so it is a speed experiment as much as an accuracy one.

The three are not equivalent inputs and the comparison is not only FPS: 512
sees the whole scene at half the detail, crop512 sees part of the scene at
full detail. If the target leaves the centre window, crop512 loses it -- that
is the trade being measured, and the mp4 is there to watch it happen.

Prompts come from `<record>/prompts.json` (full-frame coordinates, shifted
into crop coordinates automatically), or from --box for every record at once.

Usage:
    python tools/run_records.py --records frames --out frame_output
    python tools/run_records.py --records frames --out frame_output \\
        --modes crop512 --box 700,300,830,430
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_experiment import PY, find, run_step  # noqa: E402

# mode -> (backend YAML, centre crop size or None)
MODES = {
    "full1024": ("configs/edgetam_trt.yaml", None),
    "full512": ("configs/edgetam_trt_512.yaml", None),
    "crop512": ("configs/edgetam_trt_512.yaml", 512),
}


def resolve_prompts(record: Path, outdir: Path, box: str | None) -> Path | None:
    """The record's own prompts.json, or one written from --box. None if neither."""
    own = record / "prompts.json"
    if own.exists():
        return own
    if not box:
        return None
    xyxy = [float(v) for v in box.split(",")]
    path = outdir / "prompts.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"boxes": [{"obj_id": 1, "frame_idx": 0, "xyxy": xyxy}], "points": []}
    ) + "\n")
    return path


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
    p.add_argument("--box", default=None,
                   help="x1,y1,x2,y2 in full-frame coordinates, for records with "
                        "no prompts.json of their own.")
    p.add_argument("--no-video", action="store_true",
                   help="Measure only. The overlay and the mp4 are already "
                        "outside the reported frame budget, but they still run "
                        "on the same CPU and memory bus; drop them for a "
                        "measurement with nothing else competing.")
    args = p.parse_args(argv)

    records = sorted(d for d in Path(args.records).iterdir() if d.is_dir()) \
        if Path(args.records).is_dir() else []
    if not records:
        raise SystemExit(f"No record folders in {args.records}/")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in MODES:
            raise SystemExit(f"Unknown mode {m!r}; pick from {', '.join(MODES)}")

    out_root = Path(args.out)
    rows: dict[str, dict[str, dict]] = {}
    failed: list[str] = []

    for record in records:
        rows[record.name] = {}
        for mode in modes:
            config, crop = MODES[mode]
            outdir = out_root / record.name / mode
            outdir.mkdir(parents=True, exist_ok=True)

            prompts = resolve_prompts(record, outdir, args.box)
            if prompts is None:
                print(f"!! {record.name}: no prompts.json and no --box; skipped")
                failed.append(f"{record.name}/{mode}")
                continue

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

            ok, text = run_step(f"{record.name} — {mode}", cmd, outdir / "run.txt")
            if not ok:
                failed.append(f"{record.name}/{mode}")
            stages = re.search(
                r"median per frame: pre ([\d.]+) \+ inference ([\d.]+) \+ post ([\d.]+) ms",
                text,
            )
            rows[record.name][mode] = {
                "fps": find(r"avg ([\d.]+) FPS over", text) or "-",
                "stages": "  /  ".join(stages.groups()) if stages else "-",
                "demo": find(r"overlay \+ mp4 encoding: ([\d.]+) ms/frame", text) or "-",
            }

    # --------------------------------------------------------------- summary
    described = {
        "full1024": "| `full1024` | 1024x1024 | whole frame, resized |",
        "full512": "| `full512` | 512x512 | whole frame, resized |",
        "crop512": "| `crop512` | 512x512 | centred 512x512 window, **no resize** |",
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
    ]
    for name, per_mode in rows.items():
        lines += [
            f"## {name}",
            "",
            "| mode | FPS | median ms: pre / inference / post | overlay + mp4 (excluded) |",
            "|---|---|---|---|",
        ]
        for mode in modes:
            r = per_mode.get(mode)
            if r is None:
                lines.append(f"| `{mode}` | did not run | - | - |")
                continue
            lines.append(f"| `{mode}` | {r['fps']} | {r['stages']} | {r['demo']} ms |")
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
