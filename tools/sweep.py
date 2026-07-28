#!/usr/bin/env python3
"""Sweep one input condition and report where the per-frame time goes.

`benchmark.py` varies the MODEL (fp16, int8, TensorRT...). This varies the
INPUT and holds the model fixed, which answers a different set of questions
and is what the Phase 1 baseline is for:

    resolution   Does source resolution change inference cost?  (No -- every
                 frame is squashed to image_size before the encoder. It moves
                 preprocess and postprocess only. This is the headline result.)
    image-size   The model's own input size, a different knob entirely.
                 Halving it cuts encoder FLOPs ~4x and costs accuracy.
    box          Does prompt box size change anything? (No -- a box is two
                 tokens whatever it contains.) A control experiment: a
                 variable theory says cannot matter, so a difference here
                 means the measurement rig is wrong, not the model.
    objects      The prompt-side knob that does cost: each target adds a row
                 to the mask decoder and memory attention batch. Targets run
                 in disjoint bands so they never cross, which keeps this a
                 cost measurement rather than a tracking-quality one.
    frames       How many frames before the average stops moving.

Every run uses generated frames, which is valid here because no per-frame
cost depends on content. Do not use this for accuracy -- IoU needs a real
scene.

Each run writes report.md/.csv/.tex, stages.png (per-frame time by stage)
and fps.png (the per-frame trace, where a stall the median hides is visible).

    python3 tools/sweep.py --axis resolution --out runs/sweep_res
    python3 tools/sweep.py --axis box        --out runs/sweep_box
    python3 tools/sweep.py --axis objects    --out runs/sweep_obj
    python3 tools/sweep.py --axis image-size --out runs/sweep_img
    python3 tools/sweep.py --axis frames     --out runs/sweep_n
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.benchmark import (  # noqa: E402
    build_synthetic_cache,
    load_matrix,
    run_variant,
    variant_kwargs,
    variant_runnable,
)

DEFAULT_MATRIX = ROOT / "configs" / "bench_matrix.yaml"

AXES = {
    "resolution": ["640x480", "1280x720", "1920x1080", "3840x2160"],
    "image-size": ["1024", "768", "512", "256"],
    "box": ["20", "96", "400"],
    "frames": ["50", "100", "300", "500", "1000"],
    "objects": ["1", "2", "3", "4"],
}

# The stages a stacked bar is built from, in pipeline order.
STAGES = [
    ("preprocess_ms", "preprocess", "#B4551F"),
    ("inference_ms", "inference", "#2E6F8E"),
    ("postprocess_ms", "postprocess", "#33796B"),
    ("unaccounted_ms", "other", "#8A93A0"),
]


def condition_settings(axis: str, value: str, args) -> dict:
    """Turn one axis value into the knobs a single run needs."""
    settings = {
        "width": args.width,
        "height": args.height,
        "box": args.box,
        "frames": args.frames,
        "objects": args.objects,
        "image_size": None,
        "label": f"{axis}={value}",
    }
    if axis == "resolution":
        settings["width"], settings["height"] = (int(v) for v in value.lower().split("x"))
        settings["label"] = value
    elif axis == "image-size":
        settings["image_size"] = int(value)
        settings["label"] = f"image_size {value}"
    elif axis == "box":
        settings["box"] = int(value)
        settings["label"] = f"box {value}px"
    elif axis == "frames":
        settings["frames"] = int(value)
        settings["label"] = f"{value} frames"
    elif axis == "objects":
        settings["objects"] = int(value)
        settings["label"] = f"{value} object" + ("s" if int(value) > 1 else "")
    return settings


def sweep(axis: str, values: list[str], args) -> tuple[list[dict], dict]:
    defaults, matrix = load_matrix(Path(args.matrix))
    if args.variant not in matrix:
        raise SystemExit(f"Unknown variant {args.variant!r}. Available: {list(matrix)}")
    tracker_name, base_kwargs = variant_kwargs(defaults, matrix[args.variant])
    ok, reason = variant_runnable(tracker_name, base_kwargs)
    if not ok:
        raise SystemExit(f"{args.variant}: {reason}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    series: dict[str, list[float]] = {}

    for value in values:
        cfg = condition_settings(axis, value, args)
        kwargs = dict(base_kwargs)
        if cfg["image_size"] is not None:
            kwargs["image_size"] = cfg["image_size"]

        cache = out_dir / f"cache_{value.replace('x', '_')}"
        shutil.rmtree(cache, ignore_errors=True)
        frames_dir, prompts = build_synthetic_cache(
            cache, cfg["frames"], cfg["width"], cfg["height"], cfg["box"],
            objects=cfg["objects"])

        warmup = min(args.warmup, max(1, cfg["frames"] // 4))
        print(f"\n[sweep] {cfg['label']}  ({cfg['width']}x{cfg['height']}, "
              f"{cfg['frames']} frames, {cfg['objects']} obj, warmup {warmup})")

        runs, error = [], None
        for i in range(args.repeat):
            try:
                runs.append(run_variant(
                    cfg["label"], tracker_name, kwargs, frames_dir, prompts,
                    warmup, None, None, False))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                print(f"[sweep] FAILED {cfg['label']}: {error}")
                break

        row = {"condition": cfg["label"], "axis": axis, "value": value,
               "width": cfg["width"], "height": cfg["height"],
               "frames": cfg["frames"], "box": cfg["box"],
               "objects": cfg["objects"],
               "image_size": cfg["image_size"] or 1024}
        if error:
            row["error"] = error
        if runs:
            # The per-frame series drives the spike chart; it is a list, so the
            # numeric aggregation below leaves it alone and it never reaches
            # the JSON or the table.
            series[cfg["label"]] = runs[0].get("_per_frame_ms") or []
            keys = {k for r in runs for k, v in r.items() if isinstance(v, (int, float))}
            for key in keys:
                vals = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
                if vals:
                    row[key] = round(statistics.median(vals), 4)
            print(f"[sweep] {cfg['label']}: {row.get('fps', 0):.2f} FPS | "
                  f"pre {row.get('preprocess_ms') or 0:.2f} | "
                  f"infer {row.get('inference_ms') or 0:.2f} | "
                  f"post {row.get('postprocess_ms') or 0:.2f} ms")
        rows.append(row)
        (out_dir / f"{value.replace('x', '_')}.result.json").write_text(
            json.dumps(row, indent=2))

        if not args.keep_cache:
            shutil.rmtree(cache, ignore_errors=True)
    return rows, series


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

_COLUMNS = [
    ("condition", "condition", "{}"),
    ("fps loop", "fps", "{:.2f}"),
    ("fps e2e", "fps_end_to_end", "{:.2f}"),
    ("total ms", "total_ms", "{:.2f}"),
    ("pre ms", "preprocess_ms", "{:.2f}"),
    ("infer ms", "inference_ms", "{:.2f}"),
    ("post ms", "postprocess_ms", "{:.2f}"),
    ("other ms", "unaccounted_ms", "{:.2f}"),
    ("enc ms", "encoder_ms", "{:.2f}"),
    ("fps min", "fps_min", "{:.2f}"),
    ("fps max", "fps_max", "{:.2f}"),
    ("frames", "frames_measured", "{:.0f}"),
]


def _escape_tex(text: str) -> str:
    return str(text).replace("_", r"\_").replace("%", r"\%")


def _cells(row: dict, missing: str = "-") -> list[str]:
    out = []
    for _, key, fmt in _COLUMNS:
        value = row.get(key)
        out.append(missing if value is None else
                   (value if isinstance(value, str) else fmt.format(value)))
    return out


def markdown(rows: list[dict]) -> str:
    lines = ["| " + " | ".join(c[0] for c in _COLUMNS) + " |",
             "|" + "|".join("---" for _ in _COLUMNS) + "|"]
    lines += ["| " + " | ".join(_cells(r)) + " |" for r in rows]
    return "\n".join(lines)


def latex(rows: list[dict]) -> str:
    lines = [r"\begin{tabular}{l" + "r" * (len(_COLUMNS) - 1) + "}", r"\toprule",
             " & ".join(_escape_tex(c[0]) for c in _COLUMNS) + r" \\", r"\midrule"]
    for row in rows:
        # Only the first column holds free text; the rest are formatted numbers
        # or the em-dash placeholder, neither of which needs escaping.
        cells = _cells(row, missing="--")
        cells[0] = _escape_tex(cells[0])
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def csv(rows: list[dict]) -> str:
    keys = ["condition"] + sorted({k for r in rows for k, v in r.items()
                                   if k != "condition" and not isinstance(v, (dict, list))})
    out = [",".join(keys)]
    for row in rows:
        out.append(",".join("" if row.get(k) is None else str(row.get(k)) for k in keys))
    return "\n".join(out)


def stage_chart(rows: list[dict], out_path: Path, axis: str):
    """Stacked per-frame time by stage — the one picture that carries the point."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[sweep] matplotlib missing; skipping chart")
        return None

    usable = [r for r in rows if r.get("total_ms")]
    if not usable:
        return None
    labels = [r["condition"] for r in usable]
    ypos = range(len(usable))

    fig, ax = plt.subplots(figsize=(9.5, 1.9 + 0.62 * len(usable)))
    left = [0.0] * len(usable)
    total = max(float(r["total_ms"]) for r in usable)
    for key, name, color in STAGES:
        widths = [float(r.get(key) or 0.0) for r in usable]
        ax.barh(list(ypos), widths, left=left, height=0.62, label=name,
                color=color, edgecolor="white", linewidth=0.6)
        for i, (start, width) in enumerate(zip(left, widths)):
            # Label inside the segment. Skip slivers, where the text would
            # overflow into its neighbours and read as belonging to them.
            if width >= total * 0.055:
                ax.text(start + width / 2, i, f"{width:.1f}", ha="center",
                        va="center", fontsize=8.5, color="white")
        left = [a + b for a, b in zip(left, widths)]

    for i, row in enumerate(usable):
        # End-to-end, matching what the bar actually stacks. The loop-only FPS
        # would contradict the picture, since preprocess sits in the bar but
        # happens before the loop.
        ax.text(left[i] + total * 0.012, i,
                f"{row.get('fps_end_to_end', 0):.1f} FPS", va="center",
                fontsize=9, color="#333")

    ax.set_yticks(list(ypos))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("per-frame time (ms)")
    ax.set_title(f"Per-frame time by stage, varying {axis}")
    ax.set_xlim(0, total * 1.16)
    ax.grid(axis="x", alpha=0.25)
    # Outside the axes: an inside legend lands on top of the longest bar, which
    # is always the condition the chart most needs to show.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18 - 0.02 * len(usable)),
              fontsize=9, ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path.resolve()


def frame_time_chart(series: dict[str, list[float]], out_path: Path, axis: str):
    """Per-frame time for each condition, so stalls stay visible.

    Milliseconds rather than FPS: FPS is a reciprocal, which squashes the slow
    end and stretches the fast end, so a stall shows up as a hard-to-read dip.
    In milliseconds the same stall is a spike whose height is the time it
    actually cost.

    One panel per condition rather than overlaid lines, because comparing
    shapes matters more here than comparing levels.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    panels = [(label, vals) for label, vals in series.items() if vals]
    if not panels:
        return None

    fig, axes = plt.subplots(len(panels), 1, sharex=True, squeeze=False,
                             figsize=(10, 1.0 + 1.55 * len(panels)))
    for ax, (label, values) in zip(axes[:, 0], panels):
        arr = np.asarray(values, dtype=float)
        lo, hi, mean = float(arr.min()), float(arr.max()), float(arr.mean())
        ax.axhspan(lo, hi, color="#2E6F8E", alpha=0.07)
        ax.plot(arr, linewidth=0.9, color="#2E6F8E")
        ax.axhline(mean, color="#C0392B", linestyle="--", linewidth=1.1)
        ax.axhline(lo, color="#5B6B7A", linestyle=":", linewidth=0.9)
        ax.axhline(hi, color="#5B6B7A", linestyle=":", linewidth=0.9)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.2)
        ax.text(0.995, 0.9, f"min {lo:.1f}   mean {mean:.1f}   max {hi:.1f} ms",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                color="#333")
    axes[-1, 0].set_xlabel("frame index")
    axes[0, 0].set_title(f"Per-frame time (ms), varying {axis}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path.resolve()


def write_outputs(out_dir: Path, axis: str, variant: str,
                  rows: list[dict], series: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = markdown(rows)
    (out_dir / "report.md").write_text(
        f"# Input sweep: {axis}\n\nvariant: `{variant}`\n\n{table}\n")
    (out_dir / "report.csv").write_text(csv(rows) + "\n")
    (out_dir / "report.tex").write_text(latex(rows) + "\n")
    # Everything needed to redraw without measuring again. Chart tweaks are
    # frequent and a sweep is not cheap; re-running one to change an axis
    # label wastes half an hour.
    (out_dir / "sweep.json").write_text(json.dumps(
        {"axis": axis, "variant": variant, "rows": rows, "series": series}, indent=2))

    stages = stage_chart(rows, out_dir / "stages.png", axis)
    frame_time = frame_time_chart(series, out_dir / "frame_time.png", axis)

    print(f"\n{table}\n")
    print(f"[sweep] report -> {out_dir / 'report.md'} (+ .csv, .tex, .json)")
    if stages:
        print(f"[sweep] stages -> {stages}")
    if frame_time:
        print(f"[sweep] frames -> {frame_time}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--axis", choices=sorted(AXES),
                   help="Which input condition to vary.")
    p.add_argument("--replot", metavar="DIR",
                   help="Redraw the charts and tables from a finished sweep's "
                        "sweep.json, without measuring anything again.")
    p.add_argument("--values", default=None,
                   help="Comma-separated override for the axis values.")
    p.add_argument("--variant", default="pytorch_bf16",
                   help="Model variant from the matrix, held fixed across the sweep.")
    p.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    p.add_argument("--frames", type=int, default=500,
                   help="Frames per run. 500 keeps the run-to-run spread near "
                        "1-2%%, tight enough that a real 5%% difference between "
                        "conditions is still visible.")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--box", type=int, default=96)
    p.add_argument("--objects", type=int, default=1,
                   help="Targets to track. Each one adds a row to the mask "
                        "decoder and memory attention batch, so this is the "
                        "prompt-side knob that does change cost.")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--out", default="runs/sweep")
    p.add_argument("--keep-cache", action="store_true")
    args = p.parse_args()

    if args.replot:
        source = Path(args.replot)
        payload = source / "sweep.json"
        if not payload.exists():
            raise SystemExit(
                f"{payload} not found. Only sweeps run after sweep.json was "
                "introduced can be replotted; re-run the sweep once to create it.")
        data = json.loads(payload.read_text())
        write_outputs(source, data["axis"], data.get("variant", "?"),
                      data["rows"], data.get("series", {}))
        return 0

    if not args.axis:
        raise SystemExit("Pass --axis (or --replot DIR).")

    values = ([v.strip() for v in args.values.split(",") if v.strip()]
              if args.values else AXES[args.axis])
    rows, series = sweep(args.axis, values, args)
    write_outputs(Path(args.out), args.axis, args.variant, rows, series)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
