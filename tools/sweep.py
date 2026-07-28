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
    frames       How many frames before the average stops moving.

Every run uses generated frames, which is valid here because no per-frame
cost depends on content. Do not use this for accuracy -- IoU needs a real
scene.

    python3 tools/sweep.py --axis resolution --out runs/sweep_res
    python3 tools/sweep.py --axis box        --out runs/sweep_box
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
    "image-size": ["1024", "768", "512"],
    "box": ["20", "96", "400"],
    "frames": ["50", "100", "300", "500", "1000"],
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
    return settings


def sweep(axis: str, values: list[str], args) -> list[dict]:
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

    for value in values:
        cfg = condition_settings(axis, value, args)
        kwargs = dict(base_kwargs)
        if cfg["image_size"] is not None:
            kwargs["image_size"] = cfg["image_size"]

        cache = out_dir / f"cache_{value.replace('x', '_')}"
        shutil.rmtree(cache, ignore_errors=True)
        frames_dir, prompts = build_synthetic_cache(
            cache, cfg["frames"], cfg["width"], cfg["height"], cfg["box"])

        warmup = min(args.warmup, max(1, cfg["frames"] // 4))
        print(f"\n[sweep] {cfg['label']}  ({cfg['width']}x{cfg['height']}, "
              f"{cfg['frames']} frames, warmup {warmup})")

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
               "image_size": cfg["image_size"] or 1024}
        if error:
            row["error"] = error
        if runs:
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
    return rows


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

_COLUMNS = [
    ("condition", "condition", "{}"),
    ("fps", "fps", "{:.2f}"),
    ("fps min", "fps_min", "{:.2f}"),
    ("fps max", "fps_max", "{:.2f}"),
    ("frame ms", "frame_mean_ms", "{:.2f}"),
    ("pre ms", "preprocess_ms", "{:.2f}"),
    ("infer ms", "inference_ms", "{:.2f}"),
    ("post ms", "postprocess_ms", "{:.2f}"),
    ("other ms", "unaccounted_ms", "{:.2f}"),
    ("enc ms", "encoder_ms", "{:.2f}"),
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

    usable = [r for r in rows if r.get("frame_mean_ms")]
    if not usable:
        return None
    labels = [r["condition"] for r in usable]
    ypos = range(len(usable))

    fig, ax = plt.subplots(figsize=(9.5, 1.9 + 0.62 * len(usable)))
    left = [0.0] * len(usable)
    for key, name, color in STAGES:
        widths = [float(r.get(key) or 0.0) for r in usable]
        ax.barh(list(ypos), widths, left=left, height=0.62, label=name,
                color=color, edgecolor="white", linewidth=0.6)
        left = [a + b for a, b in zip(left, widths)]

    for i, row in enumerate(usable):
        ax.text(left[i] + max(left) * 0.012, i, f"{row.get('fps', 0):.1f} FPS",
                va="center", fontsize=9, color="#333")

    ax.set_yticks(list(ypos))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("per-frame time (ms)")
    ax.set_title(f"Where the frame budget goes — sweeping {axis}")
    ax.set_xlim(0, max(left) * 1.16)
    ax.grid(axis="x", alpha=0.25)
    # Outside the axes: an inside legend lands on top of the longest bar, which
    # is always the condition the chart most needs to show.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18 - 0.02 * len(usable)),
              fontsize=9, ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path.resolve()


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--axis", required=True, choices=sorted(AXES),
                   help="Which input condition to vary.")
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
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--out", default="runs/sweep")
    p.add_argument("--keep-cache", action="store_true")
    args = p.parse_args()

    values = ([v.strip() for v in args.values.split(",") if v.strip()]
              if args.values else AXES[args.axis])
    rows = sweep(args.axis, values, args)

    out_dir = Path(args.out)
    table = markdown(rows)
    (out_dir / "report.md").write_text(
        f"# Input sweep — {args.axis}\n\nvariant: `{args.variant}`\n\n{table}\n")
    (out_dir / "report.csv").write_text(csv(rows) + "\n")
    (out_dir / "report.tex").write_text(latex(rows) + "\n")
    chart = stage_chart(rows, out_dir / "stages.png", args.axis)

    print(f"\n{table}\n")
    print(f"[sweep] report -> {out_dir / 'report.md'} (+ .csv, .tex)")
    if chart:
        print(f"[sweep] chart  -> {chart}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
