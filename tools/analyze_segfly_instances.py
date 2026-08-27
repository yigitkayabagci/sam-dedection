#!/usr/bin/env python3
"""Audit SegFly semantic-to-instance conversion and render report figures.

SegFly labels pixels by semantic class.  EdgeTAM stage B needs one mask per
prompted object.  This tool runs the exact training conversion from
``src.training.aerial`` on real exported SegFly files, compares ``components``
with ``watershed``, and writes a compact JSON record plus a six-panel PNG.

The optional SAM check is a disagreement audit, not ground truth.  It prompts
SAM with each reconstructed component's box and compares SAM's mask with the
SegFly-derived component.  Agreement can rank frames for review; it cannot
prove either pseudo-label is correct.

Examples
--------
Export a small thermal slice, then render the most diagnostic frame::

    python tools/export_hf_dataset.py markus-42/SegFly \
        --dest /content/data/SegFly --modality thermal --limit 64 \
        --expect segfly
    python tools/analyze_segfly_instances.py \
        --root /content/data/SegFly \
        --figure report/figures/segfly-instance-conversion.png \
        --json report/figures/segfly-instance-conversion.json

Add an aligned-RGB SAM 2.1 disagreement audit::

    python tools/analyze_segfly_instances.py \
        --root /content/data/SegFly \
        --sam-teacher facebook/sam2.1-hiera-large \
        --sam-route pair --sam-frames 32
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.aerial import (  # noqa: E402
    Frame,
    Instance,
    InstanceGates,
    SPECS,
    Source,
    decompose,
    list_frames,
    read_mask,
)


# Published RGB colors from the official SegFly class table.  The annotation
# file itself remains a single-channel ID map; these colors are display-only.
SEGFLY_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (128, 0, 128),
    2: (204, 163, 72),
    3: (128, 0, 0),
    4: (192, 192, 192),
    6: (0, 255, 0),
    7: (112, 148, 32),
    8: (64, 64, 0),
    9: (255, 255, 0),
    13: (0, 128, 128),
    14: (0, 0, 255),
    16: (255, 0, 0),
    17: (64, 160, 120),
    33: (128, 64, 128),
    34: (240, 120, 120),
    36: (128, 128, 64),
}

INSTANCE_COLORS = (
    (230, 85, 55),
    (45, 150, 220),
    (65, 170, 95),
    (210, 155, 40),
    (145, 90, 190),
    (40, 180, 175),
    (220, 105, 155),
    (115, 120, 125),
)


@dataclass(frozen=True)
class ModeAudit:
    mode: str
    components: np.ndarray
    instances: tuple[Instance, ...]
    rejects: dict[str, int]

    def record(self, source: Source) -> dict:
        classes = Counter(source.spec.name_of(i.class_id) for i in self.instances)
        areas = [i.area for i in self.instances]
        fills = [i.fill for i in self.instances]
        return {
            "kept": len(self.instances),
            "by_class": dict(sorted(classes.items())),
            "rejected": dict(sorted(self.rejects.items())),
            "median_area_px": round(float(np.median(areas)), 1) if areas else None,
            "median_fill": round(float(np.median(fills)), 4) if fills else None,
        }


def colorize_semantic(semantic: np.ndarray) -> np.ndarray:
    """Convert a single-channel SegFly ID map to the official RGB palette."""
    semantic = np.asarray(semantic)
    out = np.zeros((*semantic.shape[:2], 3), dtype=np.uint8)
    known = np.zeros(semantic.shape[:2], dtype=bool)
    for class_id, color in SEGFLY_COLORS.items():
        selected = semantic == class_id
        out[selected] = color
        known |= selected
    # Unexpected IDs are loud in a visual audit; accepted leftovers are still
    # visible here even though the training spec ignores them.
    out[~known] = (255, 0, 255)
    return out


def read_rgb(path: Path) -> np.ndarray:
    from PIL import Image, ImageOps

    image = Image.open(path)
    if image.mode in ("I", "I;16", "F", "L"):
        image = ImageOps.autocontrast(image.convert("L")).convert("RGB")
    else:
        image = image.convert("RGB")
    return np.asarray(image)


def audit(semantic: np.ndarray, source: Source) -> dict[str, ModeAudit]:
    out = {}
    for mode in ("components", "watershed"):
        components, instances, rejects = decompose(
            semantic, source.spec, source.gates, mode
        )
        out[mode] = ModeAudit(
            mode=mode,
            components=components,
            instances=tuple(instances),
            rejects=rejects,
        )
    return out


def diagnostic_score(modes: dict[str, ModeAudit]) -> tuple[int, int, int]:
    """Prefer frames that expose the method's known fusion failure."""
    plain, water = modes["components"], modes["watershed"]
    return (
        int(plain.rejects.get("fill", 0) + water.rejects.get("fill", 0)),
        max(len(water.instances) - len(plain.instances), 0),
        len(plain.instances),
    )


def select_frame(
    frames: list[Frame], source: Source, requested: str | None, scan: int
) -> tuple[Frame, np.ndarray, dict[str, ModeAudit]]:
    if requested:
        matches = [f for f in frames if f.name == requested or f.image.name == requested]
        if not matches:
            raise SystemExit(
                f"No frame named {requested!r}. First available: "
                + ", ".join(f.name for f in frames[:5])
            )
        frame = matches[0]
        semantic = read_mask(frame.mask)
        return frame, semantic, audit(semantic, source)

    candidates = []
    for frame in frames[: max(int(scan), 1)]:
        semantic = read_mask(frame.mask)
        modes = audit(semantic, source)
        candidates.append((diagnostic_score(modes), frame, semantic, modes))
    if not candidates:
        raise SystemExit("No SegFly image/label pairs were found.")
    _, frame, semantic, modes = max(candidates, key=lambda row: row[0])
    return frame, semantic, modes


def things_view(semantic: np.ndarray, source: Source) -> np.ndarray:
    out = np.zeros((*semantic.shape[:2], 3), dtype=np.uint8)
    for class_id in source.spec.thing_ids:
        out[semantic == class_id] = SEGFLY_COLORS[class_id]
    return out


def instance_overlay(base: np.ndarray, result: ModeAudit, source: Source) -> np.ndarray:
    from PIL import Image, ImageDraw

    out = base.astype(np.float32).copy()
    for index, instance in enumerate(result.instances):
        selected = result.components == instance.label
        color = np.asarray(INSTANCE_COLORS[index % len(INSTANCE_COLORS)])
        out[selected] = 0.48 * out[selected] + 0.52 * color
    image = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    per_class: Counter[str] = Counter()
    for index, instance in enumerate(result.instances):
        name = source.spec.name_of(instance.class_id)
        per_class[name] += 1
        color = INSTANCE_COLORS[index % len(INSTANCE_COLORS)]
        draw.rectangle(instance.box, outline=color, width=2)
        draw.text(
            (instance.box[0] + 2, instance.box[1] + 2),
            f"{name[0].upper()}{per_class[name]}",
            fill=color,
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )
    return np.asarray(image)


def render_figure(
    frame: Frame,
    semantic: np.ndarray,
    modes: dict[str, ModeAudit],
    source: Source,
    path: Path,
) -> Path:
    import matplotlib.pyplot as plt

    thermal = read_rgb(frame.image)
    aligned = read_rgb(frame.pair) if frame.pair else thermal
    panels = (
        (thermal, "Thermal"),
        (aligned, "RGB aligned" if frame.pair else "RGB unavailable"),
        (colorize_semantic(semantic), "Semantic map: 15 classes + unlabeled"),
        (things_view(semantic, source), "Thing classes: vehicle + truck"),
        (
            instance_overlay(thermal, modes["components"], source),
            _mode_title(modes["components"]),
        ),
        (
            instance_overlay(thermal, modes["watershed"], source),
            _mode_title(modes["watershed"]),
        ),
    )

    figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.4), constrained_layout=True)
    for axis, (pixels, title) in zip(axes.flat, panels):
        axis.imshow(pixels)
        axis.set_title(title, fontsize=10)
        axis.set_axis_off()
    figure.suptitle(
        f"SegFly {frame.name}: semantic map to prompted instances",
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.004,
        "Source: SegFly (CC BY-NC-SA 4.0). Boxes and labels are produced by this repository.",
        ha="center",
        fontsize=8,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return path


def _mode_title(result: ModeAudit) -> str:
    rejected = sum(result.rejects.values())
    detail = ", ".join(f"{k}={v}" for k, v in sorted(result.rejects.items()))
    suffix = f"; rejected {rejected}" + (f" ({detail})" if detail else "")
    return f"{result.mode}: kept {len(result.instances)}{suffix}"


def sam_records(
    root: Path,
    source: Source,
    model_id: str,
    route: str,
    frames: int,
    instances: int,
    device: str,
    dtype: str,
) -> dict[str, list[dict]]:
    """Run the existing teacher-versus-component diagnostic on a small slice."""
    from src.training.labels import build_image_teacher
    from src.training.pool import calibrate_spec

    teacher = build_image_teacher(model_id, device=device, dtype=dtype)
    routes = ("self", "pair") if route == "both" else (route,)
    return {
        name: calibrate_spec(
            root,
            source.spec,
            teacher,
            modality="thermal",
            prompt=name,
            limit_frames=frames,
            per_frame=instances,
        )
        for name in routes
    }


def summarise_sam(routes: dict[str, list[dict]]) -> dict:
    out = {}
    for route, records in routes.items():
        values = np.asarray([row["iou"] for row in records], dtype=np.float64)
        out[route] = {
            "instances": len(records),
            "mean_component_vs_sam_iou": (
                round(float(values.mean()), 4) if values.size else None
            ),
            "median_component_vs_sam_iou": (
                round(float(np.median(values)), 4) if values.size else None
            ),
            "below_0_5": int((values < 0.5).sum()),
            "records": records,
        }
    return out


def build_record(
    frame: Frame,
    semantic: np.ndarray,
    modes: dict[str, ModeAudit],
    source: Source,
    figure: Path,
    sam: dict | None,
) -> dict:
    thing_pixels = int(np.isin(semantic, source.spec.thing_ids).sum())
    return {
        "dataset": "segfly",
        "frame": frame.name,
        "image": str(frame.image),
        "label": str(frame.mask),
        "rgb_aligned": str(frame.pair) if frame.pair else None,
        "shape": [int(semantic.shape[0]), int(semantic.shape[1])],
        "palette_values": [int(v) for v in np.unique(semantic)],
        "thing_ids": {
            source.spec.name_of(v): int(v) for v in source.spec.thing_ids
        },
        "thing_pixels": thing_pixels,
        "thing_pixel_share": round(thing_pixels / float(semantic.size), 6),
        "gates": asdict(source.gates),
        "modes": {name: result.record(source) for name, result in modes.items()},
        "sam_disagreement_audit": sam,
        "figure": str(figure),
        "interpretation": (
            "Connected components and SAM are pseudo-label diagnostics, not "
            "instance ground truth. Use SegFly for training; report held-out "
            "instance metrics on a dataset with drawn instance masks."
        ),
    }


def markdown(record: dict) -> str:
    plain = record["modes"]["components"]
    water = record["modes"]["watershed"]
    return "\n".join(
        [
            f"SegFly frame: `{record['frame']}` ({record['shape'][1]}×{record['shape'][0]})",
            f"Thing pixels: {record['thing_pixels']} ({100 * record['thing_pixel_share']:.2f}%)",
            f"Components: {plain['kept']} kept, {sum(plain['rejected'].values())} rejected",
            f"Watershed: {water['kept']} kept, {sum(water['rejected'].values())} rejected",
            f"Figure: `{record['figure']}`",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--frame", default=None, help="Frame key or image filename.")
    parser.add_argument(
        "--scan",
        type=int,
        default=128,
        help="Without --frame, scan this many rows and choose a diagnostic example.",
    )
    parser.add_argument(
        "--figure", type=Path, default=Path("report/figures/segfly-instance-conversion.png")
    )
    parser.add_argument(
        "--json", type=Path, default=Path("report/figures/segfly-instance-conversion.json")
    )
    parser.add_argument("--min-area", type=int, default=48)
    parser.add_argument("--min-side", type=int, default=4)
    parser.add_argument("--max-area", type=float, default=0.9)
    parser.add_argument("--fill", type=float, default=0.25)
    parser.add_argument(
        "--sam-teacher",
        default=None,
        help="Optional model id, e.g. facebook/sam2.1-hiera-large.",
    )
    parser.add_argument("--sam-route", choices=("self", "pair", "both"), default="pair")
    parser.add_argument("--sam-frames", type=int, default=32)
    parser.add_argument("--sam-instances", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args(argv)

    gates = InstanceGates(
        min_area=args.min_area,
        min_side=args.min_side,
        max_area=args.max_area,
        fill=args.fill,
    )
    source = Source(SPECS["segfly"], gates=gates, mode="components", role="train")
    frames = list_frames(args.root, source.spec, "thermal")
    frame, semantic, modes = select_frame(frames, source, args.frame, args.scan)
    render_figure(frame, semantic, modes, source, args.figure)

    sam = None
    if args.sam_teacher:
        sam = summarise_sam(
            sam_records(
                args.root,
                source,
                args.sam_teacher,
                args.sam_route,
                args.sam_frames,
                args.sam_instances,
                args.device,
                args.dtype,
            )
        )
    record = build_record(frame, semantic, modes, source, args.figure, sam)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(record, indent=2) + "\n")
    print(markdown(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
