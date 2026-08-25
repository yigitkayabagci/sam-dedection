#!/usr/bin/env python3
"""Rescue semantic datasets into per-instance masks, offline.

The third labeller in this repo, and the one that needs no boxes at all:

    make_masklets.py     video + per-frame box  -> masklets      (stage C)
    make_mask_pool.py    image + detection box  -> pool masks    (stage B)
    make_semantic_pool.py  image + semantic map -> instances     (stage B)

`decompose` already turns a semantic map into instances, and it fuses whatever
the annotator drew flush -- 78.5 % one-car-one-blob on SegFly, 82.0 % on
iSAID's drawn vehicles. This tool seeds the teacher inside each component and
lets a **click** separate what the label map could not, then verifies the
answer against the map itself. `src/training/semantic.py` argues every gate.

**Measure before you spend.** `measure` runs both routes over a dataset that
ships *drawn instances*, flattens them to the semantic map they imply, and
scores each route against the drawing. If the rescue does not beat plain
`decompose` on recall there, it will not beat it on a dataset where nobody can
check.

Usage:
    # the decision: both routes against drawn truth
    python tools/make_semantic_pool.py measure --truth isaid \\
        --data /content/data/iSAID --frames 40 \\
        --json /content/work/rescue.json

    # then the harvest, on a semantic set with no instances of its own
    python tools/make_semantic_pool.py label --spec kust4k \\
        --data /content/data/Kust4K --out /content/work/semantic \\
        --modality thermal

    # SAM 3 instead of the ungated default
    python tools/make_semantic_pool.py label ... --teacher facebook/sam3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.aerial import SPECS  # noqa: E402
from src.training.labels import build_image_teacher  # noqa: E402
from src.training.semantic import (  # noqa: E402
    SemanticGates,
    estimate_units,
    label_semantic_pool,
    measure_rescue,
    summarise_rescue,
    summarise_semantic_pool,
)

TRUTH = ("isaid", "vtuav_vis")


def _progress(stream, total, desc):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return stream
    return tqdm(stream, total=total, desc=desc)


# --------------------------------------------------------------------------
# Truth readers -- datasets that ship drawn instances
# --------------------------------------------------------------------------


def isaid_samples(root: Path, spec, limit: int | None = None):
    """`(pixels, instance_map, class_of)` from iSAID's instance-id PNGs.

    iSAID packs an instance id into an RGB colour (`_instance_id_RGB.png`) and
    its class into a second, semantic PNG (`_instance_color_RGB.png`). The id
    map alone cannot say which class an instance is, so the two are read
    together: the id gives the separation, the semantic colour gives the class,
    and the class of an id is the semantic value most of its pixels carry.

    Every instance is mapped onto the spec's single thing class. iSAID's own
    fifteen categories are not this project's palette and the rescue does not
    care -- what is being measured is whether a click separates two touching
    objects, and that question has no class in it.
    """
    from src.training.pool import _read_rgb

    ids = sorted(root.rglob("*_instance_id_RGB.png"))
    if not ids:
        raise FileNotFoundError(
            f"{root}: no *_instance_id_RGB.png anywhere underneath -- this "
            f"wants iSAID's Instance_masks/images/ folder.")
    thing = int(spec.classes[spec.things[0]])
    for path in ids[:limit]:
        image = path.with_name(path.name.replace("_instance_id_RGB", ""))
        if not image.is_file():
            candidates = list(root.rglob(f"{path.name.split('_instance')[0]}.png"))
            if not candidates:
                continue
            image = candidates[0]
        colours = _read_rgb(path).astype(np.int64)
        packed = (colours[..., 0] << 16) | (colours[..., 1] << 8) | colours[..., 2]
        yield _read_rgb(image), packed, {
            int(v): thing for v in np.unique(packed) if int(v) != 0}


def vtuav_samples(root: Path, spec, limit: int | None = None):
    """`(pixels, instance_map, class_of)` from VTUAV's drawn target masks.

    One target per frame, values `{0, 255}` -- so this measures the rescue's
    *precision* rather than its separation: a route that splits a single drawn
    target into two is wrong here, and that is worth knowing before trusting it
    where nobody can check.
    """
    from src.training.aerial import list_frames, read_mask
    from src.training.pool import _read_rgb

    thing = int(spec.classes[spec.things[0]])
    for frame in list_frames(root, spec, "rgb")[:limit]:
        mask = read_mask(frame.mask)
        yield _read_rgb(frame.image), (mask > 0).astype(np.int32), {1: thing}


SAMPLERS = {"isaid": isaid_samples, "vtuav_vis": vtuav_samples}


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def run_measure(args) -> int:
    spec = SPECS["vtuav_vis"] if args.truth == "vtuav_vis" else SPECS["kust4k"]
    if args.truth == "isaid":
        # iSAID is not in SPECS and does not need to be: the rescue reads a
        # semantic map and a thing class, and one synthetic class is all a
        # separation question needs. Reusing kust4k's palette would silently
        # bind this measurement to that dataset's ids.
        from src.training.aerial import DatasetSpec
        spec = DatasetSpec(
            name="isaid_truth", masks="**/*.png",
            classes={"background": 0, "object": 1}, things=("object",),
            thermal="**/*.png", rgb=None, ignore=(0,),
            palette_source="synthetic: one thing class, for the rescue "
                           "measurement only -- iSAID's own 15 categories are "
                           "not this project's palette and separation has no "
                           "class in it")
    teacher = build_image_teacher(args.teacher, device=args.device,
                                  dtype=args.dtype)
    samples = SAMPLERS[args.truth](Path(args.data), spec, args.frames)
    routes = measure_rescue(
        samples, spec, teacher, gates=SemanticGates(purity=args.purity),
        zoom=args.zoom, min_size=args.min_size, batch_size=args.batch,
        seed_cap=args.seed_cap, match_iou=args.match_iou, progress=_progress)
    table = summarise_rescue(routes)
    print()
    print(table)
    print("\nRead recall first: that is the fusion question. If `rescue` does "
          "\nnot beat `components` here, it will not beat it where nobody can "
          "\ncheck -- keep decompose and spend the GPU-hours on notebook 14.")
    if args.json:
        Path(args.json).write_text(json.dumps(routes, indent=1) + "\n")
    return 0


def run_label(args) -> int:
    spec = SPECS[args.spec]
    teacher = build_image_teacher(args.teacher, device=args.device,
                                  dtype=args.dtype)
    units = estimate_units(args.data, spec, modality=args.modality,
                           mode=args.mode, limit=args.unit_frames,
                           progress=_progress)
    print("unit area per class (px, median over the sample):")
    for class_id, area in sorted(units.items()):
        print(f"  {spec.name_of(class_id):<22} {area:8.0f}")

    report = label_semantic_pool(
        args.data, spec, teacher, args.out, dataset=args.spec,
        modality=args.modality, prompt=args.prompt,
        gates=SemanticGates(purity=args.purity, containment=args.containment),
        mode=args.mode, zoom=args.zoom, min_size=args.min_size,
        batch_size=args.batch, seed_cap=args.seed_cap, units=units,
        limit=args.limit, resume=not args.no_resume, progress=_progress)
    print()
    print(summarise_semantic_pool(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--teacher", default="facebook/sam2.1-hiera-large",
                        help="Image teacher. Anything with 'sam3' in the id "
                             "loads SAM 3 (gated, transformers>=5).")
    common.add_argument("--device", default="cuda")
    common.add_argument("--dtype", default="bfloat16")
    common.add_argument("--zoom", type=float, default=4.0)
    common.add_argument("--min-size", type=int, default=128)
    common.add_argument("--batch", type=int, default=8)
    common.add_argument("--seed-cap", type=int, default=12,
                        help="Most seeds one component may get.")
    common.add_argument("--purity", type=float, default=SemanticGates.purity)
    common.add_argument("--json", default=None)

    measure = sub.add_parser("measure", parents=[common],
                             help="Both routes against drawn instances.")
    measure.add_argument("--truth", required=True, choices=TRUTH)
    measure.add_argument("--data", required=True, type=Path)
    measure.add_argument("--frames", type=int, default=40)
    measure.add_argument("--match-iou", type=float, default=0.5)
    measure.set_defaults(func=run_measure)

    label = sub.add_parser("label", parents=[common],
                           help="Harvest a semantic dataset into instances.")
    label.add_argument("--spec", required=True, choices=sorted(SPECS))
    label.add_argument("--data", required=True, type=Path)
    label.add_argument("--out", required=True, type=Path)
    label.add_argument("--modality", choices=("thermal", "rgb"),
                       default="thermal")
    label.add_argument("--prompt", choices=("self", "pair"), default="self",
                       help="Which pixels the teacher looks at. 'pair' needs "
                            "halves of the same size; see pool.label_pool.")
    label.add_argument("--mode", default="components",
                       choices=("components", "watershed", "labels"))
    label.add_argument("--containment", type=float,
                       default=SemanticGates.containment)
    label.add_argument("--unit-frames", type=int, default=200,
                       help="Frames sampled to estimate one object's area.")
    label.add_argument("--limit", type=int, default=None)
    label.add_argument("--no-resume", action="store_true")
    label.set_defaults(func=run_label)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
