#!/usr/bin/env python3
"""Turn box-annotated datasets into teacher-mask pools, offline.

The image-side sibling of `make_masklets.py`. One command labels one dataset:
every box becomes a zoom-crop prompt to a strong image teacher (SAM 2.1 by
default, SAM 3 one string away), every returned mask runs the same four gates
the Anti-UAV410 labeller uses, and what survives lands in per-image run-length
stores under `--out`, with a `record.json` beside each saying which box each
mask answers. `src/training/pool.py` argues every decision.

**Calibrate before you spend.** Kust4K ships 4 024 registered RGB-T pairs with
real semantic masks; `calibrate` prompts the teacher with the drawn instances'
own boxes -- on the thermal frame, or on its registered RGB twin -- and prints
IoU against the drawn mask per class and size. That table, not an argument, is
what decides whether the thermal harvest prompts on thermal directly
(`--prompt self`) or rides the registration (`--prompt pair`).

Usage:
    # the measurement first: both routes over Kust4K's drawn instances
    python tools/make_mask_pool.py calibrate --spec kust4k \\
        --data /content/data/Kust4K --modality thermal --prompt self \\
        --json /content/work/cal_tir.json
    python tools/make_mask_pool.py calibrate --spec kust4k \\
        --data /content/data/Kust4K --modality thermal --prompt pair \\
        --json /content/work/cal_rgb.json

    # then the harvest
    python tools/make_mask_pool.py label --dataset hituav \\
        --data /content/data/HIT_UAV --out /content/work/pool --split train

    # SAM 3 instead: gated repo, needs transformers>=5
    python tools/make_mask_pool.py label ... --teacher facebook/sam3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.labels import Gates, build_image_teacher  # noqa: E402
from src.training.pool import (  # noqa: E402
    calibrate_spec,
    calibration_table,
    label_pool,
    pool_report,
    summarise_pool,
    write_index,
)

DATASETS = ("visdrone", "hituav", "dronevehicle", "dronevehicle_shared",
            "dronevehicle_only", "rgbtdroneperson", "vtuavdet", "birdsai",
            "vtuav")

# Which `fetch_datasets.py` recipe puts a dataset on disk. Only the readers
# that share an archive with another dataset need an entry; everything else
# fetches under its own name.
RECIPE_FOR = {"dronevehicle_shared": "dronevehicle",
              "dronevehicle_only": "dronevehicle",
              "vtuav": "vtuav_track"}


def frames_for(dataset: str, data: Path, split: str, modality: str,
               names_yaml: str | None = None):
    """The dataset's `BoxFrame` index, by its own reader."""
    from src.training import boxes

    if dataset == "visdrone":
        split_dir = next(iter(sorted(data.rglob(f"*DET-{split}"))), None)
        if split_dir is None:
            raise SystemExit(
                f"{data}: no *DET-{split} folder -- fetch it first:\n"
                f"    python tools/fetch_datasets.py visdrone --dest {data}")
        return boxes.yolo_frames(split_dir)
    if dataset == "hituav":
        return boxes.hituav_frames(data, split=split)
    if dataset == "dronevehicle":
        return boxes.dronevehicle_frames(data, modality=modality)
    if dataset == "dronevehicle_shared":
        # Boxes both halves annotate identically; `image` is the RGB frame, so
        # `--prompt self --mirror <name>` is one pass for two pools.
        return boxes.dronevehicle_shared_frames(data)
    if dataset == "dronevehicle_only":
        # The targets `--modality` sees and the other half never annotates.
        # `--prompt self` only: the other modality does not show them.
        return boxes.dronevehicle_only_frames(data, modality=modality)
    if dataset == "vtuav":
        # Its two halves carry their own boxes and disagree by a median 8.4 px,
        # so each is prompted on its own frame -- never `--prompt pair`.
        return boxes.vtuav_frames(
            data, modality="ir" if modality == "thermal" else "rgb")
    if dataset == "rgbtdroneperson":
        return boxes.rgbtdroneperson_frames(data, split=split, modality=modality)
    if dataset == "vtuavdet":
        return boxes.vtuavdet_frames(data, split=split, modality=modality)
    if dataset == "birdsai":
        # Thermal-only, so `modality` has nothing to select; `split` names the
        # archive half instead of a json (TrainReal / TestReal).
        return boxes.birdsai_frames(data, split="TestReal"
                                    if split in ("val", "test", "TestReal")
                                    else "TrainReal")
    raise SystemExit(f"dataset must be one of {DATASETS}, got {dataset!r}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--teacher", default="facebook/sam2.1-hiera-large",
                        help="facebook/sam2.1-hiera-large (default, ungated) "
                             "or facebook/sam3 (gated, transformers>=5). The "
                             "class is chosen from the id.")
    common.add_argument("--device", default="cuda")
    common.add_argument("--dtype", default="bfloat16")
    common.add_argument("--zoom", type=float, default=4.0,
                        help="Crop side, as a multiple of the box's long side.")
    common.add_argument("--min-size", type=int, default=128,
                        help="Floor on the crop side, in pixels.")
    common.add_argument("--batch", type=int, default=8,
                        help="Crops per teacher forward pass.")
    common.add_argument("--json", default=None, help="Also write results here.")

    label = sub.add_parser("label", parents=[common],
                           help="Harvest one box dataset into the pool.")
    label.add_argument("--dataset", required=True, choices=DATASETS)
    label.add_argument("--data", required=True, type=Path)
    label.add_argument("--out", required=True, type=Path,
                       help="Pool root; stores land under <out>/<dataset>/.")
    label.add_argument("--split", default="train",
                       help="visdrone/hituav/rgbtdroneperson/vtuavdet: which "
                            "split's annotations. birdsai reads it as "
                            "TrainReal unless it says val/test.")
    label.add_argument("--modality", choices=("thermal", "rgb"),
                       default="thermal",
                       help="dronevehicle/rgbtdroneperson/vtuavdet: which half "
                            "the pool supervises. birdsai is thermal-only.")
    label.add_argument("--mirror", default=None,
                       help="Second pool name to file the same masks under, "
                            "against each frame's registered twin. For "
                            "dronevehicle_shared: one teacher pass over the "
                            "RGB half supervises both modalities.")
    label.add_argument("--prompt", choices=("self", "pair"), default="self",
                       help="Which pixels the teacher looks at: the frame "
                            "itself, or its registered twin (the mask is "
                            "stored against the frame either way).")
    label.add_argument("--limit", type=int, default=None,
                       help="At most this many images.")
    label.add_argument("--frame-group", type=int, default=1,
                       help="How many frames pool their crops into one set of "
                            "teacher batches. 1 is per-frame batching; raise "
                            "it when --batch outruns a frame's box count (the "
                            "shared DroneVehicle subset has a median of 4).")
    label.add_argument("--max-boxes", type=int, default=64,
                       help="Boxes attempted per image, largest first. "
                            "VisDrone frames can carry hundreds.")
    label.add_argument("--box-iou", type=float, default=Gates.box_iou)
    label.add_argument("--teacher-iou", type=float, default=Gates.teacher_iou)
    label.add_argument("--component", type=float, default=Gates.component)
    label.add_argument("--no-resume", action="store_true",
                       help="Re-label images whose store already exists.")

    calibrate = sub.add_parser("calibrate", parents=[common],
                               help="Score the teacher against drawn masks.")
    calibrate.add_argument("--spec", default="kust4k",
                           help="Which SPECS entry's masks are the truth.")
    calibrate.add_argument("--data", required=True, type=Path)
    calibrate.add_argument("--modality", choices=("thermal", "rgb"),
                           default="thermal",
                           help="Which half's images carry the instances.")
    calibrate.add_argument("--prompt", choices=("self", "pair"),
                           default="self")
    calibrate.add_argument("--frames", type=int, default=200,
                           help="Frames sampled (seeded, reproducible).")
    calibrate.add_argument("--per-frame", type=int, default=8,
                           help="Instances per frame, largest first.")
    calibrate.add_argument("--seed", type=int, default=0)

    args = p.parse_args(argv)
    teacher = build_image_teacher(args.teacher, device=args.device,
                                  dtype=args.dtype)

    if args.command == "label":
        frames = frames_for(args.dataset, args.data, args.split, args.modality)
        print(f"{len(frames)} annotated images, teacher {args.teacher}, "
              f"prompting on {args.prompt}")
        gates = Gates(teacher_iou=args.teacher_iou, box_iou=args.box_iou,
                      component=args.component)
        dataset = (f"{args.dataset}_{args.modality}"
                   if args.dataset == "dronevehicle" else args.dataset)
        report = label_pool(
            frames, teacher, args.out, dataset=dataset, prompt=args.prompt,
            gates=gates, zoom=args.zoom, min_size=args.min_size,
            batch_size=args.batch, limit=args.limit, max_boxes=args.max_boxes,
            resume=not args.no_resume, mirror=args.mirror,
            frame_group=args.frame_group, progress=_tqdm())
        write_index(args.out)
        print(f"\n{report['accepted']}/{report['attempted']} boxes accepted "
              f"({report['acceptance_rate']:.1%}); "
              f"{report['resumed']} image(s) already done")
        print()
        print(summarise_pool(pool_report(args.out)))
        if args.json:
            _write(args.json, report)
        return 0

    from src.training.aerial import SPECS

    records = calibrate_spec(
        args.data, SPECS[args.spec], teacher, modality=args.modality,
        prompt=args.prompt, limit_frames=args.frames,
        per_frame=args.per_frame, zoom=args.zoom, min_size=args.min_size,
        batch_size=args.batch, seed=args.seed, progress=_tqdm())
    print(f"\n{len(records)} instances scored, prompting on {args.prompt}")
    print()
    print(calibration_table({args.prompt: records}))
    if args.json:
        _write(args.json, records)
    return 0


def _write(path: str, payload) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")


def _tqdm():
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - environment, not logic
        return None
    return lambda stream, total, desc: tqdm(stream, total=total, desc=desc)


if __name__ == "__main__":
    raise SystemExit(main())
