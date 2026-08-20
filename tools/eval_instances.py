#!/usr/bin/env python3
"""Score a checkpoint on held-out instances: box in, mask out, IoU.

The static counterpart of `tools/eval_antiuav.py`, and it should be read with
the same caution the module it calls states: this scores **one prompted frame**,
so it cannot see the failure this project exists to fix, which only appears
once a memory bank is in the loop. What it can see is whether the encoder now
separates one vehicle from the one parked beside it -- which is exactly what
the static stage is trying to buy, and the only thing it can be held to before
the video stage exists.

Per-class numbers and a small-object bucket are reported alongside the mean,
because they are where a change is visible: an aerial dataset's mean IoU is
dominated by its trucks, and the targets this pipeline cares about are the ones
twenty pixels across.

Usage:
    python tools/eval_instances.py --data /content/data/Kust4K --spec kust4k \\
        --checkpoint checkpoints/edgetam_aerial_512.pt --split test \\
        --json /content/instances_finetune.json
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

from src.training.aerial import (  # noqa: E402
    SPECS,
    InstanceGates,
    load_index,
    sample_windows,
    split_frames,
)

# "Small" is the bucket the deployment lives in: a target this size is a few
# hundred pixels of a 512 window, which is where a tracker starts losing things.
SMALL_SIDE = 32.0


def score(model, split, batch: int, device: str, progress=None) -> dict:
    """Mean IoU over every instance in `split`, and the breakdowns that matter."""
    from src.training.image_loop import collate, instance_iou
    from src.training.loader import batch_clips

    ious: list[float] = []
    classes: list[int] = []
    sides: list[float] = []

    chunks = list(batch_clips(split.samples, batch, seed=0, drop_last=False))
    stream = progress(chunks, total=len(chunks), desc="scoring") if progress else chunks
    for chunk in stream:
        # `split.split` is not optional here: decomposing differently from the
        # index would renumber the component labels and score every instance
        # against another instance's mask -- and still produce a plausible IoU.
        assembled = collate(chunk, split.spec, split.gates, "cpu",
                            gray=split.gray, split=split.split)
        ious.extend(instance_iou(model, assembled.to(device)).tolist())
        for sample in chunk:
            for instance in sample.instances:
                classes.append(instance.class_id)
                # In *window* pixels, which is what the model saw -- a 40-pixel
                # car in a resized 4000-wide frame is not a 40-pixel car here.
                scale = sample.size / max(sample.window)
                sides.append(max(instance.width, instance.height) * scale)

    ious = np.asarray(ious, dtype=np.float64)
    classes = np.asarray(classes)
    sides = np.asarray(sides, dtype=np.float64)
    small = sides < SMALL_SIDE

    return {
        "instances": int(ious.size),
        "mean_iou": float(ious.mean()) if ious.size else float("nan"),
        "iou_50": float((ious >= 0.5).mean()) if ious.size else float("nan"),
        "iou_75": float((ious >= 0.75).mean()) if ious.size else float("nan"),
        "small_instances": int(small.sum()),
        "small_mean_iou": float(ious[small].mean()) if small.any() else float("nan"),
        "large_mean_iou": float(ious[~small].mean()) if (~small).any() else float("nan"),
        "per_class": {int(c): {"instances": int((classes == c).sum()),
                               "mean_iou": float(ious[classes == c].mean())}
                      for c in sorted(set(classes.tolist()))},
    }


def report(result: dict, spec) -> str:
    lines = [
        f"{result['instances']} instances  mean IoU {result['mean_iou']:.4f}  "
        f"IoU>=0.5 {result['iou_50']:.3f}  IoU>=0.75 {result['iou_75']:.3f}",
        f"  small (< {SMALL_SIDE:.0f} px in the window): "
        f"{result['small_instances']} at {result['small_mean_iou']:.4f}   "
        f"larger: {result['large_mean_iou']:.4f}",
        "",
        "| class | instances | mean IoU |",
        "|---|---:|---:|",
    ]
    for class_id, row in result["per_class"].items():
        lines.append(f"| {spec.name_of(class_id)} | {row['instances']} | "
                     f"{row['mean_iou']:.4f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--spec", default="kust4k", choices=sorted(SPECS))
    p.add_argument("--modality", default="thermal", choices=("thermal", "rgb"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--index", default=None,
                   help="Reuse the index train_encoder.py wrote; built here if absent.")
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--per-image", type=int, default=1)
    p.add_argument("--max-instances", type=int, default=8)
    p.add_argument("--min-area", type=int, default=48)
    p.add_argument("--fill", type=float, default=0.25)
    p.add_argument("--split-touching", choices=("none", "watershed"), default="none",
                   help="Must match what the index and the training run used.")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)

    from src.training.image_loop import ImageSplit
    from tools.train_encoder import _tqdm, build_index, build_model

    spec = SPECS[args.spec]
    gates = InstanceGates(min_area=args.min_area, fill=args.fill)
    cache = Path(args.index) if args.index else None
    index = (load_index(cache) if cache and cache.is_file()
             else build_index(Path(args.data), spec, args.modality, gates, cache, 8,
                              split=args.split_touching))

    # The same split, from the same seed, as training used -- otherwise "held
    # out" is a claim rather than a fact.
    parts = split_frames([e.frame for e in index], seed=args.seed)
    by_name = {e.frame.name: e for e in index}
    chosen = [by_name[f.name] for f in parts[args.split]]

    split = ImageSplit(
        samples=sample_windows(chosen, size=args.size, per_image=args.per_image,
                               max_instances=args.max_instances, jitter=0,
                               seed=args.seed),
        spec=spec, gates=gates, gray=args.modality == "thermal",
        split=args.split_touching)
    print(f"{args.split}: {len(chosen)} frames, {len(split.samples)} windows, "
          f"{sum(len(s.instances) for s in split.samples)} instances")

    model = build_model(args.size, args.checkpoint, args.device)
    result = score(model, split, args.batch, args.device, progress=_tqdm())
    result |= {"checkpoint": args.checkpoint, "split": args.split,
               "dataset": spec.name, "modality": args.modality}

    print()
    print(report(result, spec))
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
