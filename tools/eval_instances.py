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

Pass the **same `--dataset` flags the training run used**, or the stratified
split will not be the same one and "held out" stops being true. Numbers are
reported per `dataset/class`, because `car` is class 5 in Kust4K and something
else elsewhere.

`--pool` is the same flag `train_encoder.py` takes and carries the same
warning: pass the run's own flags, in the same roles, or the stratified split
is a different split and "held out" stops being true.

Usage:
    python tools/eval_instances.py --dataset kust4k:/content/data/Kust4K \\
        --checkpoint checkpoints/edgetam_aerial_512.pt --split test \\
        --json /content/instances_finetune.json

    python tools/eval_instances.py \\
        --pool /content/pool/kust4k_thermal:/content/data/Kust4K:thermal:all \\
        --checkpoint third_party/EdgeTAM/checkpoints/edgetam.pt \\
        --split test --prompt point --json /content/pool_stock.json
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
    InstanceGates,
    sample_windows,
    apply_splits,
    split_index,
)
from src.training.datasets import (  # noqa: E402
    describe,
    describe_pools,
    index_pools,
    parse,
    parse_pool,
)

# "Small" is the bucket the deployment lives in: a target this size is a few
# hundred pixels of a 512 window, which is where a tracker starts losing things.
SMALL_SIDE = 32.0

# Where a target stops standing out from the ground around it, in units of that
# ground's own variation (`image_loop.instance_contrast`). Under 1 the target is
# inside the background's noise -- a grey car on a grey roof, a body at ambient
# temperature -- and over 3 it is the case every aerial demo is filmed in. The
# mean IoU of a mixed test set is dominated by the third bucket, which is how a
# model that only works on obvious targets keeps looking fine.
CONTRAST_EDGES = (1.0, 3.0)
CONTRAST_NAMES = ("in the clutter (<1)", "similar (1-3)", "standing out (>3)")


def score(model, split, batch: int, device: str, progress=None,
          prompt: str = "box", jitter: float = 0.0, seed: int = 0,
          detail: bool = False) -> dict:
    """Mean IoU over every instance in `split`, and the breakdowns that matter.

    `prompt` is what the model is told about the target -- the default `box`
    hands it the ground-truth rectangle, `jitter` loosens each edge by up to
    `jitter` of its own side, and `point` gives one click at the centre and
    nothing else. See `src/training/image_loop.instance_iou` for why the
    weaker two are the ones that can see an encoder change.

    The jitter generator is seeded per call, so two checkpoints scored with the
    same `--seed` are handed the *same* perturbed boxes and the difference
    between their numbers is still the checkpoint.

    `detail` adds a `rows` list, one entry per instance, each keyed by the
    window it was scored in. Two runs over the same split and seed produce the
    same keys in the same order, so pairing them says which *instances* a
    checkpoint gained and which it lost -- the question a mean IoU cannot
    answer, and the one that decides whether a change is worth shipping.
    """
    import torch

    from src.training.image_loop import collate, instance_contrast, instance_iou
    from src.training.loader import batch_clips

    ious: list[float] = []
    classes: list[str] = []
    sides: list[float] = []
    mods: list[str] = []
    contrasts: list[float] = []
    rows: list[dict] = []

    generator = torch.Generator(device=device).manual_seed(seed)
    chunks = list(batch_clips(split.samples, batch, seed=0, drop_last=False))
    stream = progress(chunks, total=len(chunks), desc="scoring") if progress else chunks
    for chunk in stream:
        assembled = collate(chunk, "cpu").to(device)
        ious.extend(instance_iou(model, assembled, prompt, jitter,
                                 generator).tolist())
        # Same flat row order the IoUs come back in, so the two align without a
        # join: `instance_contrast` walks `valid` exactly as `propagate_image`
        # does.
        contrasts.extend(instance_contrast(assembled).tolist())
        for sample in chunk:
            spec = sample.source.spec if sample.source else None
            for instance in sample.instances:
                # Qualified by dataset: `car` is class 5 in Kust4K and
                # something else elsewhere, and adding two unrelated
                # distributions together would report neither.
                classes.append(f"{spec.name}/{spec.name_of(instance.class_id)}"
                               if spec else str(instance.class_id))
                # In *window* pixels, which is what the model saw -- a 40-pixel
                # car in a resized 4000-wide frame is not a 40-pixel car here.
                scale = sample.size / max(sample.window)
                sides.append(max(instance.width, instance.height) * scale)
                # By input modality, so a mixed run can be read. A test split
                # holding thermal and RGB windows of the same flights blends
                # two problems into one mean; splitting on `gray` -- which is
                # what actually changed what the model saw -- unblends it.
                mods.append("thermal" if sample.source is None
                            or sample.source.gray else "rgb")
                if detail:
                    # The window is in the key because `--per-image` samples
                    # several crops of one frame and each is its own question.
                    rows.append({
                        "key": f"{classes[-1]}|{sample.frame.name}|"
                               f"{tuple(sample.window)}|{instance.label}",
                        "source": spec.name if spec else "?",
                        "frame": sample.frame.name, "label": int(instance.label),
                        "side": sides[-1], "modality": mods[-1]})

    if detail:
        for row, value, seen in zip(rows, ious, contrasts):
            row["iou"] = float(value)
            row["contrast"] = float(seen)
    ious = np.asarray(ious, dtype=np.float64)
    classes = np.asarray(classes)
    sides = np.asarray(sides, dtype=np.float64)
    mods = np.asarray(mods)
    contrasts = np.asarray(contrasts, dtype=np.float64)
    small = sides < SMALL_SIDE
    stands_out = np.digitize(contrasts, CONTRAST_EDGES)

    def bucket(mask: np.ndarray) -> dict:
        tiny = mask & small
        return {
            "instances": int(mask.sum()),
            "mean_iou": float(ious[mask].mean()) if mask.any() else float("nan"),
            "iou_50": float((ious[mask] >= 0.5).mean()) if mask.any() else float("nan"),
            "small_instances": int(tiny.sum()),
            "small_mean_iou": float(ious[tiny].mean()) if tiny.any() else float("nan"),
        }

    return {
        "instances": int(ious.size),
        **({"rows": rows} if detail else {}),
        "mean_iou": float(ious.mean()) if ious.size else float("nan"),
        "iou_50": float((ious >= 0.5).mean()) if ious.size else float("nan"),
        "iou_75": float((ious >= 0.75).mean()) if ious.size else float("nan"),
        "small_instances": int(small.sum()),
        "small_mean_iou": float(ious[small].mean()) if small.any() else float("nan"),
        "large_mean_iou": float(ious[~small].mean()) if (~small).any() else float("nan"),
        "per_class": {c: {"instances": int((classes == c).sum()),
                          "mean_iou": float(ious[classes == c].mean())}
                      for c in sorted(set(classes.tolist()))},
        "per_modality": {m: bucket(mods == m)
                         for m in sorted(set(mods.tolist()))},
        # The breakdown the mean hides: an encoder that reads a target off its
        # brightness scores well everywhere the target is bright, and a test set
        # is mostly that. Split by how far each target stands out from the
        # ground beside it and the two halves of the model become visible.
        "per_contrast": {name: {**bucket(stands_out == i),
                                "median_contrast":
                                    float(np.median(contrasts[stands_out == i]))
                                    if (stands_out == i).any() else float("nan")}
                         for i, name in enumerate(CONTRAST_NAMES)},
        "median_contrast": (float(np.median(contrasts)) if contrasts.size
                            else float("nan")),
    }


def report(result: dict) -> str:
    lines = [
        f"{result['instances']} instances  mean IoU {result['mean_iou']:.4f}  "
        f"IoU>=0.5 {result['iou_50']:.3f}  IoU>=0.75 {result['iou_75']:.3f}",
        f"  small (< {SMALL_SIDE:.0f} px in the window): "
        f"{result['small_instances']} at {result['small_mean_iou']:.4f}   "
        f"larger: {result['large_mean_iou']:.4f}",
    ]
    modalities = result.get("per_modality", {})
    if len(modalities) > 1:
        # Only a mixed run prints this: with one modality the aggregate above
        # *is* the modality and repeating it would just be noise.
        for name, row in modalities.items():
            lines.append(
                f"  {name}: {row['instances']} instances  "
                f"mean IoU {row['mean_iou']:.4f}  IoU>=0.5 {row['iou_50']:.3f}  "
                f"small {row['small_instances']} at {row['small_mean_iou']:.4f}")
    contrast = result.get("per_contrast", {})
    if contrast:
        lines += ["", "| target against its ground | instances | mean IoU | "
                      ">=0.5 | small | small IoU |", "|---|---:|---:|---:|---:|---:|"]
        for name, row in contrast.items():
            lines.append(
                f"| {name} | {row['instances']} | {row['mean_iou']:.4f} | "
                f"{row['iou_50']:.3f} | {row['small_instances']} | "
                f"{row['small_mean_iou']:.4f} |")
    lines += [
        "",
        "| dataset / class | instances | mean IoU |",
        "|---|---:|---:|",
    ]
    for name, row in result["per_class"].items():
        lines.append(f"| {name} | {row['instances']} | {row['mean_iou']:.4f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", action="append", default=[],
                   metavar="SPEC:PATH[:MODALITY[:MODE[:ROLE]]]",
                   help="Repeatable, and must match the training run's flags "
                        "for the split to be the same one.")
    p.add_argument("--pool", action="append", default=[],
                   metavar="POOL_DIR[:IMAGES_ROOT[:MODALITY[:ROLE]]]",
                   help="Repeatable. Same rule as --dataset and for the same "
                        "reason: the split is stratified per source and seeded "
                        "by its name, so a pool the training run had and this "
                        "one does not changes nothing about the others' "
                        "splits -- but a pool scored under a different `role` "
                        "than it trained under is not the same held-out set.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--index", default=None,
                   help="Directory holding the indexes train_encoder.py wrote.")
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--splits", default="",
                   help="A `save_splits` file naming the frames each "
                        "split holds, so the scored set is the one the "
                        "caller chose rather than one re-derived here.")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--window", type=int, default=None,
                   help="Crop in SOURCE pixels, resized to --size. Defaults to "
                        "--size, which is what training used and means no "
                        "resampling. A larger one takes a wider crop and "
                        "shrinks it, dividing every target's apparent side by "
                        "--window/--size -- the way to reach the small-object "
                        "bucket out of a set whose targets are all large, "
                        "without touching the annotation.")
    p.add_argument("--prompt", default="box", choices=("box", "jitter", "point"),
                   help="What the model is told. `box` is the ground-truth "
                        "rectangle and is what every training run used; it "
                        "also states most of the mask on an isolated target, "
                        "so scores taken this way move very little when the "
                        "encoder changes. `jitter` loosens it, `point` gives "
                        "one centre click and nothing else.")
    p.add_argument("--prompt-jitter", type=float, default=0.3,
                   help="With --prompt jitter: how far each edge may move, as "
                        "a fraction of that side.")
    p.add_argument("--per-image", type=int, default=1)
    p.add_argument("--max-instances", type=int, default=8)
    p.add_argument("--min-area", type=int, default=48)
    p.add_argument("--min-side", type=int, default=4)
    p.add_argument("--max-area", type=float, default=0.9)
    p.add_argument("--fill", type=float, default=0.25)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--json", default=None)
    p.add_argument("--per-instance", default=None,
                   help="Also write one row per scored instance here, keyed by "
                        "window. Two checkpoints scored with the same --seed "
                        "and --splits produce the same keys, so pairing the "
                        "two files says which instances each one won and lost.")
    args = p.parse_args(argv)

    from src.training.image_loop import ImageSplit
    from tools.train_encoder import _tqdm, build_indexes, build_model

    # All four gates, matching train_encoder.py flag for flag: an eval that
    # regated the instances differently would score a different population and
    # call it the same split. (With a cached --index the gates are already
    # baked into the instance list and these only matter on a cold rebuild.)
    gates = InstanceGates(min_area=args.min_area, min_side=args.min_side,
                          max_area=args.max_area, fill=args.fill)
    if not args.dataset and not args.pool:
        p.error("nothing to score on: pass --dataset, --pool, or both")
    requests = [parse(argument, gates) for argument in args.dataset]
    pools = [parse_pool(argument, gates) for argument in args.pool]
    cache = Path(args.index) if args.index else None
    if requests:
        print(describe(requests), "\n")
    if pools:
        print(describe_pools(pools), "\n")
    index = build_indexes(requests, cache, 8)
    index.extend(index_pools(pools, cache, 8))

    # The same stratified split, from the same seed, as training used --
    # otherwise "held out" is a claim rather than a fact.
    chosen = (apply_splits(index, args.splits) if args.splits
              else split_index(index, seed=args.seed))[args.split]
    split = ImageSplit(sample_windows(
        chosen, size=args.size, per_image=args.per_image,
        max_instances=args.max_instances, jitter=0, seed=args.seed,
        window=args.window))
    window = args.window or args.size
    print(f"{args.split}: {len(chosen)} frames, {len(split.samples)} windows, "
          f"{sum(len(s.instances) for s in split.samples)} instances")
    print(f"  {window} source px -> {args.size} model px"
          f"{'' if window == args.size else f'  (targets shrink {window / args.size:.1f}x)'}"
          f",  prompt: {args.prompt}"
          f"{f' {args.prompt_jitter:g}' if args.prompt == 'jitter' else ''}")
    for name, count in split.sources.items():
        print(f"  {name:<24} {count:>6} windows")

    model = build_model(args.size, args.checkpoint, args.device)
    result = score(model, split, args.batch, args.device, progress=_tqdm(),
                   prompt=args.prompt, jitter=args.prompt_jitter,
                   seed=args.seed, detail=bool(args.per_instance))
    rows = result.pop("rows", [])
    # The axes go in the JSON, not just the filename: the probe notebook reads
    # a directory of these back into one table, and a run that cannot say how
    # it was scored is a row that cannot be placed.
    result |= {"checkpoint": args.checkpoint, "split": args.split,
               "datasets": [r.label for r in requests],
               "sources": split.sources,
               "prompt": args.prompt,
               "prompt_jitter": args.prompt_jitter if args.prompt == "jitter" else 0.0,
               "size": args.size, "window": window}

    print()
    print(report(result))
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {out}")
    if args.per_instance:
        out = Path(args.per_instance)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "checkpoint": args.checkpoint, "split": args.split,
            "prompt": args.prompt, "seed": args.seed, "rows": rows}) + "\n")
        print(f"wrote {out} ({len(rows)} instances)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
