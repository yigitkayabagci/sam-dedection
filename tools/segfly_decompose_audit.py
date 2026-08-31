#!/usr/bin/env python3
"""Does connected-component decomposition give SegFly clean instances?

`docs/encoder_training_todo.md` §6 has carried that as open question 1 since
stage B was designed -- "the measurement is ready, the answer is still
missing". This is the measurement, run against the publisher's own bytes.

**Why this cannot read the harvested pool instead.** A pool record carries the
instances the gates *accepted*; the components they threw away are not in it,
and the whole fusion signal is a reject count -- `fill` rejections are the
"two parked cars became one component" warning `aerial.reject_reason` exists
to raise. So the only way to that number is to decompose the label maps again,
which means going back to the 178 GiB parquet the pool was built from.

**Why it is still cheap.** Parquet is columnar and a semantic map is one small
column beside two multi-megabyte image columns, so a range read pulls the
labels of a shard without its pixels -- seconds per shard instead of the
several hundred megabytes `datasets` would stream. The shard numbers are the
argument because the modality is decided per shard (see
`tools/export_hf_dataset.py`): the defaults below are three thermal ones.

It reports, per mode:

- instances kept and components rejected, **broken down by which gate**;
- the share of frames that yield nothing at all;
- size and `fill` distributions of what survived;
- raw component sizes per thing class, before any gate -- which is where a
  class that is not what its name says shows itself;
- what a `truck` component's border is made of, which separates "a shredded
  vehicle" from "a standalone blob in the vegetation";
- and how much of the fusion `fill` *cannot* see is still on the table -- an
  erosion probe for the bridged case and a shape profile for the side-by-side
  one, because a zero on `fill` alone does not answer the question.

    python tools/segfly_decompose_audit.py --shards 63 410 692
    python tools/segfly_decompose_audit.py --shards 475 550 50 --modes components
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.aerial import SPECS, InstanceGates, decompose  # noqa: E402

REPO = "markus-42/SegFly"
URL = ("https://huggingface.co/datasets/{repo}/resolve/main/data/"
       "train-{shard:05d}-of-00761.parquet")


def read_labels(repo: str, shards: list[int]) -> list[tuple[str, np.ndarray]]:
    """`(scene_altitude, semantic map)` for every row of the named shards.

    Only the `label`, `scene` and `altitude` columns are fetched; the `image`
    and `RGB_aligned` columns are the shard's bulk and are never touched.
    """
    import fsspec
    import PIL.Image
    import pyarrow.parquet as pq

    PIL.Image.MAX_IMAGE_PIXELS = None            # SegFly's RGB half is 20 MP
    handle = fsspec.filesystem("https")
    maps = []
    for shard in shards:
        url = URL.format(repo=repo, shard=shard)
        with handle.open(url) as remote:
            table = pq.ParquetFile(remote).read(
                columns=["label", "scene", "altitude"])
        for label, scene, altitude in zip(table["label"].to_pylist(),
                                          table["scene"].to_pylist(),
                                          table["altitude"].to_pylist()):
            array = np.array(PIL.Image.open(io.BytesIO(label["bytes"])))
            maps.append((f"{scene}_{altitude}", array))
        print(f"  shard {shard}: {len(table)} maps, {maps[-1][0]}, "
              f"{maps[-1][1].shape[1]}x{maps[-1][1].shape[0]}")
    return maps


def palette_check(maps, spec) -> dict:
    """Which pixel values are in these maps, and which the palette does not name."""
    seen = Counter()
    for _, array in maps:
        seen.update(np.unique(array).tolist())
    known = set(spec.classes.values())
    return {"values": dict(sorted(seen.items())),
            "outside_palette": sorted(int(v) for v in set(seen) - known)}


def raw_components(maps, spec) -> dict:
    """Component sizes per thing class **before** any gate.

    A class whose components are a fraction of another's is the finding this
    exists for: a truck is not smaller than a car, so if id 36's components
    are, the label is not carrying what its name claims.
    """
    import cv2

    out = {}
    for name in spec.things:
        class_id = spec.classes[name]
        sizes = []
        for _, array in maps:
            mask = (array == class_id).astype(np.uint8)
            if not mask.any():
                continue
            count, _, stats, _ = cv2.connectedComponentsWithStats(
                mask, connectivity=8)
            sizes += [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count)]
        sizes = np.array(sizes, dtype=float)
        out[name] = {
            "components": len(sizes),
            "frames_present": sum(1 for _, a in maps if (a == class_id).any()),
            "median_px": float(np.median(sizes)) if len(sizes) else 0.0,
            "p90_px": float(np.percentile(sizes, 90)) if len(sizes) else 0.0,
            "under_min_area": float((sizes < 48).mean()) if len(sizes) else 0.0,
        }
    return out


def fusion_probe(maps, spec, gates, name: str = "vehicle") -> dict:
    """How much of the fusion `fill` cannot see is still on the table.

    `fill` catches the *bridged* merge: two objects joined by a few pixels sit
    in a large, mostly empty box. It cannot catch the other one -- two vehicles
    touching along a full edge tile into a solid rectangle whose box is well
    filled, and that shape is identical to one larger vehicle's. No property of
    the mask separates those two, so this does not try to. It bounds them.

    Two probes, both on accepted components only:

    `erosion` -- shave 1 and 2 px off each component and count the pieces that
    survive. Anything held together by a thin neck falls apart here, so a zero
    means `fill`'s zero is not an artefact of where the threshold sits.

    `profile` -- a side-by-side merge has a signature: *one* side roughly
    doubles while the other stays the size it was. A genuinely larger vehicle
    grows in both. Counting the first shape is an upper bound on the merges,
    and it is the cut worth making if one is made -- it leaves the vans alone.
    """
    import cv2

    class_id = spec.classes[name]
    kernel = np.ones((3, 3), np.uint8)
    areas, longs, shorts, fills = [], [], [], []
    split = {1: 0, 2: 0}
    for _, array in maps:
        mask = (array == class_id).astype(np.uint8)
        if not mask.any():
            continue
        count, labelled, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            w = int(stats[index, cv2.CC_STAT_WIDTH])
            h = int(stats[index, cv2.CC_STAT_HEIGHT])
            if area < gates.min_area or min(w, h) < gates.min_side:
                continue
            fill = area / max(w * h, 1)
            if fill < gates.fill:
                continue
            areas.append(area); fills.append(fill)
            longs.append(max(w, h)); shorts.append(min(w, h))
            x = int(stats[index, cv2.CC_STAT_LEFT])
            y = int(stats[index, cv2.CC_STAT_TOP])
            blob = np.zeros((h + 8, w + 8), np.uint8)
            blob[4:4 + h, 4:4 + w] = (labelled[y:y + h, x:x + w] == index)
            for rounds in (1, 2):
                eroded = cv2.erode(blob, kernel, iterations=rounds)
                pieces, _, sub, _ = cv2.connectedComponentsWithStats(
                    eroded, connectivity=8)
                # 15 % of the parent, so a shaved corner is not a "piece"
                if sum(1 for j in range(1, pieces)
                       if sub[j, cv2.CC_STAT_AREA] >= 0.15 * area) >= 2:
                    split[rounds] += 1

    area = np.array(areas, dtype=float)
    long_side = np.array(longs, dtype=float)
    short_side = np.array(shorts, dtype=float)
    fill = np.array(fills, dtype=float)
    if not len(area):
        return {"components": 0}
    med_a, med_l, med_s = np.median(area), np.median(long_side), np.median(short_side)
    oversized = area > 1.6 * med_a
    merged = oversized & (long_side > 1.6 * med_l) & (short_side < 1.4 * med_s)
    bigger = oversized & (long_side > 1.4 * med_l) & (short_side > 1.4 * med_s)
    return {
        "components": int(len(area)),
        "split_after_1px_erosion": split[1],
        "split_after_2px_erosion": split[2],
        "oversized_share": float(oversized.mean()),
        "merge_profile_share": float(merged.mean()),
        "larger_vehicle_profile_share": float(bigger.mean()),
        "fill_median_all": float(np.median(fill)),
        # A merge with a gap between the two would sit *below* the population;
        # above it is evidence against the merge reading.
        "fill_median_merge_profile": float(np.median(fill[merged])) if merged.any() else None,
    }


def border_of(maps, spec, name: str, limit: int = 400) -> dict:
    """What surrounds a class's components -- fragments hug their parent.

    A component of a shredded vehicle has vehicle pixels around it. One that
    sits in grass does not, and that difference is the whole question of
    whether a small `truck` is a piece of something or a thing of its own.
    """
    import cv2

    class_id = spec.classes[name]
    kernel = np.ones((3, 3), np.uint8)
    majority, touching_a_thing, total = Counter(), 0, 0
    things = {spec.classes[t] for t in spec.things} - {class_id}
    for _, array in maps:
        mask = (array == class_id).astype(np.uint8)
        if not mask.any():
            continue
        count, labelled, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        for index in range(1, min(count, limit + 1)):
            x, y = stats[index, cv2.CC_STAT_LEFT], stats[index, cv2.CC_STAT_TOP]
            w, h = stats[index, cv2.CC_STAT_WIDTH], stats[index, cv2.CC_STAT_HEIGHT]
            x0, y0 = max(0, x - 3), max(0, y - 3)
            x1 = min(array.shape[1], x + w + 3)
            y1 = min(array.shape[0], y + h + 3)
            blob = (labelled[y0:y1, x0:x1] == index).astype(np.uint8)
            ring = cv2.dilate(blob, kernel, iterations=2) - blob
            values = array[y0:y1, x0:x1][ring.astype(bool)]
            if values.size == 0:
                continue
            total += 1
            majority[int(Counter(values.tolist()).most_common(1)[0][0])] += 1
            if any((values == t).any() for t in things):
                touching_a_thing += 1
    return {"components": total,
            "ring_majority": {spec.name_of(v): n for v, n in majority.most_common(8)},
            "ring_holds_another_thing_class": touching_a_thing}


def run_mode(maps, spec, gates, mode: str) -> dict:
    kept_by_class, rejects = Counter(), Counter()
    areas, fills, sides, empty = [], [], [], 0
    for _, array in maps:
        _, instances, rejected = decompose(array, spec, gates, mode)
        rejects.update(rejected)
        empty += not instances
        for instance in instances:
            kept_by_class[spec.name_of(instance.class_id)] += 1
            areas.append(instance.area)
            fills.append(instance.fill)
            sides.append(max(instance.width, instance.height))
    area = np.array(areas, dtype=float)
    fill = np.array(fills, dtype=float)
    side = np.array(sides, dtype=float)
    total = len(area) + sum(rejects.values())
    return {
        "frames": len(maps), "instances": len(area), "empty_frames": empty,
        "kept_by_class": dict(kept_by_class), "rejected": dict(rejects),
        # The fusion signal, and the reason this script exists.
        "fill_reject_share": rejects.get("fill", 0) / max(total, 1),
        "reject_share": sum(rejects.values()) / max(total, 1),
        "area_median": float(np.median(area)) if len(area) else 0.0,
        "side_median": float(np.median(side)) if len(side) else 0.0,
        "under_32px": float((side < 32).mean()) if len(side) else 0.0,
        "fill_median": float(np.median(fill)) if len(fill) else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shards", type=int, nargs="+",
                        default=[63, 410, 692],
                        help="parquet shard numbers; the defaults are thermal")
    parser.add_argument("--modes", nargs="+", default=["components", "watershed"],
                        help="watershed is impractical on the 20 MP RGB half")
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--spec", default="segfly")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--min-area", type=int, default=48)
    parser.add_argument("--min-side", type=int, default=4)
    parser.add_argument("--max-area", type=float, default=0.9)
    parser.add_argument("--fill", type=float, default=0.25)
    args = parser.parse_args()

    spec = SPECS[args.spec]
    gates = InstanceGates(min_area=args.min_area, min_side=args.min_side,
                          max_area=args.max_area, fill=args.fill)
    print(f"{args.spec}: things {spec.things} -> ids "
          f"{[spec.classes[t] for t in spec.things]}, gates {gates}")

    maps = read_labels(args.repo, args.shards)
    report = {"spec": args.spec, "shards": args.shards, "gates": gates.__dict__,
              "palette": palette_check(maps, spec),
              "raw_components": raw_components(maps, spec),
              "modes": {m: run_mode(maps, spec, gates, m) for m in args.modes}}
    for name in spec.things:
        report.setdefault("borders", {})[name] = border_of(maps, spec, name)
    report["fusion"] = {n: fusion_probe(maps, spec, gates, n) for n in spec.things}

    print(json.dumps(report, indent=2))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"{args.spec}_audit.json").write_text(
            json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.out / f'{args.spec}_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
