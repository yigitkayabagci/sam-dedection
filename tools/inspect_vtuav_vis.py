#!/usr/bin/env python3
"""Judge a VTUAV-VIS archive before trusting it, without extracting it.

Two questions, and they are different. **Is it what it says it is** -- the
member names, the sequences, whether `mask/ir/` exists beside `ir/`, whether a
mask has a frame to go with it. And **is it the data this project is short of**
-- the targets' size, and their signal-to-clutter ratio, which is the number
that decides whether a frame is the hard case or an easy one.

    python3 tools/inspect_vtuav_vis.py /path/to/train_001.zip --sample 40

The first pass reads only the ZIP central directory, so it costs a few seconds
and no disk on an archive of any size. The second inflates `--sample` mask and
image pairs into memory, decodes them, and reports the distribution -- still
nothing on disk.

Why the discipline: this repo lost frames twice by trusting a file's name over
its archive's own member names (Kust4K, then AeroVIS, whose `image_rel` paths
are relative to a directory two levels below where they looked). A zip called
`test_001.zip` whose members all say `train_003/` is the same fault waiting.
"""
from __future__ import annotations

import argparse
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
# The ring `image_loop.instance_contrast` measures the ground in, so the number
# printed here is the same quantity the training run reports.
RING = 9


def members(archive: Path) -> list[zipfile.ZipInfo]:
    with zipfile.ZipFile(archive) as zf:
        return [info for info in zf.infolist() if not info.is_dir()]


def layout(infos: list[zipfile.ZipInfo]) -> dict:
    """Sequences, and what each one carries, off the member names alone."""
    tree: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    top: dict[str, int] = defaultdict(int)
    for info in infos:
        parts = Path(info.filename).parts
        if not parts:
            continue
        top[parts[0]] += 1
        if len(parts) < 2:
            continue
        sequence = parts[0]
        kind = "/".join(parts[1:-1]) or "(root)"
        tree[sequence][kind] += 1
    return {"top": dict(top), "sequences": {k: dict(v) for k, v in tree.items()}}


def pairs(infos: list[zipfile.ZipInfo], modality: str) -> list[tuple[str, str]]:
    """`(image member, mask member)` for every mask that has a frame.

    Matched by stem within a sequence, never by name across the archive:
    `000123.png` lives in most sequences, and a match across them would pair a
    mask with a stranger's frame.
    """
    frames: dict[tuple[str, str], str] = {}
    masks: dict[tuple[str, str], str] = {}
    for info in infos:
        parts = Path(info.filename).parts
        if len(parts) < 3 or Path(info.filename).suffix.lower() not in IMAGE_SUFFIXES:
            continue
        key = (parts[0], Path(info.filename).stem)
        inner = "/".join(parts[1:-1])
        if inner == modality:
            frames[key] = info.filename
        elif inner == f"mask/{modality}":
            masks[key] = info.filename
    return [(frames[key], member) for key, member in sorted(masks.items())
            if key in frames]


def contrast(image: np.ndarray, mask: np.ndarray, ring: int = RING) -> float:
    """`|mean inside - mean in the ring| / std of the ring`, on one instance.

    The same definition `src/training/image_loop.instance_contrast` uses, so a
    number here and a number in a training report mean the same thing.
    """
    import cv2

    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grey = grey.astype(np.float64)
    inside = mask > 0
    if not inside.any():
        return float("nan")
    kernel = np.ones((ring, ring), np.uint8)
    around = cv2.dilate(inside.astype(np.uint8), kernel) > 0
    around &= ~inside
    if not around.any():
        return float("nan")
    spread = float(grey[around].std())
    return abs(float(grey[inside].mean()) - float(grey[around].mean())) / max(spread, 1e-3)


def sample(archive: Path, chosen: list[tuple[str, str]], seed: int = 0) -> list[dict]:
    """Decode the chosen pairs in memory and measure each one."""
    import cv2

    rows = []
    with zipfile.ZipFile(archive) as zf:
        for image_member, mask_member in chosen:
            image = cv2.imdecode(np.frombuffer(zf.read(image_member), np.uint8),
                                 cv2.IMREAD_UNCHANGED)
            mask = cv2.imdecode(np.frombuffer(zf.read(mask_member), np.uint8),
                                cv2.IMREAD_UNCHANGED)
            if image is None or mask is None:
                rows.append({"member": mask_member, "error": "undecodable"})
                continue
            if mask.ndim == 3:
                mask = mask[..., 0]
            labels = sorted(int(v) for v in np.unique(mask))
            positive = mask > 0
            ys, xs = np.nonzero(positive)
            row = {
                "sequence": Path(mask_member).parts[0],
                "frame": Path(mask_member).stem,
                "image_shape": tuple(int(v) for v in image.shape),
                "image_dtype": str(image.dtype),
                "mask_shape": tuple(int(v) for v in mask.shape),
                "labels": labels[:8],
                "instances": max(len(labels) - 1, 0),
                "aligned": tuple(mask.shape[:2]) == tuple(image.shape[:2]),
                "share": round(float(positive.mean()), 6),
            }
            if ys.size:
                row["box"] = (int(xs.min()), int(ys.min()), int(xs.max() + 1),
                              int(ys.max() + 1))
                row["side"] = round(float(np.sqrt(positive.sum())), 1)
                row["scr"] = round(contrast(image, positive), 3)
            else:
                row["empty_mask"] = True
            rows.append(row)
    return rows


def report(archive: Path, modality: str, count: int, seed: int) -> int:
    infos = members(archive)
    tree = layout(infos)
    print(f"== {archive}  ({archive.stat().st_size / 2**30:.1f} GiB, "
          f"{len(infos)} members)")
    names = sorted(tree["top"])
    print(f"   top level: {len(names)} entr(ies) -- {', '.join(names[:6])}"
          + (" ..." if len(names) > 6 else ""))
    # The check the file name cannot answer. A `test_001.zip` whose members all
    # say `train_003/` is a renamed or mislabelled archive, and this repo has
    # been bitten by exactly that.
    stem = re.sub(r"\.zip$", "", archive.name)
    if names and not any(name in stem or stem in name for name in names):
        print(f"   !! the file is called {stem!r} and its members say "
              f"{names[0]!r}. Trust the members, not the name.")

    kinds: dict[str, int] = defaultdict(int)
    for holds in tree["sequences"].values():
        for kind, n in holds.items():
            kinds[kind] += n
    print("   folders inside a sequence:")
    for kind, n in sorted(kinds.items(), key=lambda kv: -kv[1])[:10]:
        print(f"     {kind:<16} {n:>8} files")

    both = [s for s, holds in tree["sequences"].items()
            if f"mask/{modality}" in holds and modality in holds]
    print(f"   sequences with both {modality}/ and mask/{modality}/: "
          f"{len(both)}/{len(tree['sequences'])}")

    chosen = pairs(infos, modality)
    print(f"   masks with a frame beside them: {len(chosen)}")
    if not chosen:
        print(f"   !! nothing to measure for modality {modality!r}.")
        return 1
    per_sequence = defaultdict(int)
    for _, mask_member in chosen:
        per_sequence[Path(mask_member).parts[0]] += 1
    counts = sorted(per_sequence.values())
    print(f"   masks per sequence: min {counts[0]}, median "
          f"{counts[len(counts) // 2]}, max {counts[-1]}")

    rng = np.random.default_rng(seed)
    picked = [chosen[i] for i in rng.choice(len(chosen),
                                            size=min(count, len(chosen)),
                                            replace=False)]
    rows = sample(archive, sorted(picked))
    bad = [r for r in rows if r.get("error") or r.get("empty_mask")]
    good = [r for r in rows if "scr" in r]
    print(f"\n   sampled {len(rows)}: {len(good)} usable, {len(bad)} empty or "
          f"undecodable")
    if not good:
        return 1
    misaligned = [r for r in good if not r["aligned"]]
    if misaligned:
        print(f"   !! {len(misaligned)} mask(s) not the size of their frame, "
              f"e.g. {misaligned[0]['mask_shape']} vs {misaligned[0]['image_shape']}")
    dtypes = {r["image_dtype"] for r in good}
    shapes = {r["image_shape"] for r in good}
    print(f"   image dtype(s): {', '.join(sorted(dtypes))}; "
          f"{len(shapes)} distinct shape(s), e.g. {sorted(shapes)[0]}")
    instances = np.array([r["instances"] for r in good])
    sides = np.array([r["side"] for r in good])
    scr = np.array([r["scr"] for r in good])
    scr = scr[np.isfinite(scr)]
    print(f"   instances per mask: median {np.median(instances):.0f}, "
          f"max {instances.max()}  "
          f"({(instances > 1).mean():.0%} hold more than one)")
    print(f"   target side (sqrt area): median {np.median(sides):.1f} px, "
          f"10th {np.percentile(sides, 10):.1f}, 90th {np.percentile(sides, 90):.1f}")
    print(f"   signal-to-clutter: median {np.median(scr):.2f}, "
          f"{(scr < 1).mean():.0%} below 1, {(scr < 0.5).mean():.0%} below 0.5")
    print("     (HIT-UAV's real aerial targets sit at a median of 0.91. Above "
          "about 2 this\n      archive is easy data, and easy data is what "
          "this project already has.)")
    print("\n   hardest sampled frames:")
    for row in sorted(good, key=lambda r: r["scr"])[:5]:
        print(f"     scr {row['scr']:>6} side {row['side']:>6} px  "
              f"{row['sequence']}/{row['frame']}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--modality", default="ir", choices=("ir", "rgb"))
    parser.add_argument("--sample", type=int, default=40,
                        help="How many mask/frame pairs to decode and measure.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    worst = 0
    for archive in args.archives:
        worst = max(worst, report(archive, args.modality, args.sample, args.seed))
        print()
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
