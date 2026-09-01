#!/usr/bin/env python3
"""Read the cleaned SegFly's `index.json` and say what is in it, in this repo's terms.

The audit that produced the set kept its findings in a manifest rather than in
the pixels, and two of them are things the training path cannot see:

`kayma_suphesi`   the frame's masks are shifted off the vehicles. Measured on
                  706 frames (20.5 %), 25-50 px, direction random per frame.
                  Not corrected -- an automatic correction was written, passed
                  an injection test and damaged every case it was inspected on.
`incele`          the instance scored between the two thresholds: not clean
                  enough to trust, not bad enough for the audit to delete.

Both are frame- or instance-level facts about *label quality*, which is
exactly what `InstanceGates` is not for -- the gates ask whether a target is
worth training on, not whether its annotation is in the right place. So they
travel the one route the dataset layer already has for "the publisher says
this frame is unusable": `DatasetSpec.exclude`, a text file of frame stems at
the dataset root. Kust4K's `broken_in_*.txt` is the same mechanism.

**Dropping is a choice, and this prints its cost before making it.** The shift
rate is a floor, not an estimate: the measurement needs at least two targets
over 800 px in a frame, and 50.5 % of frames could not meet that. Excluding
what was measured therefore leaves an unknown amount of the same problem in,
and removes a fifth of the set for it.

    python tools/segfly_clean_manifest.py --set /content/data/SegFly_temiz
    python tools/segfly_clean_manifest.py --set /content/data/SegFly_temiz \
        --drop shift
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# `DatasetSpec.exclude` for `segfly_temiz` is `atlanan_*.txt`, so every file
# written here has to match that glob or it is read by nothing.
MANIFESTS = {"shift": "atlanan_kayma.txt", "review": "atlanan_incele.txt"}


def census(index: dict) -> dict:
    """Every count worth knowing, over the whole manifest."""
    frames = len(index)
    instances = sum(len(r.get("instances", ())) for r in index.values())
    shifted = [k for k, r in index.items() if r.get("kayma_suphesi")]
    measured = [k for k, r in index.items() if r.get("kayma_olculebildi")]
    review = {k for k, r in index.items()
              if any(i.get("incele") for i in r.get("instances", ()))}
    review_instances = sum(sum(1 for i in r.get("instances", ())
                               if i.get("incele"))
                           for r in index.values())
    offsets = [max(abs(v) for v in r.get("kayma_olcum", (0, 0)))
               for k, r in index.items() if r.get("kayma_suphesi")]
    return {
        "frames": frames, "instances": instances,
        "shifted": len(shifted), "shifted_keys": shifted,
        "measured": len(measured),
        "review_frames": len(review), "review_keys": sorted(review),
        "review_instances": review_instances,
        "max_offset": max(offsets) if offsets else 0,
    }


def stems(index: dict, keys) -> list[str]:
    """The frame names `excluded_keys` matches on -- `kare`, not the index key."""
    return sorted({str(index[k]["kare"]) for k in keys})


def report(counts: dict) -> list[str]:
    frames = max(counts["frames"], 1)
    lines = [
        f"{counts['frames']} frames, {counts['instances']} instances "
        f"({counts['instances'] / frames:.2f} per frame)",
        f"shift suspected     {counts['shifted']:>6} frames "
        f"({counts['shifted'] / frames:.1%}), worst offset "
        f"{counts['max_offset']} px",
        f"shift measurable    {counts['measured']:>6} frames "
        f"({counts['measured'] / frames:.1%}) -- the rest could not be "
        f"measured, so the rate above is a floor",
        f"marked for review   {counts['review_instances']:>6} instances in "
        f"{counts['review_frames']} frames "
        f"({counts['review_frames'] / frames:.1%})",
    ]
    keep = frames - counts["shifted"]
    lines.append(f"dropping the shift frames would leave {keep} frames "
                 f"({keep / frames:.1%})")
    return lines


DROPS = {"none": (), "shift": ("shift",), "review": ("review",),
         "both": ("shift", "review")}


def write_manifests(root: Path, index: dict, drop: str,
                    counts: dict | None = None) -> list[str]:
    """Write the exclusion files `drop` asks for, remove the ones it does not.

    Removing matters as much as writing: a manifest left behind from a wider
    run keeps excluding frames after the decision that made it was reversed,
    and nothing downstream would say so -- `excluded_keys` reads whatever
    matches the glob.
    """
    counts = counts or census(index)
    wanted = DROPS[drop]
    lines = []
    for kind, name in MANIFESTS.items():
        target = Path(root) / name
        if kind not in wanted:
            if target.exists():
                target.unlink()
                lines.append(f"removed {target}")
            continue
        keys = (counts["shifted_keys"] if kind == "shift"
                else counts["review_keys"])
        names = stems(index, keys)
        target.write_text("\n".join(names) + "\n", encoding="utf-8")
        lines.append(f"wrote {target} -- {len(names)} frames")
    return lines or ["no exclusion manifests -- training on the whole set"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", dest="root", type=Path, required=True,
                        help="The cleaned set's root -- the directory holding "
                             "index.json, images/ and masks/.")
    parser.add_argument("--drop", choices=tuple(DROPS),
                        default="none",
                        help="Which manifests to write. `none` (the default) "
                             "removes any this tool wrote before, so the set "
                             "goes back to training on everything.")
    args = parser.parse_args()

    path = Path(args.root) / "index.json"
    if not path.is_file():
        raise SystemExit(
            f"{path} is not there. --set takes the cleaned set's root, the "
            f"directory holding index.json beside images/ and masks/.")
    index = json.loads(path.read_text(encoding="utf-8"))
    counts = census(index)
    print("\n".join(report(counts)))

    print()
    print("\n".join(write_manifests(Path(args.root), index, args.drop,
                                    counts=counts)))

    if args.drop == "none":
        print("\nnothing excluded. `--drop shift` is the one worth taking "
              "first: a shifted mask teaches the head a target's shape in the "
              "wrong place, which is a label error and not noise the loss "
              "averages out.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
