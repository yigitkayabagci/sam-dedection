#!/usr/bin/env python3
"""Which stage A was worth it: rank them by what stage B did afterwards.

**The two pretrains cannot be compared to each other directly.** `34` minimises
a promptable-segmentation loss on pseudo-labelled masks; `36` minimises the
reconstruction error of pixels it hid from itself. Different tasks, different
units, and a lower number in one says nothing about the other. Their own logs
answer "did this run converge", never "is this the better base".

The comparison that means something is downstream and it is the only one this
tool will make: run `32` from each base, everything else identical, and read
the same held-out grade. The control is not optional -- `32` from **stock** is
what a pretrain has to beat, and a pretrain that does not beat it cost GPU
hours and bought nothing.

    python3 tools/compare_stage_a.py \\
        /content/drive/MyDrive/edgetam-stage-b/aerial_thermal_stable/verdict.json \\
        /content/drive/MyDrive/edgetam-stage-b/aerial_thermal_stable_from_*/verdict.json

It refuses to rank arms that were not run the same way. A stage B whose seed,
split sizes, gates, schedule or learning rates differ from the control's is
measuring those as well as the base it started from, and a ranking that hid
that would be worse than no ranking.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# What has to match across arms for a ranking to mean anything. Each is
# something stage B varies and something that moves the grade on its own.
FAIR = ("seed", "epochs", "steps", "batch", "min_box_iou", "prompt",
        "image_size", "method", "gates", "neighbour_weight")
# The score `eval_instances` writes, in the order they are worth reading. The
# mean is the headline; the small-instance mean is the one this project is
# actually about, because a drone is small.
METRICS = ("mean_iou", "small_mean_iou", "iou_50")


def load(path: Path) -> dict:
    body = json.loads(Path(path).read_text())
    after = body.get("after") or {}
    before = body.get("before") or {}
    prompt = sorted(after) [0] if after else None
    return {
        "path": Path(path),
        "run": body.get("run", Path(path).parent.name),
        "base": Path(body.get("run_log", {}).get("base")
                     or body.get("base") or "stock").name,
        "prompt": prompt,
        "after": after.get(prompt, {}) if prompt else {},
        "before": before.get(prompt, {}) if prompt else {},
        "fair": {key: body.get(key) for key in FAIR},
        "rates": body.get("rates", {}),
        "frames": body.get("frames", {}),
    }


def control_of(arms: list[dict]) -> dict | None:
    """The arm that started from stock: the one every other has to beat.

    Recognised by its run name carrying no `_from_` -- which is exactly how
    notebook 32 names an arm that was given no BASE_CHECKPOINT.
    """
    plain = [arm for arm in arms if "_from_" not in arm["run"]]
    return plain[0] if len(plain) == 1 else None


def unfair(arms: list[dict], control: dict) -> list[str]:
    """Every setting that differs from the control's, named."""
    out = []
    for arm in arms:
        if arm is control:
            continue
        for key, value in arm["fair"].items():
            if value != control["fair"][key] and value is not None:
                out.append(f"{arm['run']}: {key} is {value!r}, the control's "
                           f"is {control['fair'][key]!r}")
        if arm["prompt"] != control["prompt"]:
            out.append(f"{arm['run']}: scored on prompt {arm['prompt']!r}, "
                       f"the control on {control['prompt']!r}")
        if arm["frames"] != control["frames"]:
            out.append(f"{arm['run']}: split sizes {arm['frames']} against the "
                       f"control's {control['frames']}")
    return out


def table(arms: list[dict], control: dict) -> list[str]:
    width = max(len(arm["run"]) for arm in arms)
    head = f"   {'run':<{width}}  " + "  ".join(f"{m:>15}" for m in METRICS)
    lines = [head, "   " + "-" * (len(head) - 3)]
    for arm in sorted(arms, key=lambda a: -(a["after"].get("mean_iou") or 0)):
        cells = []
        for metric in METRICS:
            value = arm["after"].get(metric)
            if value is None:
                cells.append(f"{'-':>15}")
                continue
            base = control["after"].get(metric)
            delta = ("" if arm is control or base is None
                     else f" {value - base:+.4f}")
            cells.append(f"{value:.4f}{delta:>8}")
        mark = "  (control)" if arm is control else ""
        lines.append(f"   {arm['run']:<{width}}  " + "  ".join(cells) + mark)
    return lines


def report(paths: list[Path]) -> int:
    arms = [load(path) for path in paths]
    missing = [arm["run"] for arm in arms if not arm["after"]]
    if missing:
        print(f"!! no `after` score in: {', '.join(missing)}. These are runs "
              f"that did not finish scoring; rerun them before ranking.")
        return 1

    control = control_of(arms)
    if control is None:
        print("!! no single arm started from stock. A pretrain is measured "
              "against stage B *from stock*, so run that arm (32 with "
              "BASE_CHECKPOINT = \"\") before ranking these.")
        return 1

    print(f"stage B, {len(arms)} arm(s), scored on prompt "
          f"{control['prompt']!r}\n")
    print("\n".join(table(arms, control)))

    problems = unfair(arms, control)
    if problems:
        print("\n!! these arms did not run the same way as the control, so the "
              "ranking above is measuring the difference as well as the base:")
        for line in problems:
            print(f"     {line}")
        print("   Re-run them with the control's settings, or read the table "
              "as a hint rather than a result.")
        return 1

    best = max(arms, key=lambda a: a["after"].get("mean_iou") or 0)
    gain = (best["after"]["mean_iou"] - control["after"]["mean_iou"])
    print()
    if best is control or gain <= 0:
        print("   No pretrain beat stage B from stock. That is a result: the "
              "epochs bought nothing,\n   and the honest next step is to say "
              "so rather than to pick the least bad one.")
        return 0
    print(f"   {best['run']} is ahead of stock by {gain:+.4f} mean IoU "
          f"({gain / max(control['after']['mean_iou'], 1e-9):+.1%}).")
    others = [a for a in arms if a not in (best, control)]
    if others:
        second = max(others, key=lambda a: a["after"].get("mean_iou") or 0)
        margin = best["after"]["mean_iou"] - second["after"]["mean_iou"]
        print(f"   Its margin over {second['run']} is {margin:+.4f}. One seed "
              f"each: a margin\n   smaller than the gap between two seeds of "
              f"the same arm is not a ranking. Run\n   the top two again with "
              f"another SEED before reporting an ordering.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("verdicts", nargs="+", type=Path,
                        help="verdict.json from each stage B arm.")
    args = parser.parse_args(argv)
    return report(args.verdicts)


if __name__ == "__main__":
    raise SystemExit(main())
