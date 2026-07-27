#!/usr/bin/env python3
"""Compare two mask dumps produced by tools/benchmark.py --dump-masks-dir.

A quantized engine that is 3x faster and silently loses the target is not a
win, so every speed number in the matrix needs an accuracy number beside it.
The reference should be the FP32 PyTorch run (`pytorch_fp32.npz`) — that is
the model as trained, before any of our optimizations touch it.

    python3 tools/compare_masks.py runs/phase1/pytorch_fp32.npz \\
                                   runs/all/trt_int8.npz

    # per-frame IoU chart, useful for spotting drift/loss mid-clip
    python3 tools/compare_masks.py ref.npz test.npz --chart runs/iou_int8.png

Reading the output:
    mean_iou >= 0.99   indistinguishable
    0.98 - 0.99        safe for most uses
    0.95 - 0.98        visible boundary wobble; check the chart for drift
    < 0.95             investigate before shipping — usually calibration
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_dump(path: str | Path) -> dict:
    data = np.load(path, allow_pickle=False)
    required = {"masks", "frames", "obj_ids", "width"}
    missing = required - set(data.files)
    if missing:
        raise SystemExit(f"{path}: not a benchmark mask dump (missing {sorted(missing)})")
    return {
        "packed": data["masks"],           # [F, O, H, ceil(W/8)] uint8
        "frames": data["frames"],          # [F] int32
        "obj_ids": data["obj_ids"],        # [O] int32
        "width": int(data["width"]),
        "height": int(data["height"]) if "height" in data.files else 0,
        "variant": str(data["variant"]) if "variant" in data.files else Path(path).stem,
    }


def unpack(packed_frame: np.ndarray, width: int) -> np.ndarray:
    """[O, H, ceil(W/8)] uint8 -> [O, H, W] bool"""
    return np.unpackbits(packed_frame, axis=-1, count=width).astype(bool)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    if union == 0:
        return 1.0  # both empty: they agree
    return inter / union


def compare(ref: dict, test: dict) -> dict:
    common_objs = [int(o) for o in ref["obj_ids"] if o in set(test["obj_ids"].tolist())]
    if not common_objs:
        raise SystemExit(
            f"No shared object ids: ref={ref['obj_ids'].tolist()} "
            f"test={test['obj_ids'].tolist()}")
    if ref["width"] != test["width"]:
        raise SystemExit(f"Frame width differs: {ref['width']} vs {test['width']}")

    ref_index = {int(f): i for i, f in enumerate(ref["frames"])}
    test_index = {int(f): i for i, f in enumerate(test["frames"])}
    shared = sorted(set(ref_index) & set(test_index))
    if not shared:
        raise SystemExit("The two dumps share no frame indices.")

    ref_obj_pos = {int(o): i for i, o in enumerate(ref["obj_ids"])}
    test_obj_pos = {int(o): i for i, o in enumerate(test["obj_ids"])}

    per_frame: list[float] = []
    per_object: dict[int, list[float]] = {o: [] for o in common_objs}
    empty_ref = empty_test = 0

    for frame in shared:
        r = unpack(ref["packed"][ref_index[frame]], ref["width"])
        t = unpack(test["packed"][test_index[frame]], test["width"])
        scores = []
        for obj in common_objs:
            rm = r[ref_obj_pos[obj]]
            tm = t[test_obj_pos[obj]]
            score = iou(rm, tm)
            scores.append(score)
            per_object[obj].append(score)
            if not rm.any():
                empty_ref += 1
            if not tm.any():
                empty_test += 1
        per_frame.append(float(np.mean(scores)))

    arr = np.array(per_frame, dtype=np.float64)
    return {
        "reference": ref["variant"],
        "test": test["variant"],
        "frames_compared": len(shared),
        "objects": common_objs,
        "mean_iou": float(arr.mean()),
        "median_iou": float(np.median(arr)),
        "min_iou": float(arr.min()),
        "p05_iou": float(np.percentile(arr, 5)),
        "iou_below_0.99_pct": float(100 * (arr < 0.99).mean()),
        "iou_below_0.95_pct": float(100 * (arr < 0.95).mean()),
        "iou_below_0.90_pct": float(100 * (arr < 0.90).mean()),
        "empty_masks_ref": empty_ref,
        "empty_masks_test": empty_test,
        "per_object_mean_iou": {str(o): float(np.mean(v)) for o, v in per_object.items()},
        "_per_frame": arr,
    }


def write_chart(per_frame: np.ndarray, out_path: Path, title: str) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[compare] matplotlib not installed; skipping chart.")
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(per_frame, linewidth=1.0, color="#1f77b4", label="per-frame IoU")
    plt.axhline(float(per_frame.mean()), color="red", linestyle="--", linewidth=1.2,
                label=f"mean = {per_frame.mean():.4f}")
    plt.axhline(0.95, color="orange", linestyle=":", linewidth=1.0, label="0.95")
    plt.ylim(0, 1.02)
    plt.xlabel("frame")
    plt.ylabel("IoU vs reference")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    return out_path.resolve()


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("reference", help="Baseline .npz (normally pytorch_fp32.npz).")
    p.add_argument("test", help="Variant .npz to score against it.")
    p.add_argument("--chart", default=None, help="Write a per-frame IoU PNG here.")
    p.add_argument("--json", default=None, help="Write the metrics as JSON here.")
    args = p.parse_args()

    result = compare(load_dump(args.reference), load_dump(args.test))
    per_frame = result.pop("_per_frame")

    print(f"reference : {result['reference']}  ({args.reference})")
    print(f"test      : {result['test']}  ({args.test})")
    print(f"frames    : {result['frames_compared']}   objects: {result['objects']}")
    print(f"mean IoU  : {result['mean_iou']:.4f}")
    print(f"median    : {result['median_iou']:.4f}   p05: {result['p05_iou']:.4f}   "
          f"min: {result['min_iou']:.4f}")
    print(f"below .99 : {result['iou_below_0.99_pct']:.1f}%   "
          f"below .95: {result['iou_below_0.95_pct']:.1f}%   "
          f"below .90: {result['iou_below_0.90_pct']:.1f}%")
    if len(result["objects"]) > 1:
        print(f"per object: {result['per_object_mean_iou']}")
    if result["empty_masks_test"] > result["empty_masks_ref"]:
        print(f"WARNING: the test run produced {result['empty_masks_test']} empty masks "
              f"vs {result['empty_masks_ref']} in the reference — the target is being "
              "lost. For INT8 this almost always means bad/absent calibration.")

    if args.chart:
        out = write_chart(per_frame, Path(args.chart),
                          f"{result['test']} vs {result['reference']}")
        if out:
            print(f"chart     : {out}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2))
        print(f"json      : {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
