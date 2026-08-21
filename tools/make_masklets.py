#!/usr/bin/env python3
"""Turn VTUAV's tracking boxes into masklets, offline, with a video teacher.

Stage C needs video with **instance identity**, and VTUAV has 1.7 M registered
RGB-T frames whose only annotation is a per-frame box. A video teacher --
SAM 2.1 by default, SAM 3 one string away -- is prompted with the box on the
first frame of each chunk and carries the mask through the rest; every frame's
result is gated against that frame's own annotated box, and what survives is
written to the same run-length store the Anti-UAV410 pseudo-labels use
(`src/training/masklets.py` argues each decision).

**Calibrate before you spend.** The VIS split's ~100 sequences carry drawn
masks; run this with `--calibrate` on a few of them first and read the
masklet-versus-drawn IoU. That number -- not an argument -- is what says
whether the other ~400 sequences are worth teacher-hours, and whether
`--frames rgb` (masks made on the registered RGB half) beats `--frames ir`
(the teacher looking at thermal directly).

Usage:
    # the measurement first: sequences that have drawn masks
    python tools/make_masklets.py --data /content/data/VTUAV_VIS \\
        --out /content/work/masklets --calibrate --limit 3

    # then the harvest, on sequences that have only boxes
    python tools/make_masklets.py --data /content/data/VTUAV_ST \\
        --out /content/work/masklets --chunk 200

    # SAM 3 instead: gated repo, needs transformers>=5
    python tools/make_masklets.py ... --teacher facebook/sam3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.labels import Gates  # noqa: E402
from src.training.masklets import (  # noqa: E402
    build_video_teacher,
    calibrate,
    find_sequences,
    load_store,
    masklet_sequence,
    summarise_masklets,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True,
                   help="Root holding VTUAV-shaped sequences (rgb/, ir/, rgb.txt).")
    p.add_argument("--out", required=True, help="Where the mask stores go.")
    p.add_argument("--teacher", default="facebook/sam2.1-hiera-large",
                   help="facebook/sam2.1-hiera-large (default, ungated) or "
                        "facebook/sam3 (gated, transformers>=5). The class is "
                        "chosen from the id.")
    p.add_argument("--frames", choices=("rgb", "ir"), default="rgb",
                   help="Which modality the teacher looks at. rgb rides the "
                        "registration; ir asks the teacher to read thermal "
                        "directly. --calibrate measures which is better.")
    p.add_argument("--modality", choices=("ir", "rgb"), default="ir",
                   help="Whose drawn masks calibrate the result.")
    p.add_argument("--sequences", action="append", default=None,
                   help="Repeatable. Only these sequence names.")
    p.add_argument("--limit", type=int, default=None,
                   help="At most this many sequences, in name order.")
    p.add_argument("--chunk", type=int, default=200,
                   help="Frames per propagation before re-prompting with the "
                        "ground-truth box. Bounds drift and host memory alike.")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Cap on visible frames per sequence -- for smoke runs.")
    p.add_argument("--box-iou", type=float, default=Gates.box_iou,
                   help="Reject a propagated mask whose box-IoU with that "
                        "frame's annotation falls below this. The drift gate.")
    p.add_argument("--component", type=float, default=Gates.component,
                   help="Reject a mask whose largest connected component holds "
                        "less than this share of its area.")
    p.add_argument("--calibrate", action="store_true",
                   help="Score accepted masklets against drawn masks where a "
                        "sequence has them.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--json", default=None, help="Also write the summary here.")
    args = p.parse_args(argv)

    sequences = find_sequences(args.data, names=args.sequences)
    if args.limit is not None:
        sequences = sequences[:args.limit]
    if not sequences:
        raise SystemExit(f"no VTUAV-shaped sequences under {args.data} -- a "
                         f"sequence is a directory holding rgb/ and rgb.txt.")
    print(f"{len(sequences)} sequence(s), teacher {args.teacher}, "
          f"prompting on {args.frames} frames")

    teacher = build_video_teacher(args.teacher, device=args.device,
                                  dtype=args.dtype)
    gates = Gates(box_iou=args.box_iou, component=args.component)

    reports = []
    for sequence in sequences:
        report = masklet_sequence(
            sequence, teacher, args.out, gates=gates,
            prompt_frames=args.frames, chunk=args.chunk,
            max_frames=args.max_frames, progress=_tqdm())
        if args.calibrate:
            store = load_store(Path(args.out) / sequence.name)
            report["calibration"] = calibrate(sequence, store, args.modality)
        reports.append(report)
        rate = report["acceptance_rate"]
        cal = report.get("calibration")
        note = (f"  IoU all {cal['iou_all']:.3f} / accepted "
                f"{cal['iou_accepted']:.3f}" if cal else "")
        print(f"  {sequence.name}: {report['accepted']}/{report['attempted']} "
              f"accepted ({rate:.1%}){note}")

    print()
    print(summarise_masklets(reports))
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(reports, indent=2) + "\n")
        print(f"\nwrote {out}")
    return 0


def _tqdm():
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - environment, not logic
        return None
    return lambda stream, total, desc: tqdm(stream, total=total, desc=desc)


if __name__ == "__main__":
    raise SystemExit(main())
