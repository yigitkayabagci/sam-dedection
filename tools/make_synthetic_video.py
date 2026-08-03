#!/usr/bin/env python3
"""Write a long synthetic clip + matching prompt file for a visual FPS run.

`samples/synthetic_car.mp4` is 30 frames -- enough to prove the pipeline runs,
not enough to plot an FPS curve or eyeball tracking drift over hundreds of
frames. This reuses the same moving-blob generator `benchmark_tracking.py`
uses for its throughput numbers, but writes it to disk with a prompt file next
to it so `cli.py` can run on it directly and produce an annotated video plus
an `--fps-chart` PNG -- the same clip, same prompts, for both backends.

Usage:
    python tools/make_synthetic_video.py --frames 750 --outdir data/synth_750

    # --frame-pattern is required: frames-dir mode defaults to *.tif*, and
    # this writes the same JPGs benchmark_tracking.py does. --fps-chart and
    # --stage-chart are independent -- pass either, both, or neither.
    python cli.py --tracker edgetam_trt --config configs/edgetam_trt.yaml \\
        --frames-dir data/synth_750 --frame-pattern "*.jpg" --fps 30 \\
        --output outputs/synth_trt.mp4 \\
        --prompt file --prompt-file data/synth_750/prompts.json \\
        --fps-warmup 20 --fps-chart outputs/synth_trt_fps.png \\
        --stage-chart outputs/synth_trt_stages.png

    python cli.py --tracker edgetam --config configs/edgetam.yaml \\
        --frames-dir data/synth_750 --frame-pattern "*.jpg" --fps 30 \\
        --output outputs/synth_torch.mp4 \\
        --prompt file --prompt-file data/synth_750/prompts.json \\
        --fps-warmup 20 --fps-chart outputs/synth_torch_fps.png \\
        --stage-chart outputs/synth_torch_stages.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.benchmark_tracking import write_synthetic_clip  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--frames", type=int, default=750)
    p.add_argument("--size", default="1280x720", help="WxH.")
    p.add_argument("--objects", type=int, default=1)
    p.add_argument("--outdir", default="data/synth_clip")
    args = p.parse_args(argv)

    width, height = (int(v) for v in args.size.lower().split("x"))
    outdir = Path(args.outdir)

    print(f">> Writing {args.frames} frames at {width}x{height} to {outdir}")
    prompts = write_synthetic_clip(outdir, args.frames, width, height, args.objects)

    prompt_path = outdir / "prompts.json"
    prompt_path.write_text(
        json.dumps(
            {
                "boxes": [
                    {"obj_id": b.obj_id, "frame_idx": b.frame_idx, "xyxy": list(b.xyxy)}
                    for b in prompts.boxes
                ],
                "points": [],
            },
            indent=2,
        )
        + "\n"
    )
    print(f">> wrote {prompt_path}")

    print(
        "\n>> Next, run both backends on it (--frame-pattern is required -- "
        "frames-dir mode\n"
        "   defaults to *.tif*, and this wrote JPGs):\n"
        f"   python cli.py --tracker edgetam_trt --config configs/edgetam_trt.yaml \\\n"
        f"       --frames-dir {outdir} --frame-pattern \"*.jpg\" --fps 30 \\\n"
        f"       --output outputs/synth_trt.mp4 \\\n"
        f"       --prompt file --prompt-file {prompt_path} \\\n"
        f"       --fps-warmup 20 --fps-chart outputs/synth_trt_fps.png \\\n"
        f"       --stage-chart outputs/synth_trt_stages.png\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
