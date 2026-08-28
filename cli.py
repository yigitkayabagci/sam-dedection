#!/usr/bin/env python3
"""Modular SAM-style video tracking CLI.

Examples
--------
Interactive bounding box (default), EdgeTAM backend on CUDA:

    python cli.py --video samples/road.mp4 --output outputs/road_tracked.mp4 \
        --tracker edgetam --prompt box

Click-based prompts (left=fg, right=bg, ENTER to finish):

    python cli.py --video samples/road.mp4 --output outputs/road_tracked.mp4 \
        --prompt point

Headless: load prompts from JSON (e.g. on Orin AGX without a display):

    python cli.py --video samples/road.mp4 --output outputs/road_tracked.mp4 \
        --prompt file --prompt-file prompts/car_box.json

Frame sequence input (no video file; e.g. frame_000000.tiff, ...):

    python cli.py --frames-dir frames/ --frame-pattern 'frame_*.tiff' \
        --output outputs/tracked.mp4 --fps 30 --prompt box
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from src.pipeline import (
    PipelineConfig,
    grab_first_frame,
    grab_first_frame_dir,
    run,
)
from src.prompts import PromptSet
from src.prompts.file_source import load_prompts
from src.trackers import available_trackers, build_tracker

ROOT = Path(__file__).resolve().parent


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p).resolve()


def _load_backend_config(name: str, override: str | None) -> dict:
    cfg_path = _resolve(override) if override else _resolve(f"configs/{name}.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    return yaml.safe_load(cfg_path.read_text())


def _resolve_paths(cfg: dict, keys: tuple[str, ...]) -> dict:
    out = dict(cfg)
    for k in keys:
        if k in out and out[k]:
            out[k] = str(_resolve(out[k]))
    return out


def _collect_prompts(args: argparse.Namespace) -> PromptSet:
    if args.prompt == "file":
        if not args.prompt_file:
            raise SystemExit("--prompt file requires --prompt-file <path>")
        return load_prompts(args.prompt_file)

    if args.frames_dir:
        first = grab_first_frame_dir(args.frames_dir, args.frame_pattern)
    else:
        first = grab_first_frame(args.video)
    if args.prompt == "box":
        if args.multi:
            from src.prompts.interactive import pick_boxes
            return pick_boxes(first, start_obj_id=args.obj_id)
        from src.prompts.interactive import pick_box
        return pick_box(first, obj_id=args.obj_id)
    if args.prompt == "point":
        if args.multi:
            from src.prompts.interactive import pick_points_multi
            return pick_points_multi(first, start_obj_id=args.obj_id)
        from src.prompts.interactive import pick_points
        return pick_points(first, obj_id=args.obj_id)
    raise SystemExit(f"Unknown prompt mode: {args.prompt}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Modular SAM-style video tracking pipeline.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="Input video path.")
    src.add_argument("--frames-dir", help="Directory of pre-extracted frames "
                                          "(e.g. frame_000000.tiff, frame_000001.tiff, ...).")
    p.add_argument("--frame-pattern", default="*.tif*",
                   help="Glob for frames inside --frames-dir (default: *.tif*).")
    p.add_argument("--center-crop", type=int, default=None,
                   help="Track a centred NxN window instead of the whole frame "
                        "(--frames-dir only). Set N to the model input size and "
                        "preprocessing stops resizing: native pixels reach the "
                        "model, which is what a camera that already centres and "
                        "focuses on the target delivers. Prompts stay in "
                        "full-frame coordinates and are shifted for you.")
    p.add_argument("--fps", type=float, default=30.0,
                   help="Output FPS for --frames-dir mode (image sequences have no inherent fps).")
    p.add_argument("--frame-skip", "--frameskip", dest="frame_skip", type=int,
                   nargs="?", const=2, default=1, metavar="N",
                   help="Run inference on one source frame in N and hold that mask over "
                        "the rest (bare flag = 2, i.e. infer a frame, skip a frame). "
                        "Every source frame still gets a mask and the output clip keeps "
                        "its full length -- a 500-frame clip stays 500 frames, decided by "
                        "250 inferences. Skipped frames never reach the model, so this "
                        "does not make a frame faster: it gives each inference N frame "
                        "periods to finish in, 66.7 ms instead of 33.3 at 30 fps. The "
                        "cost is temporal resolution: the target moves N times further "
                        "between inferences, and a held mask is up to N-1 frames stale. "
                        "Not available with --video-mode mp4 (decord gets the whole "
                        "file; use jpg mode or --frames-dir).")
    p.add_argument("--output", default=None, help="Output video (.mp4) path. Omit "
                                                  "with --no-video to measure only.")
    p.add_argument("--no-video", action="store_true",
                   help="Track without producing a video: skip re-reading each frame "
                        "for the overlay, skip drawing it, and skip mp4 encoding. "
                        "Those three exist to make something watchable, not to track, "
                        "so this is the configuration that reflects a real-time edge "
                        "deployment and the one to quote a frame budget from.")
    p.add_argument("--tracker", default="edgetam", help=f"Backend (one of {available_trackers()}).")
    p.add_argument("--config", default=None, help="Optional backend YAML override.")
    p.add_argument("--strict", action="store_true",
                   help="Refuse a silent PyTorch fallback (TensorRT backends). "
                        "The shipped TRT configs set strict: false, so a missing "
                        "or unloadable engine prints one line and the run "
                        "continues on PyTorch -- correct weights, different "
                        "backend, and a speed number that is not what it says "
                        "it is. Pass this when the run is a measurement rather "
                        "than a look.")
    p.add_argument("--prompt", choices=("box", "point", "file"), default="box")
    p.add_argument("--prompt-file", default=None, help="JSON prompt file (used with --prompt file).")
    p.add_argument("--multi", action="store_true",
                   help="Multi-target interactive selection. With --prompt box: draw several "
                        "boxes (each a new object). With --prompt point: left=fg, right=bg, "
                        "'n'=next object, 1-9=pick object id, ENTER=done.")
    p.add_argument("--obj-id", type=int, default=1,
                   help="Object id for interactive prompts (start id in --multi mode).")
    p.add_argument("--video-mode", choices=("auto", "mp4", "jpg"), default="auto",
                   help="auto = use mp4 directly if decord is available, else fall back to jpg.")
    p.add_argument("--keep-frames", action="store_true", help="Do not delete extracted frames (jpg mode).")
    p.add_argument("--no-bbox", action="store_true", help="Disable bounding-box overlay.")
    p.add_argument("--alpha", type=float, default=0.5, help="Mask overlay alpha.")
    p.add_argument("--fps-warmup", type=int, default=0,
                   help="Exclude the first N frames (model load / CUDA warm-up) from the "
                        "average FPS report.")
    p.add_argument("--fps-chart", default=None,
                   help="Save a per-frame latency chart (ms, with max/min/avg) to this "
                        "PNG path (needs matplotlib).")
    p.add_argument("--stage-chart", default=None,
                   help="Save a per-frame decode/inference/render breakdown chart to this "
                        "PNG path (needs matplotlib).")
    p.add_argument("--offload-video", dest="offload_video", action="store_true", default=None,
                   help="Keep decoded frames on the CPU, overriding the backend YAML. "
                        "EdgeTAM otherwise preloads every frame to the GPU as fp32 at "
                        "model resolution (~6.3 GB for 500 frames at 1024x1024), which "
                        "on a Jetson's shared memory slows the model down. Match this to "
                        "tools/benchmark_tracking.py's --offload-video when comparing.")
    p.add_argument("--no-offload-video", dest="offload_video", action="store_false",
                   help="Force GPU-resident frames, overriding the backend YAML.")
    args = p.parse_args(argv)

    # Before building anything: an unusable flag combination should not cost the
    # user a model load and a prompt read first.
    if args.no_video:
        if args.output:
            raise SystemExit("--no-video produces no video; drop --output.")
    elif not args.output:
        raise SystemExit("--output is required (or pass --no-video to measure only).")
    if args.center_crop and not args.frames_dir:
        raise SystemExit("--center-crop applies to --frames-dir mode only.")
    if args.frame_skip < 1:
        raise SystemExit(f"--frame-skip takes 1 (every frame) or more, got {args.frame_skip}.")
    if args.frame_skip > 1 and args.video_mode == "mp4":
        raise SystemExit("--frame-skip cannot decimate the direct-mp4 path; use "
                         "--video-mode jpg or --frames-dir.")

    backend_cfg = _load_backend_config(args.tracker, args.config)
    if args.strict:
        backend_cfg["strict"] = True
    if args.offload_video is not None:
        backend_cfg["offload_video_to_cpu"] = args.offload_video
    # Resolve filesystem paths; model_cfg is a Hydra config name, not a path.
    backend_cfg = _resolve_paths(
        backend_cfg,
        keys=(
            "checkpoint",
            "engine_path",
            "image_encoder_engine",
            "memory_attention_engine",
            "memory_encoder_engine",
            "sam_head_engine",
        ),
    )
    tracker = build_tracker(args.tracker, **backend_cfg)

    prompts = _collect_prompts(args)

    cfg = PipelineConfig(
        output_path=Path(args.output) if args.output else None,
        video_path=Path(args.video) if args.video else None,
        frames_dir=Path(args.frames_dir) if args.frames_dir else None,
        frame_pattern=args.frame_pattern,
        center_crop=args.center_crop,
        fps=args.fps,
        frame_stride=args.frame_skip,
        video_mode=args.video_mode,
        keep_frames=args.keep_frames,
        draw_bbox=not args.no_bbox,
        mask_alpha=args.alpha,
        fps_warmup=args.fps_warmup,
        fps_chart=Path(args.fps_chart) if args.fps_chart else None,
        stage_chart=Path(args.stage_chart) if args.stage_chart else None,
    )
    out = run(tracker, prompts, cfg)
    print(f"wrote {out}" if out else "done (no video written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
