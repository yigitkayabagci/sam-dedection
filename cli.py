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
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from src.pipeline import PipelineConfig, grab_first_frame, run
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

    first = grab_first_frame(args.video)
    if args.prompt == "box":
        from src.prompts.interactive import pick_box
        return pick_box(first, obj_id=args.obj_id)
    if args.prompt == "point":
        from src.prompts.interactive import pick_points
        return pick_points(first, obj_id=args.obj_id)
    raise SystemExit(f"Unknown prompt mode: {args.prompt}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Modular SAM-style video tracking pipeline.")
    p.add_argument("--video", required=True, help="Input video path.")
    p.add_argument("--output", required=True, help="Output video (.mp4) path.")
    p.add_argument("--tracker", default="edgetam", help=f"Backend (one of {available_trackers()}).")
    p.add_argument("--config", default=None, help="Optional backend YAML override.")
    p.add_argument("--prompt", choices=("box", "point", "file"), default="box")
    p.add_argument("--prompt-file", default=None, help="JSON prompt file (used with --prompt file).")
    p.add_argument("--obj-id", type=int, default=1, help="Object id for interactive prompts.")
    p.add_argument("--keep-frames", action="store_true", help="Do not delete extracted frames.")
    p.add_argument("--no-bbox", action="store_true", help="Disable bounding-box overlay.")
    p.add_argument("--alpha", type=float, default=0.5, help="Mask overlay alpha.")
    args = p.parse_args(argv)

    backend_cfg = _load_backend_config(args.tracker, args.config)
    backend_cfg = _resolve_paths(backend_cfg, keys=("model_cfg", "checkpoint"))
    tracker = build_tracker(args.tracker, **backend_cfg)

    prompts = _collect_prompts(args)

    cfg = PipelineConfig(
        video_path=Path(args.video),
        output_path=Path(args.output),
        keep_frames=args.keep_frames,
        draw_bbox=not args.no_bbox,
        mask_alpha=args.alpha,
    )
    out = run(tracker, prompts, cfg)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
