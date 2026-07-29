#!/usr/bin/env python3
"""Deprecated: use `tools/export_edgetam_onnx.py`.

This used to be the only exporter, back when TensorRT covered just the image
encoder. It now forwards to the unified exporter, which also emits the memory
attention, memory encoder and SAM head graphs plus the `.spec.json` files
`tools/build_trt_engines.py` needs to build optimisation profiles.

The image encoder graph itself changed too: it now folds in the mask decoder's
conv_s0/conv_s1 projections, which shrinks its per-frame output from ~45 MB to
~6 MB. Pass --no-fuse-high-res-proj to the new exporter for the old behaviour.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.export_edgetam_onnx import main as export_main  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="third_party/EdgeTAM/checkpoints/edgetam.pt")
    p.add_argument("--model-cfg", default="configs/edgetam.yaml")
    p.add_argument("--output", default="models/edgetam_image_encoder.onnx")
    p.add_argument("--input-size", type=int, default=None,
                   help="Ignored; the resolution comes from the model config.")
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args(argv)

    outdir = Path(args.output).parent or Path("models")
    print(
        "tools/export_image_encoder_onnx.py is deprecated. Forwarding to:\n"
        f"  python tools/export_edgetam_onnx.py --module image_encoder "
        f"--outdir {outdir}\n"
        "Run it without --module to export all four graphs.\n"
    )
    if args.input_size is not None:
        print(f"NOTE: --input-size {args.input_size} ignored; using the model's image_size.\n")

    forwarded = [
        "--module", "image_encoder",
        "--checkpoint", args.checkpoint,
        "--model-cfg", args.model_cfg,
        "--outdir", str(outdir),
        "--opset", str(args.opset),
    ]
    if args.verify:
        forwarded.append("--verify")
    return export_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
