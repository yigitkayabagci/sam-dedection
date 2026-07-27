#!/usr/bin/env python3
"""Export EdgeTAM's image encoder to ONNX.

The image encoder is the heaviest per-frame component; replacing just it
with a TensorRT engine is what unlocks big speedups on Jetson Orin AGX.
The rest of EdgeTAM (memory module, mask decoder) stays in PyTorch.

Output: a single ONNX graph
    Input:  image  [1, 3, 1024, 1024]   (static shape; faster TRT engine)
    Outputs (raw FPN levels — all 256-channel because EdgeTAM's FpnNeck
    projects every level to d_model=256; the downstream `forward_image`
    later applies conv_s0/conv_s1 to project the high-res ones to 32/64 ch):
        backbone_fpn_0  [1, 256, 256, 256]   (highest resolution)
        backbone_fpn_1  [1, 256, 128, 128]
        backbone_fpn_2  [1, 256,  64,  64]   (lowest resolution = vision_features)

Positional encodings are NOT exported — they are deterministic functions
of spatial size and re-generated in PyTorch on the consumer side.

The export runs on CPU in FP32 on purpose. Do NOT convert the ONNX to fp16
here — precision is TensorRT's decision to make per layer at build time, and
a pre-cast graph takes that choice away. The resulting .onnx is hardware
independent (export it on an x86 box and copy it over if that is faster);
only the .engine is tied to the specific GPU and TensorRT version.

Usage:
    python3 tools/export_image_encoder_onnx.py \\
        --checkpoint third_party/EdgeTAM/checkpoints/edgetam.pt \\
        --output     models/edgetam_image_encoder.onnx --verify
"""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn


class ImageEncoderWrapper(nn.Module):
    """Wrap EdgeTAM's image encoder so it returns a tuple instead of a dict.

    ONNX export does not support dict outputs. The wrapper calls the
    image_encoder (which already applies the configured `scalp` discard
    inside its forward) and returns the post-scalp backbone_fpn levels as
    a tuple. We expect exactly 3 levels — that is what the SAM2 base
    consumes downstream.
    """

    def __init__(self, image_encoder: nn.Module) -> None:
        super().__init__()
        self.image_encoder = image_encoder

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.image_encoder(image)
        feats = out["backbone_fpn"]
        if len(feats) != 3:
            raise RuntimeError(
                f"Expected 3 post-scalp FPN levels, got {len(feats)}. "
                "Adjust the wrapper if you use a non-default scalp."
            )
        return feats[0], feats[1], feats[2]


def build(checkpoint: str, model_cfg: str = "configs/edgetam.yaml") -> nn.Module:
    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device="cpu")
    predictor.eval()
    return ImageEncoderWrapper(predictor.image_encoder)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="third_party/EdgeTAM/checkpoints/edgetam.pt")
    p.add_argument("--model-cfg", default="configs/edgetam.yaml",
                   help="Hydra config name resolved by the sam2 package.")
    p.add_argument("--output", default="models/edgetam_image_encoder.onnx")
    p.add_argument("--input-size", type=int, default=1024)
    p.add_argument("--opset", type=int, default=18,
                   help="ONNX opset. 18 is the lowest version torch's exporter "
                        "natively emits; lower values trigger a version-converter "
                        "round-trip with noisy warnings. TRT 10 supports >=11.")
    p.add_argument("--verify", action="store_true",
                   help="Run onnxruntime CPU and assert numerical parity with PyTorch.")
    args = p.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    model = build(args.checkpoint, args.model_cfg)
    model.eval()
    dummy = torch.randn(1, 3, args.input_size, args.input_size, dtype=torch.float32)

    # Sanity check: run once eagerly so any shape mismatch surfaces here.
    with torch.no_grad():
        sample = model(dummy)
    for i, t in enumerate(sample):
        print(f"output[{i}] shape={tuple(t.shape)} dtype={t.dtype}")

    export_kwargs = dict(
        input_names=["image"],
        output_names=["backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2"],
        opset_version=args.opset,
        do_constant_folding=True,
        # Static shape; TensorRT builds a faster engine without dynamic axes.
        dynamic_axes=None,
    )
    # Newer torch releases flipped torch.onnx.export's default to the dynamo
    # exporter, which emits a different op set and carries dynamic-shape
    # metadata. Pin the TorchScript tracer so the graph TensorRT sees does not
    # change under us when the Jetson torch wheel is upgraded.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(model, dummy, args.output, **export_kwargs)
    print(f"wrote {args.output}  ({Path(args.output).stat().st_size / 1e6:.1f} MB)")

    if args.verify:
        try:
            import onnxruntime as ort
        except ImportError:
            print("WARN: onnxruntime not installed; skipping --verify.")
            return 0
        import numpy as np
        sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"image": dummy.numpy()})
        worst = 0.0
        for py, o in zip(sample, ort_out):
            worst = max(worst, float(np.abs(py.detach().numpy() - o).max()))
        print(f"PyTorch vs ONNX Runtime max abs diff = {worst:.2e}")
        if worst > 1e-3:
            raise SystemExit(f"Parity check failed (diff={worst:.2e} > 1e-3).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
