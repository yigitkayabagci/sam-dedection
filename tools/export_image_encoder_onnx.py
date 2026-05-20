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

Usage:
    python tools/export_image_encoder_onnx.py \\
        --checkpoint third_party/EdgeTAM/checkpoints/edgetam.pt \\
        --output     models/edgetam_image_encoder.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn


class ImageEncoderWrapper(nn.Module):
    """Wrap EdgeTAM's image encoder so it returns a tuple instead of a dict.

    ONNX export does not support dict outputs, and we only need the FPN
    feature maps anyway.
    """

    def __init__(self, image_encoder: nn.Module, scalp: int) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.scalp = scalp

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.image_encoder(image)
        feats = out["backbone_fpn"]
        # feats is in coarse->fine order (lowest-res first). The SAM2 base
        # expects 3 levels after scalp discard; pick the last 3.
        feats = feats[-3:]
        return feats[0], feats[1], feats[2]


def build(checkpoint: str, model_cfg: str = "configs/edgetam.yaml") -> nn.Module:
    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device="cpu")
    predictor.eval()
    scalp = getattr(predictor.image_encoder, "scalp", 1)
    return ImageEncoderWrapper(predictor.image_encoder, scalp=scalp)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="third_party/EdgeTAM/checkpoints/edgetam.pt")
    p.add_argument("--model-cfg", default="configs/edgetam.yaml",
                   help="Hydra config name resolved by the sam2 package.")
    p.add_argument("--output", default="models/edgetam_image_encoder.onnx")
    p.add_argument("--input-size", type=int, default=1024)
    p.add_argument("--opset", type=int, default=17)
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

    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["image"],
        output_names=["backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2"],
        opset_version=args.opset,
        do_constant_folding=True,
        # Static shape; TensorRT builds a faster engine without dynamic axes.
        dynamic_axes=None,
    )
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
