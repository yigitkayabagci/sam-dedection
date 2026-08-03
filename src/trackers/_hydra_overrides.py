"""Hydra overrides for building EdgeTAM at a non-default `image_size`.

`model.image_size` alone is not enough. The memory attention's cross-attention
(`RoPEAttentionv2`) precomputes its rotary tables once at construction from a
separately hardcoded `q_sizes` config field (`[64, 64]` for the stock 1024
config) and never revisits them -- unlike the self-attention's plain
`RoPEAttention`, which recomputes its table on the fly if the input shape
doesn't match. Overriding `image_size` without also fixing `q_sizes` leaves a
stale rotary table sized for the old resolution, which either throws a shape
error or, worse, applies the wrong rotation silently.

`q_sizes` tracks `image_size // backbone_stride` -- the exact quantity
`SAM2Base` itself computes at runtime as `sam_image_embedding_size`
(`modeling/sam2_base.py`), so this override keeps the two in lockstep instead
of introducing a second, independent formula.

`k_sizes` (the memory side) is left alone: it sizes the 2D Spatial Perceiver's
compressed output, which is a fixed latent count by design, not a function of
image_size.
"""
from __future__ import annotations

BACKBONE_STRIDE = 16  # SAM2Base's default; fixed by the backbone architecture.


def image_size_overrides(image_size: int | None) -> list[str]:
    if not image_size:
        return []
    side = image_size // BACKBONE_STRIDE
    return [
        f"++model.image_size={image_size}",
        f"model.memory_attention.layer.cross_attention.q_sizes=[{side},{side}]",
    ]
