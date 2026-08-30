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

# What a usable `image_size` has to be a multiple of. **Not** the backbone
# stride: the 2D spatial perceiver partitions the stride-16 feature map into
# 16x16 windows without padding, so the feature side must itself be a multiple
# of 16 and the image size a multiple of 16 * 16. See `check_image_size`.
WINDOW_ALIGNMENT = 256


def check_image_size(image_size: int) -> str | None:
    """Why this resolution cannot be built, or None when it can.

    The binding constraint is the **2D spatial perceiver**, not the FPN and not
    the backbone: it partitions the stride-16 feature map into exactly 16x16
    windows with a partition that does not pad
    (`sam2/modeling/perceiver.py`), so the *feature side* has to be a multiple
    of 16 -- and the feature side is `image_size // 16`. Hence
    `image_size % 256 == 0`, which is a stronger rule than the one it looks
    like it should be. `tests/test_hydra_overrides.py` carries the two
    measured crashes that pin it:

        640  RuntimeError: size of tensor a (256) must match tensor b (400)
        896  RuntimeError: shape '[1,18,3,18,3,64]' is invalid for input of ...

    640 / 16 = 40 and 896 / 16 = 56, and neither is divisible by 16. 256, 512,
    768 (3 * 256) and 1024 are, which is the set this repo ships.

    A check and not a paragraph, because the failure is silent in the worst
    way: a size the arithmetic does not admit is written into a config, and
    what comes back is a shape error deep in the first forward pass -- or, for
    a stale rotary table, no error at all and the wrong rotation.
    """
    size = int(image_size)
    if size <= 0:
        return f"image_size must be positive, got {size}"
    if size % BACKBONE_STRIDE:
        return (f"image_size {size} is not a multiple of the backbone stride "
                f"{BACKBONE_STRIDE}, so the stride-16 feature map has no "
                f"integer side and the cross-attention rotary table cannot be "
                f"sized to match it")
    if size % WINDOW_ALIGNMENT:
        side = size // BACKBONE_STRIDE
        return (f"image_size {size} gives a stride-16 map of {side}x{side}, and "
                f"{side} is not a multiple of 16 -- the spatial perceiver "
                f"partitions that map into 16x16 windows without padding, so it "
                f"fails at the first forward pass. Sizes must be multiples of "
                f"{WINDOW_ALIGNMENT}: 256, 512, 768, 1024")
    return None


def image_size_overrides(image_size: int | None) -> list[str]:
    if not image_size:
        return []
    refusal = check_image_size(image_size)
    if refusal:
        raise ValueError(refusal)
    side = image_size // BACKBONE_STRIDE
    return [
        f"++model.image_size={image_size}",
        f"model.memory_attention.layer.cross_attention.q_sizes=[{side},{side}]",
    ]
