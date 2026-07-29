"""Fixed-slot layout of EdgeTAM's memory bank.

Shared by the ONNX exporters (`tools/edgetam_graphs.py`) and the runtime
tracker, because both have to agree on the token order byte for byte: the
exporter bakes the RoPE tables and the slot split into the graph, the tracker
fills the buffer the graph expects.

Upstream concatenates however many memories exist on the current frame, so the
key sequence grows over the first frames of a video and then plateaus. A
TensorRT engine inside a CUDA graph cannot follow that: it has one shape,
forever. So we always hand it a full-size buffer and switch unused positions
off with an additive attention mask.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

# Additive logit applied to masked-out memory positions. Must be finite (a real
# -inf would produce NaN if a whole row were masked) and representable in fp16
# (max 65504). exp(-30000) underflows to exactly 0 in every dtype we use, so a
# masked position contributes exactly nothing -- masking is not an
# approximation of the shorter sequence, it is equal to it.
MASKED_LOGIT = -30000.0


@dataclass(frozen=True)
class MemoryLayout:
    """Token order of the padded memory bank:

        [ slot 0 | slot 1 | ... | slot N-1 | object-pointer region ]
          <----------- spatial_tokens -----------> <-- ptr_tokens -->

    Each spatial slot holds one frame's compressed memory: `tokens_per_slot`
    tokens, of which the *trailing* `rope_tokens_per_slot` are the Perceiver's
    2D latents (spatially structured, so they get RoPE) and the leading ones
    are its 1D latents (no RoPE). That split is EdgeTAM's -- see
    `apply_rotary_enc_v2` -- not something introduced here.
    """

    num_slots: int
    tokens_per_slot: int
    rope_tokens_per_slot: int
    ptr_tokens: int

    @property
    def spatial_tokens(self) -> int:
        return self.num_slots * self.tokens_per_slot

    @property
    def total_tokens(self) -> int:
        return self.spatial_tokens + self.ptr_tokens

    @property
    def no_rope_tokens_per_slot(self) -> int:
        return self.tokens_per_slot - self.rope_tokens_per_slot

    @classmethod
    def from_model(cls, model, ptr_tokens: int | None = None) -> "MemoryLayout":
        """Derive the layout from a built EdgeTAM predictor.

        `ptr_tokens` is the only free parameter: it is the *capacity* reserved
        for object-pointer tokens, not a fixed count. Upstream emits
        hidden_dim/mem_dim tokens per pointer, and the pointer count is
        `n_cond_frames + min(max_obj_ptrs - 1, n_past_frames)` -- unbounded
        above, because a user may prompt on arbitrarily many frames. The
        default reserves room for 8 conditioning frames beyond the 16 pointers
        upstream targets; frames that would overflow fall back to PyTorch.
        """
        perceiver = getattr(model, "spatial_perceiver", None)
        if perceiver is None:
            raise ValueError(
                "This model has no spatial_perceiver; the fixed-slot memory "
                "layout is specific to EdgeTAM's compressed memory."
            )
        if perceiver.num_latents <= 0 or perceiver.num_latents_2d <= 0:
            raise ValueError(
                "Expected the Perceiver to emit both 1D and 2D latents "
                f"(got num_latents={perceiver.num_latents}, "
                f"num_latents_2d={perceiver.num_latents_2d})."
            )
        cross_attn = model.memory_attention.layers[0].cross_attn_image
        rope_tokens = int(cross_attn.freqs_cis_k.shape[0])
        if rope_tokens != perceiver.num_latents_2d:
            raise ValueError(
                f"Cross-attention RoPE table covers {rope_tokens} tokens but the "
                f"Perceiver emits {perceiver.num_latents_2d} 2D latents."
            )
        if ptr_tokens is None:
            per_ptr = model.hidden_dim // model.mem_dim
            ptr_tokens = per_ptr * (model.max_obj_ptrs_in_encoder + 8)
        return cls(
            num_slots=model.num_maskmem,
            tokens_per_slot=perceiver.num_latents + perceiver.num_latents_2d,
            rope_tokens_per_slot=rope_tokens,
            ptr_tokens=int(ptr_tokens),
        )

    @classmethod
    def from_model_and_engine(cls, model, total_tokens: int) -> "MemoryLayout":
        """Recover the layout an engine was built with, from its memory length.

        Everything but the pointer capacity is fixed by the model, so the
        engine's own `memory` sequence length is enough to pin down the rest.
        This is how the tracker stays in sync with an engine without a sidecar
        file to lose.
        """
        layout = cls.from_model(model, ptr_tokens=0)
        ptr_tokens = total_tokens - layout.spatial_tokens
        if ptr_tokens < 0:
            raise ValueError(
                f"Engine memory length {total_tokens} is shorter than this model's "
                f"{layout.num_slots} x {layout.tokens_per_slot} spatial memory "
                "slots. The engine was built for a different configuration."
            )
        return cls(
            num_slots=layout.num_slots,
            tokens_per_slot=layout.tokens_per_slot,
            rope_tokens_per_slot=layout.rope_tokens_per_slot,
            ptr_tokens=ptr_tokens,
        )


def build_memory_mask(
    layout: MemoryLayout,
    num_spatial_mem: int,
    num_ptr_tokens: int,
    batch: int = 1,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Additive attention mask for a partially filled memory bank."""
    mask = torch.full(
        (batch, 1, 1, layout.total_tokens), MASKED_LOGIT, device=device, dtype=dtype
    )
    mask[..., : num_spatial_mem * layout.tokens_per_slot] = 0.0
    if num_ptr_tokens > 0:
        end = layout.spatial_tokens + num_ptr_tokens
        mask[..., layout.spatial_tokens : end] = 0.0
    return mask
