"""Export-friendly re-implementations of EdgeTAM's per-frame modules.

EdgeTAM's PyTorch modules are written for flexibility: variable-length memory,
complex-number RoPE, dict outputs, Python-level branching. None of that
survives an ONNX export, and none of it can live inside a CUDA graph (which
needs one static shape and one static set of device pointers, forever).

This module rebuilds them as fixed-shape `nn.Module`s that reuse the *original
weights and submodules* — no re-training, no re-derivation of layer math beyond
what is documented per class:

    ImageEncoderGraph      RepViT trunk + FPN neck, with the mask decoder's two
                           high-res 1x1 projections folded in.
    MemoryAttentionGraph   the 2-layer self+cross attention that conditions the
                           current frame on the memory bank. The dominant cost
                           of an EdgeTAM frame.
    MemoryEncoderGraph     memory encoder (mask downsample + ConvNeXt fuser)
                           followed by the 2D spatial Perceiver.
    SamHeadGraph           the SAM prompt-encoder/mask-decoder head, specialised
                           to the propagation path (no clicks, multimask=True).

Two transformations are what make the memory attention exportable:

1. Fixed memory slots + additive mask.
   Upstream concatenates however many memories exist this frame, so the key
   length grows from 512 to `num_maskmem * 512 + n_ptr_tokens` over the first
   ~16 frames of a video. We instead always hand the engine a full-size buffer
   of `num_slots` spatial slots plus a pointer region, and pass an additive
   attention mask that drives unused positions to zero weight. Because
   softmax(x + -30000) is exactly 0 in fp16 and fp32 alike, the masked result
   is numerically identical to attending over the shorter sequence -- see
   `tests/test_edgetam_graphs.py::test_padded_memory_matches_variable_length`.

   The padding is free of RoPE side effects: EdgeTAM tiles the same 16x16
   frequency table across every spatial slot, so `tile(f, num_slots)` is
   slot-position independent and can be folded into the graph as a constant.

2. Real-arithmetic RoPE.
   `apply_rotary_enc*` multiplies `torch.view_as_complex(x)` by a complex
   table. ONNX has no complex dtype. `(a + ib)(cos + i sin)` expanded into
   two real multiply-adds is exactly the same arithmetic, and exports to plain
   Mul/Sub/Add.

Nothing here is EdgeTAM-config-agnostic on purpose: every architectural
assumption is asserted in `__init__`, so a config this code cannot faithfully
reproduce fails loudly at export time instead of silently producing a wrong
engine.
"""
from __future__ import annotations

import contextlib
import math
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.trackers._trt_layout import MASKED_LOGIT, MemoryLayout, build_memory_mask

__all__ = [
    "MASKED_LOGIT",
    "ImageEncoderGraph",
    "MemoryAttentionGraph",
    "MemoryEncoderGraph",
    "MemoryLayout",
    "SamHeadGraph",
    "apply_rope_real",
    "build_memory_mask",
    "rope_tables",
]

# --------------------------------------------------------------------------
# Image encoder
# --------------------------------------------------------------------------


class ImageEncoderGraph(nn.Module):
    """Image encoder plus the two high-res projections `forward_image` applies.

    `SAM2Base.forward_image` runs `sam_mask_decoder.conv_s0` / `conv_s1` over
    FPN levels 0 and 1 right after the encoder. Folding those 1x1 convolutions
    into the engine costs nothing and shrinks its outputs from 256 channels to
    32 and 64 -- at 1024x1024 that is ~45 MB of writes per frame down to ~6 MB.
    The projected levels feed only the mask decoder's high-res path, so no
    consumer wants the unprojected ones.

    Outputs are the post-`scalp` FPN levels as a tuple (ONNX has no dict
    outputs). Positional encodings are left out: they are a pure function of
    spatial size, so the tracker regenerates and caches them once.
    """

    def __init__(self, model, fuse_high_res_proj: bool = True) -> None:
        super().__init__()
        self.image_encoder = model.image_encoder
        self.fuse = bool(fuse_high_res_proj and model.use_high_res_features_in_sam)
        if self.fuse:
            self.conv_s0 = model.sam_mask_decoder.conv_s0
            self.conv_s1 = model.sam_mask_decoder.conv_s1

    def forward(self, image: torch.Tensor):
        feats = self.image_encoder(image)["backbone_fpn"]
        if len(feats) != 3:
            raise RuntimeError(
                f"Expected 3 post-scalp FPN levels, got {len(feats)}. "
                "Adjust this wrapper if you use a non-default scalp."
            )
        if self.fuse:
            return self.conv_s0(feats[0]), self.conv_s1(feats[1]), feats[2]
        return feats[0], feats[1], feats[2]


# --------------------------------------------------------------------------
# RoPE without complex numbers
# --------------------------------------------------------------------------


def rope_tables(freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a complex `freqs_cis` table into real (cos) and imaginary (sin)."""
    return freqs_cis.real.float().contiguous(), freqs_cis.imag.float().contiguous()


def apply_rope_real(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotary embedding on real tensors.

    `x` is [..., N, C] with C even and interpreted as C/2 (even, odd) pairs;
    `cos`/`sin` are [N, C/2]. Identical arithmetic to
    `view_as_real(view_as_complex(x) * freqs_cis)`, which is what upstream does.

    Upstream promotes to float32 for the rotation regardless of the activation
    dtype and casts back; we keep that so an fp16 engine still computes the
    rotation the way the reference does.
    """
    x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    even = x_pairs[..., 0]
    odd = x_pairs[..., 1]
    rot_even = even * cos - odd * sin
    rot_odd = even * sin + odd * cos
    return torch.stack((rot_even, rot_odd), dim=-1).flatten(-2).type_as(x)


class _HeadShaper:
    """Static-shape stand-in for Attention._separate_heads/_recombine_heads.

    Upstream reads every dimension off `x.shape`, which bakes the batch size
    into the traced graph. We keep token count and head geometry as Python
    constants (they are static by construction) and leave batch as -1, so the
    exported graph stays batch-dynamic.
    """

    def __init__(self, num_heads: int, head_dim: int) -> None:
        self.num_heads = num_heads
        self.head_dim = head_dim

    def split(self, x: torch.Tensor, num_tokens: int) -> torch.Tensor:
        x = x.reshape(-1, num_tokens, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # [B, heads, tokens, head_dim]

    def merge(self, x: torch.Tensor, num_tokens: int) -> torch.Tensor:
        x = x.transpose(1, 2)
        return x.reshape(-1, num_tokens, self.num_heads * self.head_dim)


class _SelfAttentionGraph(nn.Module):
    """`RoPEAttention` with a precomputed real-valued table for a fixed length."""

    def __init__(self, attn: nn.Module, num_tokens: int) -> None:
        super().__init__()
        if attn.rope_k_repeat:
            raise ValueError("Self-attention is not expected to repeat RoPE keys.")
        side = int(round(math.sqrt(num_tokens)))
        if side * side != num_tokens:
            raise ValueError(f"Self-attention token count {num_tokens} is not square.")
        self.attn = attn
        self.num_tokens = num_tokens
        self.shaper = _HeadShaper(attn.num_heads, attn.internal_dim // attn.num_heads)
        cos, sin = rope_tables(attn.compute_cis(end_x=side, end_y=side))
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        a = self.attn
        n = self.num_tokens
        q = self.shaper.split(a.q_proj(q), n)
        k = self.shaper.split(a.k_proj(k), n)
        v = self.shaper.split(a.v_proj(v), n)
        q = apply_rope_real(q, self.cos, self.sin)
        k = apply_rope_real(k, self.cos, self.sin)
        out = F.scaled_dot_product_attention(q, k, v)
        return a.out_proj(self.shaper.merge(out, n))


class _CrossAttentionGraph(nn.Module):
    """`RoPEAttentionv2` over a fixed-length, padded, masked memory bank."""

    def __init__(self, attn: nn.Module, layout: MemoryLayout, num_q_tokens: int) -> None:
        super().__init__()
        if int(attn.freqs_cis_q.shape[0]) != num_q_tokens:
            raise ValueError(
                f"Query RoPE table covers {int(attn.freqs_cis_q.shape[0])} tokens "
                f"but the feature map has {num_q_tokens}."
            )
        self.attn = attn
        self.layout = layout
        self.num_q_tokens = num_q_tokens
        self.shaper = _HeadShaper(attn.num_heads, attn.internal_dim // attn.num_heads)

        cos_q, sin_q = rope_tables(attn.freqs_cis_q)
        self.register_buffer("cos_q", cos_q, persistent=False)
        self.register_buffer("sin_q", sin_q, persistent=False)
        # Upstream tiles the single-slot key table across every spatial slot.
        # Tiling a fixed `num_slots` times makes it a graph constant; because
        # every slot gets the same block, which slots are occupied never
        # changes the table.
        cos_k, sin_k = rope_tables(attn.freqs_cis_k)
        self.register_buffer("cos_k", cos_k.repeat(layout.num_slots, 1), persistent=False)
        self.register_buffer("sin_k", sin_k.repeat(layout.num_slots, 1), persistent=False)

    def _rope_spatial(self, k: torch.Tensor) -> torch.Tensor:
        """Rotate only the 2D-latent tail of each slot, as `apply_rotary_enc_v2` does."""
        lay = self.layout
        heads, head_dim = self.shaper.num_heads, self.shaper.head_dim
        k = k.reshape(-1, heads, lay.num_slots, lay.tokens_per_slot, head_dim)
        plain = k[:, :, :, : lay.no_rope_tokens_per_slot, :]
        rotated = k[:, :, :, lay.no_rope_tokens_per_slot :, :].reshape(
            -1, heads, lay.num_slots * lay.rope_tokens_per_slot, head_dim
        )
        rotated = apply_rope_real(rotated, self.cos_k, self.sin_k)
        rotated = rotated.reshape(
            -1, heads, lay.num_slots, lay.rope_tokens_per_slot, head_dim
        )
        return torch.cat((plain, rotated), dim=3).reshape(
            -1, heads, lay.spatial_tokens, head_dim
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        a = self.attn
        lay = self.layout
        nq, nk = self.num_q_tokens, lay.total_tokens
        q = self.shaper.split(a.q_proj(q), nq)
        k = self.shaper.split(a.k_proj(k), nk)
        v = self.shaper.split(a.v_proj(v), nk)
        q = apply_rope_real(q, self.cos_q, self.sin_q)
        # Object-pointer tokens are excluded from RoPE upstream via
        # `num_k_exclude_rope`; here they are simply the tail of the buffer.
        k = torch.cat(
            (self._rope_spatial(k[:, :, : lay.spatial_tokens]), k[:, :, lay.spatial_tokens :]),
            dim=2,
        )
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return a.out_proj(self.shaper.merge(out, nq))


# --------------------------------------------------------------------------
# Memory attention
# --------------------------------------------------------------------------


class MemoryAttentionGraph(nn.Module):
    """Fixed-shape, mask-aware `MemoryAttention`.

    Inputs are shaped so the runtime never has to make a strided copy:
        curr        [B, C, H, W]    straight off the image encoder engine
        curr_pos    [B, C, H, W]    constant across frames, written once
        memory      [B, L, mem_dim] padded memory bank, see `MemoryLayout`
        memory_pos  [B, L, mem_dim] its positional encoding (temporal PE folded in)
        memory_mask [B, 1, 1, L]    0 for live positions, MASKED_LOGIT for padding

    Output:
        pix_feat    [B, C, H, W]    memory-conditioned features, already in the
                                    BCHW layout `_prepare_memory_conditioned_features`
                                    reshapes to.

    The BCHW <-> token-sequence transposes upstream does in Python happen
    inside the graph instead, where TensorRT folds them into the neighbouring
    layer norm / matmul.
    """

    def __init__(self, model, layout: MemoryLayout) -> None:
        super().__init__()
        mem_attn = model.memory_attention
        if not mem_attn.batch_first:
            raise ValueError("Expected a batch-first MemoryAttention.")
        if not mem_attn.pos_enc_at_input:
            raise ValueError("Expected pos_enc_at_input=True.")
        for layer in mem_attn.layers:
            if layer.pos_enc_at_attn:
                raise ValueError("pos_enc_at_attn=True is not reproduced here.")
            if layer.pos_enc_at_cross_attn_queries:
                raise ValueError(
                    "pos_enc_at_cross_attn_queries=True is not reproduced here."
                )
            if not layer.pos_enc_at_cross_attn_keys:
                raise ValueError(
                    "pos_enc_at_cross_attn_keys=False is not reproduced here."
                )

        self.mem_attn = mem_attn
        self.layout = layout
        self.feat_side = model.image_size // model.backbone_stride
        self.num_tokens = self.feat_side * self.feat_side
        self.hidden_dim = model.hidden_dim
        self.self_attn = nn.ModuleList(
            [_SelfAttentionGraph(l.self_attn, self.num_tokens) for l in mem_attn.layers]
        )
        self.cross_attn = nn.ModuleList(
            [
                _CrossAttentionGraph(l.cross_attn_image, layout, self.num_tokens)
                for l in mem_attn.layers
            ]
        )

    def forward(
        self,
        curr: torch.Tensor,
        curr_pos: torch.Tensor,
        memory: torch.Tensor,
        memory_pos: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        # [B, C, H, W] -> [B, HW, C], the layout MemoryAttention works in.
        out = (curr + 0.1 * curr_pos).flatten(2).transpose(1, 2)
        # Upstream recomputes `memory + pos` inside every layer; it is the same
        # value each time, so hoist it.
        keys = memory + memory_pos
        for idx, layer in enumerate(self.mem_attn.layers):
            normed = layer.norm1(out)
            out = out + self.self_attn[idx](normed, normed, normed)
            normed = layer.norm2(out)
            out = out + self.cross_attn[idx](normed, keys, memory, memory_mask)
            normed = layer.norm3(out)
            out = out + layer.linear2(layer.activation(layer.linear1(normed)))
        out = self.mem_attn.norm(out)
        # [B, HW, C] -> [B, C, H, W], matching the caller's reshape.
        return out.transpose(1, 2).reshape(
            -1, self.hidden_dim, self.feat_side, self.feat_side
        )


# --------------------------------------------------------------------------
# Memory encoder + spatial Perceiver
# --------------------------------------------------------------------------


class MemoryEncoderGraph(nn.Module):
    """`MemoryEncoder` followed by the 2D spatial Perceiver, as one graph.

    Inputs:
        pix_feat     [B, C, H, W]        raw top-level vision features
        mask_for_mem [B, 1, img, img]    sigmoid-scaled mask, exactly what
                                         `_encode_new_memory` feeds the encoder
    Outputs:
        maskmem      [B, tokens_per_slot, mem_dim]
        maskmem_pos  [B, tokens_per_slot, mem_dim]

    The two run back to back on every tracked frame and nothing between them is
    reusable, so fusing removes an intermediate [B, mem_dim, H, W] round trip.
    """

    def __init__(self, model) -> None:
        super().__init__()
        if model.no_obj_embed_spatial is not None:
            raise ValueError(
                "no_obj_embed_spatial is applied between the encoder and the "
                "Perceiver and is not folded into this graph."
            )
        self.memory_encoder = model.memory_encoder
        self.spatial_perceiver = model.spatial_perceiver

    def forward(
        self, pix_feat: torch.Tensor, mask_for_mem: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.memory_encoder(pix_feat, mask_for_mem, skip_mask_sigmoid=True)
        feats = out["vision_features"]
        pos = out["vision_pos_enc"][0]
        return self.spatial_perceiver(feats, pos)


# --------------------------------------------------------------------------
# SAM head (propagation path)
# --------------------------------------------------------------------------


@contextlib.contextmanager
def _repeat_interleave_as_expand():
    """Make `repeat_interleave` along a size-1 axis export as `Expand`.

    `MaskDecoder.predict_masks` broadcasts the positional encoding across the
    batch with `torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)`.
    With a dynamic batch, torch exports that as OneHot -> Tile, and TensorRT
    refuses it:

        an IIOneHotLayer cannot be used to compute a shape tensor
        In node ... and operator: Tile (parseNode): INVALID_NODE

    Repeating a size-1 axis n times *is* expanding it, so swapping in `expand`
    changes no arithmetic while exporting to a plain Expand node that TensorRT
    parses. The substitution is scoped to this call and only fires when the
    axis really is size 1; anything else falls through to the original.

    Reading the shape makes the tracer warn that it is baking in a constant.
    That is the intent: whether the axis is 1 is a property of the module, not
    of the input -- `predict_masks` asserts `image_pe.size(0) == 1` two lines
    above the call -- so freezing that decision is correct. The batch *count*
    stays dynamic; it flows into Expand's shape input.
    """
    original = torch.repeat_interleave

    def patched(input, repeats, dim=None, **kwargs):
        if (
            dim is not None
            and isinstance(input, torch.Tensor)
            and input.shape[dim] == 1
        ):
            sizes = [-1] * input.dim()
            sizes[dim] = repeats
            return input.expand(sizes)
        return original(input, repeats, dim=dim, **kwargs)

    torch.repeat_interleave = patched
    try:
        yield
    finally:
        torch.repeat_interleave = original


class SamHeadGraph(nn.Module):
    """`SAM2Base._forward_sam_heads` specialised to propagation.

    During `propagate_in_video` every tracked frame calls the head with
    `point_inputs=None`, `mask_inputs=None` and (for EdgeTAM's config)
    `multimask_output=True`. That makes the whole prompt encoder a constant:
    two `not_a_point` sparse tokens and the `no_mask` dense embedding, identical
    for every batch row. We evaluate it once here and carry the result as a
    buffer, so the engine contains only the mask decoder.

    Prompted frames (the ones that carry the user's box/clicks) do *not* match
    these assumptions; the tracker keeps sending those through PyTorch.

    Inputs:
        pix_feat     [B, C, H, W]        memory-conditioned features
        high_res_0   [B, C/8, 4H, 4W]    conv_s0-projected FPN level 0
        high_res_1   [B, C/4, 2H, 2W]    conv_s1-projected FPN level 1
    Outputs:
        low_res_multimasks   [B, M, 4H, 4W]
        ious                 [B, M]
        low_res_masks        [B, 1, 4H, 4W]
        high_res_masks       [B, 1, img, img]
        obj_ptr              [B, C]
        object_score_logits  [B, 1]
        obj_ptr_all          [B, M, C]        (only with `emit_all_pointers`)

    `obj_ptr_all` exists for motion-aware selection (`src/trackers/samurai.py`).
    The engine picks its mask by `argmax(ious)` internally, so a caller that
    wants a *different* candidate needs that candidate's object pointer too --
    the pointer written to the memory bank has to describe the mask that was
    kept. Emitting all of them costs one extra pass of a 256-wide MLP over
    three tokens and keeps the choice on the Python side, where the motion
    model already lives.
    """

    def __init__(self, model, emit_all_pointers: bool = False) -> None:
        super().__init__()
        self.emit_all_pointers = bool(emit_all_pointers)
        if not model.pred_obj_scores:
            raise ValueError("Expected pred_obj_scores=True.")
        if model.soft_no_obj_ptr:
            raise ValueError("soft_no_obj_ptr=True is not reproduced here.")
        if not model.fixed_no_obj_ptr:
            raise ValueError("Expected fixed_no_obj_ptr=True.")
        if not model.use_high_res_features_in_sam:
            raise ValueError("Expected use_high_res_features_in_sam=True.")
        if not model._use_multimask(is_init_cond_frame=False, point_inputs=None):
            raise ValueError(
                "This config does not use multimask output while tracking; the "
                "specialised head would not match the PyTorch path."
            )

        self.mask_decoder = model.sam_mask_decoder
        self.obj_ptr_proj = model.obj_ptr_proj
        self.image_size = model.image_size
        self.no_obj_score = -1024.0  # sam2_base.NO_OBJ_SCORE

        device = next(model.parameters()).device
        prompt_encoder = model.sam_prompt_encoder
        with torch.no_grad():
            # The exact "no prompt" padding `_forward_sam_heads` builds.
            coords = torch.zeros(1, 1, 2, device=device)
            labels = -torch.ones(1, 1, dtype=torch.int32, device=device)
            sparse, dense = prompt_encoder(points=(coords, labels), boxes=None, masks=None)
            image_pe = prompt_encoder.get_dense_pe()
        self.register_buffer("sparse_prompt", sparse.detach(), persistent=False)
        self.register_buffer("dense_prompt", dense.detach(), persistent=False)
        self.register_buffer("image_pe", image_pe.detach(), persistent=False)
        self.register_buffer("no_obj_ptr", model.no_obj_ptr.detach(), persistent=False)

    def forward(
        self,
        pix_feat: torch.Tensor,
        high_res_0: torch.Tensor,
        high_res_1: torch.Tensor,
    ):
        sparse = self.sparse_prompt.expand(pix_feat.shape[0], -1, -1)
        dense = self.dense_prompt.expand(pix_feat.shape[0], -1, -1, -1)

        with _repeat_interleave_as_expand():
            low_res_multimasks, ious, sam_tokens, object_score_logits = self.mask_decoder(
                image_embeddings=pix_feat,
                image_pe=self.image_pe,
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=True,
                repeat_image=False,
                high_res_features=[high_res_0, high_res_1],
            )

        is_obj = object_score_logits > 0
        low_res_multimasks = torch.where(
            is_obj[..., None, None], low_res_multimasks, self.no_obj_score
        )

        # Select the highest-IoU mask. `torch.gather` instead of upstream's
        # `x[arange(B), idx]`: advanced indexing exports to a GatherND with a
        # batch-size-dependent index tensor, gather stays batch-dynamic.
        best = torch.argmax(ious, dim=-1)
        h, w = low_res_multimasks.shape[-2:]
        mask_idx = best.reshape(-1, 1, 1, 1).expand(-1, 1, h, w)
        low_res_masks = torch.gather(low_res_multimasks, 1, mask_idx)

        # Upstream upsamples all M candidates and then selects; selecting first
        # is identical (interpolation is per-channel) and 4x cheaper here.
        high_res_masks = F.interpolate(
            low_res_masks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        token_dim = sam_tokens.shape[-1]
        token_idx = best.reshape(-1, 1, 1).expand(-1, 1, token_dim)
        # `[:, 0]` rather than `.squeeze(1)`: torch cannot prove at trace time
        # that axis 1 is 1, so it exports squeeze as an If whose branches differ
        # in rank ([B, C] vs [B, 1, C]) -- which TensorRT rejects. Indexing is
        # unconditional and gives the same [B, C] upstream produces.
        sam_output_token = torch.gather(sam_tokens, 1, token_idx)[:, 0]

        obj_ptr = self.obj_ptr_proj(sam_output_token)
        appearing = is_obj.to(obj_ptr.dtype)
        obj_ptr = appearing * obj_ptr + (1.0 - appearing) * self.no_obj_ptr

        outputs = (
            low_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )
        if not self.emit_all_pointers:
            return outputs
        # Every candidate's pointer, blended the same way, so a caller that
        # overrides the mask choice can take the matching one. `obj_ptr` above
        # is row `argmax(ious)` of this, by construction.
        all_ptrs = self.obj_ptr_proj(sam_tokens)
        appearing = is_obj.to(all_ptrs.dtype)[..., None]
        return outputs + (appearing * all_ptrs + (1.0 - appearing) * self.no_obj_ptr,)
