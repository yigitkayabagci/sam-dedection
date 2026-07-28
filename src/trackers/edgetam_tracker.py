from __future__ import annotations

from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Iterator

import numpy as np

from ..prompts import PromptSet
from .base import TrackingResult, VideoTracker
from .registry import register


# Aliases accepted on the YAML side ("fp16", "bf16", "fp32").
_PRECISION_ALIASES = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "fp32": "float32",
}


@register("edgetam")
class EdgeTAMTracker(VideoTracker):
    """EdgeTAM (SAM 2 variant) video tracker.

    Mirrors the canonical inference pattern from
    third_party/EdgeTAM/tools/vos_inference.py:

        @torch.inference_mode()
        @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        def run(...):
            state = predictor.init_state(...)
            predictor.add_new_points_or_box(...)
            for f, ids, logits in predictor.propagate_in_video(...):
                ...

    Each public method enters both `inference_mode` and `autocast` contexts
    so callers get the same behaviour regardless of call order.
    """

    def __init__(
        self,
        model_cfg: str,
        checkpoint: str,
        device: str = "cuda",
        precision: str = "bfloat16",
        mask_threshold: float = 0.0,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        compile_image_encoder: bool = False,
        compile_mode: str = "default",
        image_size: int | None = None,
        fill_hole_area: int | None = None,
    ) -> None:
        self.model_cfg = model_cfg
        self.checkpoint = checkpoint
        self.device = device
        self.precision = _PRECISION_ALIASES.get(precision, precision)
        if self.precision not in ("bfloat16", "float16", "float32"):
            raise ValueError(
                f"precision must be one of bfloat16|float16|float32 (got {precision})"
            )
        self.mask_threshold = mask_threshold
        self.offload_video_to_cpu = offload_video_to_cpu
        self.offload_state_to_cpu = offload_state_to_cpu
        self.compile_image_encoder = compile_image_encoder
        self.compile_mode = compile_mode
        self.compiled = False
        # Model-side input resolution. Distinct from the source video's
        # resolution: every frame is squashed to image_size x image_size before
        # the encoder, so this drives inference cost while the source resolution
        # drives preprocess and postprocess cost.
        self.image_size = image_size
        self.fill_hole_area = fill_hole_area
        self._predictor = None
        self._state = None

    def _hydra_overrides(self) -> list[str]:
        """Config overrides passed to build_sam2_video_predictor."""
        overrides: list[str] = []
        if self.image_size is not None:
            overrides.append(f"++model.image_size={self.image_size}")
            # image_size alone is not enough. EdgeTAM's RoPEAttentionv2
            # precomputes its rotary frequency table from the config's
            # q_sizes, which is the stride-16 feature grid (64x64 for the
            # stock 1024). Unlike SAM2's v1 attention it does NOT recompute
            # on a size mismatch, so a different image_size reaches it with
            # the wrong number of tokens and dies on a bare assert. The
            # memory-side k_sizes stays put: those keys are the Perceiver's
            # fixed 256 (16x16) latents, which do not scale with input size.
            grid = self.image_size // 16
            overrides.append(
                "++model.memory_attention.layer.cross_attention.q_sizes="
                f"[{grid},{grid}]"
            )
            print(f"[edgetam] image_size={self.image_size} -> RoPE q_sizes "
                  f"[{grid},{grid}]. Note the rotary embeddings were trained at "
                  "64x64, so expect accuracy loss beyond what the lower "
                  "resolution alone costs — measure IoU before trusting it.")

        if self.fill_hole_area is not None:
            overrides.append(f"++model.fill_hole_area={self.fill_hole_area}")
        else:
            # build_sam2_video_predictor's apply_postprocessing default sets
            # fill_hole_area=8, and filling holes needs the compiled sam2._C
            # extension. When that extension was not built the code path is
            # still entered on EVERY frame, throws ImportError, and is caught
            # and skipped -- paying the cost of failing without ever doing the
            # work. Turning it off is strictly cheaper for identical output.
            try:
                from sam2 import _C  # noqa: F401
            except ImportError:
                overrides.append("++model.fill_hole_area=0")
                print("[edgetam] sam2._C is not built; disabling fill_hole_area "
                      "(the hole filling was being skipped every frame anyway).")
        return overrides

    def _ensure_predictor(self) -> None:
        if self._predictor is not None:
            return
        try:
            import torch
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as exc:
            raise ImportError(
                "EdgeTAM is not installed. Run scripts/setup_edgetam.sh first."
            ) from exc
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "device=cuda requested but torch reports cuda.is_available()=False. "
                "On Jetson Orin AGX install the JetPack-matched torch wheel "
                "(see scripts/setup_edgetam.sh comments). For dev machines without "
                "CUDA, pass --config configs/edgetam_cpu.yaml."
            )
        self._predictor = build_sam2_video_predictor(
            self.model_cfg, self.checkpoint, device=self.device,
            hydra_overrides_extra=self._hydra_overrides(),
        )
        if self.compile_image_encoder:
            # A TensorRT-free speed knob for the same block TRT targets, so the
            # two are directly comparable. Note that torch.compile is lazy: if
            # inductor/triton is broken on this aarch64 build the error surfaces
            # on the first forward, not here. That is on purpose — a benchmark
            # variant should fail loudly rather than quietly run eager.
            try:
                self._predictor.image_encoder = torch.compile(
                    self._predictor.image_encoder, mode=self.compile_mode
                )
                self.compiled = True
            except Exception as exc:
                print(f"[edgetam] torch.compile setup failed ({exc}); running eager.")

    def _inference_ctx(self):
        """Enter inference_mode + autocast together, mirroring vos_inference.py."""
        import torch

        stack = ExitStack()
        stack.enter_context(torch.inference_mode())
        if self.precision == "float32" or self.device == "cpu":
            stack.enter_context(nullcontext())
        else:
            dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[self.precision]
            stack.enter_context(torch.autocast(device_type=self.device, dtype=dtype))
        return stack

    def prepare(self, frames_dir: str | Path) -> None:
        self._ensure_predictor()
        with self._inference_ctx():
            self._state = self._predictor.init_state(
                video_path=str(frames_dir),
                offload_video_to_cpu=self.offload_video_to_cpu,
                offload_state_to_cpu=self.offload_state_to_cpu,
            )

    def set_prompts(self, prompts: PromptSet) -> None:
        if self._state is None:
            raise RuntimeError("Call prepare() before set_prompts().")
        if prompts.is_empty():
            raise ValueError("PromptSet is empty.")

        # Group point prompts per (frame_idx, obj_id); EdgeTAM's
        # add_new_points_or_box clears prior points on each call, so all
        # points for a (frame, obj) must be sent in one shot.
        point_groups: dict[tuple[int, int], list] = {}
        for p in prompts.points:
            point_groups.setdefault((p.frame_idx, p.obj_id), []).append(p)

        with self._inference_ctx():
            for box in prompts.boxes:
                self._predictor.add_new_points_or_box(
                    inference_state=self._state,
                    frame_idx=box.frame_idx,
                    obj_id=box.obj_id,
                    box=np.array(box.xyxy, dtype=np.float32),
                )
            for (frame_idx, obj_id), group in point_groups.items():
                coords = np.array([p.xy for p in group], dtype=np.float32)
                labels = np.array([p.label for p in group], dtype=np.int32)
                self._predictor.add_new_points_or_box(
                    inference_state=self._state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=coords,
                    labels=labels,
                )

    def propagate(self) -> Iterator[TrackingResult]:
        if self._state is None:
            raise RuntimeError("Call prepare() and set_prompts() before propagate().")
        with self._inference_ctx():
            for frame_idx, obj_ids, mask_logits in self._predictor.propagate_in_video(self._state):
                # mask_logits: [N, 1, H, W] at *video* resolution, on the model's
                # device. Threshold on the GPU and copy bytes rather than floats —
                # the comparison works directly on bf16/fp16 logits, and it makes
                # the per-frame device->host transfer 4x smaller.
                binary = mask_logits.detach() > self.mask_threshold
                masks_np = binary[:, 0].cpu().numpy()
                masks: dict[int, np.ndarray] = {
                    int(obj_id): masks_np[i] for i, obj_id in enumerate(obj_ids)
                }
                yield TrackingResult(frame_idx=int(frame_idx), masks=masks)

    def reset(self) -> None:
        if self._predictor is not None and self._state is not None:
            try:
                self._predictor.reset_state(self._state)
            except Exception:
                pass
        self._state = None
