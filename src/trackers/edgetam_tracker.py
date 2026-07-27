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
        self._predictor = None
        self._state = None

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
            self.model_cfg, self.checkpoint, device=self.device
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
                # mask_logits: [N, 1, H, W] on the model's device.
                # .float() is important when autocast returned bf16/fp16 logits.
                logits_np = mask_logits.detach().float().cpu().numpy()
                masks: dict[int, np.ndarray] = {
                    int(obj_id): logits_np[i, 0] > self.mask_threshold
                    for i, obj_id in enumerate(obj_ids)
                }
                yield TrackingResult(frame_idx=int(frame_idx), masks=masks)

    def reset(self) -> None:
        if self._predictor is not None and self._state is not None:
            try:
                self._predictor.reset_state(self._state)
            except Exception:
                pass
        self._state = None
