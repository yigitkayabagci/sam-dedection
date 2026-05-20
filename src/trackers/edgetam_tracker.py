from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Iterator

import numpy as np

from ..prompts import PromptSet
from .base import TrackingResult, VideoTracker
from .registry import register


@register("edgetam")
class EdgeTAMTracker(VideoTracker):
    """EdgeTAM (SAM 2 variant) video tracker.

    Loads `build_sam2_video_predictor` from the EdgeTAM package that ships in
    third_party/EdgeTAM. The import is deferred so the rest of the pipeline
    is usable for development / tests without EdgeTAM installed.
    """

    def __init__(
        self,
        model_cfg: str,
        checkpoint: str,
        device: str = "cuda",
        fp16: bool = True,
        mask_threshold: float = 0.0,
    ) -> None:
        self.model_cfg = model_cfg
        self.checkpoint = checkpoint
        self.device = device
        self.fp16 = fp16
        self.mask_threshold = mask_threshold
        self._predictor = None
        self._state = None

    def _ensure_predictor(self) -> None:
        if self._predictor is not None:
            return
        try:
            import torch  # noqa: F401
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as exc:
            raise ImportError(
                "EdgeTAM is not installed. Run scripts/setup_edgetam.sh first."
            ) from exc
        self._predictor = build_sam2_video_predictor(
            self.model_cfg, self.checkpoint, device=self.device
        )

    def _autocast(self):
        if not self.fp16 or self.device == "cpu":
            return nullcontext()
        import torch
        # Orin AGX has good fp16 throughput; bfloat16 also works but fp16 is safer.
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def prepare(self, frames_dir: str | Path) -> None:
        self._ensure_predictor()
        with self._autocast():
            self._state = self._predictor.init_state(video_path=str(frames_dir))

    def set_prompts(self, prompts: PromptSet) -> None:
        if self._state is None:
            raise RuntimeError("Call prepare() before set_prompts().")
        if prompts.is_empty():
            raise ValueError("PromptSet is empty.")

        import numpy as _np

        # Group point prompts per (frame_idx, obj_id) so we can pass them in one call.
        point_groups: dict[tuple[int, int], list] = {}
        for p in prompts.points:
            point_groups.setdefault((p.frame_idx, p.obj_id), []).append(p)

        with self._autocast():
            for box in prompts.boxes:
                self._predictor.add_new_points_or_box(
                    inference_state=self._state,
                    frame_idx=box.frame_idx,
                    obj_id=box.obj_id,
                    box=_np.array(box.xyxy, dtype=_np.float32),
                )
            for (frame_idx, obj_id), group in point_groups.items():
                coords = _np.array([p.xy for p in group], dtype=_np.float32)
                labels = _np.array([p.label for p in group], dtype=_np.int32)
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
        with self._autocast():
            for frame_idx, obj_ids, mask_logits in self._predictor.propagate_in_video(self._state):
                masks: dict[int, np.ndarray] = {}
                # mask_logits: [N, 1, H, W] tensor on device
                logits_np = mask_logits.detach().float().cpu().numpy()
                for i, obj_id in enumerate(obj_ids):
                    masks[int(obj_id)] = logits_np[i, 0] > self.mask_threshold
                yield TrackingResult(frame_idx=int(frame_idx), masks=masks)

    def reset(self) -> None:
        if self._predictor is not None and self._state is not None:
            try:
                self._predictor.reset_state(self._state)
            except Exception:
                pass
        self._state = None
