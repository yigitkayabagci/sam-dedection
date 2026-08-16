from __future__ import annotations

from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Iterator

import numpy as np

from ..prompts import PromptSet
from ._hydra_overrides import image_size_overrides
from .base import TrackingResult, VideoTracker
from .registry import register


# Aliases accepted on the YAML side ("fp16", "bf16", "fp32").
_PRECISION_ALIASES = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "fp32": "float32",
}


def _block_config(block: dict | None, kind: str):
    """Turn a config block into its dataclass, or None when it is switched off.

    Validated here rather than at first use, so a typo in a YAML key is a
    startup error naming the key instead of a silently ignored setting that
    makes the run look like it did nothing.
    """
    if not block:
        return None
    if kind == "samurai":
        from .samurai import SamuraiConfig as Config
    else:
        from .sam2long import Sam2LongConfig as Config

    unknown = set(block) - set(Config.__dataclass_fields__)
    if unknown:
        raise ValueError(
            f"unknown {kind} option(s) {sorted(unknown)}; valid keys are "
            f"{sorted(Config.__dataclass_fields__)}"
        )
    config = Config(**block)
    return config if config.enabled else None


def _samurai_config(samurai: dict | None):
    return _block_config(samurai, "samurai")


def _replicate_prompts(prompts: PromptSet, pathways: int) -> PromptSet:
    """Every prompt repeated once per SAM2Long pathway, under its own object id."""
    from dataclasses import replace

    from .sam2long import expand_ids

    return PromptSet(
        boxes=[replace(b, obj_id=new_id)
               for b in prompts.boxes for new_id in expand_ids(b.obj_id, pathways)],
        points=[replace(p, obj_id=new_id)
                for p in prompts.points for new_id in expand_ids(p.obj_id, pathways)],
    )


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
        image_size: int | None = None,
        samurai: dict | None = None,
        sam2long: dict | None = None,
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
        # Overrides the Hydra config's own `model.image_size` (1024). The
        # checkpoint's backbone (RepViT) and neck are fully convolutional and
        # its positional encodings are computed per-frame from the actual
        # feature-map size, not a fixed learned table, so a different input
        # resolution loads and runs -- verified against the real checkpoint at
        # 512 (proportional FPN shapes, no shape errors), once
        # `_hydra_overrides.image_size_overrides` also fixes up the memory
        # attention's cross-attention RoPE table (see that module -- it does
        # not self-heal like the self-attention one does). It is still the
        # same weights evaluated off their training resolution, so accuracy at
        # a new size is a measurement, not a given: compare_backends.py
        # answers "did TensorRT change the masks", not "does 512 track as well
        # as 1024" -- that second question needs ground truth, not a backend
        # A/B.
        self.image_size = image_size
        # Motion-aware memory (`samurai.py`): a training-free inference wrapper,
        # off unless the config asks for it. None and `{enabled: false}` both
        # leave the model exactly as upstream wrote it.
        self.samurai_config = _samurai_config(samurai)
        # SAM2Long: N competing memory pathways per object. Experimental, and
        # unlike SAMURAI it costs real time -- the memory path runs once per
        # pathway. Off unless asked for. The two compose: SAMURAI chooses
        # within a pathway, SAM2Long chooses between them.
        self.sam2long_config = _block_config(sam2long, "sam2long")
        self._samurai = None
        self._sam2long = None
        self._predictor = None
        self._state = None

    def _install_samurai(self) -> None:
        if self.samurai_config is not None and self._samurai is None:
            from .samurai import install

            self._samurai = install(self._predictor, self.samurai_config)
        if self.sam2long_config is not None and self._sam2long is None:
            from .sam2long import install as install_sam2long

            self._sam2long = install_sam2long(self._predictor, self.sam2long_config)

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
        overrides = image_size_overrides(self.image_size)
        self._predictor = build_sam2_video_predictor(
            self.model_cfg, self.checkpoint, device=self.device,
            hydra_overrides_extra=overrides,
        )

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
        self._install_samurai()
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

        if self._sam2long is not None:
            # One object becomes N identically-prompted ones. SAM 2 already
            # gives every batch row its own memory bank, object pointer and
            # object score, so N pathways ride the machinery N objects would.
            prompts = _replicate_prompts(prompts, self.sam2long_config.pathways)

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

        if self._sam2long is not None:
            # Read the batch order off the state rather than assuming it
            # follows insertion: the mask decoder sees rows, everything else
            # here is keyed by object id, and the two must not drift apart.
            self._sam2long.set_rows(self._state["obj_ids"])

    def propagate(self) -> Iterator[TrackingResult]:
        if self._state is None:
            raise RuntimeError("Call prepare() and set_prompts() before propagate().")
        with self._inference_ctx():
            for frame_idx, obj_ids, mask_logits in self._predictor.propagate_in_video(self._state):
                # mask_logits: [N, 1, H, W] on the model's device.
                # .float() is important when autocast returned bf16/fp16 logits.
                logits_np = mask_logits.detach().float().cpu().numpy()
                if self._sam2long is not None:
                    masks = {
                        user: logits_np[row, 0] > self.mask_threshold
                        for user, row in self._sam2long.winners().items()
                    }
                else:
                    masks = {
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
        # The motion model, the per-frame record and the pathway scores all
        # belong to one video.
        if self._samurai is not None:
            self._samurai.reset()
        if self._sam2long is not None:
            self._sam2long.reset()
        self._state = None
