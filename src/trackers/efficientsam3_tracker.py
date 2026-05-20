from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..prompts import PromptSet
from .base import TrackingResult, VideoTracker
from .registry import register


@register("efficientsam3")
class EfficientSAM3Tracker(VideoTracker):
    """Placeholder for EfficientSAM3.

    Upstream (SimonZeng7108/efficientsam3) only ships Stage 1 image weights as
    of this writing — the Perceiver-based memory module needed for video
    tracking is listed as Stage 2 / planned. Fill in `prepare`/`set_prompts`/
    `propagate` once those weights are released; the rest of the pipeline
    (frame I/O, prompts, visualization) is already backend-agnostic.
    """

    def __init__(self, **_: object) -> None:
        pass

    def prepare(self, frames_dir: str | Path) -> None:
        raise NotImplementedError(
            "EfficientSAM3 video tracking is not yet supported upstream. "
            "Use --tracker edgetam for now."
        )

    def set_prompts(self, prompts: PromptSet) -> None:
        raise NotImplementedError

    def propagate(self) -> Iterator[TrackingResult]:
        raise NotImplementedError
