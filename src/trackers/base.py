from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ..prompts import PromptSet


@dataclass(frozen=True)
class TrackingResult:
    frame_idx: int
    masks: dict[int, np.ndarray]  # obj_id -> HxW bool mask


class VideoTracker(ABC):
    """Backend-agnostic interface every SAM-style video tracker must implement.

    Lifecycle:
        tracker.prepare(frames_dir)
        tracker.set_prompts(prompts)
        for result in tracker.propagate(): ...
        tracker.reset()
    """

    name: str = "base"

    @abstractmethod
    def prepare(self, frames_dir: str | Path) -> None:
        """Initialise internal state from a directory of frames (00000.jpg, ...)."""

    @abstractmethod
    def set_prompts(self, prompts: PromptSet) -> None:
        """Register prompts (boxes and/or points) on their declared frames."""

    @abstractmethod
    def propagate(self) -> Iterator[TrackingResult]:
        """Yield per-frame masks in chronological order."""

    def reset(self) -> None:
        """Optional hook to clear cached state between videos."""
