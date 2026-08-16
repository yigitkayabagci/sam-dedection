"""Size-adaptive inference: spend the latency budget where the pixels are.

The Orin runs a 512x512 frame in about 10 ms, and roughly 20-30 ms is
affordable. This is what that margin can buy for targets that are a handful of
pixels -- which, on Anti-UAV410, most of them are.

**Not SAHI as published.** Slicing Aided Hyper Inference tiles a frame and runs
a *detector* on each tile. This pipeline has no detector: it has one prompted
target and a memory bank. The equivalent for a tracker is a **window that
follows the target and sizes itself to it** -- crop a region a fixed multiple
of the target's size, feed that to the model at 512, and a 10-pixel drone
arrives as a 100-pixel one. Nothing about the model changes; it simply stops
being asked to segment something smaller than its own stride.

Tiling does still earn a place, but only where a detector-shaped technique
belongs: **re-acquisition**. After the target has been lost for a while there
is nothing to follow, and sweeping tiles to find it again is affordable at, say,
one frame in thirty in a way that it never is every frame.

### The hazard, stated first

SAM 2's memory bank stores features in *input* coordinates. Move the window and
every stored memory is in a different frame of reference from the features
reading it. This module's answer is that **the window is a segment boundary,
not a per-frame adjustment**:

* it moves as rarely as possible (a size ladder, a position grid, and a margin
  before either is touched), and
* when it does move, the caller is told, and is expected to re-anchor -- start a
  fresh memory bank at the new window, re-prompted from the last mask.

That makes the coordinate change explicit and correct rather than silent and
approximately wrong. It is also why everything here is pure geometry: the
decision is testable on its own, and the tracker keeps its existing interface.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AdaptiveConfig:
    """When to move the window, and how big to make it.

    `target_fraction`  how much of the window the target should occupy, by its
                       longer side. 0.08 puts a 10 px drone at ~40 px once the
                       125 px window is resized to 512 -- enough for a
                       stride-16 backbone to have something to work with.
    `scale_step`       the window sizes form a geometric ladder with this
                       ratio, and the size only changes when the measured
                       fraction leaves one full rung of tolerance. That is the
                       hysteresis: without it, a target drifting across a
                       threshold would resize the window every other frame and
                       invalidate the memory bank each time.
    `margin`           how close to an edge the target may come before the
                       window re-centres, as a fraction of the window side.
    `grid`             window origins snap to this, so sub-grid motion never
                       moves anything.
    """

    enabled: bool = True
    size: int = 512
    target_fraction: float = 0.08
    min_window: int = 128
    max_window: int | None = None
    scale_step: float = 1.5
    margin: float = 0.2
    grid: int = 32


def box_side(box: np.ndarray) -> float:
    """The longer side of an xyxy box, or NaN if there is no box."""
    box = np.asarray(box, dtype=np.float64)
    if np.isnan(box).any():
        return float("nan")
    return float(max(box[2] - box[0], box[3] - box[1]))


def ladder_side(desired: float, cfg: AdaptiveConfig, frame_size: tuple[int, int]) -> int:
    """The smallest ladder rung at or above `desired`, clamped to the frame.

    Discrete sizes rather than a continuous one, so that a window which is
    already about right is left exactly alone.
    """
    ceiling = cfg.max_window or min(frame_size)
    ceiling = min(ceiling, min(frame_size))
    floor = min(cfg.min_window, ceiling)
    if not math.isfinite(desired) or desired <= floor:
        return int(floor)
    steps = math.ceil(math.log(desired / floor) / math.log(cfg.scale_step))
    return int(min(floor * cfg.scale_step ** steps, ceiling))


class AdaptiveWindow:
    """The crop the model sees, and the decision to move it.

    Coordinates are the source frame's throughout; `rect` is `(x0, y0, side)`.
    """

    def __init__(self, frame_size: tuple[int, int], config: AdaptiveConfig | None = None):
        self.frame_size = frame_size
        self.config = config or AdaptiveConfig()
        self.side = min(self.config.size, min(frame_size))
        self.origin = (0, 0)
        self.moves = 0

    # -- geometry ---------------------------------------------------------

    @property
    def rect(self) -> tuple[int, int, int]:
        return (*self.origin, self.side)

    @property
    def scale(self) -> float:
        """How much the crop is magnified on its way to the model input."""
        return self.config.size / self.side

    def to_window(self, box: np.ndarray) -> np.ndarray:
        """Frame coordinates -> model-input coordinates."""
        out = np.asarray(box, dtype=np.float64).copy()
        out[0::2] = (out[0::2] - self.origin[0]) * self.scale
        out[1::2] = (out[1::2] - self.origin[1]) * self.scale
        return out

    def to_frame(self, box: np.ndarray) -> np.ndarray:
        """Model-input coordinates -> frame coordinates."""
        out = np.asarray(box, dtype=np.float64).copy()
        out[0::2] = out[0::2] / self.scale + self.origin[0]
        out[1::2] = out[1::2] / self.scale + self.origin[1]
        return out

    def contains(self, box: np.ndarray) -> bool:
        x0, y0, side = self.rect
        box = np.asarray(box, dtype=np.float64)
        return bool(x0 <= box[0] and box[2] <= x0 + side
                    and y0 <= box[1] and box[3] <= y0 + side)

    # -- the decision -----------------------------------------------------

    def _place(self, box: np.ndarray, side: int) -> tuple[int, int]:
        width, height = self.frame_size
        grid = max(self.config.grid, 1)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        x0 = int(round((cx - side / 2) / grid)) * grid
        y0 = int(round((cy - side / 2) / grid)) * grid
        return (int(np.clip(x0, 0, max(width - side, 0))),
                int(np.clip(y0, 0, max(height - side, 0))))

    def needs_move(self, box: np.ndarray) -> str | None:
        """Why the window should move for this target, or None to leave it.

        Returning the reason rather than a bool is what makes a run
        diagnosable: a window moving every few frames on `size` means
        `scale_step` is too tight, and one moving on `edge` means `margin` is.
        """
        if np.isnan(np.asarray(box, dtype=np.float64)).any():
            return None  # nothing to follow; hold position and wait

        cfg = self.config
        low = cfg.target_fraction / cfg.scale_step
        high = cfg.target_fraction * cfg.scale_step
        fraction = box_side(box) / max(self.side, 1)
        if not low <= fraction <= high:
            wanted = ladder_side(box_side(box) / cfg.target_fraction, cfg, self.frame_size)
            if wanted != self.side:
                return "size"

        edge = cfg.margin * self.side
        x0, y0, side = self.rect
        if (box[0] < x0 + edge or box[2] > x0 + side - edge
                or box[1] < y0 + edge or box[3] > y0 + side - edge):
            return "edge"
        return None

    def move_to(self, box: np.ndarray) -> tuple[int, int, int]:
        """Re-size and re-centre on `box`. Returns the new rect.

        The caller must treat this as a segment boundary: everything in the
        memory bank was encoded in the old window's coordinates.
        """
        cfg = self.config
        self.side = ladder_side(box_side(box) / cfg.target_fraction, cfg, self.frame_size)
        self.origin = self._place(np.asarray(box, dtype=np.float64), self.side)
        self.moves += 1
        return self.rect


# --------------------------------------------------------------------------
# Re-acquisition
# --------------------------------------------------------------------------


def tile_grid(frame_size: tuple[int, int], tile: int,
              overlap: float = 0.2) -> list[tuple[int, int, int]]:
    """SAHI's tiling, for the one job it is right for here: finding the target
    again after it has been lost.

    Overlapping, because a target sitting on a tile boundary is split between
    two tiles and convincing in neither -- which is the failure tiling exists
    to avoid. The last tile on each axis is pulled back inside the frame rather
    than padded, so every tile is a real crop of real pixels.
    """
    width, height = frame_size
    tile = int(min(tile, width, height))
    step = max(int(tile * (1 - overlap)), 1)

    def starts(extent: int) -> list[int]:
        positions = list(range(0, max(extent - tile, 0) + 1, step))
        if positions[-1] != extent - tile and extent > tile:
            positions.append(extent - tile)
        return positions

    return [(x, y, tile) for y in starts(height) for x in starts(width)]


@dataclass
class Reacquisition:
    """When to stop following and start searching.

    A tile sweep costs one model pass per tile, so it is affordable at one
    frame in tens and never every frame. `patience` is what keeps it that way:
    a target lost for two frames is usually just occluded and will come back on
    its own; one lost for thirty is not.
    """

    patience: int = 30
    cooldown: int = 30
    overlap: float = 0.2
    lost: int = 0
    since_sweep: int = 10 ** 9

    def observe(self, present: bool) -> None:
        self.lost = 0 if present else self.lost + 1
        self.since_sweep += 1

    def should_sweep(self) -> bool:
        return self.lost >= self.patience and self.since_sweep >= self.cooldown

    def swept(self) -> None:
        self.since_sweep = 0
        self.lost = 0
