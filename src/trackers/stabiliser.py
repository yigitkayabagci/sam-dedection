"""Classical guards around a learned tracker: motion, plausibility, re-acquisition.

`samurai.py` fixes what the *model* remembers. This fixes what the *system*
accepts, and it shares none of its machinery: no network, no gradients, no
weights, nothing but optical flow, thresholds and template matching. That is
the point. A learned tracker fails in ways it cannot notice -- the mask spreads
over a field, the box jumps to a distractor, the target is lost the frame after
the camera slid -- and none of those need a network to *detect*. They need a
motion model, three ratios and a state machine, which is what forty years of
tracking literature says and what this module is.

The three failures it was written against, all observed on one recording:

**The camera slides with the target.** A drone yaws a degree and the whole
scene translates; a filter that works in image coordinates reads that as the
target accelerating, and then cannot tell a real jump from the ego-motion it
never modelled. `estimate_shift` measures the background's own displacement --
sparse Lucas-Kanade on a grid, the target's neighbourhood excluded, robust
median over what survives -- and every prediction below is made in the frame
the background sits still in.

**The box balloons.** Contrast drops, the mask leaks into terrain that looks
like the target, and the box grows to cover a field. Nothing in a promptable
segmenter refuses this: it was asked for a mask and it gave one.
`PlausibilityGate` refuses it, on the three numbers a target's geometry cannot
break between two frames at 30 Hz -- its area, its aspect and how far its
centre may travel once the camera's own motion is taken out.

**It jumps, and never comes back.** Once the tracker locks onto the wrong
thing, its own memory confirms it. The answer is the oldest one there is: keep
the last good appearance, and when the track is declared lost, look for it
again -- normalised cross-correlation over a search region that grows every
frame the target stays missing and that travels with the camera.

What this module deliberately does *not* do is decide anything about pixels the
model is better at. It never edits a mask, never moves a box it accepted, and
never invents a target: a rejected frame yields the *predicted* box and a state
of `suspect`, which a caller is free to ignore. Everything it returns says
which of the four states the track is in and why.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

# How much a candidate shift has to lower the mean absolute frame difference
# before it is believed, in 8-bit levels. Small because it is a floor against
# noise rather than a confidence threshold -- the separation between a right
# shift and an invented one on real footage is a whole level, not a hundredth.
MIN_ALIGNMENT_GAIN = 0.05

# States, in the order a track passes through them. `suspect` is not a failure
# yet: it is one frame the gates refused, and a single refused frame is far more
# often a motion-blurred one than a lost target -- which is what the hysteresis
# in `GuardConfig.suspect_frames` exists to say.
TRACKING = "tracking"
SUSPECT = "suspect"
LOST = "lost"
REACQUIRED = "reacquired"
STATES = (TRACKING, SUSPECT, LOST, REACQUIRED)


@dataclass(frozen=True)
class GuardConfig:
    """Every threshold, with what it guards and why it has the value it has.

    The ratios are deliberately loose. A gate that fires on a real target's
    honest change is worse than no gate at all: it turns a tracker that is
    occasionally wrong into one that is confidently stuck on its own
    prediction. These refuse the *impossible*, not the unusual.
    """

    # Geometry. A target's area between two frames at video rate cannot triple
    # or quarter, whatever the sensor did -- 30 Hz and a drone's approach speed
    # put the honest bound near 10 % a frame.
    max_area_ratio: float = 3.0
    min_area_ratio: float = 0.3
    # Aspect is the balloon's other signature: a mask leaking along a road
    # keeps its area for a frame or two while its shape stops being a vehicle.
    max_aspect_ratio: float = 2.5
    # How far the centre may travel, in units of the target's own size, once
    # the background's motion has been removed. A car two body-lengths away
    # from where it was is a different car.
    max_jump: float = 2.5
    # Frames a target may cover of the whole frame before the box is called a
    # balloon regardless of the ratios -- the case where the very first frames
    # are already wrong and the running median is poisoned from the start.
    max_frame_share: float = 0.25

    # Hysteresis. One refused frame is blur; three in a row is a lost target.
    suspect_frames: int = 3
    recover_frames: int = 2
    # How many frames of history the medians are taken over. Long enough to be
    # robust, short enough to follow a target that really is approaching.
    history: int = 12

    # Re-acquisition. The search region starts at twice the target's size,
    # grows while the target stays missing, and stops growing at `search_max`
    # so a long loss does not turn into a whole-frame search for a 20-pixel car.
    search_scale: float = 2.0
    search_growth: float = 1.4
    search_max: float = 8.0
    match_threshold: float = 0.45
    # Frames after which a lost track stops being re-acquired at all. The
    # appearance kept from before the loss is stale by then, and matching it is
    # how a tracker locks onto a bush that happens to correlate.
    give_up_after: int = 90

    # Contrast. Below this signal-to-clutter ratio -- the same number
    # `image_loop.instance_contrast` reports, in numpy here -- the target is
    # inside the clutter and the tracker is in the regime it is known to fail
    # in, so the gates tighten by `caution_factor` for that frame.
    #
    # **The gates tighten; the pixels do not change.** Contrast enhancement was
    # measured before it was rejected: CLAHE multiplies the target's signal and
    # the ground's clutter by the same factor and leaves the ratio where it was
    # (2.61 -> 2.50 on this repo's synthetic low-contrast case), and a top-hat
    # only raises it when the clutter sits at a scale different from the
    # target's (2.61 -> 2.58 when it does not). Neither of them changes what
    # the model has to work with, and both would put a histogram in front of it
    # that no training window had. The contrast answer that does work is on the
    # training side -- `src/training/photometric.py`.
    #
    # 1.0 and not 2.0, which was the first guess: on 4 421 real annotated
    # targets (HIT-UAV, 60-130 m) the median signal-to-clutter ratio is 0.91,
    # so a threshold of 2 would have tightened the gates on roughly seven
    # frames in ten and stopped being a caution at all. At 1.0 it fires on the
    # harder half, which is what "the regime this fails in" should mean.
    caution_below: float = 1.0
    caution_factor: float = 0.6

    # Re-acquisition matches a band-passed frame rather than the raw one, cut
    # to the target's own scale -- see `bandpass`, which measured what the raw
    # matcher actually does on a faint target. 0 turns it off and matches the
    # pixels as they are.
    match_bandpass: float = 0.55


@dataclass(frozen=True)
class Decision:
    """What the guard made of one frame."""

    box: np.ndarray | None          # xyxy, or None when nothing is claimed
    state: str
    reason: str = ""
    shift: tuple[float, float] = (0.0, 0.0)
    match: float = float("nan")     # the re-acquisition score, when one ran

    @property
    def accepted(self) -> bool:
        """Whether this box came back believed.

        True for the tracker's own box (`tracking`) and for one the guard found
        again by appearance (`reacquired`); false while the guard is only
        holding a prediction (`suspect`, `lost`), which is the caller's cue
        that the position is dead reckoning and not a measurement.
        """
        return self.state in (TRACKING, REACQUIRED)


# --------------------------------------------------------------------------
# Ego-motion
# --------------------------------------------------------------------------


def estimate_shift(previous: np.ndarray, current: np.ndarray,
                   exclude: np.ndarray | None = None,
                   points: int = 12, margin: float = 1.5) -> tuple[float, float]:
    """How far the *background* moved between two greyscale frames, in pixels.

    Sparse Lucas-Kanade on a regular grid rather than dense flow or a full
    homography: the whole point is to be cheap enough to run every frame beside
    a network, and a translation is what a drone's small yaw and pitch actually
    produce over a few tens of milliseconds. Rotation and scale change too
    slowly at video rate to be worth the conditioning problems of estimating
    them from a few dozen points.

    `exclude` is the target's box, and excluding it is not a refinement: the
    target is the one thing in the frame that is *not* background, and a grid
    that samples it measures the target's motion plus the camera's and calls
    the sum ego-motion -- which would let a real jump pass as camera shake.

    The median over surviving points, not the mean: half the grid usually lands
    on sky, water or a featureless field where flow is undefined, and one such
    point at the wrong scale moves a mean by more than the shift being
    measured. Falls back to phase correlation when too few points track, and to
    `(0, 0)` -- the honest "no estimate" -- when that fails too.
    """
    import cv2

    previous = np.ascontiguousarray(previous)
    current = np.ascontiguousarray(current)
    if previous.shape != current.shape or previous.ndim != 2:
        raise ValueError("estimate_shift takes two greyscale frames of one size")

    height, width = previous.shape
    ys = np.linspace(height * 0.1, height * 0.9, points)
    xs = np.linspace(width * 0.1, width * 0.9, points)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2).astype(np.float32)

    if exclude is not None:
        x0, y0, x1, y1 = _padded(exclude, margin, width, height)
        outside = ~((grid[:, 0] >= x0) & (grid[:, 0] <= x1)
                    & (grid[:, 1] >= y0) & (grid[:, 1] <= y1))
        grid = grid[outside]
    if len(grid) < 4:
        return _phase_shift(previous, current)

    moved, status, error = cv2.calcOpticalFlowPyrLK(
        previous, current, grid.reshape(-1, 1, 2), None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if moved is None or status is None:
        return _phase_shift(previous, current)

    keep = status.reshape(-1).astype(bool)
    if error is not None:
        # Points whose patch never matched well are noise wearing a flow
        # vector's clothes; the median survives a few, not a third of them.
        keep &= error.reshape(-1) < 20.0
    if keep.sum() < 4:
        return _phase_shift(previous, current)

    flow = moved.reshape(-1, 2)[keep] - grid[keep]
    shift = np.median(flow, axis=0)
    # A translation is a claim that the points *agree*. On a featureless frame
    # -- sky, water, a uniform field -- Lucas-Kanade still returns a vector per
    # point and a status of 1 for each, and the median of that is a confident
    # number with nothing behind it. The spread is what separates the two: real
    # ego-motion moves every background point the same way.
    spread = float(np.median(np.abs(flow - shift).sum(axis=1)))
    candidate = ((float(shift[0]), float(shift[1]))
                 if spread <= 2.0 + 0.25 * float(np.abs(shift).sum())
                 else _phase_shift(previous, current))
    return _supported(previous, current, candidate)


def _supported(previous: np.ndarray, current: np.ndarray,
               shift: tuple[float, float]) -> tuple[float, float]:
    """`shift` if undoing it actually aligns the two frames, else no motion.

    The check that turned out to matter, and it took real footage to find:
    on HIT-UAV's own thermal drone videos, `cv2.phaseCorrelate` -- the natural
    fallback and the one this used unguarded -- returns a confident **zero** on
    frames where the camera really moved thirteen pixels. Across 1 978 frame
    pairs the flow estimate improved the alignment on 67 % of them, and the
    fallback was reached on 6.3 %; a false zero there is worse than no estimate,
    because a Kalman filter told the camera was still treats that as a
    measurement and the target's apparent motion becomes the target's.

    So nothing is returned on trust. Warping the second frame back by the
    candidate and comparing it with the first is one warp and one subtraction
    on a half-size copy -- a few tenths of a millisecond -- and it is decisive:
    a shift that is right lowers the residual, a shift that is invented raises
    it, and a shift of zero is what "no measurement" already means to every
    consumer here.
    """
    import cv2

    dx, dy = shift
    if abs(dx) < 0.25 and abs(dy) < 0.25:
        return 0.0, 0.0
    matrix = np.float32([[1, 0, -dx], [0, 1, -dy]])
    undone = cv2.warpAffine(current, matrix, (current.shape[1], current.shape[0]),
                            borderMode=cv2.BORDER_REPLICATE)
    edge = max(int(0.1 * min(previous.shape[:2])), 4)
    view = (slice(edge, -edge), slice(edge, -edge))
    base = previous[view].astype(np.float32)
    with_shift = float(np.abs(base - undone[view].astype(np.float32)).mean())
    without = float(np.abs(base - current[view].astype(np.float32)).mean())
    return (dx, dy) if with_shift < without - MIN_ALIGNMENT_GAIN else (0.0, 0.0)


def _phase_shift(previous: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Whole-frame translation by phase correlation -- the fallback."""
    import cv2

    if float(previous.std()) < 1.0 or float(current.std()) < 1.0:
        # No texture, no measurement. Zero is the honest answer here, and the
        # one that leaves the caller's prediction where it was rather than
        # translating it by a number the pixels do not support.
        return 0.0, 0.0
    try:
        (dx, dy), response = cv2.phaseCorrelate(previous.astype(np.float32),
                                                current.astype(np.float32))
    except cv2.error:                                    # pragma: no cover
        return 0.0, 0.0
    height, width = previous.shape
    if (not np.isfinite(dx) or not np.isfinite(dy) or response < 0.05
            or abs(dx) > 0.25 * width or abs(dy) > 0.25 * height):
        # A quarter of the frame between two frames at video rate is not a
        # camera, it is a failed correlation reporting its own ambiguity.
        return 0.0, 0.0
    return float(dx), float(dy)


def _padded(box, margin: float, width: int, height: int) -> tuple[int, int, int, int]:
    """`box` grown by `margin` of its own size and clipped to the frame."""
    x0, y0, x1, y1 = (float(v) for v in box)
    pad_x = (x1 - x0) * margin
    pad_y = (y1 - y0) * margin
    return (int(max(0, x0 - pad_x)), int(max(0, y0 - pad_y)),
            int(min(width, x1 + pad_x)), int(min(height, y1 + pad_y)))


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def box_of(mask: np.ndarray) -> np.ndarray | None:
    """The tight xyxy box around a boolean mask, or None when it is empty."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    y = np.flatnonzero(rows)
    x = np.flatnonzero(cols)
    return np.array([x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=np.float64)


def centre_of(box) -> np.ndarray:
    x0, y0, x1, y1 = (float(v) for v in box)
    return np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])


def size_of(box) -> tuple[float, float]:
    x0, y0, x1, y1 = (float(v) for v in box)
    return max(x1 - x0, 0.0), max(y1 - y0, 0.0)


def area_of(box) -> float:
    width, height = size_of(box)
    return width * height


def aspect_of(box) -> float:
    width, height = size_of(box)
    return width / height if height > 0 else float("inf")


def local_contrast(frame: np.ndarray, box, ring: int = 9) -> float:
    """Signal-to-clutter for one box: `|mean inside - mean ring| / std ring`.

    The numpy twin of `image_loop.instance_contrast`, on a box rather than a
    mask. One number, defined the same way in training, in evaluation and here,
    so "the bucket this run is weak in" and "the frames this tracker should
    enhance" are the same statement.
    """
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = (int(round(float(v))) for v in box)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, width), min(y1, height)
    if x1 <= x0 or y1 <= y0:
        return float("nan")

    inner = frame[y0:y1, x0:x1].astype(np.float64)
    rx0, ry0 = max(x0 - ring, 0), max(y0 - ring, 0)
    rx1, ry1 = min(x1 + ring, width), min(y1 + ring, height)
    outer = frame[ry0:ry1, rx0:rx1].astype(np.float64)
    if outer.size <= inner.size:
        return float("nan")

    ring_sum = outer.sum() - inner.sum()
    ring_count = outer.size - inner.size
    ring_mean = ring_sum / ring_count
    ring_sq = (outer ** 2).sum() - (inner ** 2).sum()
    spread = max(ring_sq / ring_count - ring_mean ** 2, 0.0) ** 0.5
    return float(abs(inner.mean() - ring_mean) / max(spread, 1e-3))


# --------------------------------------------------------------------------
# Re-acquisition
# --------------------------------------------------------------------------


def bandpass(frame: np.ndarray, size, low: float = 0.55,
             grain: float = 1.5) -> np.ndarray:
    """`frame` with everything coarser than the target -- and finer -- removed.

    **Why re-acquisition needs this, measured rather than assumed.** The
    template is a box around the target, so it holds the target *and* the
    ground under it. On this footage the ground is most of what is in it: a
    thermal target sitting 5 levels above its own surroundings, on terrain
    whose texture runs 27 levels, puts roughly a twentieth of the template's
    variance in the target. Normalised cross-correlation then finds the
    *ground*, and it finds it exactly where the template was cut from -- on a
    synthetic bench of 36 low-contrast cases the peak landed on the target 0
    times, scoring 0.64 at the template's old position against 0.17 at the
    target's new one. That is not an occasional wrong answer; it is what the
    matcher does by default, and `give_up_after` was hiding it as "not found".

    A difference of Gaussians cut to the target's own scale fixes it, because
    the ground's variation and the target's are at different scales and nothing
    else about them differs. `low` is the outer sigma as a fraction of the
    target's longer side -- the ground goes with it; `grain` is the inner one in
    pixels, which is the sensor's own noise. Over three background recipes and
    twelve seeds each, counting a hit as the peak landing on the target:

        lift over its own ground     3          5          8         14
        raw                       0/36       0/36       0/35      36/36
        band-passed              11/36      20/36      35/36      36/36

    It is a real improvement and not a cure: at three levels it still misses
    two thirds of the time. At fourteen, where the raw matcher already found
    everything, the peak rises from 0.63 to 0.92 rather than falling -- never
    worse is why it is on by default.

    **What was tried first and did not work.** CLAHE is the usual suggestion.
    It does raise the signal-to-clutter ratio on ground that has slow variation
    to remove -- 1.2 to 1.4x on this bench, and nothing at all on flat ground,
    which is the one case the earlier measurement here used -- but it does not
    move the matcher: it lifts the target and the clutter beside it together,
    so the template stays mostly ground. Across the same cases it went from
    0/36 to 5/36 where the band-pass reached 35/36. A top-hat is the same
    story. The scale separation is the mechanism; the contrast stretch is not,
    which is also why nothing here enhances the pixels the *model* sees.

    Returns float32, which `match_template` correlates directly. Quantising
    back to 8 bits would throw away most of what the band-pass just isolated.
    """
    import cv2

    wide = max(float(size[0]), float(size[1])) if size is not None else 8.0
    outer = max(float(low) * wide, float(grain) * 2.0)
    field = np.asarray(frame, dtype=np.float32)
    if field.ndim > 2:
        field = field.mean(axis=2)
    return (cv2.GaussianBlur(field, (0, 0), float(grain))
            - cv2.GaussianBlur(field, (0, 0), outer))


def match_template(frame: np.ndarray, template: np.ndarray,
                   region) -> tuple[np.ndarray | None, float]:
    """Best normalised cross-correlation of `template` inside `region`.

    `cv2.TM_CCOEFF_NORMED`, which subtracts each patch's own mean before
    correlating -- the reason it is the right classical matcher for thermal:
    the sensor's level drifts with its own temperature and an unnormalised
    correlation would rank a brighter patch above the right one.

    Returns `(box, score)` for the best position, or `(None, nan)` when the
    region cannot hold the template.
    """
    import cv2

    height, width = frame.shape[:2]
    x0, y0, x1, y1 = (int(round(float(v))) for v in region)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, width), min(y1, height)
    patch = frame[y0:y1, x0:x1]
    if (patch.shape[0] < template.shape[0] or patch.shape[1] < template.shape[1]
            or template.size == 0):
        return None, float("nan")
    # The refusal threshold is relative to the patch being searched, because a
    # band-passed frame is not in 8-bit levels any more and a fixed `1.0` would
    # refuse every template on one and no template on the other.
    floor = max(float(np.asarray(patch, dtype=np.float64).std()) * 0.02, 1e-3)
    if template.dtype == np.uint8 and patch.dtype == np.uint8:
        floor = 1.0
    if float(np.asarray(template, dtype=np.float64).std()) < floor:
        # A template with no variation correlates perfectly with *everything*:
        # `TM_CCOEFF_NORMED` divides by the patch's own standard deviation, and
        # OpenCV returns 1.0 across the whole search region rather than a
        # division by zero. A saturated target -- every pixel at 255, which a
        # thermal sensor produces regularly -- is exactly that case, so this is
        # not a synthetic worry. Refusing is the only safe answer: there is
        # nothing in those pixels to find the target by.
        return None, float("nan")

    kind = (np.uint8 if patch.dtype == np.uint8 and template.dtype == np.uint8
            else np.float32)
    scores = cv2.matchTemplate(patch.astype(kind), template.astype(kind),
                               cv2.TM_CCOEFF_NORMED)
    _, best, _, position = cv2.minMaxLoc(scores)
    top_left = (x0 + position[0], y0 + position[1])
    box = np.array([top_left[0], top_left[1],
                    top_left[0] + template.shape[1],
                    top_left[1] + template.shape[0]], dtype=np.float64)
    return box, float(best)


class FrameMotion:
    """The camera's displacement per frame, read off a directory of frames.

    One decoded frame is held at a time and each is read once, so a whole video
    costs one extra grayscale decode per frame and one sparse flow -- around a
    millisecond beside a network that takes tens. The decode is done at a
    reduced size on purpose (`cv2.IMREAD_REDUCED_GRAYSCALE_*`, which decodes
    fewer coefficients rather than resizing after the fact) and the answer is
    returned as a **fraction of the frame**, so the reduction cancels and the
    consumer never has to know which resolution was decoded.

    Frames are asked for in order but not necessarily every one of them, and a
    gap is not an error: a shift measured over two frames is still the camera's
    displacement between the two frames the caller is comparing.
    """

    def __init__(self, frames_dir, reduce: int = 4,
                 suffixes: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")):
        from pathlib import Path as _Path

        self.paths = sorted(p for p in _Path(frames_dir).iterdir()
                            if p.suffix.lower() in suffixes)
        self.reduce = int(reduce)
        self._index: int | None = None
        self._frame: np.ndarray | None = None

    def frame(self, index: int) -> np.ndarray | None:
        """The greyscale frame at `index`, decoded small, or None if unreadable."""
        import cv2

        if not 0 <= index < len(self.paths):
            return None
        flag = {1: cv2.IMREAD_GRAYSCALE, 2: cv2.IMREAD_REDUCED_GRAYSCALE_2,
                4: cv2.IMREAD_REDUCED_GRAYSCALE_4,
                8: cv2.IMREAD_REDUCED_GRAYSCALE_8}.get(self.reduce,
                                                       cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(str(self.paths[index]), flag)
        return None if image is None else image

    def shift(self, index: int, exclude=None) -> tuple[float, float] | None:
        """`(dx, dy)` from the previously asked-for frame to `index`, normalised.

        None for the first frame of a video, for an unreadable one, and for a
        pair whose sizes disagree -- all three are "no measurement", which a
        Kalman filter should be told rather than handed a zero it would treat
        as a measured standstill.
        """
        current = self.frame(index)
        if current is None:
            return None
        previous, before = self._frame, self._index
        self._frame, self._index = current, index
        if previous is None or before is None or previous.shape != current.shape:
            return None
        dx, dy = estimate_shift(previous, current, exclude)
        height, width = current.shape[:2]
        return dx / float(width), dy / float(height)

    def reset(self) -> None:
        self._index = None
        self._frame = None


# --------------------------------------------------------------------------
# The guard
# --------------------------------------------------------------------------


@dataclass
class Stabiliser:
    """One track's classical guard: ego-motion, plausibility, re-acquisition.

    Call `update` once per frame with the greyscale frame and whatever box the
    tracker produced (`None` when its mask was empty). What comes back says
    whether that box was taken, held or replaced, and why.

    It holds no model and no frames -- one previous greyscale frame, one
    template, and a short history of areas and aspects. Everything it decides
    is reproducible from those, which is what makes it testable frame by frame
    against a recording.
    """

    config: GuardConfig = field(default_factory=GuardConfig)
    state: str = TRACKING
    box: np.ndarray | None = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    template: np.ndarray | None = None
    missing: int = 0
    healthy: int = 0
    frames: int = 0
    # True while the box being reported came from a template match rather than
    # from the tracker. It is what stops the guard flapping: a tracker that is
    # persistently wrong would otherwise be re-searched only every
    # `suspect_frames`, and dead-reckoned in between.
    driven: bool = False
    _previous: np.ndarray | None = None
    _areas: deque = field(default_factory=lambda: deque(maxlen=12))
    _aspects: deque = field(default_factory=lambda: deque(maxlen=12))

    def __post_init__(self) -> None:
        self._areas = deque(maxlen=max(self.config.history, 1))
        self._aspects = deque(maxlen=max(self.config.history, 1))

    # -- public ----------------------------------------------------------

    def reset(self) -> None:
        """Forget the track. One `Stabiliser` belongs to one video."""
        self.state = TRACKING
        self.box = None
        self.velocity = np.zeros(2)
        self.template = None
        self.missing = self.healthy = self.frames = 0
        self.driven = False
        self._previous = None
        self._areas.clear()
        self._aspects.clear()

    def update(self, frame: np.ndarray, box=None) -> Decision:
        """One frame. `box` is the tracker's claim, `None` if it made none."""
        frame = np.ascontiguousarray(frame)
        if frame.ndim != 2:
            raise ValueError("Stabiliser.update takes a greyscale frame")
        self.frames += 1

        shift = (0.0, 0.0)
        if self._previous is not None:
            shift = estimate_shift(self._previous, frame, self.box)
        self._previous = frame

        if self.box is None:
            # Nothing to compare against yet: the first box a tracker gives is
            # the one it was prompted with, and refusing that would leave the
            # guard with no reference for every frame after it.
            return self._adopt(frame, box, shift)

        predicted = self._predict(shift)
        if box is None:
            return self._refuse(frame, predicted, shift, "no mask")

        verdict = self._implausible(frame, box, predicted, self._gates(frame, predicted))
        if verdict:
            return self._refuse(frame, predicted, shift, verdict)
        return self._accept(frame, np.asarray(box, dtype=np.float64), shift)

    # -- internals -------------------------------------------------------

    def _predict(self, shift) -> np.ndarray:
        """Where the box should be, given the camera's motion and the target's."""
        step = np.asarray(shift, dtype=np.float64) + self.velocity
        return np.asarray(self.box, dtype=np.float64) + np.concatenate([step, step])

    def _gates(self, frame: np.ndarray, predicted: np.ndarray) -> GuardConfig:
        """The thresholds for *this* frame, tightened where the target is faint.

        A tracker's mistakes are not uniformly distributed: they cluster on the
        frames where the target has stopped standing out from the ground, which
        is a number this can measure before deciding anything. So on those
        frames the guard asks for more before believing a large move or a large
        change of size -- adaptive gating, the oldest trick in the file.
        """
        config = self.config
        if config.caution_factor >= 1.0:
            return config
        contrast = local_contrast(frame, predicted)
        if not np.isfinite(contrast) or contrast >= config.caution_below:
            return config
        return replace(config,
                       max_area_ratio=1.0 + (config.max_area_ratio - 1.0)
                       * config.caution_factor,
                       max_jump=config.max_jump * config.caution_factor)

    def _implausible(self, frame: np.ndarray, box, predicted,
                     config: GuardConfig | None = None) -> str:
        """The first gate this box fails, or `""` when it passes them all."""
        config = config or self.config
        box = np.asarray(box, dtype=np.float64)
        area, aspect = area_of(box), aspect_of(box)
        if area <= 0:
            return "empty box"
        if area > config.max_frame_share * frame.shape[0] * frame.shape[1]:
            return "covers the frame"

        typical_area = float(np.median(self._areas)) if self._areas else area
        typical_aspect = float(np.median(self._aspects)) if self._aspects else aspect
        if typical_area > 0:
            ratio = area / typical_area
            if ratio > config.max_area_ratio:
                return f"area x{ratio:.1f}"
            if ratio < config.min_area_ratio:
                return f"area x{ratio:.2f}"
        if np.isfinite(aspect) and typical_aspect > 0:
            shape = max(aspect / typical_aspect, typical_aspect / max(aspect, 1e-6))
            if shape > config.max_aspect_ratio:
                return f"aspect x{shape:.1f}"

        width, height = size_of(predicted)
        reach = max(width, height, 1.0) * config.max_jump
        travelled = float(np.linalg.norm(centre_of(box) - centre_of(predicted)))
        if travelled > reach:
            return f"jumped {travelled / max(width, height, 1.0):.1f} sizes"
        return ""

    def _accept(self, frame: np.ndarray, box: np.ndarray, shift) -> Decision:
        """Take the tracker's box, and remember what it looked like."""
        moved = centre_of(box) - centre_of(self.box) - np.asarray(shift)
        # A quarter of the new estimate, so one frame of blur does not steer
        # the prediction that judges the next one.
        self.velocity = 0.75 * self.velocity + 0.25 * moved
        self.box = box
        self._areas.append(area_of(box))
        self._aspects.append(aspect_of(box))
        self.template = _crop(self._view(frame, size_of(box)), box)
        self.missing = 0
        self.driven = False
        was = self.state
        self.healthy += 1
        if was in (SUSPECT, LOST, REACQUIRED):
            if self.healthy >= self.config.recover_frames:
                self.state = TRACKING
            else:
                self.state = REACQUIRED
        else:
            self.state = TRACKING
        return Decision(box=box, state=self.state, shift=tuple(shift))

    def _refuse(self, frame: np.ndarray, predicted: np.ndarray, shift,
                reason: str) -> Decision:
        """Hold the prediction, count the miss, and search once lost."""
        self.healthy = 0
        self.missing += 1
        self.box = predicted
        if self.missing < self.config.suspect_frames and not self.driven:
            self.state = SUSPECT
            return Decision(box=predicted, state=SUSPECT, reason=reason,
                            shift=tuple(shift))

        self.state = LOST
        found, score = self._search(frame, predicted)
        if found is None:
            return Decision(box=predicted, state=LOST, reason=reason,
                            shift=tuple(shift), match=score)
        # A re-acquisition is a claim, not a certainty: it re-enters through
        # the same door every accepted frame does, so the recovery hysteresis
        # still has to agree before the track is called healthy again.
        self.box = found
        self._areas.append(area_of(found))
        self._aspects.append(aspect_of(found))
        self.velocity = np.zeros(2)
        self.missing = 0
        self.driven = True
        self.healthy = 0
        self.state = REACQUIRED
        return Decision(box=found, state=REACQUIRED, reason=f"{reason} -> matched",
                        shift=tuple(shift), match=score)

    def _view(self, frame: np.ndarray, size) -> np.ndarray:
        """The frame the matcher sees: band-passed, or raw when turned off."""
        if not self.config.match_bandpass:
            return frame
        return bandpass(frame, size, low=self.config.match_bandpass)

    def _search(self, frame: np.ndarray, predicted: np.ndarray
                ) -> tuple[np.ndarray | None, float]:
        """Normalised cross-correlation in a region that grows while lost."""
        if self.template is None or self.missing > self.config.give_up_after:
            return None, float("nan")
        grown = min(self.config.search_scale
                    * self.config.search_growth ** max(self.missing - 1, 0),
                    self.config.search_max)
        region = _grown(predicted, grown, frame.shape[1], frame.shape[0])
        # The template was cut from a band-passed frame, so the search has to
        # run on one cut the same way -- correlating a band-passed template
        # against raw pixels compares two different pictures.
        shape = (self.template.shape[1], self.template.shape[0])
        found, score = match_template(self._view(frame, shape), self.template,
                                      region)
        if found is None or not np.isfinite(score) or score < self.config.match_threshold:
            return None, score
        return found, score

    def _adopt(self, frame: np.ndarray, box, shift) -> Decision:
        """The first box of a track, taken as given."""
        if box is None:
            return Decision(box=None, state=LOST, reason="no first box",
                            shift=tuple(shift))
        box = np.asarray(box, dtype=np.float64)
        self.box = box
        self._areas.append(area_of(box))
        self._aspects.append(aspect_of(box))
        self.template = _crop(self._view(frame, size_of(box)), box)
        self.state = TRACKING
        self.healthy = 1
        return Decision(box=box, state=TRACKING, shift=tuple(shift))


def _crop(frame: np.ndarray, box) -> np.ndarray | None:
    """The target's own pixels -- the appearance a re-acquisition matches."""
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = (int(round(float(v))) for v in box)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, width), min(y1, height)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return frame[y0:y1, x0:x1].copy()


def _grown(box, scale: float, width: int, height: int) -> np.ndarray:
    """`box` scaled about its own centre and clipped to the frame."""
    centre = centre_of(box)
    half_w, half_h = (np.array(size_of(box)) * scale) / 2.0
    return np.array([max(centre[0] - half_w, 0), max(centre[1] - half_h, 0),
                     min(centre[0] + half_w, width),
                     min(centre[1] + half_h, height)], dtype=np.float64)


__all__ = ["Decision", "FrameMotion", "GuardConfig", "LOST", "REACQUIRED",
           "STATES", "SUSPECT",
           "Stabiliser", "TRACKING", "area_of", "aspect_of", "box_of",
           "bandpass", "centre_of", "estimate_shift", "local_contrast",
           "match_template",
           "size_of"]
