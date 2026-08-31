"""Making the hard case out of the easy one: contrast, polarity, noise.

A thermal encoder can learn a shortcut its training data offers: answer "the
bright blob" and score well wherever the target is hot against cold ground.
`image_loop.instance_contrast` puts a number on how far a target stands out --
its signal against the clutter of the ground beside it -- and this module makes
that number worse on purpose, so the shortcut stops paying.

**How much of a shortcut there is to break is a property of the source, and it
was measured rather than assumed.** On HIT-UAV -- 4 421 annotated targets over
500 of its frames, drone at 60-130 m -- the median target sits at a
signal-to-clutter ratio of **0.91**, 53 % of them are *below* 1, and only 15 %
stand out at 3 or more. People are the visible ones (median 1.58); cars (0.55)
and bicycles (0.34) are mostly inside the clutter already. That set does not
offer the shortcut, and this module is nearly a no-op on it: the aggressive
setting moved its median from 0.85 to 0.79 and shrank the easy tail from 15 %
to 10 %. (Measured on boxes rather than masks, since HIT-UAV ships no masks, so
those are a lower bound -- a mask-tight measure reads higher.)

Which is the argument for the knobs rather than against them: what they are
worth depends on the pool, so a run should read the per-band table its own
evaluation prints instead of trusting an augmentation to have created
difficulty. On a source whose targets really are hot against cold ground, the
collapse has something to work on; on one like HIT-UAV, the polarity flip and
the gamma are the terms still doing work, and they do it at any contrast.

**Collapsing contrast alone does nothing.** Scaling a window's pixels toward
their mean divides the target-to-background difference *and* the background's
own variation by the same factor, so the signal-to-clutter ratio comes out
unchanged and the model sees the same problem in a darker image. What lowers it
is collapsing the scene and then adding sensor noise on top, because the noise
is not scaled with it. That is why `noise` is not a garnish here: it is half of
the mechanism, and a `Photometric` with `collapse` set and `noise` at zero is a
no-op dressed as an augmentation.

The other three are cheaper and answer the same shortcut from other sides:

`invert`  a polarity flip. A thermal frame's grey is a sensor convention --
          white-hot and black-hot are the same scene -- so a model that has
          only ever seen one has learned half a rule. Flipping the window keeps
          every mask exactly where it was and makes "bright" meaningless.
`gamma`   a non-linear stretch, which changes local contrast unevenly: it moves
          the target's separation from its ground without moving the histogram
          as a whole, the way a different sensor's transfer curve would.
`noise`   read noise. Also, per the above, what gives `collapse` its teeth.

Nothing here touches the mask, the box, or the class. The target is exactly
where it was; only the evidence for it gets worse.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The 8-bit range everything below works in. The windows arrive as uint8 from
# `load_image`, whatever the sensor's own depth was.
LEVELS = 255.0


@dataclass(frozen=True)
class Photometric:
    """How often, and how far, a training window is made harder to read.

    `collapse` is a probability and `floor` the weakest scale it draws: 0.35
    means a window can come out with a third of its contrast. `invert` is a
    probability and applies to thermal windows only -- an RGB frame's polarity
    is not a convention. `gamma` is the half-width of a log-uniform exponent,
    and `noise` is a standard deviation in 8-bit levels, applied last so that a
    collapsed window really does hold a fainter target rather than a dimmer
    copy of the same one.
    """

    collapse: float = 0.0
    floor: float = 0.35
    invert: float = 0.0
    gamma: float = 0.0
    noise: float = 0.0

    @property
    def active(self) -> bool:
        """Whether this would change a window at all."""
        return bool(self.collapse or self.invert or self.gamma or self.noise)

    def describe(self) -> str:
        if not self.active:
            return "photometric augmentation off"
        return (f"collapse {self.collapse:.0%} to >={self.floor:g}x, "
                f"polarity flip {self.invert:.0%}, gamma +-{self.gamma:g}, "
                f"noise {self.noise:g} levels")


def harden(pixels: np.ndarray, gray: bool, config: Photometric,
           rng: np.random.Generator) -> np.ndarray:
    """One window, made harder to read. `pixels` is HxWx3 uint8, and so is the result.

    Order is the sensor's: the transfer curve first, then the scene's own
    contrast, then the polarity convention, then the read noise the sensor adds
    to whatever came out. Noise last is the load-bearing part -- see the module
    docstring.
    """
    if not config.active:
        return pixels
    out = pixels.astype(np.float32)

    if config.gamma:
        exponent = float(np.exp(rng.uniform(-config.gamma, config.gamma)))
        out = LEVELS * np.power(np.clip(out / LEVELS, 0.0, 1.0), exponent)
    if config.collapse and rng.random() < config.collapse:
        scale = float(rng.uniform(config.floor, 1.0))
        out = out.mean() + (out - out.mean()) * scale
    if gray and config.invert and rng.random() < config.invert:
        out = LEVELS - out
    if config.noise:
        # One draw across the channels, not three: a thermal window is one
        # measurement replicated, and independent noise per channel would hand
        # the model a colour cue no sensor gives it.
        grain = rng.normal(0.0, config.noise, out.shape[:2]).astype(np.float32)
        out = out + grain[..., None]
    return np.clip(out, 0.0, LEVELS).astype(np.uint8)


def augmenter(config: Photometric, seed: int = 0):
    """A `(pixels, gray) -> pixels` callable with its own stream of draws.

    Seeded per call rather than per process: a run is reproducible from its
    seed, and two runs that differ only in the checkpoint see the same windows
    made hard in the same way. `collate` applies it in sample order, outside
    the thread pool that decodes, so the draws do not depend on which decode
    finished first.
    """
    rng = np.random.default_rng(seed)

    def apply(pixels: np.ndarray, gray: bool = True) -> np.ndarray:
        return harden(pixels, gray, config, rng)

    return apply


__all__ = ["LEVELS", "Photometric", "augmenter", "harden"]
