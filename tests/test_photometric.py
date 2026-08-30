"""Making the low-contrast case, and measuring that it was made.

The failure this covers is a *tracker* failure: the encoder reads a thermal
target off its brightness, which is free on the sets it is trained on -- almost
every annotated target in them is hot against cold ground -- and then loses a
parked car on warm concrete. The two halves of the answer are here: a metric
that says how far a target stands out from the ground beside it, and an
augmentation that manufactures the hard end of that distribution from the easy
end the data actually holds.

The one non-obvious property, and the reason this file exists rather than a
docstring: **collapsing a window's contrast on its own changes nothing.** It
divides the target's signal and the background's clutter by the same number, so
the ratio -- which is what the model has to work with -- comes out where it
started. Only the noise added after it moves the number.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
except ImportError:                                      # pragma: no cover
    torch = None

from src.training.photometric import (LEVELS, Photometric,  # noqa: E402
                                      augmenter, harden)


def window(background: float, target: float, noise: float = 6.0,
           seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A grey window with one rectangular target on it, and its mask."""
    rng = np.random.default_rng(seed)
    pixels = np.full((96, 96), float(background)) + rng.normal(0, noise, (96, 96))
    mask = np.zeros((96, 96), dtype=bool)
    mask[40:52, 40:58] = True
    pixels[mask] = target
    grey = np.clip(pixels, 0, LEVELS).astype(np.uint8)
    return np.repeat(grey[..., None], 3, axis=2), mask


def contrast_of(pixels: np.ndarray, mask: np.ndarray) -> float:
    """`instance_contrast` on a one-window, one-instance batch."""
    from src.training.aerial import normalise
    from src.training.image_loop import ImageBatch, instance_contrast

    batch = ImageBatch(images=normalise(pixels[None]),
                       boxes=torch.zeros(1, 1, 4),
                       masks=torch.from_numpy(mask[None, None]),
                       valid=torch.ones(1, 1, dtype=torch.bool))
    return float(instance_contrast(batch)[0])


@unittest.skipIf(torch is None, "torch is needed for the batch metric")
class ContrastMetricTest(unittest.TestCase):
    def test_a_target_on_its_own_ground_ranks_by_how_far_it_stands_out(self):
        easy = contrast_of(*window(60, 210))
        middling = contrast_of(*window(90, 140))
        camouflaged = contrast_of(*window(120, 124))
        self.assertGreater(easy, middling)
        self.assertGreater(middling, camouflaged)
        # The bucket edges `eval_instances` reports on have to mean something:
        # a target four levels from its ground is inside the clutter.
        self.assertLess(camouflaged, 1.0)
        self.assertGreater(easy, 3.0)

    def test_the_measure_is_blind_to_an_affine_change_of_the_pixels(self):
        """It is computed on the normalised tensor the model was fed, and
        `normalise` is affine -- so the number has to be the one the raw window
        would have given, or it would be reporting ImageNet statistics."""
        pixels, mask = window(70, 180)
        brighter = np.clip(pixels.astype(np.float32) * 0.5 + 60,
                           0, LEVELS).astype(np.uint8)
        # `delta` and not `places`: halving the range and storing it back as
        # 8 bits is itself a quantisation, so the two windows are not quite the
        # same picture any more. 0.1 of a ratio near 20 is that rounding.
        self.assertAlmostEqual(contrast_of(pixels, mask),
                               contrast_of(brighter, mask), delta=0.1)

    def test_a_polarity_flip_is_the_same_problem(self):
        """White-hot and black-hot are one scene. The target stands out exactly
        as far in both, which is why the flip is an augmentation and not a way
        of making a window harder."""
        pixels, mask = window(70, 190)
        flipped = harden(pixels, True, Photometric(invert=1.0),
                         np.random.default_rng(0))
        self.assertAlmostEqual(contrast_of(pixels, mask),
                               contrast_of(flipped, mask), places=2)


@unittest.skipIf(torch is None, "torch is needed for the batch metric")
class HardeningTest(unittest.TestCase):
    def test_collapsing_contrast_alone_does_not_make_a_target_harder(self):
        """The whole reason `noise` is not optional."""
        pixels, mask = window(70, 190)
        collapsed = harden(pixels, True, Photometric(collapse=1.0, floor=0.35),
                           np.random.default_rng(0))
        self.assertAlmostEqual(contrast_of(pixels, mask),
                               contrast_of(collapsed, mask), delta=1.5)

    def test_a_collapse_with_noise_after_it_does(self):
        pixels, mask = window(70, 190)
        before = contrast_of(pixels, mask)
        after = np.mean([
            contrast_of(harden(pixels, True,
                               Photometric(collapse=1.0, floor=0.2, noise=6.0),
                               np.random.default_rng(seed)), mask)
            for seed in range(8)])
        self.assertLess(after, before / 1.5)

    def test_noise_alone_lowers_it_too(self):
        pixels, mask = window(70, 190)
        noisy = harden(pixels, True, Photometric(noise=8.0),
                       np.random.default_rng(0))
        self.assertLess(contrast_of(noisy, mask), contrast_of(pixels, mask))


class HardeningShapeTest(unittest.TestCase):
    """These need no torch: they are about what the augmentation may touch."""

    def test_an_inactive_config_hands_the_window_back_untouched(self):
        pixels, _ = window(70, 190)
        self.assertIs(harden(pixels, True, Photometric(), np.random.default_rng(0)),
                      pixels)
        self.assertFalse(Photometric().active)

    def test_the_window_keeps_its_shape_and_its_type(self):
        pixels, _ = window(70, 190)
        out = harden(pixels, True,
                     Photometric(collapse=1.0, invert=1.0, gamma=0.4, noise=5.0),
                     np.random.default_rng(0))
        self.assertEqual((out.shape, out.dtype), (pixels.shape, np.uint8))
        self.assertGreaterEqual(int(out.min()), 0)
        self.assertLessEqual(int(out.max()), 255)

    def test_the_three_channels_stay_one_measurement(self):
        """A thermal window is one channel replicated. Independent noise per
        channel would hand the model a colour cue no sensor gives it."""
        pixels, _ = window(70, 190)
        out = harden(pixels, True, Photometric(noise=9.0),
                     np.random.default_rng(0))
        self.assertTrue((out[..., 0] == out[..., 1]).all())
        self.assertTrue((out[..., 1] == out[..., 2]).all())

    def test_colour_windows_are_never_flipped(self):
        """`invert` is a thermal convention. An RGB frame has no other polarity."""
        pixels, _ = window(70, 190)
        out = harden(pixels, False, Photometric(invert=1.0),
                     np.random.default_rng(0))
        self.assertTrue((out == pixels).all())

    def test_one_seed_is_one_sequence_of_windows(self):
        pixels, _ = window(70, 190)
        config = Photometric(collapse=0.5, invert=0.5, gamma=0.3, noise=5.0)
        one, other = augmenter(config, 11), augmenter(config, 11)
        first = [one(pixels) for _ in range(4)]
        again = [other(pixels) for _ in range(4)]
        for mine, theirs in zip(first, again):
            self.assertTrue((mine == theirs).all())
        # One augmenter is a *stream* of draws, not one draw repeated: the
        # windows of an epoch have to be hardened differently from each other.
        self.assertFalse(all((frame == first[0]).all() for frame in first[1:]))


if __name__ == "__main__":
    unittest.main()
