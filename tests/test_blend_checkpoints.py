"""The blend is a checkpoint, or it raises -- it is never a quiet half-merge."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
except ImportError:                          # pragma: no cover - CPU-only CI
    torch = None

from tools.blend_checkpoints import blend, state_dict_of  # noqa: E402


@unittest.skipIf(torch is None, "torch is not installed")
class BlendTest(unittest.TestCase):
    def setUp(self):
        self.base = {"w": torch.zeros(4), "n": torch.tensor(3)}
        self.tuned = {"w": torch.ones(4), "n": torch.tensor(7)}

    def test_the_ends_of_the_sweep_are_the_two_checkpoints(self):
        self.assertTrue(torch.equal(blend(self.base, self.tuned, 0.0)["w"],
                                    self.base["w"]))
        self.assertTrue(torch.equal(blend(self.base, self.tuned, 1.0)["w"],
                                    self.tuned["w"]))

    def test_the_middle_is_the_weighted_average(self):
        out = blend(self.base, self.tuned, 0.25)["w"]
        self.assertTrue(torch.allclose(out, torch.full((4,), 0.25)))

    def test_a_counter_is_carried_not_averaged(self):
        # num_batches_tracked and friends: an average of two counters is not
        # a counter.
        self.assertEqual(int(blend(self.base, self.tuned, 0.5)["n"]), 7)

    def test_a_key_only_one_side_has_is_refused(self):
        with self.assertRaises(ValueError):
            blend(self.base, {**self.tuned, "extra": torch.ones(1)}, 0.5)

    def test_a_shape_that_changed_is_refused(self):
        with self.assertRaises(ValueError):
            blend(self.base, {**self.tuned, "w": torch.ones(5)}, 0.5)

    def test_alpha_outside_the_segment_is_refused(self):
        for alpha in (-0.1, 1.5):
            with self.assertRaises(ValueError):
                blend(self.base, self.tuned, alpha)

    def test_a_checkpoint_is_read_whichever_way_it_was_written(self):
        weights = {"w": torch.zeros(2)}
        self.assertIs(state_dict_of({"model": weights, "meta": {}}), weights)
        self.assertIs(state_dict_of(weights), weights)

    def test_the_dtype_survives_the_average(self):
        base = {"w": torch.zeros(4, dtype=torch.bfloat16)}
        tuned = {"w": torch.ones(4, dtype=torch.bfloat16)}
        self.assertEqual(blend(base, tuned, 0.5)["w"].dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
