"""SAM2Long's pathway bookkeeping and its branching rule.

The claim being pinned is that this is *off by default and neutral when on*:
with one pathway, or on frames where the candidates are clearly separated, it
has to behave exactly like stock SAM 2. An experimental variant that quietly
changes the baseline would make every comparison against it meaningless.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prompts import BoxPrompt, PointPrompt, PromptSet  # noqa: E402
from src.trackers.edgetam_tracker import _block_config, _replicate_prompts  # noqa: E402
from src.trackers.sam2long import (  # noqa: E402
    PATHWAY_STRIDE,
    Sam2Long,
    Sam2LongConfig,
    expand_ids,
    split_id,
)


def state(pathways=3, **kwargs):
    long = Sam2Long(Sam2LongConfig(enabled=True, pathways=pathways, **kwargs))
    long.set_rows(expand_ids(1, pathways))
    return long


class TestIds(unittest.TestCase):
    def test_expand_and_split_are_inverse(self):
        for user in (1, 2, 17):
            for rank, internal in enumerate(expand_ids(user, 4)):
                self.assertEqual(split_id(internal), (user, rank))

    def test_ids_of_different_objects_never_collide(self):
        self.assertEqual(len(set(expand_ids(1, 3) + expand_ids(2, 3))), 6)

    def test_the_stride_leaves_room_for_realistic_object_counts(self):
        self.assertGreaterEqual(PATHWAY_STRIDE, 100)


class TestReplicatePrompts(unittest.TestCase):
    def test_a_box_becomes_one_per_pathway_with_the_same_geometry(self):
        original = PromptSet(boxes=[BoxPrompt(obj_id=1, frame_idx=0,
                                              xyxy=(1.0, 2.0, 3.0, 4.0))])
        replicated = _replicate_prompts(original, 3)

        self.assertEqual(len(replicated.boxes), 3)
        self.assertEqual([b.obj_id for b in replicated.boxes], expand_ids(1, 3))
        self.assertTrue(all(b.xyxy == (1.0, 2.0, 3.0, 4.0) for b in replicated.boxes))

    def test_points_are_replicated_too(self):
        original = PromptSet(points=[PointPrompt(obj_id=2, frame_idx=0,
                                                 xy=(5.0, 6.0), label=1)])
        replicated = _replicate_prompts(original, 2)

        self.assertEqual([p.obj_id for p in replicated.points], expand_ids(2, 2))
        self.assertTrue(all(p.xy == (5.0, 6.0) for p in replicated.points))

    def test_multiple_objects_stay_separable(self):
        original = PromptSet(boxes=[
            BoxPrompt(1, 0, (0.0, 0.0, 1.0, 1.0)),
            BoxPrompt(2, 0, (2.0, 2.0, 3.0, 3.0)),
        ])
        users = [split_id(b.obj_id)[0] for b in _replicate_prompts(original, 3).boxes]
        self.assertEqual(users, [1, 1, 1, 2, 2, 2])


class TestBranching(unittest.TestCase):
    def _assign(self, long, ious, object_score=5.0):
        rows = torch.tensor(ious, dtype=torch.float32)
        scores = torch.full((rows.shape[0], 1), float(object_score))
        return long.assign(rows, scores).argmax(dim=-1).tolist()

    def test_a_clear_frame_makes_every_pathway_agree(self):
        # Nothing to disagree about; branching here would only add noise.
        long = state(uncertainty_margin=0.05, object_score_floor=0.0)
        self.assertEqual(self._assign(long, [[0.2, 0.9, 0.4]] * 3), [1, 1, 1])
        self.assertEqual(long.branches, 0)

    def test_an_ambiguous_frame_makes_them_diverge(self):
        long = state(uncertainty_margin=0.1, object_score_floor=0.0)
        chosen = self._assign(long, [[0.80, 0.78, 0.10]] * 3)
        # Rank 0 takes the head's own pick, rank 1 the runner-up, rank 2 next.
        self.assertEqual(chosen, [0, 1, 2])
        self.assertGreater(long.branches, 0)

    def test_a_low_object_score_counts_as_ambiguous(self):
        # The case where SAM 2 is about to write no_obj_ptr and being wrong is
        # most expensive -- exactly where a second opinion is worth having.
        long = state(uncertainty_margin=0.0, object_score_floor=1.0)
        self.assertEqual(self._assign(long, [[0.9, 0.5, 0.1]] * 3, object_score=-2.0),
                         [0, 1, 2])

    def test_more_pathways_than_candidates_clamps_instead_of_failing(self):
        long = state(pathways=5, uncertainty_margin=1.0)
        chosen = self._assign(long, [[0.5, 0.4, 0.3]] * 5)
        self.assertEqual(chosen, [0, 1, 2, 2, 2])

    def test_one_pathway_is_exactly_stock_behaviour(self):
        long = state(pathways=1, uncertainty_margin=1.0, object_score_floor=10.0)
        for ious in ([[0.2, 0.9, 0.4]], [[0.5, 0.49, 0.48]]):
            self.assertEqual(self._assign(long, ious), [int(np.argmax(ious[0]))])

    def test_the_single_mask_path_is_untouched(self):
        long = state()
        ious = torch.tensor([[0.7], [0.7], [0.7]])
        self.assertIs(long.assign(ious, torch.zeros(3, 1)), ious)

    def test_rows_not_registered_yet_behave_as_stock(self):
        # prepare() runs before set_prompts(); nothing may change in between.
        long = Sam2Long(Sam2LongConfig(enabled=True))
        ious = torch.tensor([[0.1, 0.9, 0.2]])
        self.assertIs(long.assign(ious, torch.zeros(1, 1)), ious)


class TestScoringAndCollapse(unittest.TestCase):
    def test_the_best_scoring_pathway_wins(self):
        long = state()
        long.scores = np.array([0.1, 0.9, 0.4])
        self.assertEqual(long.winners(), {1: 1})

    def test_ties_go_to_the_pathway_that_never_deviated(self):
        # With everything else equal, the answer is stock SAM 2's.
        long = state()
        long.scores = np.array([0.5, 0.5, 0.5])
        self.assertEqual(long.winners(), {1: 0})

    def test_each_object_is_collapsed_independently(self):
        long = Sam2Long(Sam2LongConfig(enabled=True, pathways=2))
        long.set_rows(expand_ids(1, 2) + expand_ids(2, 2))
        long.scores = np.array([0.1, 0.8, 0.9, 0.2])
        self.assertEqual(long.winners(), {1: 1, 2: 2})

    def test_scores_accumulate_across_frames(self):
        long = state(uncertainty_margin=0.0, object_score_floor=0.0)
        before = long.scores.copy()
        for _ in range(3):
            long.assign(torch.tensor([[0.9, 0.1, 0.1]] * 3), torch.full((3, 1), 5.0))
        self.assertTrue((long.scores > before).all())
        # 3 frames x (iou 0.9 + sigmoid(5) ~ 0.993)
        self.assertAlmostEqual(float(long.scores[0]), 3 * (0.9 + 0.9933), places=2)

    def test_reset_clears_everything(self):
        long = state()
        long.scores = np.array([1.0, 2.0, 3.0])
        long.branches = 7
        long.reset()
        self.assertEqual(long.ids, [])
        self.assertEqual(long.branches, 0)
        self.assertEqual(len(long.scores), 0)


class TestConfig(unittest.TestCase):
    def test_it_is_off_unless_asked_for(self):
        self.assertFalse(Sam2LongConfig().enabled)
        self.assertIsNone(_block_config(None, "sam2long"))
        self.assertIsNone(_block_config({}, "sam2long"))
        self.assertIsNone(_block_config({"enabled": False}, "sam2long"))

    def test_an_unknown_key_is_rejected_at_startup(self):
        with self.assertRaises(ValueError) as ctx:
            _block_config({"enabled": True, "pathwayz": 3}, "sam2long")
        self.assertIn("pathwayz", str(ctx.exception))

    def test_zero_pathways_is_rejected(self):
        with self.assertRaises(ValueError):
            Sam2LongConfig(enabled=True, pathways=0)

    def test_a_valid_block_is_accepted(self):
        config = _block_config({"enabled": True, "pathways": 4}, "sam2long")
        self.assertEqual(config.pathways, 4)


if __name__ == "__main__":
    unittest.main()
