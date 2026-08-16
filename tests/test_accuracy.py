"""Tracking-accuracy metrics, against hand-computed cases.

The interesting ones are the absent-target cases: a metric that only averages
IoU over visible frames cannot distinguish a tracker that correctly reports
"gone" from one that keeps hallucinating a box, and cannot distinguish one long
loss from many short ones. Both distinctions are the point.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.accuracy import (  # noqa: E402
    box_from_mask,
    box_iou,
    dropout_episodes,
    mask_iou,
    per_frame_iou,
    report,
    score_sequence,
    state_accuracy,
    success_auc,
    success_curve,
)

ABSENT = [np.nan] * 4


class TestBoxIoU(unittest.TestCase):
    def test_identical_boxes(self):
        b = np.array([[0.0, 0.0, 10.0, 10.0]])
        np.testing.assert_allclose(box_iou(b, b), [1.0])

    def test_half_overlap(self):
        a = np.array([[0.0, 0.0, 10.0, 10.0]])
        b = np.array([[5.0, 0.0, 15.0, 10.0]])
        # intersection 50, union 150
        np.testing.assert_allclose(box_iou(a, b), [50 / 150])

    def test_disjoint(self):
        a = np.array([[0.0, 0.0, 10.0, 10.0]])
        b = np.array([[20.0, 20.0, 30.0, 30.0]])
        np.testing.assert_allclose(box_iou(a, b), [0.0])


class TestMaskIoU(unittest.TestCase):
    def test_two_empty_masks_agree(self):
        empty = np.zeros((4, 4), dtype=bool)
        self.assertEqual(mask_iou(empty, empty), 1.0)

    def test_partial_overlap(self):
        a = np.zeros((4, 4), dtype=bool)
        b = np.zeros((4, 4), dtype=bool)
        a[0:2, 0:2] = True   # 4 px
        b[1:3, 1:3] = True   # 4 px, 1 px shared
        self.assertAlmostEqual(mask_iou(a, b), 1 / 7)


class TestBoxFromMask(unittest.TestCase):
    def test_tight_box_spans_the_pixels(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:6, 2:7] = True
        np.testing.assert_allclose(box_from_mask(mask), [2, 3, 7, 6])

    def test_single_pixel_spans_one_pixel(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 3] = True
        np.testing.assert_allclose(box_from_mask(mask), [3, 2, 4, 3])

    def test_empty_mask_is_nan(self):
        self.assertTrue(np.isnan(box_from_mask(np.zeros((5, 5), dtype=bool))).all())


class TestStateAccuracy(unittest.TestCase):
    def test_perfect_tracking_scores_one(self):
        gt = np.array([[0.0, 0.0, 10.0, 10.0]] * 4)
        self.assertAlmostEqual(state_accuracy(gt, gt), 1.0)

    def test_correctly_reporting_an_absent_target_scores_one(self):
        gt = np.array([ABSENT, ABSENT])
        pred = np.array([ABSENT, ABSENT])
        self.assertAlmostEqual(state_accuracy(pred, gt), 1.0)

    def test_hallucinating_a_box_on_an_absent_target_scores_zero(self):
        # This is the case mean-IoU-over-visible-frames cannot see at all.
        gt = np.array([ABSENT, ABSENT])
        pred = np.array([[0.0, 0.0, 10.0, 10.0]] * 2)
        self.assertAlmostEqual(state_accuracy(pred, gt), 0.0)

    def test_missing_a_visible_target_scores_zero_on_that_frame(self):
        gt = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
        pred = np.array([[0.0, 0.0, 10.0, 10.0], ABSENT])
        self.assertAlmostEqual(state_accuracy(pred, gt), 0.5)

    def test_mixed_sequence_matches_the_definition(self):
        gt = np.array([
            [0.0, 0.0, 10.0, 10.0],     # tracked exactly    -> 1.0
            [0.0, 0.0, 10.0, 10.0],     # half overlap       -> 1/3
            ABSENT,                     # absent, agreed     -> 1.0
            ABSENT,                     # absent, box emitted-> 0.0
        ])
        pred = np.array([
            [0.0, 0.0, 10.0, 10.0],
            [5.0, 0.0, 15.0, 10.0],
            ABSENT,
            [1.0, 1.0, 2.0, 2.0],
        ])
        self.assertAlmostEqual(state_accuracy(pred, gt), (1.0 + 1 / 3 + 1.0 + 0.0) / 4)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            state_accuracy(np.zeros((2, 4)), np.zeros((3, 4)))

    def test_empty_sequence_is_zero_not_nan(self):
        self.assertEqual(state_accuracy(np.zeros((0, 4)), np.zeros((0, 4))), 0.0)


class TestSuccessAuc(unittest.TestCase):
    def test_perfect_tracking_is_one(self):
        gt = np.array([[0.0, 0.0, 10.0, 10.0]] * 3)
        self.assertAlmostEqual(success_auc(gt, gt), 1.0)

    def test_ignores_frames_where_the_target_is_absent(self):
        gt = np.array([[0.0, 0.0, 10.0, 10.0], ABSENT])
        pred = np.array([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 2.0, 2.0]])
        self.assertAlmostEqual(success_auc(pred, gt), 1.0)

    def test_all_absent_is_zero(self):
        self.assertEqual(success_auc(np.array([ABSENT]), np.array([ABSENT])), 0.0)

    def test_auc_is_the_limit_of_the_plotted_curve(self):
        # The exact AUC and a fine threshold grid have to agree, or the number
        # reported is not the area under the curve that gets plotted.
        rng = np.random.default_rng(0)
        gt = np.tile([0.0, 0.0, 10.0, 10.0], (200, 1))
        pred = gt + np.concatenate(
            [rng.uniform(0, 5, (200, 2)), np.zeros((200, 2))], axis=1
        )
        grid, rates = success_curve(pred, gt, np.linspace(0, 1, 2001))
        self.assertAlmostEqual(success_auc(pred, gt), float(np.trapezoid(rates, grid)),
                               places=2)

    def test_curve_is_monotonically_non_increasing(self):
        gt = np.tile([0.0, 0.0, 10.0, 10.0], (5, 1))
        pred = np.array([[0.0, 0.0, 10.0, 10.0], [2.0, 0.0, 12.0, 10.0],
                         [5.0, 0.0, 15.0, 10.0], [8.0, 0.0, 18.0, 10.0], ABSENT])
        _, rates = success_curve(pred, gt)
        self.assertTrue(np.all(np.diff(rates) <= 0))


class TestDropoutEpisodes(unittest.TestCase):
    def _gt(self, n):
        return np.array([[0.0, 0.0, 10.0, 10.0]] * n)

    def test_no_dropouts_when_tracking_holds(self):
        gt = self._gt(5)
        self.assertEqual(dropout_episodes(gt, gt).episodes, 0)

    def test_one_long_loss_and_many_short_ones_are_told_apart(self):
        # Same number of lost frames, same mean IoU, completely different
        # failure. One episode of 6 is memory poisoning; six of 1 is noise.
        gt = self._gt(12)
        long_loss = np.array([gt[0]] * 3 + [ABSENT] * 6 + [gt[0]] * 3)
        short_losses = np.array(
            [gt[0] if i % 2 else ABSENT for i in range(12)]
        )

        one = dropout_episodes(long_loss, gt)
        many = dropout_episodes(short_losses, gt)

        self.assertEqual(one.episodes, 1)
        self.assertEqual(one.longest, 6)
        self.assertEqual(many.episodes, 6)
        self.assertEqual(many.longest, 1)
        self.assertEqual(one.frames, many.frames)

    def test_a_dropout_running_to_the_end_is_still_counted(self):
        gt = self._gt(5)
        pred = np.array([gt[0], gt[0], ABSENT, ABSENT, ABSENT])
        self.assertEqual(dropout_episodes(pred, gt).lengths, (3,))

    def test_a_missed_box_counts_as_lost_not_just_an_absent_prediction(self):
        gt = self._gt(3)
        pred = np.array([gt[0], [100.0, 100.0, 110.0, 110.0], gt[0]])
        self.assertEqual(dropout_episodes(pred, gt).lengths, (1,))

    def test_frames_where_the_target_is_absent_are_not_dropouts(self):
        gt = np.array([ABSENT] * 4)
        pred = np.array([ABSENT] * 4)
        self.assertEqual(dropout_episodes(pred, gt).episodes, 0)


class TestReport(unittest.TestCase):
    def test_totals_are_frame_weighted_not_sequence_averaged(self):
        # A short perfect clip must not outvote a long failing one.
        short_gt = np.array([[0.0, 0.0, 10.0, 10.0]] * 10)
        long_gt = np.array([[0.0, 0.0, 10.0, 10.0]] * 1000)
        scores = [
            score_sequence("short", short_gt, short_gt),
            score_sequence("long", np.array([ABSENT] * 1000), long_gt),
        ]
        text = report(scores)

        self.assertIn("all (2 seq)", text)
        self.assertIn("1010", text)
        # 10 frames at 1.0 and 1000 at 0.0 -> ~0.0099, not the 0.5 a
        # sequence-level average would give.
        self.assertIn("0.0099", text)

    def test_per_frame_iou_zero_when_only_one_side_is_present(self):
        gt = np.array([[0.0, 0.0, 10.0, 10.0], ABSENT])
        pred = np.array([ABSENT, [0.0, 0.0, 10.0, 10.0]])
        np.testing.assert_allclose(per_frame_iou(pred, gt), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
