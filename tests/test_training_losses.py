"""The training objective, against hand-computed cases.

Torch only -- no EdgeTAM, no GPU. Every term is checked for the property it
exists to provide, not just for running without an exception.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.losses import (  # noqa: E402
    Weights,
    box_projection_loss,
    dice_loss,
    focal_loss,
    frame_loss,
    iou_head_loss,
    object_score_loss,
)

BIG = 20.0  # a logit saturated enough that sigmoid is 1.0 to float precision


def mask_with_box(shape, box):
    mask = torch.zeros(shape)
    x0, y0, x1, y1 = box
    mask[..., y0:y1, x0:x1] = 1.0
    return mask


class TestFocalLoss(unittest.TestCase):
    def test_perfect_prediction_is_near_zero(self):
        targets = mask_with_box((1, 1, 16, 16), (4, 4, 12, 12))
        logits = (targets * 2 - 1) * BIG
        self.assertLess(float(focal_loss(logits, targets)), 1e-6)

    def test_confidently_wrong_costs_far_more_than_uncertain(self):
        targets = mask_with_box((1, 1, 16, 16), (4, 4, 12, 12))
        wrong = (targets * 2 - 1) * -BIG
        uncertain = torch.zeros_like(targets)
        self.assertGreater(float(focal_loss(wrong, targets)),
                           10 * float(focal_loss(uncertain, targets)))

    def test_it_is_per_sample_not_reduced(self):
        targets = mask_with_box((3, 1, 8, 8), (2, 2, 6, 6))
        self.assertEqual(focal_loss(torch.zeros_like(targets), targets).shape, (3,))

    def test_predicting_all_background_still_costs_on_a_tiny_target(self):
        # The reason focal loss is here: one drone-sized blob in a 512 frame.
        # An unweighted BCE would be almost satisfied by predicting nothing.
        targets = mask_with_box((1, 1, 512, 512), (250, 250, 262, 260))
        background = torch.full_like(targets, -BIG)
        self.assertGreater(float(focal_loss(background, targets)), 0.0)


class TestDiceLoss(unittest.TestCase):
    def test_perfect_prediction_is_near_zero(self):
        targets = mask_with_box((1, 1, 16, 16), (4, 4, 12, 12))
        self.assertLess(float(dice_loss((targets * 2 - 1) * BIG, targets)), 1e-3)

    def test_disjoint_prediction_is_near_one(self):
        targets = mask_with_box((1, 1, 16, 16), (0, 0, 4, 4))
        predicted = mask_with_box((1, 1, 16, 16), (10, 10, 14, 14))
        self.assertGreater(float(dice_loss((predicted * 2 - 1) * BIG, targets)), 0.9)

    def test_half_overlap_matches_the_formula(self):
        # 2*|A n B| / (|A| + |B|) with 8 shared of 16 + 16.
        targets = mask_with_box((1, 1, 16, 16), (0, 0, 4, 4))
        predicted = mask_with_box((1, 1, 16, 16), (2, 0, 6, 4))
        loss = float(dice_loss((predicted * 2 - 1) * BIG, targets, eps=0.0))
        self.assertAlmostEqual(loss, 1 - 2 * 8 / 32, places=4)


class TestIouHeadLoss(unittest.TestCase):
    def test_a_head_reporting_the_truth_pays_nothing(self):
        targets = mask_with_box((1, 1, 16, 16), (4, 4, 12, 12))
        logits = (targets * 2 - 1) * BIG
        self.assertLess(float(iou_head_loss(torch.ones(1, 1), logits, targets)), 1e-4)

    def test_a_head_overstating_its_quality_is_penalised(self):
        targets = mask_with_box((1, 1, 16, 16), (0, 0, 4, 4))
        logits = (mask_with_box((1, 1, 16, 16), (10, 10, 14, 14)) * 2 - 1) * BIG
        self.assertAlmostEqual(float(iou_head_loss(torch.ones(1, 1), logits, targets)),
                               1.0, places=3)

    def test_no_gradient_flows_through_the_target(self):
        targets = mask_with_box((1, 1, 8, 8), (2, 2, 6, 6))
        logits = torch.zeros_like(targets, requires_grad=True)
        pred_iou = torch.full((1, 1), 0.5, requires_grad=True)
        iou_head_loss(pred_iou, logits, targets).sum().backward()

        self.assertIsNotNone(pred_iou.grad)
        # The mask must not be pushed around to make the IoU report easier.
        self.assertTrue(logits.grad is None or float(logits.grad.abs().sum()) == 0.0)


class TestObjectScoreLoss(unittest.TestCase):
    def test_confident_and_correct_costs_nothing(self):
        logits = torch.tensor([[BIG], [-BIG]])
        exist = torch.tensor([1.0, 0.0])
        self.assertLess(float(object_score_loss(logits, exist).max()), 1e-6)

    def test_confidently_claiming_a_departed_target_is_expensive(self):
        loss = object_score_loss(torch.tensor([[BIG]]), torch.tensor([0.0]))
        self.assertGreater(float(loss), 10.0)

    def test_it_is_per_sample(self):
        logits = torch.zeros(4, 1)
        self.assertEqual(object_score_loss(logits, torch.ones(4)).shape, (4,))


class TestBoxProjectionLoss(unittest.TestCase):
    def test_a_mask_exactly_filling_its_box_is_near_zero(self):
        box = torch.tensor([[4.0, 6.0, 12.0, 10.0]])
        logits = (mask_with_box((1, 1, 16, 16), (4, 6, 12, 10)) * 2 - 1) * BIG
        self.assertLess(float(box_projection_loss(logits, box)), 0.1)

    def test_a_thin_diagonal_inside_the_box_also_passes(self):
        # The honest limit of a box label: projections cannot see shape. This
        # is asserted so the weakness is documented rather than assumed away.
        box = torch.tensor([[0.0, 0.0, 8.0, 8.0]])
        mask = torch.zeros(1, 1, 16, 16)
        for i in range(8):
            mask[0, 0, i, i] = 1.0
        self.assertLess(float(box_projection_loss((mask * 2 - 1) * BIG, box)), 0.1)

    def test_a_mask_spilling_outside_the_box_is_penalised(self):
        box = torch.tensor([[4.0, 4.0, 8.0, 8.0]])
        logits = (mask_with_box((1, 1, 16, 16), (0, 0, 16, 16)) * 2 - 1) * BIG
        self.assertGreater(float(box_projection_loss(logits, box)), 0.5)

    def test_an_empty_mask_is_penalised(self):
        box = torch.tensor([[4.0, 4.0, 8.0, 8.0]])
        self.assertGreater(
            float(box_projection_loss(torch.full((1, 1, 16, 16), -BIG), box)), 0.5)

    def test_it_is_per_sample(self):
        boxes = torch.tensor([[0.0, 0.0, 4.0, 4.0], [8.0, 8.0, 12.0, 12.0]])
        self.assertEqual(box_projection_loss(torch.zeros(2, 1, 16, 16), boxes).shape, (2,))


def outputs_for(masks, ious=None, object_logits=None):
    n = masks.shape[0]
    return {
        "pred_masks_high_res": (masks.unsqueeze(1) * 2 - 1) * BIG,
        "ious": torch.ones(n, 1) if ious is None else ious,
        "object_score_logits": torch.full((n, 1), BIG) if object_logits is None
        else object_logits,
    }


class TestFrameLoss(unittest.TestCase):
    def test_a_perfectly_tracked_labelled_frame_costs_almost_nothing(self):
        masks = mask_with_box((2, 16, 16), (4, 4, 12, 12))
        loss, terms = frame_loss(outputs_for(masks), masks, None, torch.ones(2))

        self.assertLess(float(loss), 0.05)
        self.assertIn("mask", terms)
        self.assertNotIn("box_projection", terms)

    def test_unlabelled_frames_fall_back_to_the_box_term(self):
        masks = mask_with_box((1, 16, 16), (4, 4, 12, 12))
        boxes = torch.tensor([[4.0, 4.0, 12.0, 12.0]])
        loss, terms = frame_loss(outputs_for(masks), None, boxes, torch.ones(1))

        self.assertIn("box_projection", terms)
        self.assertNotIn("mask", terms)
        self.assertLess(float(loss), 0.2)

    def test_a_batch_mixing_labelled_and_unlabelled_uses_both(self):
        # The decision is per sample. Collapsing it to the batch would either
        # throw away a good mask or invent one that does not exist.
        masks = mask_with_box((2, 16, 16), (4, 4, 12, 12))
        masks[1] = 0  # sample 1 has no teacher mask
        boxes = torch.tensor([[4.0, 4.0, 12.0, 12.0]] * 2)
        _, terms = frame_loss(outputs_for(masks), masks, boxes, torch.ones(2))

        self.assertIn("mask", terms)
        self.assertIn("box_projection", terms)

    def test_an_absent_target_is_scored_only_on_the_object_head(self):
        masks = torch.zeros(1, 16, 16)
        outputs = outputs_for(masks, object_logits=torch.tensor([[-BIG]]))
        loss, terms = frame_loss(outputs, masks, None, torch.zeros(1))

        self.assertEqual(set(terms), {"object_score"})
        self.assertLess(float(loss), 1e-5)

    def test_claiming_a_departed_target_dominates_the_loss(self):
        # The failure this project actually hit, expressed as a cost.
        masks = torch.zeros(1, 16, 16)
        outputs = outputs_for(masks, object_logits=torch.tensor([[BIG]]))
        loss, _ = frame_loss(outputs, masks, None, torch.zeros(1))
        self.assertGreater(float(loss), 10.0)

    def test_weights_are_honoured(self):
        masks = mask_with_box((1, 16, 16), (4, 4, 12, 12))
        outputs = outputs_for(masks, object_logits=torch.zeros(1, 1))
        light, _ = frame_loss(outputs, masks, None, torch.ones(1),
                              Weights(object_score=1.0))
        heavy, _ = frame_loss(outputs, masks, None, torch.ones(1),
                              Weights(object_score=10.0))
        self.assertGreater(float(heavy), float(light))

    def test_gradients_reach_the_mask_logits(self):
        masks = mask_with_box((1, 16, 16), (4, 4, 12, 12))
        logits = torch.zeros(1, 1, 16, 16, requires_grad=True)
        outputs = {"pred_masks_high_res": logits, "ious": torch.ones(1, 1),
                   "object_score_logits": torch.zeros(1, 1)}
        frame_loss(outputs, masks, None, torch.ones(1))[0].backward()

        self.assertIsNotNone(logits.grad)
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
