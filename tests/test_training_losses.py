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
    instance_loss,
    iou_head_loss,
    neighbour_terms,
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


def two_objects(size=16):
    """One window, two separated blobs: the target and the neighbour."""
    target = torch.zeros(size, size, dtype=torch.bool)
    other = torch.zeros(size, size, dtype=torch.bool)
    target[4:8, 2:6] = True
    other[4:8, 10:14] = True
    return target, other


def logits_over(*masks, size=16):
    out = torch.full((size, size), -BIG)
    for mask in masks:
        out[mask] = BIG
    return out[None]


class TestNeighbourLeak(unittest.TestCase):
    """The dense-cluster term: what share of the objects beside it it claims."""

    def test_a_clean_mask_leaks_nothing(self):
        target, other = two_objects()
        claimed, penalty, crowded = neighbour_terms(logits_over(target),
                                                    other[None])
        self.assertTrue(bool(crowded[0]))
        self.assertAlmostEqual(float(claimed[0]), 0.0, places=5)
        self.assertAlmostEqual(float(penalty[0]), 0.0, places=5)

    def test_swallowing_the_neighbour_leaks_everything(self):
        target, other = two_objects()
        claimed, penalty, _ = neighbour_terms(logits_over(target, other),
                                              other[None])
        self.assertAlmostEqual(float(claimed[0]), 1.0, places=4)
        self.assertAlmostEqual(float(penalty[0]), BIG, places=3)

    def test_it_is_a_fraction_of_the_neighbour_not_of_the_window(self):
        """Half the neighbour is 0.5 whatever the window around it is.

        A term measured against the window would report the same failure as
        smaller on a bigger crop, which is exactly the comparison this project
        makes when it changes SIZE.
        """
        for size in (16, 64):
            target = torch.zeros(size, size, dtype=torch.bool)
            other = torch.zeros(size, size, dtype=torch.bool)
            target[4:8, 2:6] = True
            other[4:8, 10:14] = True
            half = torch.zeros(size, size, dtype=torch.bool)
            half[4:6, 10:14] = True
            claimed, _, _ = neighbour_terms(
                logits_over(target, half, size=size), other[None])
            self.assertAlmostEqual(float(claimed[0]), 0.5, places=4,
                                   msg=str(size))

    def test_an_instance_with_no_neighbour_is_flagged_not_scored_as_zero(self):
        """Isolated targets must not average the problem away.

        A set of lonely objects would otherwise report a leak of 0 and read as
        a model that separates well, when nothing has been asked of it.
        """
        target, _ = two_objects()
        empty = torch.zeros(16, 16, dtype=torch.bool)
        claimed, _, crowded = neighbour_terms(logits_over(target), empty[None])
        self.assertFalse(bool(crowded[0]))
        self.assertAlmostEqual(float(claimed[0]), 0.0, places=5)

    def test_the_trained_term_still_has_gradient_when_the_mask_is_confident(self):
        """Why the penalty is cross-entropy and not the fraction itself.

        A mask that is *confidently* swallowing the cluster is the failure. At
        a saturated logit `sigmoid` is flat, so the fraction hands back
        essentially no gradient exactly there; the cross-entropy's gradient is
        `sigmoid(logit)`, which is ~1.
        """
        target, other = two_objects()
        logits = logits_over(target, other).clone().requires_grad_(True)
        claimed, penalty, _ = neighbour_terms(logits, other[None])

        claimed.sum().backward(retain_graph=True)
        from_fraction = logits.grad[0][other].abs().max()
        logits.grad = None
        penalty.sum().backward()
        from_penalty = logits.grad[0][other].abs().max()

        self.assertLess(float(from_fraction), 1e-6)
        self.assertGreater(float(from_penalty), 1e-3)
        self.assertTrue(bool((logits.grad[0][target] == 0).all()))


class TestInstanceLossNeighbour(unittest.TestCase):
    def outputs(self, logits):
        return {"pred_masks_high_res": logits[:, None],
                "ious": torch.zeros(logits.shape[0], 1)}

    def test_the_leak_is_reported_at_weight_zero(self):
        """Measure before training against it -- the term is a diagnostic first."""
        target, other = two_objects()
        logits = logits_over(target, other)
        loss, terms = instance_loss(self.outputs(logits), target[None].float(),
                                    Weights(), other[None])
        without, _ = instance_loss(self.outputs(logits), target[None].float(),
                                   Weights())
        self.assertAlmostEqual(terms["neighbour"], 1.0, places=4)
        self.assertAlmostEqual(float(loss), float(without), places=6)

    def test_a_non_zero_weight_makes_the_swallowing_mask_cost_more(self):
        target, other = two_objects()
        weights = Weights(neighbour=5.0)
        swallowed, _ = instance_loss(
            self.outputs(logits_over(target, other)), target[None].float(),
            weights, other[None])
        clean, _ = instance_loss(
            self.outputs(logits_over(target)), target[None].float(),
            weights, other[None])
        self.assertGreater(float(swallowed) - float(clean), 4.9)

    def test_omitting_others_leaves_the_published_objective_untouched(self):
        """Every run before this term exists has to be reproducible."""
        target, _ = two_objects()
        logits = logits_over(target)
        loss, terms = instance_loss(self.outputs(logits), target[None].float())
        self.assertNotIn("neighbour", terms)
        self.assertTrue(torch.isfinite(loss))
