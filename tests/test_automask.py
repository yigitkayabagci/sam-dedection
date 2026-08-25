"""The mask arm: what gets hidden, what is scored, and whether it collapsed.

The claims this arm rests on are all checkable without a GPU or a trained
model. That exactly `ratio` of the grid is hidden in every mode, so the five
modes differ in *which* cells and in nothing else. That the saliency-driven
mode really does pick the most distinctive cells, and that `hint` really does
leave the very top visible. That the loss carries gradient only where the
pixels were blanked. And that the collapse detector fires on a collapsed map,
which is the one failure a falling loss curve hides completely.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.automask import (  # noqa: E402
    MODES,
    TARGETS,
    Target,
    blank,
    collapse,
    green_noise,
    masked_loss,
    saliency,
    sample_mask,
    summarise_collapse,
)


class TestSaliency(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.features = torch.randn(3, 16, 8, 8)

    def test_both_scores_produce_one_number_per_cell(self):
        for mode in ("distinct", "norm"):
            self.assertEqual(tuple(saliency(self.features, mode).shape), (3, 8, 8))

    def test_an_unknown_score_is_refused_rather_than_defaulted(self):
        with self.assertRaises(ValueError):
            saliency(self.features, "attention")

    def test_distinct_is_scale_free_and_norm_is_not(self):
        """A conv trunk has no attention map, so `distinct` stands in for it:
        how far a position's feature points away from the frame's average
        direction. Feature *magnitude* varies for reasons that have nothing to
        do with content, which is why it is the alternative rather than the
        default."""
        louder = self.features * 7.0
        self.assertTrue(torch.allclose(saliency(self.features, "distinct"),
                                       saliency(louder, "distinct"), atol=1e-4))
        self.assertFalse(torch.allclose(saliency(self.features, "norm"),
                                        saliency(louder, "norm"), atol=1e-3))

    def test_a_uniform_map_has_no_distinctive_cell(self):
        flat = torch.ones(2, 4, 6, 6)
        self.assertLess(float(saliency(flat, "distinct").abs().max()), 1e-4)


class TestMask(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.scores = torch.randn(4, 8, 8)

    def test_every_mode_hides_the_same_number_of_cells(self):
        """The modes have to differ in *which* cells and in nothing else, or
        the comparison between them also varies the loss's denominator."""
        for mode in MODES:
            mask = sample_mask(self.scores, ratio=0.5, mode=mode)
            self.assertEqual(tuple(mask.shape), (4, 8, 8), mode)
            self.assertEqual(mask.dtype, torch.bool, mode)
            for row in mask:
                self.assertEqual(int(row.sum()), 32, mode)

    def test_the_count_follows_the_ratio(self):
        for ratio, wanted in ((0.25, 16), (0.5, 32), (0.75, 48)):
            mask = sample_mask(self.scores, ratio=ratio, mode="random")
            self.assertEqual(int(mask[0].sum()), wanted)

    def test_a_ratio_outside_the_open_unit_interval_is_refused(self):
        for bad in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                sample_mask(self.scores, ratio=bad, mode="random")

    def test_an_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            sample_mask(self.scores, mode="attention")

    def test_attentive_hides_exactly_the_most_salient_cells(self):
        mask = sample_mask(self.scores, ratio=0.25, mode="attentive")
        flat = self.scores.reshape(4, 64)
        for row in range(4):
            hidden = flat[row][mask[row].reshape(64)]
            visible = flat[row][~mask[row].reshape(64)]
            self.assertGreater(float(hidden.min()), float(visible.max()))

    def test_hint_leaves_the_very_top_visible(self):
        """AttMask's own remedy for a task that becomes unsolvable rather than
        hard: something of the object is always left."""
        ratio, hint = 0.5, 0.25
        mask = sample_mask(self.scores, ratio=ratio, mode="hint", hint=hint)
        flat = self.scores.reshape(4, 64)
        for row in range(4):
            order = flat[row].argsort(descending=True)
            revealed = order[:int(round(hint * 32))]
            self.assertFalse(bool(mask[row].reshape(64)[revealed].any()))

    def test_block_masking_produces_contiguity_that_scattered_cells_do_not(self):
        """A scattered single-cell mask is 16 source pixels wide and a 3x3
        kernel two layers up already spans it."""
        torch.manual_seed(2)
        scattered = sample_mask(torch.randn(1, 16, 16), ratio=0.5, mode="random")
        blocks = sample_mask(torch.randn(1, 16, 16), ratio=0.5, mode="block",
                             block=4)

        def neighbours(mask):
            m = mask.float()
            return float((m[:, :-1, :] * m[:, 1:, :]).sum()
                         + (m[:, :, :-1] * m[:, :, 1:]).sum())

        self.assertGreater(neighbours(blocks), neighbours(scattered))

    def test_a_generator_makes_the_random_modes_reproducible(self):
        for mode in ("random", "green"):
            first = sample_mask(self.scores, mode=mode,
                                generator=torch.Generator().manual_seed(7))
            again = sample_mask(self.scores, mode=mode,
                                generator=torch.Generator().manual_seed(7))
            self.assertTrue(torch.equal(first, again), mode)

    def test_green_noise_is_a_field_the_size_of_the_grid(self):
        field = green_noise((3, 8, 8), generator=torch.Generator().manual_seed(0))
        self.assertEqual(tuple(field.shape), (3, 8, 8))
        self.assertTrue(torch.isfinite(field).all())

    def test_green_masking_ignores_the_scores_entirely(self):
        """That is the point of it: ColorMAE costs no parameters and no extra
        forward pass, and it is what a learned mask has to beat to have earned
        its own."""
        one = sample_mask(self.scores, mode="green",
                          generator=torch.Generator().manual_seed(3))
        other = sample_mask(torch.randn(4, 8, 8), mode="green",
                            generator=torch.Generator().manual_seed(3))
        self.assertTrue(torch.equal(one, other))


class TestBlank(unittest.TestCase):
    def test_the_mask_tiles_the_image_at_the_strides_ratio(self):
        images = torch.randn(2, 3, 64, 64)
        mask = sample_mask(torch.randn(2, 4, 4), ratio=0.5, mode="random")
        blanked = blank(images, mask)
        self.assertEqual(blanked.shape, images.shape)
        grown = mask.repeat_interleave(16, 1).repeat_interleave(16, 2)
        self.assertTrue(torch.equal(blanked[:, 0][grown],
                                    torch.zeros_like(blanked[:, 0][grown])))
        self.assertTrue(torch.equal(blanked[:, 0][~grown], images[:, 0][~grown]))

    def test_the_fill_is_the_mean_pixel_not_black(self):
        """Zero after ImageNet normalisation *is* the mean pixel -- the least
        informative value available. Filling with black would be filling with
        a strong, learnable signal."""
        blanked = blank(torch.randn(1, 3, 32, 32),
                        torch.ones(1, 2, 2, dtype=torch.bool))
        self.assertEqual(float(blanked.abs().max()), 0.0)

    def test_a_mask_that_does_not_tile_the_image_is_refused(self):
        with self.assertRaises(ValueError):
            blank(torch.randn(1, 3, 30, 30), torch.ones(1, 4, 4, dtype=torch.bool))


class TestMaskedLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.target = torch.randn(2, 8, 6, 6)
        self.mask = sample_mask(torch.randn(2, 6, 6), ratio=0.5, mode="random")

    def test_only_the_masked_positions_carry_gradient(self):
        """The unmasked positions are the ones whose pixels the student can
        see, so agreeing there is a copy rather than an inference -- and a
        dense convolution can see across a mask boundary, which is exactly
        what it must not be rewarded for."""
        student = torch.randn(2, 8, 6, 6, requires_grad=True)
        loss, _ = masked_loss(student, self.target, self.mask)
        loss.backward()
        per_cell = student.grad.abs().sum(dim=1)
        self.assertGreater(float(per_cell[self.mask].min()), 0.0)
        self.assertEqual(float(per_cell[~self.mask].abs().max()), 0.0)

    def test_the_visible_cosine_is_reported_but_not_optimised(self):
        _, terms = masked_loss(torch.randn(2, 8, 6, 6), self.target, self.mask)
        self.assertEqual(set(terms), {"masked", "visible"})

    def test_a_perfect_student_scores_zero_on_the_whitened_target(self):
        from src.training.pretrain import centre

        loss, terms = masked_loss(centre(self.target), self.target, self.mask)
        self.assertAlmostEqual(float(loss), 0.0, places=5)
        self.assertAlmostEqual(terms["visible"], 0.0, places=5)

    def test_whitening_can_be_turned_off_and_changes_the_number(self):
        with_norm, _ = masked_loss(torch.randn(2, 8, 6, 6), self.target, self.mask)
        without, _ = masked_loss(torch.randn(2, 8, 6, 6), self.target, self.mask,
                                 whiten=False)
        self.assertNotAlmostEqual(float(with_norm), float(without), places=5)

    def test_disagreeing_grids_are_refused_rather_than_resampled(self):
        with self.assertRaises(ValueError):
            masked_loss(torch.randn(2, 8, 3, 3), self.target, self.mask)


class TestCollapse(unittest.TestCase):
    def test_a_constant_map_reads_as_collapsed(self):
        """The failure a loss curve hides completely: the loss falls to zero
        while every position becomes the same vector."""
        stats = collapse(torch.ones(2, 8, 6, 6))
        self.assertGreater(stats["similarity"], 0.99)
        self.assertLess(stats["channel_std"], 1e-5)

    def test_a_healthy_map_has_positions_that_disagree(self):
        torch.manual_seed(4)
        stats = collapse(torch.randn(2, 32, 8, 8))
        self.assertLess(abs(stats["similarity"]), 0.3)
        self.assertGreater(stats["channel_std"], 0.5)

    def test_it_subsamples_rather_than_building_a_quadratic_matrix(self):
        stats = collapse(torch.randn(1, 4, 40, 40), positions=16)
        self.assertTrue(all(v == v for v in stats.values()))   # not NaN

    def test_the_verdict_names_collapse_drift_and_stability(self):
        collapsed = summarise_collapse([
            {"epoch": 0, "similarity": 0.2, "channel_std": 1.0},
            {"epoch": 1, "similarity": 0.97, "channel_std": 0.01}])
        self.assertIn("COLLAPSED", collapsed)

        drifting = summarise_collapse([
            {"epoch": 0, "similarity": 0.10, "channel_std": 1.0},
            {"epoch": 1, "similarity": 0.40, "channel_std": 0.8}])
        self.assertIn("drifting", drifting)

        steady = summarise_collapse([
            {"epoch": 0, "similarity": 0.10, "channel_std": 1.0},
            {"epoch": 1, "similarity": 0.12, "channel_std": 1.0}])
        self.assertIn("stable", steady)

    def test_a_run_with_no_statistics_says_so_rather_than_indexing_nothing(self):
        self.assertIn("no collapse", summarise_collapse([{"epoch": 0, "loss": 1.0}]))


class Tiny(nn.Module):
    """Stands in for the encoder: `Target` deep-copies and EMAs parameters, and
    neither operation cares what the module computes."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4))


class TestTarget(unittest.TestCase):
    def test_the_two_targets_are_the_two_the_design_argues_for(self):
        self.assertEqual(set(TARGETS), {"ema", "frozen"})

    def test_a_frozen_target_never_moves(self):
        """It cannot collapse, by construction -- the objective becomes
        recovering the stock encoder's clean features from a masked frame."""
        model = Tiny()
        target = Target(model, mode="frozen")
        with torch.no_grad():
            model.weight.fill_(5.0)
        target.update(model)
        self.assertEqual(float(target.model.weight.abs().max()), 0.0)

    def test_an_ema_target_moves_toward_the_student_at_its_decay(self):
        model = Tiny()
        target = Target(model, decay=0.9, mode="ema")
        with torch.no_grad():
            model.weight.fill_(1.0)
        target.update(model)
        self.assertAlmostEqual(float(target.model.weight[0]), 0.1, places=5)
        target.update(model)
        self.assertAlmostEqual(float(target.model.weight[0]), 0.19, places=5)

    def test_the_target_is_never_optimised(self):
        target = Target(Tiny(), mode="ema")
        self.assertFalse(any(p.requires_grad for p in target.model.parameters()))
        self.assertFalse(target.model.training)

    def test_an_unknown_target_is_refused(self):
        with self.assertRaises(ValueError):
            Target(Tiny(), mode="teacher")


if __name__ == "__main__":
    unittest.main()
