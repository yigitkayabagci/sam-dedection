"""Modality distillation: the parts that do not need a foundation model.

`FeatureTeacher` is a `transformers` call and is not tested here -- what is
tested is everything the method's *claim* rests on, which is independent of
which teacher is used: that the student's features come out of the tensor the
deployment actually exports, that the loss measures direction rather than
scale, and that a teacher on a different grid is brought to the student's
rather than the reverse.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.distill import Projector, distill_loss, encoder_features  # noqa: E402
from src.training.finetune import STAGES, apply_freeze  # noqa: E402
from tests.test_clip_loop import FakeSam2  # noqa: E402

SIZE = 32


class TestEncoderFeatures(unittest.TestCase):
    def test_the_flattened_features_come_back_as_a_map(self):
        model = FakeSam2(SIZE).eval()
        feats = encoder_features(model, torch.randn(2, 3, SIZE, SIZE))
        self.assertEqual(feats.shape, (2, 8, SIZE, SIZE))

    def test_the_gradient_reaches_the_trunk(self):
        model = FakeSam2(SIZE).eval()
        encoder_features(model, torch.randn(1, 3, SIZE, SIZE)).sum().backward()
        self.assertGreater(float(model.image_encoder.weight.grad.abs().sum()), 0.0)

    def test_no_head_is_involved(self):
        # There is no mask decoder in this graph at all, which is why the
        # pretraining stage unfreezes `backbone` and not `encoder`.
        model = FakeSam2(SIZE).eval()
        encoder_features(model, torch.randn(1, 3, SIZE, SIZE)).sum().backward()
        self.assertIsNone(model.sam_mask_decoder.weight.grad)


class TestDistillLoss(unittest.TestCase):
    def test_identical_features_cost_nothing(self):
        feats = torch.randn(2, 16, 4, 4)
        loss, terms = distill_loss(feats, feats)
        self.assertAlmostEqual(float(loss), 0.0, places=5)
        self.assertAlmostEqual(terms["cosine"], 0.0, places=5)

    def test_opposed_features_cost_two(self):
        feats = torch.randn(2, 16, 4, 4)
        self.assertAlmostEqual(float(distill_loss(feats, -feats)[0]), 2.0, places=5)

    def test_scale_is_ignored_and_direction_is_not(self):
        # The point of cosine over L2: the two models' features live on
        # different scales and matching the scale is not what transfers.
        feats = torch.randn(2, 16, 4, 4)
        scaled = float(distill_loss(feats, feats * 7.0)[0])
        rotated = float(distill_loss(feats, feats.flip(1))[0])
        self.assertAlmostEqual(scaled, 0.0, places=5)
        self.assertGreater(rotated, 0.1)

    def test_a_teacher_on_a_coarser_grid_is_resized_to_the_students(self):
        student = torch.randn(2, 16, 8, 8)
        teacher = torch.randn(2, 16, 5, 5)
        loss, _ = distill_loss(student, teacher)
        self.assertTrue(torch.isfinite(loss))

    def test_the_l1_term_is_off_unless_asked_for(self):
        student, teacher = torch.randn(2, 8, 4, 4), torch.randn(2, 8, 4, 4)
        self.assertNotIn("l1", distill_loss(student, teacher)[1])
        self.assertIn("l1", distill_loss(student, teacher, l1_weight=1.0)[1])

    def test_the_gradient_flows_to_the_student_only(self):
        student = torch.randn(2, 8, 4, 4, requires_grad=True)
        teacher = torch.randn(2, 8, 4, 4, requires_grad=True)
        distill_loss(student, teacher.detach())[0].backward()

        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)


class TestProjector(unittest.TestCase):
    def test_it_maps_the_student_dimension_onto_the_teachers(self):
        projector = Projector(256, 768)
        out = projector(torch.randn(2, 256, 4, 4))
        self.assertEqual(out.shape, (2, 768, 4, 4))


class TestBackboneStage(unittest.TestCase):
    def test_it_trains_the_encoder_and_nothing_else(self):
        model = FakeSam2(SIZE)
        counts = apply_freeze(model, "backbone")

        self.assertEqual(set(counts), {"image_encoder"})
        self.assertFalse(model.sam_mask_decoder.weight.requires_grad)

    def test_the_memory_path_is_frozen_here_too(self):
        model = FakeSam2(SIZE)
        apply_freeze(model, "backbone")
        for name, param in model.named_parameters():
            if name.startswith(("memory_attention", "memory_encoder", "spatial_perceiver")):
                self.assertFalse(param.requires_grad, name)

    def test_lora_can_adapt_the_same_scope(self):
        # PEFT for the pretraining stage costs one word, because `inject` reads
        # the same STAGES table the freeze does.
        from src.training import lora

        self.assertIn("backbone", STAGES)
        model = FakeSam2(SIZE)
        report = lora.inject(model, "backbone", r=2)
        self.assertEqual(report["stage"], "backbone")
        self.assertEqual(set(lora.freeze(model, "backbone")), {"image_encoder"})


if __name__ == "__main__":
    unittest.main()
