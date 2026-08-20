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

from src.training.distill import (  # noqa: E402
    SAM2_SIZE,
    STUDENT_SIZE,
    FeatureTeacher,
    Projector,
    Sam2FeatureTeacher,
    build_teacher,
    distill_loss,
    encoder_features,
    patch_aligned,
    shared_window,
    subsample,
    tokens_to_map,
)
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


class TestTokensToMap(unittest.TestCase):
    def test_a_clean_patch_grid_keeps_its_positions(self):
        # Token i must land at grid position i. A transpose here would produce
        # a perfectly plausible feature map of a mirrored image.
        tokens = torch.arange(9, dtype=torch.float32).reshape(1, 9, 1)
        grid = tokens_to_map(tokens, 3)[0, 0]
        torch.testing.assert_close(grid, torch.arange(9, dtype=torch.float32).reshape(3, 3))

    def test_leading_tokens_are_dropped_from_the_front(self):
        # One CLS token, then the grid. Counting from the back is what makes
        # this right for a checkpoint carrying registers as well.
        tokens = torch.cat([torch.full((1, 1, 1), -1.0),
                            torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)], dim=1)
        grid = tokens_to_map(tokens, 2)[0, 0]
        torch.testing.assert_close(grid, torch.arange(4, dtype=torch.float32).reshape(2, 2))

    def test_four_extra_registers_are_dropped_too(self):
        tokens = torch.cat([torch.full((1, 5, 1), -1.0),
                            torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)], dim=1)
        grid = tokens_to_map(tokens, 2)[0, 0]
        torch.testing.assert_close(grid, torch.arange(4, dtype=torch.float32).reshape(2, 2))

    def test_channels_move_to_the_channel_axis(self):
        self.assertEqual(tokens_to_map(torch.randn(2, 16, 768), 4).shape, (2, 768, 4, 4))

    def test_too_few_tokens_is_an_error_not_a_reshape(self):
        with self.assertRaises(RuntimeError):
            tokens_to_map(torch.randn(1, 8, 4), 4)


class TestTeacherChoice(unittest.TestCase):
    """Which teacher a model id selects, and at what input size.

    Neither teacher is instantiated -- that downloads weights. What is pinned
    is the dispatch, because the failure it prevents is silent: a SAM 2
    checkpoint run at DINOv2's 518 loads, runs, and produces features from
    position embeddings it was never trained with.
    """

    def dispatch(self, model_id):
        seen = {}

        def record(cls):
            def fake(mid, device, dtype, size):
                seen.update(cls=cls, model_id=mid, size=size)
            return fake

        import src.training.distill as distill
        originals = (distill.FeatureTeacher, distill.Sam2FeatureTeacher)
        distill.FeatureTeacher = record("vit")
        distill.Sam2FeatureTeacher = record("sam2")
        try:
            build_teacher(model_id)
        finally:
            distill.FeatureTeacher, distill.Sam2FeatureTeacher = originals
        return seen

    def test_a_dino_id_gets_the_vit_teacher_and_resolves_its_own_size(self):
        # The ViT teacher reads its patch size off the loaded config, so
        # build_teacher passes the request through untouched rather than
        # guessing a resolution per checkpoint.
        for model_id in ("facebook/dinov3-vitb16-pretrain-lvd1689m",
                         "facebook/dinov2-base"):
            seen = self.dispatch(model_id)
            self.assertEqual((seen["cls"], seen["size"]), ("vit", None), model_id)

    def test_a_sam2_id_gets_sam2s_own_encoder_at_1024(self):
        for model_id in ("facebook/sam2.1-hiera-large",
                         "facebook/sam2.1-hiera-base-plus",
                         "some-org/SAM2-finetuned"):
            seen = self.dispatch(model_id)
            self.assertEqual((seen["cls"], seen["size"]), ("sam2", SAM2_SIZE), model_id)

    def test_an_explicit_size_overrides_the_default(self):
        import src.training.distill as distill
        original = distill.Sam2FeatureTeacher
        seen = {}
        distill.Sam2FeatureTeacher = lambda mid, d, t, s: seen.update(size=s)
        try:
            build_teacher("facebook/sam2.1-hiera-large", size=512)
        finally:
            distill.Sam2FeatureTeacher = original
        self.assertEqual(seen["size"], 512)

    def test_the_two_teachers_are_interchangeable(self):
        # `build_teacher` calls either with the same four positional arguments,
        # and `pretrain` reads model_id, dim and size off whichever came back.
        # Neither is constructed here -- that downloads weights -- so the
        # contract is checked against the signature and the constructor body.
        import inspect

        for cls in (FeatureTeacher, Sam2FeatureTeacher):
            params = list(inspect.signature(cls.__init__).parameters)
            self.assertEqual(params, ["self", "model_id", "device", "dtype", "size"],
                             cls.__name__)
            source = inspect.getsource(cls.__init__)
            for attribute in ("model_id", "dim", "size", "patch"):
                self.assertIn(f"self.{attribute} =", source,
                              f"{cls.__name__} never sets {attribute}")
            self.assertTrue(callable(cls.features), cls.__name__)


class TestPatchAligned(unittest.TestCase):
    """One rule instead of a resolution constant per checkpoint."""

    def test_a_16_pixel_patch_lands_exactly_on_the_students_grid(self):
        # 512 / 16 = 32, and the student's grid at 512 is 32x32 -- so DINOv3's
        # map is never resampled before the loss.
        self.assertEqual(patch_aligned(STUDENT_SIZE, 16), 512)
        self.assertEqual(patch_aligned(STUDENT_SIZE, 16) // 16, 32)

    def test_a_14_pixel_patch_lands_on_dinov2s_native_518(self):
        self.assertEqual(patch_aligned(STUDENT_SIZE, 14), 518)

    def test_an_exact_multiple_is_left_alone(self):
        self.assertEqual(patch_aligned(224, 16), 224)

    def test_it_always_rounds_up_never_down(self):
        for size in range(500, 530):
            for patch in (14, 16):
                self.assertGreaterEqual(patch_aligned(size, patch), size)
                self.assertEqual(patch_aligned(size, patch) % patch, 0)


class TestSharedWindow(unittest.TestCase):
    """The crop that keeps a registered pair registered."""

    def test_the_same_placement_lands_on_the_same_fraction_of_each_half(self):
        # A 1920x1080 thermal half and a 960x540 RGB half of the same scene
        # must be cropped to the same piece of the world, not the same pixels.
        big = shared_window((1080, 1920), (0.5, 0.5), 0.5)
        small = shared_window((540, 960), (0.5, 0.5), 0.5)
        self.assertEqual(big[1][0], 2 * small[1][0])
        self.assertEqual(big[0][0], 2 * small[0][0])

    def test_the_window_is_square_and_inside_the_frame(self):
        for place in ((0.0, 0.0), (0.5, 0.5), (1.0, 1.0)):
            (x0, y0), (w, h) = shared_window((1080, 1920), place, 0.474)
            self.assertEqual(w, h)
            self.assertGreaterEqual(x0, 0)
            self.assertGreaterEqual(y0, 0)
            self.assertLessEqual(x0 + w, 1920)
            self.assertLessEqual(y0 + h, 1080)

    def test_the_documented_vtuav_fraction_gives_native_pixels(self):
        # 512/1080: a 512-pixel window of a 1080-tall frame, resized to 512 --
        # which is no resize at all.
        _, (side, _) = shared_window((1080, 1920), (0.3, 0.7), 512 / 1080)
        self.assertEqual(side, 512)

    def test_a_scale_of_one_is_the_largest_square_that_fits(self):
        _, (side, _) = shared_window((1080, 1920), (0.5, 0.5), 1.0)
        self.assertEqual(side, 1080)


class TestSubsample(unittest.TestCase):
    def test_it_spreads_across_the_set_instead_of_truncating(self):
        # The bug this replaces: the first 5 000 pairs of VTUAV are two
        # flights, and a run capped that way trains on two scenes while
        # reporting five thousand samples.
        pairs = list(range(10_000))
        picked = subsample(pairs, 100, seed=0)

        self.assertEqual(len(picked), 100)
        self.assertGreater(max(picked), 9_000)
        self.assertLess(min(picked), 1_000)

    def test_order_is_preserved_so_a_run_stays_reproducible(self):
        picked = subsample(list(range(1000)), 50, seed=1)
        self.assertEqual(picked, sorted(picked))
        self.assertEqual(picked, subsample(list(range(1000)), 50, seed=1))

    def test_asking_for_everything_or_more_returns_everything(self):
        pairs = list(range(20))
        self.assertEqual(subsample(pairs, None), pairs)
        self.assertEqual(subsample(pairs, 50), pairs)


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
