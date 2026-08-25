"""Modality distillation: the parts that do not need a foundation model.

`FeatureTeacher` is a `transformers` call and is not tested here -- what is
tested is everything the method's *claim* rests on, which is independent of
which teacher is used: that the student's features come out of the tensor the
deployment actually exports, that the loss measures direction rather than
scale, and that a teacher on a different grid is brought to the student's
rather than the reverse.
"""
from __future__ import annotations

import inspect
import sys
import textwrap
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.distill import (  # noqa: E402
    CONVNEXT_IDS,
    DINO_ID,
    Pair,
    collate_pairs,
    SAM2_SIZE,
    STUDENT_SIZE,
    TEACHER_ID,
    FeatureTeacher,
    Projector,
    Sam2FeatureTeacher,
    build_teacher,
    distill_loss,
    encoder_features,
    moment_loss,
    patch_aligned,
    probe_geometry,
    shared_window,
    teacher_input,
    teacher_maps,
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

    def test_the_moment_term_is_off_unless_asked_for(self):
        student, teacher = torch.randn(2, 8, 4, 4), torch.randn(2, 8, 4, 4)
        self.assertNotIn("moments", distill_loss(student, teacher)[1])
        self.assertIn("moments", distill_loss(student, teacher, moment_weight=1.0)[1])

    def test_the_moment_term_catches_a_channel_the_cosine_term_cannot(self):
        # The failure it exists for: a student that matches direction position
        # by position while letting half its channels go constant across the
        # map. Those channels carry nothing for the decoder to read, and the
        # per-position cosine never notices.
        torch.manual_seed(0)
        teacher = torch.randn(4, 32, 8, 8)
        collapsed = teacher.clone()
        collapsed[:, 16:] = collapsed[:, 16:].mean(dim=(2, 3), keepdim=True)

        self.assertAlmostEqual(float(moment_loss(*(F.normalize(x.float(), dim=1)
                                                   for x in (teacher, teacher)))),
                               0.0, places=6)
        self.assertGreater(
            float(moment_loss(*(F.normalize(x.float(), dim=1)
                                for x in (collapsed, teacher)))), 1e-4)

    def test_a_globally_scaled_teacher_is_the_same_field(self):
        # The moment term runs on the normalised maps on purpose, so a teacher
        # scaled by a constant costs nothing -- matching a foundation model's
        # absolute magnitude would drag the student away from the scale
        # EdgeTAM's own decoder was trained to read.
        feats = torch.randn(2, 16, 6, 6)
        self.assertAlmostEqual(
            float(distill_loss(feats, feats * 7.0, moment_weight=1.0)[0]),
            0.0, places=5)

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


class TestConvolutionalTeacher(unittest.TestCase):
    """A teacher that returns a map, not tokens -- DINOv3's ConvNeXt students."""

    class FakeConvModel:
        dtype = torch.float32

        def __call__(self, pixel_values):
            batch = pixel_values.shape[0]
            out = type("Output", (), {})()
            # Stride 32, the ConvNeXt geometry: 512 in, 16x16 out, [B, C, h, w].
            out.last_hidden_state = torch.randn(batch, 768, 16, 16)
            return out

    def test_a_4d_hidden_state_is_the_feature_map_itself(self):
        teacher = FeatureTeacher.__new__(FeatureTeacher)
        teacher.model = self.FakeConvModel()
        teacher.size, teacher.patch = 512, 14      # patch is a ViT notion; unused here
        maps = teacher.features(torch.randn(2, 3, 512, 512))
        self.assertEqual(tuple(maps.shape), (2, 768, 16, 16))
        self.assertEqual(maps.dtype, torch.float32)

    def test_its_grid_still_meets_the_student_through_the_loss(self):
        # Stride 32 against the student's 16: distill_loss resamples the
        # teacher to the student's grid, so nothing upstream has to care.
        teacher_map = torch.randn(1, 8, 16, 16)
        student_map = torch.randn(1, 8, 32, 32)
        loss, _ = distill_loss(student_map, teacher_map)
        self.assertTrue(torch.isfinite(loss))


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

    def test_the_default_teacher_is_sam2_and_the_vits_default_is_a_vit(self):
        # Two defaults, and they are allowed to differ. `TEACHER_ID` is what
        # the pipeline runs -- SAM 2, because the task is class-agnostic single
        # object tracking and EdgeTAM's own trunk is a fit to SAM 2's stride-16
        # map. `FeatureTeacher`'s default has to stay a ViT whatever that is
        # set to, or constructing the ViT class directly would hand a Hiera
        # checkpoint to AutoModel and fail somewhere further in.
        self.assertEqual(self.dispatch(TEACHER_ID)["cls"], "sam2")
        self.assertEqual(self.dispatch(DINO_ID)["cls"], "vit")
        signature = inspect.signature(FeatureTeacher.__init__)
        self.assertEqual(signature.parameters["model_id"].default, DINO_ID)

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
            # Matched on the assignment *target* rather than on the text
            # `self.x =`: FeatureTeacher now unpacks its geometry from
            # `probe_geometry` in one statement, which is still an assignment
            # to self.patch and would fail a substring check.
            import ast

            tree = ast.parse(textwrap.dedent(inspect.getsource(cls.__init__)))
            assigned = {
                node.attr
                for statement in ast.walk(tree)
                if isinstance(statement, ast.Assign)
                for target in statement.targets
                for node in ast.walk(target)
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name) and node.value.id == "self"
            }
            for attribute in ("model_id", "dim", "size", "patch"):
                self.assertIn(attribute, assigned,
                              f"{cls.__name__} never sets {attribute}")
            self.assertTrue(callable(cls.features), cls.__name__)


class Tokens:
    """A stand-in backbone that returns `prefix + (S/stride)^2` tokens.

    Enough to exercise `probe_geometry`, which is the whole point: the thing
    being tested is arithmetic over token counts, and downloading a foundation
    model to check arithmetic would mean it never got checked.
    """

    def __init__(self, stride: int, prefix: int, channels: int = 64,
                 four_d: bool = False) -> None:
        self.stride, self.prefix, self.channels, self.four_d = (
            stride, prefix, channels, four_d)
        self.dtype = torch.float32

    def __call__(self, pixel_values):
        side = int(pixel_values.shape[-1]) // self.stride
        if self.four_d:
            return type("Out", (), {
                "last_hidden_state": torch.zeros(1, self.channels, side, side)})
        count = self.prefix + side * side
        return type("Out", (), {
            "last_hidden_state": torch.zeros(1, count, self.channels)})


class TestProbeGeometry(unittest.TestCase):
    """Read off the config this goes wrong quietly: DINOv3's ConvNeXt configs
    carry no `patch_size` at all, so a `getattr(config, "patch_size", 14)`
    answers 14 for a trunk whose real stride is 32."""

    def test_it_recovers_the_stride_and_the_prefix_of_a_token_backbone(self):
        for stride, prefix in ((16, 5), (16, 1), (32, 1), (8, 0), (14, 5)):
            found = probe_geometry(Tokens(stride, prefix), device="cpu")
            self.assertEqual(found[0], stride, f"stride {stride}")
            self.assertEqual(found[1], prefix, f"prefix {prefix} at {stride}")

    def test_dinov3s_convnext_shape_specifically(self):
        # 4x stem then three 2x downsamples, and one pooled token in front.
        stride, prefix, channels = probe_geometry(Tokens(32, 1, 768), device="cpu")
        self.assertEqual((stride, prefix, channels), (32, 1, 768))

    def test_a_backbone_that_returns_a_map_needs_no_arithmetic(self):
        self.assertEqual(probe_geometry(Tokens(16, 0, 256, four_d=True),
                                        device="cpu"), (16, 0, 256))

    def test_a_backbone_that_does_not_scale_as_a_grid_says_so(self):
        class Odd(Tokens):
            def __call__(self, pixel_values):
                return type("Out", (), {"last_hidden_state": torch.zeros(1, 7, 8)})

        with self.assertRaises(RuntimeError) as caught:
            probe_geometry(Odd(16, 0), device="cpu")
        self.assertIn("stride", str(caught.exception))


class TestTeacherInput(unittest.TestCase):
    """One rule: never *upsample* the teacher to the student's grid."""

    def test_a_stride_16_teacher_runs_at_the_students_own_resolution(self):
        self.assertEqual(teacher_input(512, 16), 512)
        self.assertEqual(teacher_input(512, 16) // 16, 512 // 16)

    def test_a_stride_14_teacher_keeps_the_old_snap_up_rule(self):
        # Finer than the student, so downsampling it loses nothing the student
        # could have represented anyway.
        self.assertEqual(teacher_input(512, 14), 518)

    def test_a_stride_32_convnext_is_fed_twice_the_input(self):
        """Fed 512 it would produce 16x16 against the student's 32x32, and the
        loss would upsample it -- supervising four student cells with one
        teacher cell, which is the resolution that matters when a target is
        twenty pixels across."""
        self.assertEqual(teacher_input(512, 32), 1024)
        self.assertEqual(teacher_input(512, 32) // 32, 512 // 16)

    def test_the_teachers_grid_is_never_coarser_than_the_students(self):
        # The invariant, stated as an inequality rather than an equality: a
        # teacher finer than the student is left alone and downsampled at the
        # loss, which loses nothing the student could have represented. A
        # teacher coarser than the student is what the rule exists to prevent.
        for stride in (8, 14, 16, 32, 64):
            self.assertGreaterEqual(teacher_input(512, stride) // stride, 32,
                                    f"stride {stride}")


class TestTeacherMaps(unittest.TestCase):
    def test_one_layer_needs_nothing_but_features(self):
        class Single:
            def features(self, images):
                return torch.zeros(1, 8, 4, 4)

        maps = teacher_maps(Single(), torch.zeros(1, 3, 64, 64), layers=1)
        self.assertEqual(len(maps), 1)

    def test_asking_a_single_map_teacher_for_blocks_is_refused(self):
        """`Sam2FeatureTeacher` reads an FPN level rather than a block, so
        "the last two layers" has no meaning for it -- and answering with one
        map twice would be a constraint that is not one."""
        class Single:
            def features(self, images):
                return torch.zeros(1, 8, 4, 4)

        with self.assertRaises(ValueError):
            teacher_maps(Single(), torch.zeros(1, 3, 64, 64), layers=2)


class TestConvNextTeachers(unittest.TestCase):
    def test_the_four_convnext_students_are_named_not_spelled_out(self):
        self.assertEqual(sorted(CONVNEXT_IDS), ["base", "large", "small", "tiny"])
        for size, model_id in CONVNEXT_IDS.items():
            self.assertTrue(model_id.startswith("facebook/dinov3-convnext-"), size)
            self.assertIn(size, model_id)

    def test_build_teacher_does_not_route_a_convnext_to_the_sam_class(self):
        # Dispatch is on the id, and "sam2" is the only substring that means
        # the SAM class; a dinov3 id must never reach it.
        for model_id in CONVNEXT_IDS.values():
            self.assertNotIn("sam2", model_id.lower())


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


class TestBorderCrop(unittest.TestCase):
    """DroneVehicle pads both halves with 100 px of pure white per side.

    Measured on the real archive, not read off the paper: every image in both
    modalities is 840x712, and the picture inside the band is 640x512 -- this
    project's native resolution. Left in, the band is a third of the frame, so
    the teacher spends a third of its patch tokens describing white and
    `shared_window` places crops that are partly margin.

    PNG rather than JPEG for the fixtures: a hard white edge is exactly what
    JPEG rings around, and the ringing would be indistinguishable from the
    margin leaking through.
    """

    INTERIOR = 60

    def frames(self, tmp, border=100, inner=(512, 640)):
        import cv2
        import numpy as np

        height, width = inner[0] + 2 * border, inner[1] + 2 * border
        paths = []
        for name in ("a.png", "b.png"):
            image = np.full((height, width, 3), 255, dtype=np.uint8)
            image[border:height - border, border:width - border] = self.INTERIOR
            path = Path(tmp) / name
            cv2.imwrite(str(path), image)
            paths.append(path)
        return [(paths[0], paths[1])]

    def unique(self, half):
        return half.flatten().unique()

    def test_the_white_band_is_gone_from_both_halves(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            batch = collate_pairs(self.frames(tmp), size=64, device="cpu",
                                  border=100)
        # A constant interior normalises to one value per channel, so three
        # distinct numbers and no more. White surviving anywhere would add a
        # fourth.
        for name, half in (("thermal", batch.thermal), ("rgb", batch.rgb)):
            self.assertEqual(len(self.unique(half)), 3,
                             f"{name}: {self.unique(half)}")

    def test_without_the_border_the_white_survives(self):
        # The check above is only meaningful if the same call keeps the band
        # when it is not asked to crop -- otherwise it could be passing because
        # the resize washed the margin out.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            kept = collate_pairs(self.frames(tmp), size=64, device="cpu",
                                 border=0)
        self.assertGreater(len(self.unique(kept.thermal)), 3)

    def test_a_crop_lands_inside_the_picture_wherever_it_is_placed(self):
        # `shared_window` works in the inset frame, so no placement can reach
        # the margin -- including the extremes, which is where it would.
        import tempfile

        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            pairs = self.frames(tmp)
            class Fixed:                      # a Generator's `random` is read-only
                def __init__(self, value):
                    self.value = value

                def random(self, shape):
                    return np.full(shape, self.value)

            for place in (0.0, 0.5, 1.0):
                batch = collate_pairs(pairs, size=64, device="cpu", crop=1.0,
                                      rng=Fixed(place), border=100)
                self.assertEqual(len(self.unique(batch.thermal)), 3,
                                 f"placement {place} reached the margin")

    def test_a_border_that_would_eat_the_image_is_an_error(self):
        # Better than silently returning an empty crop, which is what a spec
        # whose border does not match the download would otherwise produce.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            pairs = self.frames(tmp, border=10, inner=(20, 20))   # 40x40
            with self.assertRaises(ValueError):
                collate_pairs(pairs, size=64, device="cpu", border=100)


class TestMixedSources(unittest.TestCase):
    """Two datasets in one stage-A batch, needing opposite treatment.

    VTUAV is 1920x1080 with no border and has to be cropped to keep native
    pixels; DroneVehicle is 840x712 with a 100 px white border and is already
    640x512 underneath, so cropping it would throw resolution away. One `crop`
    and one `border` for the batch cannot express both, and getting it wrong is
    silent: either a frame is squashed into a square or the teacher reads
    margin.
    """

    def make(self, tmp, name, size, border, value):
        import cv2
        import numpy as np

        height, width = size
        image = np.full((height, width, 3), 255, dtype=np.uint8)
        image[border:height - border, border:width - border] = value
        path = Path(tmp) / name
        cv2.imwrite(str(path), image)
        return path

    def mixed(self, tmp):
        # A bordered 840x712 (interior 640x512) and a clean tall 300x600.
        bordered = [self.make(tmp, f"b{i}.png", (712, 840), 100, 60)
                    for i in range(2)]
        plain = [self.make(tmp, f"p{i}.png", (600, 300), 0, 90) for i in range(2)]
        return [Pair(bordered[0], bordered[1], None, 100),
                Pair(plain[0], plain[1], 128 / 300, 0)]

    def test_each_pair_is_read_by_its_own_rules(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            batch = collate_pairs(self.mixed(tmp), size=64, device="cpu")
        self.assertEqual(batch.thermal.shape, (2, 3, 64, 64))
        # Neither sample may contain white: the bordered one because its border
        # was cropped, the plain one because it has none.
        for index in range(2):
            self.assertEqual(len(batch.thermal[index].flatten().unique()), 3,
                             f"sample {index} carries more than its interior")

    def test_the_batch_level_arguments_are_only_a_fallback(self):
        # Passing border=0 must not undo a Pair that says 100 -- otherwise
        # mixing sources would silently depend on argument order.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            batch = collate_pairs(self.mixed(tmp), size=64, device="cpu",
                                  crop=None, border=0)
        self.assertEqual(len(batch.thermal[0].flatten().unique()), 3)

    def test_plain_tuples_still_behave_as_before(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            a = self.make(tmp, "a.png", (712, 840), 100, 60)
            b = self.make(tmp, "b.png", (712, 840), 100, 60)
            kept = collate_pairs([(a, b)], size=64, device="cpu")
            cropped = collate_pairs([(a, b)], size=64, device="cpu", border=100)
        self.assertGreater(len(kept.thermal.flatten().unique()), 3)
        self.assertEqual(len(cropped.thermal.flatten().unique()), 3)


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
