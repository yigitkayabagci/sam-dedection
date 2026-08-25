"""Stage A's hub: the routing, the index, the losses and the handoff.

No foundation model and no EdgeTAM here. What is tested is everything the
stage's *claims* rest on and that a Colab run cannot check for you: that a
corpus routed wrongly fails loudly rather than contributing nothing, that the
student's half of an RGB corpus really is single-channel, that the teacher's
window follows the student's except where jitter is asked for, that each loss
is zero on identical inputs and positive otherwise, and that all four arms
leave one kind of file behind.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.pretrain import (  # noqa: E402
    ARMS,
    DEFAULT_ROUTE,
    LOSSES,
    MODALITIES,
    ROUTES,
    STUDENTS,
    Batch,
    Corpus,
    Item,
    baseline,
    centre,
    channel_normalise,
    feature_loss,
    for_student,
    frequency_loss,
    from_pixels,
    gram_loss,
    index,
    photometric,
    read_manifest,
    stage_b_command,
    subsample,
    summarise,
    summarise_gap,
    to_luminance,
    to_pixels,
    with_teacher,
    write_manifest,
)

try:
    import cv2  # noqa: F401 -- the index and the batcher decode real files
    HAVE_CV2 = True
except ImportError:  # pragma: no cover - environment, not logic
    HAVE_CV2 = False


def write_images(folder: Path, count: int, seed: int = 0,
                 shape: tuple[int, int] = (64, 80), colour: bool = True,
                 stem: str = "") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(count):
        size = (*shape, 3) if colour else shape
        frame = (rng.random(size) * 255).astype(np.uint8)
        cv2.imwrite(str(folder / f"{stem}{i:04d}.png"), frame)


class TestCorpus(unittest.TestCase):
    def test_every_modality_has_a_default_route_and_every_route_is_known(self):
        # A modality with no default would fail only for the person who added
        # it, at index time, on a dataset that had already downloaded.
        self.assertEqual(sorted(DEFAULT_ROUTE), sorted(MODALITIES))
        for modality, route in DEFAULT_ROUTE.items():
            self.assertIn(route, ROUTES, modality)

    def test_a_typo_in_a_modality_or_route_or_spec_is_refused_at_construction(self):
        with self.assertRaises(ValueError):
            Corpus("x", "/tmp", "rgbt")
        with self.assertRaises(ValueError):
            Corpus("x", "/tmp", "paired", route="rgb->depth")
        with self.assertRaises(ValueError):
            Corpus("x", "/tmp", "paired", spec="no-such-set")

    def test_a_spec_fills_the_globs_and_an_explicit_value_wins(self):
        borrowed = Corpus("k", "/tmp", "paired", spec="kust4k")
        thermal, rgb = borrowed.globs()
        self.assertTrue(thermal and rgb)
        overridden = Corpus("k", "/tmp", "paired", spec="kust4k",
                            thermal="**/other/*.png")
        self.assertEqual(overridden.globs()[0], "**/other/*.png")
        self.assertEqual(overridden.globs()[1], rgb)

    def test_the_border_comes_from_the_spec_and_zero_is_not_none(self):
        # DroneVehicle's 100 px white band is the case this exists for, and
        # `border=0` has to mean "no border" rather than "ask the spec".
        self.assertEqual(Corpus("d", "/tmp", "paired", spec="dronevehicle").padding(),
                         100)
        self.assertEqual(
            Corpus("d", "/tmp", "paired", spec="dronevehicle", border=0).padding(), 0)

    def test_a_route_asking_for_a_half_the_modality_lacks_is_refused(self):
        """The silent failure this prevents: a thermal-only corpus routed
        cross-modally finds no RGB, pairs nothing, and contributes zero items
        to a run that prints a total and looks healthy."""
        with self.assertRaises(ValueError) as caught:
            Corpus("t", "/tmp", "thermal", thermal="**/*.png",
                   route="rgb->thermal").check()
        self.assertIn("has none", str(caught.exception))

        with self.assertRaises(ValueError):
            Corpus("r", "/tmp", "rgb", rgb="**/*.png",
                   route="rgb->thermal").check()

    def test_the_default_routes_are_the_ones_the_design_argues_for(self):
        # paired: teacher on RGB, student on thermal and single-channel.
        # rgb:    teacher on colour, student on the same file in luminance.
        # thermal: both sides on the thermal frame.
        self.assertEqual(Corpus("a", "/tmp", "paired").routing, "rgb->thermal")
        self.assertEqual(Corpus("b", "/tmp", "rgb").routing, "rgb->gray")
        self.assertEqual(Corpus("c", "/tmp", "thermal").routing, "thermal->thermal")
        student, gray, teacher, teacher_gray = ROUTES["rgb->gray"]
        self.assertEqual((student, teacher), ("rgb", "rgb"))
        self.assertTrue(gray)
        self.assertFalse(teacher_gray)


class TestForStudent(unittest.TestCase):
    """Two models are two pretrainings, and the difference between them is
    entirely in what the student's half is."""

    def setUp(self):
        self.rows = [
            Corpus("paired", "/a", "paired", spec="dronevehicle"),
            Corpus("colour", "/b", "rgb", rgb="**/*.jpg"),
            Corpus("heat", "/c", "thermal", thermal="**/*.jpg"),
        ]

    def test_the_two_students_are_the_two_the_design_argues_for(self):
        self.assertEqual(set(STUDENTS), {"thermal", "rgb"})

    def test_a_thermal_student_keeps_every_corpus_and_its_default_route(self):
        kept = for_student(self.rows, "thermal", log=lambda *_: None)
        self.assertEqual([c.routing for c in kept],
                         ["rgb->thermal", "rgb->gray", "thermal->thermal"])

    def test_an_rgb_student_gets_same_modality_distillation_everywhere(self):
        """The canonical published recipe: teacher and student on the identical
        view, the student trained to reproduce the teacher's map."""
        kept = for_student(self.rows, "rgb", log=lambda *_: None)
        self.assertEqual([c.name for c in kept], ["paired", "colour"])
        self.assertTrue(all(c.routing == "rgb->rgb" for c in kept))
        for corpus in kept:
            student, gray, teacher, teacher_gray = ROUTES[corpus.routing]
            self.assertEqual((student, teacher), ("rgb", "rgb"))
            self.assertFalse(gray or teacher_gray)

    def test_a_paired_corpus_contributes_only_its_rgb_half_to_an_rgb_student(self):
        kept = for_student(self.rows, "rgb", log=lambda *_: None)
        self.assertEqual(kept[0].wants(), ("rgb", "rgb"))
        kept[0].check()                       # and the route is satisfiable

    def test_dropping_a_thermal_corpus_is_announced_not_silent(self):
        """A corpus list that silently shrinks is how a run reports a sample
        count nobody checked."""
        said = []
        for_student(self.rows, "rgb", log=said.append)
        self.assertTrue(any("heat" in line for line in said))

    def test_rerouting_twice_changes_nothing(self):
        # The notebook reroutes before indexing and the CLI reroutes again from
        # the JSON it was handed; the second pass has to be a no-op.
        once = for_student(self.rows, "rgb", log=lambda *_: None)
        twice = for_student(once, "rgb", log=lambda *_: None)
        self.assertEqual([(c.name, c.routing) for c in once],
                         [(c.name, c.routing) for c in twice])

    def test_an_all_thermal_list_refuses_rather_than_returning_nothing(self):
        with self.assertRaises(SystemExit):
            for_student([self.rows[2]], "rgb", log=lambda *_: None)

    def test_an_unknown_student_is_refused(self):
        with self.assertRaises(ValueError):
            for_student(self.rows, "visible")


@unittest.skipUnless(HAVE_CV2, "indexing reads real image files")
class TestIndex(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        write_images(self.root / "pair" / "ir", 6, seed=1, colour=False)
        write_images(self.root / "pair" / "rgb", 6, seed=2)
        write_images(self.root / "colour" / "images", 5, seed=3)
        write_images(self.root / "heat" / "ir", 4, seed=4, colour=False)

    def corpora(self):
        return [
            Corpus("pair", self.root / "pair", "paired",
                   thermal="**/ir/*.png", rgb="**/rgb/*.png"),
            Corpus("colour", self.root / "colour", "rgb", rgb="**/images/*.png"),
            Corpus("heat", self.root / "heat", "thermal", thermal="**/ir/*.png"),
        ]

    def test_each_modality_lands_on_the_files_its_route_names(self):
        items = {i.corpus: i for i in index(self.corpora(), size=32, log=lambda *_: None)}
        self.assertEqual(sorted(items), ["colour", "heat", "pair"])

        # paired reads two different files, and the student's is the thermal one
        self.assertNotEqual(items["pair"].student, items["pair"].teacher)
        self.assertIn("/ir/", items["pair"].student.as_posix())
        self.assertIn("/rgb/", items["pair"].teacher.as_posix())
        self.assertTrue(items["pair"].student_gray)
        self.assertFalse(items["pair"].teacher_gray)

        # rgb reads one file twice, grey for the student and colour for the teacher
        self.assertEqual(items["colour"].student, items["colour"].teacher)
        self.assertTrue(items["colour"].student_gray)
        self.assertFalse(items["colour"].teacher_gray)

        # thermal reads one file twice, grey on both sides
        self.assertEqual(items["heat"].student, items["heat"].teacher)
        self.assertTrue(items["heat"].student_gray and items["heat"].teacher_gray)

    def test_a_dropped_route_keeps_the_corpus_for_the_mask_arm_only(self):
        rows = self.corpora()
        rows[2] = Corpus("heat", self.root / "heat", "thermal",
                         thermal="**/ir/*.png", route="drop")
        items = index(rows, size=32, log=lambda *_: None)
        heat = [i for i in items if i.corpus == "heat"]
        self.assertTrue(heat)
        self.assertTrue(all(i.teacher is None for i in heat))
        # ...and the teacher arm then sees the rest, and says what it skipped.
        said = []
        usable = with_teacher(items, log=said.append)
        self.assertEqual(len(usable), len(items) - len(heat))
        self.assertIn("heat", " ".join(said))

    def test_the_crop_is_derived_from_the_pictures_that_are_there(self):
        """The most error-prone hand-set knob in the stage, and the one whose
        wrong value is invisible: too high on a large source and every small
        target is resampled away before training sees it."""
        items = index(self.corpora(), size=32, log=lambda *_: None)
        # 64x80 images at size 32 -> shorter side 64 -> 32/64
        self.assertAlmostEqual(items[0].crop, 0.5, places=6)
        # and at a size the source is already below, no crop at all
        big = index(self.corpora(), size=128, log=lambda *_: None)
        self.assertIsNone(big[0].crop)

    def test_frames_the_dataset_marks_unusable_are_dropped_on_both_paths(self):
        """Kust4K ships five manifests naming **1 160 of its 4 024 frames** as
        broken -- 29 % -- where one modality is corrupt. They decode fine and
        are simply not pictures of the scene. `list_pairs` already drops them;
        a corpus routed `thermal->thermal` reads a flat listing instead and
        would otherwise train on them silently."""
        (self.root / "heat" / "broken_in_night.txt").write_text(
            "0000.png\n0002\n")
        said = []
        rows = [Corpus("heat", self.root / "heat", "thermal",
                       thermal="**/ir/*.png", crop=None,
                       exclude="broken_in_*.txt")]
        items = index(rows, size=32, log=said.append)
        stems = {i.student.stem for i in items}
        self.assertEqual(stems, {"0001", "0003"})
        self.assertTrue(any("unusable" in line for line in said))

    def test_the_exclusion_glob_is_inherited_from_a_named_spec(self):
        # The real case: `spec="kust4k"` must bring `broken_in_*.txt` with it,
        # so the row that reads simplest is also the row that is correct.
        self.assertEqual(Corpus("k", "/tmp", "paired", spec="kust4k").excludes(),
                         "broken_in_*.txt")
        self.assertEqual(Corpus("x", "/tmp", "rgb", rgb="**/*.png").excludes(), "")

    def test_a_missing_root_and_an_empty_glob_are_reported_not_raised(self):
        said = []
        rows = [Corpus("gone", self.root / "nope", "rgb", rgb="**/*.png"),
                Corpus("empty", self.root / "colour", "rgb", rgb="**/none/*.png")]
        self.assertEqual(index(rows, size=32, log=said.append), [])
        self.assertEqual(sum("!!" in line for line in said), 2)

    def test_limit_spreads_across_a_corpus_rather_than_truncating_it(self):
        rows = [Corpus("colour", self.root / "colour", "rgb",
                       rgb="**/images/*.png", limit=3)]
        items = index(rows, size=32, log=lambda *_: None)
        self.assertEqual(len(items), 3)
        # Most of these sets come from video, so the first N files are two
        # flights: a capped run must not be a prefix.
        stems = [i.student.stem for i in items]
        self.assertEqual(stems, sorted(stems))
        self.assertNotEqual(stems, ["0000", "0001", "0002"])

    def test_summarise_counts_what_has_a_teacher_separately(self):
        rows = self.corpora()
        rows[2] = Corpus("heat", self.root / "heat", "thermal",
                         thermal="**/ir/*.png", route="drop")
        text = summarise(index(rows, size=32, log=lambda *_: None))
        self.assertIn("total", text)
        self.assertIn("with teacher", text)


@unittest.skipUnless(HAVE_CV2, "the batcher decodes real image files")
class TestCollate(unittest.TestCase):
    def setUp(self):
        from src.training.pretrain import collate

        self.collate = collate
        self.root = Path(tempfile.mkdtemp())
        write_images(self.root / "pair" / "ir", 4, seed=5, colour=False)
        write_images(self.root / "pair" / "rgb", 4, seed=6)
        self.items = index(
            [Corpus("pair", self.root / "pair", "paired",
                    thermal="**/ir/*.png", rgb="**/rgb/*.png")],
            size=32, log=lambda *_: None)

    def test_both_halves_come_back_at_the_models_input_size(self):
        batch = self.collate(self.items, size=32)
        self.assertEqual(tuple(batch.student.shape), (4, 3, 32, 32))
        self.assertEqual(tuple(batch.teacher.shape), (4, 3, 32, 32))
        self.assertEqual(len(batch), 4)

    def test_the_students_half_is_one_channel_replicated(self):
        """Not a formality: the checkpoint's stem takes three channels, so the
        single thermal channel is replicated rather than the stem rewritten."""
        # Compared in *pixel* space: the replication happens before
        # `normalise`, which then applies a different ImageNet mean and
        # standard deviation to each of the three channels.
        batch = self.collate(self.items, size=32)
        first = to_pixels(batch.student)[0]
        self.assertTrue(torch.allclose(first[0], first[1], atol=1e-5))
        self.assertTrue(torch.allclose(first[1], first[2], atol=1e-5))
        # the teacher's is not
        teacher = to_pixels(batch.teacher)[0]
        self.assertFalse(torch.allclose(teacher[0], teacher[1], atol=1e-3))

    def test_wanting_a_teacher_that_is_not_there_says_so_rather_than_guessing(self):
        blind = [i._replace(teacher=None) for i in self.items]
        with self.assertRaises(ValueError) as caught:
            self.collate(blind, size=32, want_teacher=True)
        self.assertIn("no teacher half", str(caught.exception))
        # ...and the mask arm reads the same index without complaint
        self.assertIsNone(self.collate(blind, size=32, want_teacher=False).teacher)

    def test_without_jitter_the_two_halves_see_the_same_window(self):
        # Same placement seed, so any difference is the window and not the draw.
        quiet = self.collate(self.items, size=32,
                             rng=np.random.default_rng(0), jitter=0)
        again = self.collate(self.items, size=32,
                             rng=np.random.default_rng(0), jitter=0)
        self.assertTrue(torch.equal(quiet.teacher, again.teacher))
        moved = self.collate(self.items, size=32,
                             rng=np.random.default_rng(0), jitter=6)
        # The student is never jittered -- only the teacher's window moves.
        self.assertTrue(torch.equal(quiet.student, moved.student))
        self.assertFalse(torch.equal(quiet.teacher, moved.teacher))

    def test_jitter_does_not_touch_a_corpus_whose_halves_are_one_file(self):
        """A `rgb->gray` corpus reads one image twice and is registered by
        construction; jittering it would be inventing a problem."""
        same = [i._replace(teacher=i.student, teacher_gray=False) for i in self.items]
        quiet = self.collate(same, size=32, rng=np.random.default_rng(0), jitter=0)
        moved = self.collate(same, size=32, rng=np.random.default_rng(0), jitter=6)
        self.assertTrue(torch.equal(quiet.teacher, moved.teacher))

    def test_a_batch_moves_to_a_device_with_or_without_a_teacher(self):
        batch = self.collate(self.items, size=32)
        self.assertEqual(batch.to("cpu").teacher.shape, batch.teacher.shape)
        blind = Batch(student=batch.student, teacher=None)
        self.assertIsNone(blind.to("cpu").teacher)


class TestLosses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.student = torch.randn(2, 8, 6, 6)
        self.other = torch.randn(2, 8, 6, 6)

    def test_the_loss_names_are_the_ones_the_cli_offers(self):
        self.assertEqual(set(LOSSES), {"cosine", "feature", "frequency"})

    def test_centre_removes_the_per_position_offset_and_nothing_else(self):
        moved = centre(self.student)
        self.assertEqual(moved.shape, self.student.shape)
        self.assertLess(float(moved.mean(dim=1).abs().max()), 1e-4)

    def test_channel_normalise_leaves_each_channel_zero_mean_unit_norm(self):
        normalised = channel_normalise(self.student).flatten(2)
        self.assertLess(float(normalised.mean(dim=2).abs().max()), 1e-5)
        self.assertLess(float((normalised.norm(dim=2) - 1).abs().max()), 1e-4)

    def test_the_frequency_loss_is_zero_on_identical_maps_and_positive_otherwise(self):
        same, terms = frequency_loss(self.student, self.student.clone())
        self.assertAlmostEqual(float(same), 0.0, places=6)
        self.assertEqual(set(terms), {"low", "high"})
        different, _ = frequency_loss(self.student, self.other)
        self.assertGreater(float(different), 0.0)

    def test_the_frequency_loss_is_harder_on_the_low_band_than_the_high_one(self):
        """The whole point of the split: two modalities' features diverge far
        more in the high band than the low, and weighting both equally is what
        made plain MSE score below not distilling at all."""
        strict, _ = frequency_loss(self.student, self.other, high_weight=0.0)
        forgiving, _ = frequency_loss(self.student, self.other, high_weight=0.1)
        self.assertGreater(float(forgiving), float(strict))
        # ...and the high band's contribution is saturating, so raising its
        # weight tenfold does not raise the loss tenfold.
        loud, _ = frequency_loss(self.student, self.other, high_weight=1.0)
        self.assertLess(float(loud), 10 * float(forgiving))

    def test_the_gram_term_is_zero_on_identical_maps_and_ignores_the_channel_count(self):
        self.assertAlmostEqual(
            float(gram_loss(self.student, self.student.clone())), 0.0, places=6)
        wide = torch.randn(2, 32, 6, 6)
        self.assertGreater(float(gram_loss(self.student, wide)), 0.0)

    def test_the_gram_term_survives_a_rotation_of_the_teachers_channels(self):
        """A relational loss constrains the pattern of agreements between
        positions, not the coordinates -- which is why it needs no projector
        and why it tolerates a teacher trained by somebody else."""
        rotation, _ = torch.linalg.qr(torch.randn(8, 8))
        turned = torch.einsum("ij,bjhw->bihw", rotation, self.other)
        self.assertAlmostEqual(float(gram_loss(self.student, self.other)),
                               float(gram_loss(self.student, turned)), places=4)

    def test_the_feature_loss_resamples_the_teacher_to_the_students_grid(self):
        """The student's resolution is fixed by the deployment; upsampling it
        to compare would supervise positions the model never produces."""
        coarse = torch.randn(2, 8, 3, 3)
        value, terms = feature_loss(self.student, coarse)
        self.assertTrue(np.isfinite(float(value)))
        self.assertEqual(set(terms), {"smooth_l1", "cosine"})
        wide, _ = frequency_loss(self.student, coarse)
        self.assertTrue(np.isfinite(float(wide)))


class TestPixels(unittest.TestCase):
    def test_normalised_pixels_round_trip(self):
        pixels = torch.rand(2, 3, 8, 8)
        self.assertTrue(torch.allclose(to_pixels(from_pixels(pixels)), pixels,
                                       atol=1e-5))

    def test_luminance_collapses_colour_and_keeps_three_channels(self):
        colour = from_pixels(torch.rand(2, 3, 8, 8))
        grey = to_pixels(to_luminance(colour))
        self.assertEqual(grey.shape, colour.shape)
        self.assertTrue(torch.allclose(grey[:, 0], grey[:, 1], atol=1e-5))
        self.assertTrue(torch.allclose(grey[:, 1], grey[:, 2], atol=1e-5))

    def test_a_photometric_nudge_stays_inside_the_valid_range(self):
        pixels = to_pixels(photometric(from_pixels(torch.rand(4, 3, 8, 8))))
        self.assertGreaterEqual(float(pixels.min()), -1e-5)
        self.assertLessEqual(float(pixels.max()), 1 + 1e-5)


class TestModalityGap(unittest.TestCase):
    """The probe's arithmetic and its verdict, without a foundation model."""

    def test_the_verdict_reads_the_thermal_row_against_chance_and_the_floor(self):
        near_chance = summarise_gap({"floor": 0.95, "luminance": 0.9,
                                     "thermal": 0.15, "chance": 0.05,
                                     "samples": 64, "teacher": "t"})
        self.assertIn("near chance", near_chance)
        self.assertIn("mask arm", near_chance)

        middling = summarise_gap({"floor": 0.95, "luminance": 0.9,
                                  "thermal": 0.45, "chance": 0.05,
                                  "samples": 64, "teacher": "t"})
        self.assertIn("partial", middling)

        healthy = summarise_gap({"floor": 0.95, "luminance": 0.9,
                                 "thermal": 0.80, "chance": 0.05,
                                 "samples": 64, "teacher": "t"})
        self.assertIn("close to the floor", healthy)

    def test_a_partial_report_prints_what_it_has_without_inventing_a_verdict(self):
        text = summarise_gap({"floor": 0.9, "samples": 8, "teacher": "t"})
        self.assertIn("0.9000", text)
        self.assertNotIn("->", text)

    def test_a_flat_scale_does_not_divide_by_zero(self):
        text = summarise_gap({"floor": 0.5, "thermal": 0.5, "chance": 0.5,
                              "luminance": 0.5, "samples": 8, "teacher": "t"})
        self.assertIn("%", text)


class TestHandoff(unittest.TestCase):
    """All four arms leave one kind of file, including the one that trains
    nothing -- a baseline reached by a different code path is a baseline that
    drifts."""

    def test_the_none_arm_copies_the_stock_weights_and_writes_a_manifest(self):
        work = Path(tempfile.mkdtemp())
        stock = work / "edgetam.pt"
        stock.write_bytes(b"not really a checkpoint")
        out = baseline(stock, work / "staged" / "stage_a_none.pt",
                       {"image_size": 512, "seed": 0})
        self.assertEqual(out.read_bytes(), stock.read_bytes())
        manifest = read_manifest(out)
        self.assertEqual(manifest["arm"], "none")
        self.assertEqual(manifest["steps"], 0)
        self.assertEqual(manifest["checkpoint"], "stage_a_none.pt")

    def test_a_manifest_travels_beside_its_checkpoint_and_survives_json(self):
        work = Path(tempfile.mkdtemp())
        target = work / "stage_a_distil.pt"
        target.write_bytes(b"x")
        path = write_manifest(target, {"arm": "distil", "root": Path("/data")})
        self.assertEqual(path.name, "stage_a_distil.stage_a.json")
        self.assertEqual(json.loads(path.read_text())["root"], "/data")

    def test_a_checkpoint_with_no_manifest_reads_as_empty_rather_than_raising(self):
        self.assertEqual(read_manifest(Path(tempfile.mkdtemp()) / "nothing.pt"), {})

    def test_the_stage_b_command_names_the_checkpoint_it_was_given(self):
        line = stage_b_command("/drive/stage_a_mask.pt")
        self.assertIn("train_encoder.py", line)
        self.assertIn("--base /drive/stage_a_mask.pt", line)

    def test_the_arms_are_the_four_the_design_argues_for(self):
        self.assertEqual(set(ARMS), {"none", "mask", "distil", "both"})


class TestSampling(unittest.TestCase):
    def test_subsample_spreads_and_is_reproducible_from_its_seed(self):
        pool = list(range(100))
        picked = subsample(pool, 10, seed=3)
        self.assertEqual(len(picked), 10)
        self.assertEqual(picked, sorted(picked))
        self.assertNotEqual(picked, pool[:10])
        self.assertEqual(picked, subsample(pool, 10, seed=3))

    def test_subsample_returns_everything_when_the_cap_is_not_binding(self):
        pool = list(range(5))
        self.assertEqual(subsample(pool, None), pool)
        self.assertEqual(subsample(pool, 99), pool)

    def test_the_probes_chunker_does_not_swallow_a_short_list(self):
        """Dropping the tail is right for training and wrong for a probe: a
        run asked for 64 samples with a batch of 8 would otherwise average
        over an empty list when the index happened to hold 7."""
        from src.training.pretrain import _chunks

        self.assertEqual(list(_chunks([1, 2, 3], 8)), [[1, 2, 3]])
        self.assertEqual(list(_chunks([], 8)), [])
        self.assertEqual(list(_chunks(list(range(9)), 4)), [[0, 1, 2, 3], [4, 5, 6, 7]])


class TestItem(unittest.TestCase):
    def test_an_item_is_a_flat_tuple_so_a_batch_can_mix_corpora(self):
        """A 1920x1080 set and a bordered 640x512 one share a batch only
        because each sample carries its own crop and border."""
        item = Item(Path("a.png"), Path("b.png"), 0.5, 100, "mixed")
        self.assertEqual(item[0], Path("a.png"))
        self.assertEqual(item.crop, 0.5)
        self.assertEqual(item.border, 100)
        self.assertTrue(item.student_gray)
        self.assertFalse(item.teacher_gray)


if __name__ == "__main__":
    unittest.main()
