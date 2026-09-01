"""The consecutive-run viewer for one sequence.

What this tool is for is judging, by eye, whether a store's masks are real and
whether they stay on one object. Every part of that judgement rests on the
viewer having picked the right frames and paired each one with its own mask, so
those are the parts pinned here: a run broken by a gap is two runs, the longest
run wins over an earlier short one, frame *i* is drawn with mask *i* and never
mask *i+1*, and the area share is of the whole frame -- the number that says
"small target" -- and not of the mask's own box, which would be near 1 always.

No network and no downloads: every fixture is a handful of numpy arrays written
to a temporary directory.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import cv2
except ImportError:                                   # pragma: no cover
    cv2 = None

from src.training.antiuav import Sequence, SequenceLabels  # noqa: E402
from tools.preview_sequence import (MASK, align, contact_sheet,  # noqa: E402
                                    longest_run, mask_stats, panel,
                                    pick_sequence, preview, stretch)


def labels(count: int) -> SequenceLabels:
    boxes = np.full((count, 4), np.nan, dtype=np.float32)
    boxes[:] = (8, 8, 24, 24)
    return SequenceLabels(exist=np.ones(count, bool), boxes=boxes)


class TestLongestRun(unittest.TestCase):
    def test_a_gap_breaks_a_run(self):
        """The store's keys are positions in `Sequence.frames`; a missing one
        is a real discontinuity, and a preview that stitched across it would
        show a jump the data does not contain."""
        picked = longest_run([0, 1, 2, 7, 8, 9, 10], count=100)
        self.assertEqual(list(picked), [7, 8, 9, 10])

    def test_the_longest_run_wins_over_an_earlier_short_one(self):
        """Taking the first run found is the easy mistake: sequences often
        open with two or three masked frames and carry the real stretch much
        later, and previewing the first three shows nothing."""
        picked = longest_run([0, 1, 2, 50, 51, 52, 53, 54, 55], count=100)
        self.assertEqual(list(picked), [50, 51, 52, 53, 54, 55])

    def test_ties_go_to_the_earlier_run(self):
        picked = longest_run([0, 1, 2, 10, 11, 12], count=100)
        self.assertEqual(list(picked), [0, 1, 2])

    def test_only_count_frames_are_taken_and_they_stay_consecutive(self):
        picked = longest_run(range(0, 40), count=5)
        self.assertEqual(list(picked), [0, 1, 2, 3, 4])

    def test_a_mapping_is_read_by_its_keys(self):
        """It is handed the mask store itself, so the keys are the frames that
        actually carry supervision -- not `range(len(sequence))`."""
        store = {4: None, 5: None, 6: None, 20: None}
        self.assertEqual(list(longest_run(store, count=100)), [4, 5, 6])

    def test_an_empty_store_produces_nothing_rather_than_raising(self):
        self.assertEqual(list(longest_run({}, count=10)), [])

    def test_one_masked_frame_is_a_run_of_one(self):
        self.assertEqual(list(longest_run([9], count=10)), [9])


class TestPickSequence(unittest.TestCase):
    def test_the_sequence_with_the_most_masks_is_the_one_offered(self):
        """With no --sequence the point is to land on something worth
        watching; first-by-name usually lands on a sequence with three
        masks in it."""
        thin = Sequence(name="vtuav_vis__a", split="unsplit", frames=(),
                        labels=labels(0))
        fat = Sequence(name="vtuav_vis__z", split="unsplit", frames=(),
                       labels=labels(0))
        chosen = pick_sequence([thin, fat],
                               {"vtuav_vis__a": {0: None},
                                "vtuav_vis__z": {0: None, 1: None, 2: None}})
        self.assertEqual(chosen.name, "vtuav_vis__z")


class TestMaskStats(unittest.TestCase):
    def test_the_share_is_of_the_frame_and_not_of_the_mask_box(self):
        """A share of the target's own bounding box is 1.0 for every target
        ever drawn. The number the user reads to tell a small target from a
        medium one is the share of the whole frame."""
        mask = np.zeros((50, 100), bool)
        mask[10:20, 10:20] = True
        stats = mask_stats(mask, (50, 100))
        self.assertEqual(stats["area"], 100)
        self.assertAlmostEqual(stats["share"], 100 / 5000)
        self.assertEqual(stats["box"], (10, 10, 20, 20))

    def test_a_non_square_frame_divides_by_its_own_area(self):
        mask = np.zeros((768, 1280), bool)
        mask[0:16, 0:16] = True
        stats = mask_stats(mask, (768, 1280))
        self.assertAlmostEqual(stats["share"], 256 / (768 * 1280))

    def test_a_mask_at_another_resolution_is_counted_in_frame_pixels(self):
        """Areas from two resolutions are not comparable, so the resize comes
        before the count, never after."""
        mask = np.zeros((25, 50), bool)
        mask[0:5, 0:5] = True
        stats = mask_stats(mask, (50, 100))
        self.assertEqual(stats["area"], 100)
        self.assertAlmostEqual(stats["share"], 100 / 5000)

    def test_an_empty_mask_is_reported_as_empty_and_not_as_a_tiny_target(self):
        stats = mask_stats(np.zeros((32, 32), bool), (32, 32))
        self.assertTrue(stats["empty"])
        self.assertEqual(stats["area"], 0)
        self.assertIsNone(stats["box"])

    def test_a_missing_mask_is_empty_rather_than_a_crash(self):
        self.assertTrue(mask_stats(None, (16, 16))["empty"])

    def test_align_keeps_a_mask_that_already_fits(self):
        mask = np.zeros((8, 8), bool)
        mask[1, 1] = True
        self.assertTrue(np.array_equal(align(mask, (8, 8)), mask))


class TestStretch(unittest.TestCase):
    def test_a_16_bit_frame_becomes_visible_rather_than_black(self):
        frame = np.linspace(3000, 3400, 64 * 64).reshape(64, 64).astype(np.uint16)
        out = stretch(frame)
        self.assertEqual(out.shape, (64, 64, 3))
        self.assertGreater(int(out.max()) - int(out.min()), 200)

    def test_colour_survives_because_the_rgb_half_is_previewed_too(self):
        """`inspect_stage_c.stretch` greys everything, which is right for
        thermal and wrong here: judging whether a mask is on the target is
        much easier in colour."""
        frame = np.zeros((16, 16, 3), np.uint8)
        frame[..., 0] = 200
        frame[..., 2] = 20
        out = stretch(frame)
        self.assertGreater(int(out[..., 0].mean()), int(out[..., 2].mean()))

    def test_a_flat_frame_does_not_divide_by_zero(self):
        self.assertTrue(np.isfinite(stretch(np.full((16, 16), 7, np.uint16))).all())


@unittest.skipIf(cv2 is None, "OpenCV is not installed")
class TestPanel(unittest.TestCase):
    def test_the_two_halves_are_the_same_frame_and_only_one_is_drawn_on(self):
        frame = np.full((64, 64), 120, np.uint8)
        mask = np.zeros((64, 64), bool)
        mask[10:20, 10:20] = True
        image, stats = panel(frame, mask, index=3, half=64)
        left, right = image[:64, :64], image[:64, 64:128]
        self.assertFalse((left == np.array(MASK)).all(axis=2).any(),
                         "the left half is the untouched frame")
        self.assertTrue((right == np.array(MASK)).all(axis=2).any(),
                        "the right half carries the mask and its box")
        self.assertEqual(stats["area"], 100)

    def test_an_empty_mask_draws_no_box_on_a_frame_with_nothing_in_it(self):
        image, stats = panel(np.full((64, 64), 120, np.uint8),
                             np.zeros((64, 64), bool), index=0, half=64)
        self.assertTrue(stats["empty"])
        self.assertFalse((image[:64] == np.array(MASK)).all(axis=2).any())

    def test_the_panel_is_two_halves_wide_plus_a_caption_strip(self):
        image, _ = panel(np.full((40, 80), 90, np.uint8), None, 0, half=80)
        self.assertEqual(image.shape[1], 160)
        self.assertGreater(image.shape[0], 40)


@unittest.skipIf(cv2 is None, "OpenCV is not installed")
class TestPreview(unittest.TestCase):
    """The whole path, on a sequence whose frames and masks are keyed to their
    own index, so an off-by-one anywhere is visible in the output."""

    def setUp(self):
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.root = Path(holder.name)
        folder = self.root / "frames"
        folder.mkdir()
        frames = []
        for i in range(30):
            path = folder / f"{i * 30:06d}.png"
            # Frame i is a flat image of value i: the value identifies which
            # frame was read, so a pairing bug is a number, not an impression.
            cv2.imwrite(str(path), np.full((40, 80), 4 + i * 8, np.uint8))
            frames.append(path)
        self.sequence = Sequence(name="vtuav_vis__train_001", split="unsplit",
                                 frames=tuple(frames), labels=labels(30))
        # Mask i is a square of side i+2 -- a different area per frame, again
        # so the pairing can be checked by arithmetic.
        self.store = {}
        for i in range(4, 20):
            mask = np.zeros((40, 80), bool)
            side = i + 2
            mask[2:2 + side, 2:2 + side] = True
            self.store[i] = mask

    def test_it_previews_the_run_and_writes_both_a_video_and_a_sheet(self):
        report = preview(self.sequence, self.store, self.root / "out",
                         count=8, quiet=True)
        self.assertEqual(report["run"], list(range(4, 12)))
        self.assertTrue(Path(report["sheet"]).is_file())
        if report["video"] is not None:
            self.assertTrue(Path(report["video"]).is_file())

    def test_a_sheet_is_still_written_when_the_video_cannot_be(self):
        """A headless image without the mp4v codec must still leave something
        to look at, or the preview silently produces nothing."""
        import tools.preview_sequence as module

        original = module.write_video
        module.write_video = lambda *a, **k: False
        self.addCleanup(setattr, module, "write_video", original)
        report = preview(self.sequence, self.store, self.root / "out_novideo",
                         count=6, quiet=True)
        self.assertIsNone(report["video"])
        self.assertTrue(Path(report["sheet"]).is_file())

    def test_each_row_carries_its_own_frames_mask_never_the_next_ones(self):
        """The pairing test. Mask i has area (i+2)^2 by construction, so a
        row whose area belongs to i+1 is an off-by-one caught by arithmetic."""
        report = preview(self.sequence, self.store, self.root / "out_pair",
                         count=10, quiet=True)
        for row in report["frames"]:
            self.assertEqual(row["area"], (row["index"] + 2) ** 2)
            self.assertAlmostEqual(row["share"], row["area"] / (40 * 80))
        self.assertEqual([row["index"] for row in report["frames"]],
                         report["run"])

    def test_the_run_skips_a_gap_in_the_store_rather_than_crossing_it(self):
        store = {i: self.store[i] for i in (4, 5, 6, 10, 11, 12, 13, 14)}
        report = preview(self.sequence, store, self.root / "out_gap",
                         count=100, quiet=True)
        self.assertEqual(report["run"], [10, 11, 12, 13, 14])

    def test_an_out_path_ending_in_mp4_names_both_files(self):
        report = preview(self.sequence, self.store,
                         self.root / "named" / "run.mp4", count=4, quiet=True)
        self.assertTrue(str(report["sheet"]).endswith("run.png"))

    def test_the_printed_table_leads_with_the_share_of_the_frame(self):
        """The user's stated interest is small and medium targets, so the
        share is the column they read; it has to actually be printed."""
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            preview(self.sequence, self.store, self.root / "out_print",
                    count=3, quiet=False)
        text = buffer.getvalue()
        self.assertIn("SHARE OF FRAME", text)
        self.assertIn("consecutive EXTRACTED frames", text)
        self.assertIn("small", text)

    def test_an_empty_sheet_is_not_written_as_a_one_pixel_png(self):
        report = preview(self.sequence, {}, self.root / "out_empty",
                         count=8, quiet=True)
        self.assertEqual(report["run"], [])
        self.assertIsNone(report["sheet"])


@unittest.skipIf(cv2 is None, "OpenCV is not installed")
class TestContactSheet(unittest.TestCase):
    def test_every_panel_of_the_run_is_in_the_sheet(self):
        panels = [np.full((30, 60, 3), v, np.uint8) for v in (10, 20, 30)]
        sheet = contact_sheet(panels, columns=2)
        self.assertEqual(sheet.shape[1], 520 * 2)
        self.assertGreater(sheet.shape[0], 0)

    def test_no_panels_is_a_stub_rather_than_a_crash(self):
        self.assertEqual(contact_sheet([]).shape, (1, 1, 3))


if __name__ == "__main__":
    unittest.main()
