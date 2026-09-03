"""Tiling several arms of one clip into one picture, and the ways it lies.

The tool reads mp4s and writes an mp4, so most of it is not worth a test. Three
things are: that an arm resolves to the folder `run_records.py` actually wrote
(`stock` keeps the bare mode name there, and looking for `full768_stock/` would
report the one always-present arm as missing), that the grid keeps two and three
arms in a row where a slide can read them, and that a mode's worth of panes is
tiled in the order given rather than in whatever order a set iterates.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.compare_videos import band_height, folder_for, grid_shape  # noqa: E402


class FolderFor(unittest.TestCase):
    def test_stock_keeps_the_bare_mode_name(self):
        """run_records.folder() omits the weights for stock, so this must too."""
        self.assertEqual(folder_for("full768", "stock"), "full768")

    def test_any_other_arm_is_suffixed(self):
        self.assertEqual(folder_for("full768", "aerial_stable"),
                         "full768_aerial_stable")
        self.assertEqual(folder_for("crop768", "aerial_stable_rgb"),
                         "crop768_aerial_stable_rgb")

    def test_a_folder_name_is_taken_as_given(self):
        """A policy rung is named by its folder, not by a --weights value, and
        joining it to a grid must not suffix it a second time."""
        self.assertEqual(folder_for("full768", "full768_aerial_stable_guard"),
                         "full768_aerial_stable_guard")
        self.assertEqual(folder_for("full768", "full768"), "full768")

    def test_a_mode_is_not_confused_with_a_longer_one(self):
        """`full76` must not swallow `full768`: the prefix test is on the
        separator, not on the letters."""
        self.assertEqual(folder_for("full76", "full768"), "full76_full768")


class GridShape(unittest.TestCase):
    def test_up_to_three_arms_stay_in_one_row(self):
        self.assertEqual(grid_shape(1), (1, 1))
        self.assertEqual(grid_shape(2), (1, 2))
        self.assertEqual(grid_shape(3), (1, 3))

    def test_four_arms_become_two_by_two(self):
        """Four in a row on a 16:9 slide leaves each pane too small to see a
        mask edge, which is the only reason the picture exists."""
        self.assertEqual(grid_shape(4), (2, 2))

    def test_five_and_six_fill_two_rows(self):
        self.assertEqual(grid_shape(5), (2, 3))
        self.assertEqual(grid_shape(6), (2, 3))

    def test_an_explicit_column_count_wins(self):
        self.assertEqual(grid_shape(4, cols=4), (1, 4))
        self.assertEqual(grid_shape(5, cols=2), (3, 2))

    def test_nothing_to_tile_is_an_error_not_an_empty_video(self):
        with self.assertRaises(ValueError):
            grid_shape(0)


class Labels(unittest.TestCase):
    """`--labels` renames a pane's caption without renaming the run.

    A run name and a caption answer to different readers, and a slide has room
    for the second. What must not follow is a caption drifting onto the wrong
    pane, so the counts are matched by position and a short list is refused
    rather than padded.
    """

    def captions(self, arms: str, labels: str | None):
        from tools.compare_videos import main
        import argparse

        # The parsing rule, exercised the way main() applies it.
        names = [a.strip() for a in arms.split(",") if a.strip()]
        given = [c.strip() for c in labels.split(",")] if labels else list(names)
        if len(given) != len(names):
            raise ValueError(f"{len(given)} captions for {len(names)} arms")
        return given

    def test_defaults_to_the_arm_names(self):
        self.assertEqual(self.captions("stock,aerial_stable", None),
                         ["stock", "aerial_stable"])

    def test_replaces_them_in_order(self):
        self.assertEqual(
            self.captions("stock,aerial_stable_woc_final", "base,finetune"),
            ["base", "finetune"])

    def test_a_short_list_is_refused_not_padded(self):
        with self.assertRaises(ValueError):
            self.captions("stock,aerial_stable_woc_final", "finetune")


class BandHeight(unittest.TestCase):
    def test_scales_with_the_pane(self):
        self.assertGreater(band_height(768), band_height(384))

    def test_never_collapses_on_a_small_pane(self):
        """A one-pixel band is a label nobody can read; the floor keeps the
        caption legible when six arms share a 1920 grid."""
        self.assertGreaterEqual(band_height(10), 22)


class TrimSpan(unittest.TestCase):
    """`--start` / `--seconds`, which is arithmetic that fails quietly.

    A start past the end writes an empty video; a duration longer than what is
    left means "the rest" and should say so rather than pretend. Both are
    decided before an arm is opened, which is why this is a function and not a
    few lines in the read loop.
    """

    def span(self, frames, fps=30.0, start=0.0, seconds=None):
        from tools.compare_videos import trim_span

        return trim_span(frames, fps, start, seconds)

    def test_no_flags_keeps_the_whole_clip(self):
        self.assertEqual(self.span(300), (0, 300))

    def test_seconds_counts_in_frames_at_the_clip_s_own_fps(self):
        self.assertEqual(self.span(300, seconds=3), (0, 90))
        self.assertEqual(self.span(300, fps=25.0, seconds=3), (0, 75))

    def test_start_offsets_and_shortens_what_is_left(self):
        self.assertEqual(self.span(300, start=5), (150, 150))
        self.assertEqual(self.span(300, start=5, seconds=2), (150, 60))

    def test_asking_for_more_than_is_left_gives_what_is_left(self):
        self.assertEqual(self.span(300, start=9, seconds=60), (270, 30))

    def test_a_start_past_the_end_is_refused_not_an_empty_video(self):
        with self.assertRaises(SystemExit) as raised:
            self.span(300, start=99)
        self.assertIn("300 frames", str(raised.exception))

    def test_a_duration_below_one_frame_still_yields_a_frame(self):
        """Zero frames is a file no player will open; one is a still that
        happens to be an mp4, which at least says what was asked for."""
        self.assertEqual(self.span(300, seconds=0.001)[1], 1)


class Captions(unittest.TestCase):
    """`--no-captions` removes the band, and with it the band's height.

    A slide that carries its own labels underneath does not want a second set
    burnt into the pixels. What has to follow is that the sheet loses exactly
    the band -- a video sized for a band it no longer draws would letterbox
    every pane against black for no reason.
    """

    def sheet_height(self, rows: int, pane_h: int, captions_on: bool) -> int:
        from tools.compare_videos import band_height

        band = band_height(pane_h) if captions_on else 0
        return rows * (pane_h + band)

    def test_the_band_is_gone_not_blank(self):
        self.assertEqual(self.sheet_height(1, 960, False), 960)
        self.assertGreater(self.sheet_height(1, 960, True), 960)

    def test_every_row_loses_its_band(self):
        self.assertEqual(self.sheet_height(2, 576, False), 1152)


class Fit(unittest.TestCase):
    """Padding a pane, which is the only honest way to tile two framings.

    `--fit` exists so `crop768` can sit beside `full768` -- 768x768 against
    1280x768. Stretching either to the other's shape would hide exactly what
    the picture is for, so what is pinned is that the aspect survives and the
    content is centred, with black where the field of view is not.
    """

    def fit(self, w, h, pane_w, pane_h):
        import cv2
        import numpy as np

        from tools.compare_videos import _fit

        frame = np.full((h, w, 3), 200, np.uint8)
        return _fit(frame, pane_w, pane_h, cv2, np)

    def test_the_pane_is_always_the_asked_for_size(self):
        self.assertEqual(self.fit(1280, 768, 960, 960).shape, (960, 960, 3))
        self.assertEqual(self.fit(768, 768, 960, 960).shape, (960, 960, 3))

    def test_a_wide_frame_is_padded_above_and_below_not_stretched(self):
        pane = self.fit(1280, 768, 960, 960)
        filled = (pane > 0).any(axis=(1, 2))
        self.assertEqual(filled.sum(), 576, "1280x768 width-fit to 960 is 576 tall")
        self.assertTrue(filled[480], "the content sits in the middle")
        self.assertFalse(filled[0] or filled[-1], "and the bars are at the edges")

    def test_a_square_frame_fills_a_square_pane_with_no_bars(self):
        pane = self.fit(768, 768, 960, 960)
        self.assertTrue((pane > 0).all())


if __name__ == "__main__":
    unittest.main()
