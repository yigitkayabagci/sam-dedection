"""Contact sheets for Stage C's video data.

The parts worth pinning are the ones that decide *what gets looked at*: a
sampler that misses every disappearance produces a sheet that shows nothing,
and a tile that draws a box on an absent frame invents a target.
"""
from __future__ import annotations

import json
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
from tools.inspect_stage_c import (ABSENT, MASKED, PRESENT,  # noqa: E402
                                   draw, render, sheet, stretch,
                                   windows_around_gaps)


def labels(exist):
    exist = np.asarray(exist, dtype=bool)
    boxes = np.full((exist.shape[0], 4), np.nan, dtype=np.float32)
    boxes[exist] = (8, 8, 24, 24)
    return SequenceLabels(exist=exist, boxes=boxes)


class TestWindows(unittest.TestCase):
    def test_a_window_is_centred_on_the_disappearance(self):
        exist = np.ones(60, bool)
        exist[30:40] = False
        picked = windows_around_gaps(exist, span=10, count=1)
        self.assertEqual(len(picked), 1)
        self.assertIn(30, picked[0], "the frame the target vanishes on is "
                                     "the one the sheet exists to show")

    def test_the_return_is_a_window_of_its_own(self):
        """Coming back is a different event from going away, and it is the one
        a memory bank fails at."""
        exist = np.ones(60, bool)
        exist[30:40] = False
        picked = windows_around_gaps(exist, span=8, count=2)
        self.assertEqual(len(picked), 2)
        self.assertIn(40, picked[1])

    def test_a_sequence_that_never_disappears_is_still_sampled(self):
        picked = windows_around_gaps(np.ones(100, bool), span=10, count=2)
        self.assertEqual(len(picked), 1)
        self.assertGreater(picked[0].size, 1)
        self.assertLess(picked[0][0], picked[0][-1])

    def test_windows_stay_inside_the_sequence(self):
        exist = np.ones(12, bool)
        exist[1] = False
        for indices in windows_around_gaps(exist, span=10, count=2):
            self.assertGreaterEqual(int(indices.min()), 0)
            self.assertLess(int(indices.max()), 12)

    def test_an_empty_sequence_produces_nothing_rather_than_raising(self):
        self.assertEqual(windows_around_gaps(np.zeros(0, bool), 8, 2), [])


class TestStretch(unittest.TestCase):
    def test_a_16_bit_frame_becomes_visible_rather_than_black(self):
        frame = np.linspace(3000, 3400, 64 * 64).reshape(64, 64).astype(np.uint16)
        out = stretch(frame)
        self.assertEqual(out.shape, (64, 64, 3))
        self.assertGreater(int(out.max()) - int(out.min()), 200)

    def test_a_flat_frame_does_not_divide_by_zero(self):
        out = stretch(np.full((16, 16), 7, np.uint16))
        self.assertTrue(np.isfinite(out).all())


@unittest.skipIf(cv2 is None, "OpenCV is not installed")
class TestTiles(unittest.TestCase):
    def frame(self):
        return np.full((64, 64), 100, np.uint8)

    def test_an_absent_frame_gets_no_box(self):
        """`exist` False means there is nothing there; drawing the last known
        box would show a target the annotation says is gone."""
        tile = draw(self.frame(), np.array([8, 8, 24, 24]), None, present=False)
        self.assertFalse((tile == np.array(PRESENT)).all(axis=2).any())

    def test_an_absent_frame_is_marked_so_it_cannot_be_mistaken(self):
        tile = draw(self.frame(), np.array([8, 8, 24, 24]), None, present=False)
        self.assertTrue((tile[0] == np.array(ABSENT)).all(axis=1).any())

    def test_a_masked_frame_is_drawn_differently_from_a_box_only_one(self):
        """The distinction is the point: a run can look supervised in the frame
        count and be box-only exactly where it matters."""
        box = np.array([8, 8, 24, 24])
        mask = np.zeros((64, 64), bool)
        mask[10:20, 10:20] = True
        masked = draw(self.frame(), box, mask, present=True)
        box_only = draw(self.frame(), box, None, present=True)
        self.assertTrue((masked == np.array(MASKED)).all(axis=2).any())
        self.assertFalse((box_only == np.array(MASKED)).all(axis=2).any())

    def test_a_mask_at_another_resolution_is_resized_not_refused(self):
        mask = np.zeros((32, 32), bool)
        mask[4:12, 4:12] = True
        tile = draw(self.frame(), np.array([8, 8, 24, 24]), mask, present=True)
        self.assertTrue((tile == np.array(MASKED)).all(axis=2).any())


@unittest.skipIf(cv2 is None, "OpenCV is not installed")
class TestRender(unittest.TestCase):
    def setUp(self):
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.root = Path(holder.name)
        frames = []
        folder = self.root / "frames"
        folder.mkdir()
        for i in range(24):
            path = folder / f"{i:04d}.png"
            cv2.imwrite(str(path), np.full((48, 48), 40 + i * 4, np.uint8))
            frames.append(path)
        exist = np.ones(24, bool)
        exist[10:14] = False
        self.sequence = Sequence(name="vtuav__flight_a__track_1",
                                 split="unsplit", frames=tuple(frames),
                                 labels=labels(exist))

    def test_it_writes_a_sheet_per_window_and_an_index(self):
        out = self.root / "out"
        report = render([self.sequence], {self.sequence.name: {}}, out,
                        per_source=1, span=6, windows=2, quiet=True)
        self.assertIn("vtuav", report)
        self.assertEqual(len(report["vtuav"]), 2)
        self.assertTrue((out / "index.json").is_file())
        for row in report["vtuav"]:
            self.assertTrue(Path(row["sheet"]).is_file())

    def test_the_index_records_how_much_mask_supervision_a_sequence_has(self):
        """The number that decides whether Stage C is training on masks or on
        box projection for this sequence."""
        store = {2: np.ones((48, 48), bool)}
        report = render([self.sequence], {self.sequence.name: store},
                        self.root / "out2", per_source=1, span=6, windows=1,
                        quiet=True)
        self.assertEqual(report["vtuav"][0]["masks"], 1)
        self.assertEqual(report["vtuav"][0]["visible"], 20)

    def test_the_index_says_where_the_disappearances_come_from(self):
        """Only VTUAV reads a real `exist` column; VTUAV-VIS and BIRDSAI both
        hand back `exist=np.ones`. A source with no absent frame trains the
        object-score head on nothing, and that is worth reading off the sheet
        rather than off three function bodies."""
        never = Sequence(name="birdsai__flight_b__track_9", split="unsplit",
                         frames=self.sequence.frames,
                         labels=labels(np.ones(24, bool)))
        out = self.root / "out4"
        render([self.sequence, never],
               {self.sequence.name: {}, never.name: {}}, out,
               per_source=1, span=6, windows=1, quiet=True)
        index = json.loads((out / "index.json").read_text())
        self.assertEqual(index["absence"]["vtuav"]["absent"], 4)
        self.assertEqual(index["absence"]["birdsai"]["absent"], 0)

    def test_sheets_are_grouped_by_source(self):
        other = Sequence(name="birdsai__flight_b__track_9", split="unsplit",
                         frames=self.sequence.frames,
                         labels=self.sequence.labels)
        out = self.root / "out3"
        report = render([self.sequence, other],
                        {self.sequence.name: {}, other.name: {}}, out,
                        per_source=1, span=6, windows=1, quiet=True)
        self.assertEqual(sorted(report), ["birdsai", "vtuav"])
        self.assertTrue((out / "vtuav").is_dir())
        self.assertTrue((out / "birdsai").is_dir())


if __name__ == "__main__":
    unittest.main()
