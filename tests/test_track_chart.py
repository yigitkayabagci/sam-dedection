"""The chart for footage that has no ground truth.

Every measurement in this repo so far compares against a drawn answer:
`eval_instances` needs masks, `eval_antiuav` needs boxes. A recording off a
drone has neither, and that is the footage the failures were reported on --
"the box spreads over a field", "it jumps". Both are visible in the tracker's
own output without any label at all: one as the mask's share of the frame
climbing, the other as its centre travelling further than a target could.

These tests hold the two halves: that the geometry is read off the masks
correctly, and that the pipeline writes the chart for a real run and puts the
guard's verdicts on it when the guard is on.
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
except ImportError:                                      # pragma: no cover
    cv2 = None

from src.metrics import track_geometry, write_track_chart  # noqa: E402
from src.pipeline import PipelineConfig  # noqa: E402
from src.prompts import BoxPrompt, PromptSet  # noqa: E402


class GeometryTest(unittest.TestCase):
    def test_the_share_and_the_centre_come_off_the_mask(self):
        """The centre is the box's, which is the point the guard measures a
        jump from -- so the curve and `verdicts.json` describe one quantity."""
        from src.trackers.stabiliser import box_of, centre_of

        mask = np.zeros((40, 40), bool)
        mask[10:20, 10:22] = True
        share, centre = track_geometry({1: mask})
        self.assertAlmostEqual(share, 120 / 1600)
        self.assertEqual(centre, (16.0, 15.0))
        self.assertEqual(centre, tuple(float(v) for v in centre_of(box_of(mask))))

    def test_it_costs_almost_nothing_at_a_real_mask_size(self):
        """It runs on every frame of every run, including the ones with no
        policy at all, so it must not be a tax on the baseline it exists to
        compare against. `np.nonzero` here was 1.9 ms; this is a tenth of it."""
        import time

        mask = np.zeros((768, 768), bool)
        mask[300:326, 400:418] = True
        masks = {1: mask}
        track_geometry(masks)
        start = time.perf_counter()
        for _ in range(50):
            track_geometry(masks)
        self.assertLess((time.perf_counter() - start) / 50 * 1000.0, 0.6)

    def test_several_objects_are_one_union(self):
        """The failure being watched for is a mask swelling over a field, and a
        union that swelled is a union that swelled whichever object did it."""
        left, right = np.zeros((20, 20), bool), np.zeros((20, 20), bool)
        left[2:6, 2:6] = True
        right[2:6, 14:18] = True
        self.assertAlmostEqual(track_geometry({1: left, 2: right})[0], 32 / 400)

    def test_a_refused_mask_reads_as_nothing_returned(self):
        """The guard hands back an empty mask rather than inventing pixels, so
        this is what a refusal looks like from the chart's side."""
        self.assertEqual(track_geometry({1: np.zeros((8, 8), bool)}), (0.0, None))
        self.assertEqual(track_geometry({}), (0.0, None))
        self.assertEqual(track_geometry(None), (0.0, None))

    def test_a_stacked_mask_is_flattened_rather_than_miscounted(self):
        """Some backends return `(1, H, W)`. Counting that as area would still
        be right; counting its `size` as the frame would not."""
        mask = np.zeros((1, 10, 10), bool)
        mask[0, 0:5, 0:4] = True
        self.assertAlmostEqual(track_geometry({1: mask})[0], 20 / 100)


class ChartTest(unittest.TestCase):
    def setUp(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:                              # pragma: no cover
            self.skipTest("matplotlib is not installed")

    def test_a_chart_is_written_for_a_run_with_no_labels_at_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "track.png"
            shares = [0.02] * 10 + [0.3, 0.5] + [0.02] * 5
            centres = [(10.0 + i, 10.0) for i in range(len(shares))]
            self.assertEqual(write_track_chart(shares, centres, out), out.resolve())
            self.assertTrue(out.stat().st_size > 0)

    def test_a_run_that_returned_nothing_writes_no_chart(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(write_track_chart([], [], Path(tmp) / "t.png"))

    def test_frames_the_guard_refused_do_not_break_the_jump_curve(self):
        """A refused frame has no centre, so the jump across it is undefined
        rather than a leap from the last known position to the origin."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "track.png"
            centres = [(1.0, 1.0), (2.0, 2.0), None, (4.0, 4.0)]
            self.assertIsNotNone(write_track_chart(
                [0.1, 0.1, 0.0, 0.1], centres, out,
                verdicts={2: "lost"}))


@unittest.skipIf(cv2 is None, "cv2 is needed to write the frames")
class PipelineTest(unittest.TestCase):
    """The chart has to come out of a real `run()`, not just a direct call."""

    class _Result:
        def __init__(self, idx, mask):
            self.frame_idx, self.masks = idx, {1: mask}

    class _Tracker:
        name, precision = "fake", "float32"

        def __init__(self, verdicts=None, guard=False):
            self.seen: list[Path] = []
            self.verdicts = verdicts
            self.guard_config = object() if guard else None

        def prepare(self, root):
            self.seen = sorted(Path(root).glob("*.jpg"))

        def set_prompts(self, prompts): pass

        def reset(self): pass

        def propagate(self):
            for i in range(len(self.seen)):
                # A mask that grows: the balloon, manufactured.
                mask = np.zeros((16, 16), dtype=bool)
                mask[1:2 + i, 1:2 + i] = True
                yield PipelineTest._Result(i, mask)

    def run_one(self, tracker, chart: bool):
        import src.pipeline as P

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for i in range(6):
                cv2.imwrite(str(folder / f"frame_{i:06d}.tiff"),
                            np.full((16, 16, 3), 40 + i, dtype=np.uint8))
            out = folder / "track.png"
            real_report = P._report_timing
            P._report_timing = lambda *a, **k: None
            try:
                P.run(tracker,
                      PromptSet(boxes=[BoxPrompt(1, 0, (1, 1, 3, 3))]),
                      PipelineConfig(output_path=None, frames_dir=folder,
                                     frame_pattern="*.tif*",
                                     track_chart=out if chart else None))
            finally:
                P._report_timing = real_report
            return out, out.is_file()

    def numbers(self, kind):
        """Run a tracker that fails in one named way, and read `track.json`."""
        import json
        import src.pipeline as P

        class Tracker(PipelineTest._Tracker):
            def propagate(inner):
                for i in range(len(inner.seen)):
                    mask = np.zeros((64, 64), dtype=bool)
                    if kind == "good":
                        mask[20:32, 20:32] = True
                    elif kind == "balloon":
                        side = 12 + (i * 3 if i > 8 else 0)
                        mask[10:10 + min(side, 50), 10:10 + min(side, 50)] = True
                    elif not 10 <= i < 20:          # loses it for ten frames
                        mask[20:32, 20:32] = True
                    yield PipelineTest._Result(i, mask)

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for i in range(30):
                cv2.imwrite(str(folder / f"frame_{i:06d}.tiff"),
                            np.full((64, 64, 3), 80, dtype=np.uint8))
            chart = folder / "track.png"
            real = P._report_timing
            P._report_timing = lambda *a, **k: None
            try:
                P.run(Tracker(),
                      PromptSet(boxes=[BoxPrompt(1, 0, (20, 20, 32, 32))]),
                      PipelineConfig(output_path=None, frames_dir=folder,
                                     frame_pattern="*.tif*", track_chart=chart))
            finally:
                P._report_timing = real
            return json.loads(chart.with_suffix(".json").read_text())

    def test_the_numbers_tell_a_balloon_from_a_loss_with_no_labels(self):
        """What makes several policy folders comparable at all. There is no
        drawn answer on this footage, so these are the only two failures that
        can be seen from the tracker's own output -- and they have to separate,
        or the summary table is decoration."""
        good = self.numbers("good")
        balloon = self.numbers("balloon")
        lost = self.numbers("lost")

        self.assertEqual((good["held"], good["longest_gap"]), (30, 0))
        # The balloon holds every frame and is still the worse run.
        self.assertEqual(balloon["held"], 30)
        self.assertGreater(balloon["share_max"], 0.5)
        self.assertLess(good["share_max"], 0.1)
        # The loss holds fewer, and the gap is the length of it.
        self.assertEqual((lost["held"], lost["longest_gap"]), (20, 10))
        self.assertLess(lost["share_max"], 0.1)

    def test_no_numbers_are_written_without_the_chart(self):
        path, _ = self.run_one(self._Tracker(), chart=False)
        self.assertFalse(path.with_suffix(".json").is_file())

    def test_the_chart_lands_where_the_run_was_told_to_put_it(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:                              # pragma: no cover
            self.skipTest("matplotlib is not installed")
        _, written = self.run_one(self._Tracker(), chart=True)
        self.assertTrue(written)

    def test_nothing_is_written_and_nothing_is_measured_without_the_flag(self):
        """The collector sits inside the timed region, so it has to cost
        nothing at all when no chart was asked for."""
        path, written = self.run_one(self._Tracker(), chart=False)
        self.assertFalse(written)
        self.assertFalse(path.is_file())

    def test_the_guards_verdicts_reach_the_chart(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:                              # pragma: no cover
            self.skipTest("matplotlib is not installed")
        import src.metrics as M

        class _Decision:
            def __init__(self, state):
                self.state = state

        seen = {}
        real = M.write_track_chart
        M.write_track_chart = lambda *a, **k: seen.update(k) or real(*a, **k)
        import src.pipeline as P
        P.write_track_chart = M.write_track_chart
        try:
            tracker = self._Tracker(verdicts={3: {1: _Decision("suspect")},
                                              4: {1: _Decision("lost")}},
                                    guard=True)
            self.run_one(tracker, chart=True)
        finally:
            M.write_track_chart = real
            P.write_track_chart = real
        self.assertEqual(seen.get("verdicts"), {3: "suspect", 4: "lost"})


if __name__ == "__main__":
    unittest.main()
