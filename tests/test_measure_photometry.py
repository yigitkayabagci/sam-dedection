"""Measuring a run that was made without `--photometry`.

The measurement is a property of the frames, not of the tracker, so a folder
already on disk can be diagnosed without re-running the model. What has to be
right is that it re-reads the frames the way the *pipeline* did -- the run's own
crop, the model's input size, `to_rgb8` last -- because a number measured on the
sensor's pixels describes the sensor, not what the encoder was given.
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
except ImportError:                                      # pragma: no cover
    cv2 = None

from tools.measure_photometry import main as measure_main  # noqa: E402


@unittest.skipIf(cv2 is None, "OpenCV is not installed")
class MeasureTest(unittest.TestCase):
    def build(self, folder: Path, frames=12, depth="uint8", step_at=None):
        source = folder / "frames"
        source.mkdir(parents=True)
        for index in range(frames):
            level = 150 if step_at is not None and index >= step_at else 40
            image = np.clip(np.random.randn(240, 320) * 5 + level, 0, 255)
            cv2.imwrite(str(source / f"frame_{index:06d}.tiff"),
                        image.astype(np.uint16 if depth == "uint16" else np.uint8))
        run = folder / "run"
        run.mkdir()
        (run / "provenance.json").write_text(json.dumps({
            "image_size": 128, "center_crop": 160,
            "command": ["python3", "cli.py", "--frames-dir", str(source),
                        "--frame-pattern", "*.tif*"]}))
        return source, run

    def measure(self, **kwargs):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        folder = Path(tmp.name)
        source, run = self.build(folder, **kwargs)
        self.assertEqual(measure_main([str(run)]), 0)
        return json.loads((run / "photometry.json").read_text()), source, run

    def test_it_writes_a_row_per_frame_and_changes_nothing(self):
        body, _, _ = self.measure(frames=12)
        self.assertEqual(body["frames"], 12)
        self.assertEqual(body["stretched"], 0)
        self.assertEqual(body["floor"], 0)
        self.assertEqual(len(body["rows"]), 12)

    def test_it_finds_the_frames_from_the_run_s_own_provenance(self):
        """Not from a path typed again, which is how a diagnosis ends up
        describing a different clip than the one that broke."""
        body, source, run = self.measure(frames=6)
        self.assertIn("--frames-dir", (run / "provenance.json").read_text())
        self.assertEqual(body["frames"], len(list(source.glob("*.tiff"))))

    def test_an_exposure_step_shows_up_as_drift(self):
        body, _, _ = self.measure(frames=20, step_at=10)
        self.assertGreater(body["drift_max"], 50)

    def test_a_steady_clip_does_not(self):
        body, _, _ = self.measure(frames=20)
        self.assertLess(body["drift_max"], 15)

    def test_a_16_bit_source_is_measured_after_to_rgb8_not_before(self):
        """`to_rgb8` is itself a per-frame min/max stretch on 16-bit input, and
        it runs before the encoder -- so the span the encoder saw is the one
        after it, which is wide even when the sensor's was narrow."""
        body, _, _ = self.measure(frames=8, depth="uint16")
        self.assertGreater(body["span_median"], 100)

    def test_only_the_first_n_frames_when_asked(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        folder = Path(tmp.name)
        _, run = self.build(folder, frames=12)
        self.assertEqual(measure_main([str(run), "--limit", "4"]), 0)
        self.assertEqual(json.loads((run / "photometry.json").read_text())["frames"], 4)

    def test_it_says_so_when_the_frames_have_moved(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        folder = Path(tmp.name)
        source, run = self.build(folder, frames=4)
        import shutil

        shutil.rmtree(source)
        with self.assertRaises(SystemExit) as caught:
            measure_main([str(run)])
        self.assertIn("--records", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
