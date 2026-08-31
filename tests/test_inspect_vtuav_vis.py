"""Judging an archive before trusting it.

Two faults this repo has actually paid for: a file whose name disagrees with
its own members (frames lost twice, in Kust4K and AeroVIS), and data accepted
because it looked plausible rather than because anyone measured whether it was
the hard case the project is short of.
"""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import cv2
except ImportError:                                      # pragma: no cover
    cv2 = None

from tools.inspect_vtuav_vis import contrast, layout, main, pairs  # noqa: E402


@unittest.skipIf(cv2 is None, "OpenCV is not installed")
class ArchiveTest(unittest.TestCase):
    def build(self, folder: Path, name="train_003.zip", sequences=("train_003_001",),
              lift=6, side=14, masks=6, modality="ir"):
        path = folder / name
        rng = np.random.default_rng(0)
        with zipfile.ZipFile(path, "w") as zf:
            for sequence in sequences:
                for index in range(masks):
                    frame = index * 30
                    image = np.clip(rng.normal(60, 7, (128, 160)), 0, 255).astype(np.uint8)
                    mask = np.zeros((128, 160), np.uint8)
                    x, y = 40 + index, 50
                    image[y:y + side, x:x + side] = np.clip(
                        image[y:y + side, x:x + side] + lift, 0, 255)
                    mask[y:y + side, x:x + side] = 1
                    zf.writestr(f"{sequence}/{modality}/{frame:06d}.jpg",
                                cv2.imencode(".jpg", image)[1].tobytes())
                    zf.writestr(f"{sequence}/mask/{modality}/{frame:06d}.png",
                                cv2.imencode(".png", mask)[1].tobytes())
        return path

    def run_on(self, path, *extra):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main([str(path), "--sample", "6", *extra])
        return code, out.getvalue()

    def tmp(self):
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        return Path(holder.name)

    def test_it_reads_the_layout_off_the_member_names(self):
        folder = self.tmp()
        code, text = self.run_on(self.build(folder, sequences=("a", "b")))
        self.assertEqual(code, 0)
        self.assertIn("sequences with both ir/ and mask/ir/: 2/2", text)
        self.assertIn("masks with a frame beside them: 12", text)

    def test_a_mislabelled_archive_is_called_out(self):
        """A zip named test_001 whose members all say train_003. The repo lost
        frames twice by trusting the name."""
        folder = self.tmp()
        path = self.build(folder, name="test_001.zip", sequences=("train_003_001",))
        _, text = self.run_on(path)
        self.assertIn("Trust the members, not the name", text)

    def test_a_matching_name_is_not_flagged(self):
        folder = self.tmp()
        _, text = self.run_on(self.build(folder, name="train_003.zip"))
        self.assertNotIn("Trust the members", text)

    def test_a_mask_is_paired_only_inside_its_own_sequence(self):
        """`000123.png` is in most sequences; pairing across them would put a
        mask on a stranger's frame."""
        folder = self.tmp()
        path = folder / "x.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("seq_a/ir/000000.jpg", b"x")
            zf.writestr("seq_b/mask/ir/000000.png", b"x")
        with zipfile.ZipFile(path) as zf:
            self.assertEqual(pairs(zf.infolist(), "ir"), [])

    def test_it_reports_the_contrast_the_project_is_short_of(self):
        """A faint target reads under 1; a bright one reads far above it. That
        distinction is the whole reason to look at a new archive."""
        folder = self.tmp()
        _, faint = self.run_on(self.build(folder, name="faint.zip", lift=5))
        _, bright = self.run_on(self.build(folder, name="bright.zip", lift=90))

        def median(text):
            line = next(l for l in text.splitlines() if "signal-to-clutter" in l)
            return float(line.split("median")[1].split(",")[0])

        self.assertLess(median(faint), 1.5)
        self.assertGreater(median(bright), 5.0)

    def test_a_modality_that_is_not_there_is_a_refusal_not_a_zero(self):
        folder = self.tmp()
        code, text = self.run_on(self.build(folder), "--modality", "rgb")
        self.assertEqual(code, 1)
        self.assertIn("nothing to measure", text)

    def test_the_contrast_definition_matches_the_training_run_s(self):
        """`|mean inside - mean in the ring| / std of the ring`, so a number
        here and a number in a training report mean the same thing."""
        image = np.full((64, 64), 50, np.uint8)
        image[30:40, 30:40] = 80
        mask = np.zeros((64, 64), bool)
        mask[30:40, 30:40] = True
        flat = contrast(image, mask)
        self.assertTrue(np.isfinite(flat))
        noisy = np.clip(np.random.default_rng(0).normal(50, 10, (64, 64)), 0, 255)
        noisy[30:40, 30:40] = 80
        self.assertLess(contrast(noisy.astype(np.uint8), mask), flat)

    def test_layout_counts_every_folder_a_sequence_holds(self):
        folder = self.tmp()
        with zipfile.ZipFile(self.build(folder)) as zf:
            tree = layout(zf.infolist())
        holds = tree["sequences"]["train_003_001"]
        self.assertEqual(holds["ir"], 6)
        self.assertEqual(holds["mask/ir"], 6)


if __name__ == "__main__":
    unittest.main()
