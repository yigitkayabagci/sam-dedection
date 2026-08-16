"""Pseudo-label geometry, gates and the mask store.

The teacher itself is not exercised here -- it needs a GPU and a 2 GB
checkpoint. Everything around it is, which is the point of keeping the
`transformers` call down to one short method.
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

from src.training.labels import (  # noqa: E402
    Gates,
    Measurement,
    load_masks,
    reject_reason,
    rle_decode,
    rle_encode,
    save_masks,
    summarise,
    zoom_window,
)

FRAME = (640, 512)


def good() -> Measurement:
    return Measurement(teacher_iou=0.9, box_iou=0.8, area_ratio=0.6, component=1.0)


class TestZoomWindow(unittest.TestCase):
    def test_crop_is_square_and_centred_on_the_box(self):
        box = np.array([300.0, 240.0, 340.0, 280.0])  # 40 px, centre (320, 260)
        x0, y0, w, h = zoom_window(box, FRAME, zoom=4.0, min_size=32)

        self.assertEqual(w, h)
        self.assertEqual(w, 160)  # 40 * 4
        self.assertAlmostEqual(x0 + w / 2, 320, delta=1)
        self.assertAlmostEqual(y0 + h / 2, 260, delta=1)

    def test_tiny_target_gets_the_minimum_crop_not_a_zoomed_speck(self):
        # A 4 px drone at zoom 4 would give a 16 px crop with no context in it.
        box = np.array([300.0, 240.0, 304.0, 244.0])
        _, _, w, _ = zoom_window(box, FRAME, zoom=4.0, min_size=128)
        self.assertEqual(w, 128)

    def test_crop_stays_inside_the_frame_at_the_corners(self):
        for box in (np.array([0.0, 0.0, 20.0, 20.0]),
                    np.array([620.0, 492.0, 640.0, 512.0])):
            x0, y0, w, h = zoom_window(box, FRAME, zoom=8.0, min_size=128)
            self.assertGreaterEqual(x0, 0)
            self.assertGreaterEqual(y0, 0)
            self.assertLessEqual(x0 + w, FRAME[0])
            self.assertLessEqual(y0 + h, FRAME[1])

    def test_crop_never_exceeds_the_shorter_frame_side(self):
        box = np.array([200.0, 200.0, 400.0, 400.0])
        _, _, w, h = zoom_window(box, FRAME, zoom=10.0)
        self.assertLessEqual(max(w, h), min(FRAME))

    def test_the_box_is_inside_its_own_crop(self):
        box = np.array([600.0, 20.0, 630.0, 50.0])
        x0, y0, w, h = zoom_window(box, FRAME, zoom=4.0, min_size=128)
        self.assertTrue(x0 <= box[0] and box[2] <= x0 + w)
        self.assertTrue(y0 <= box[1] and box[3] <= y0 + h)


class TestGates(unittest.TestCase):
    def test_a_good_mask_passes(self):
        self.assertIsNone(reject_reason(good()))

    def test_each_gate_names_itself(self):
        from dataclasses import replace
        cases = {
            "teacher_iou": replace(good(), teacher_iou=0.5),
            "box_iou": replace(good(), box_iou=0.3),
            "area": replace(good(), area_ratio=0.05),
            "component": replace(good(), component=0.4),
        }
        for expected, measurement in cases.items():
            self.assertEqual(reject_reason(measurement), expected)

    def test_a_mask_far_larger_than_its_box_is_background_bleed(self):
        from dataclasses import replace
        self.assertEqual(reject_reason(replace(good(), area_ratio=3.0)), "area")

    def test_an_empty_mask_is_rejected_rather_than_scored(self):
        self.assertEqual(
            reject_reason(Measurement(np.nan, np.nan, np.nan, np.nan)), "empty"
        )

    def test_gates_are_configurable(self):
        from dataclasses import replace
        loose = Gates(teacher_iou=0.1, box_iou=0.1, area=(0.0, 10.0), component=0.0)
        self.assertIsNone(reject_reason(replace(good(), teacher_iou=0.2), loose))


class TestRle(unittest.TestCase):
    def _roundtrip(self, mask):
        np.testing.assert_array_equal(rle_decode(rle_encode(mask), mask.shape), mask)

    def test_roundtrip_empty(self):
        self._roundtrip(np.zeros((8, 8), dtype=bool))

    def test_roundtrip_full(self):
        self._roundtrip(np.ones((8, 8), dtype=bool))

    def test_roundtrip_mask_starting_on_a_true_pixel(self):
        # The encoding starts with a False run by convention, so a mask whose
        # very first pixel is set needs a zero-length run in front of it.
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 0] = True
        self.assertEqual(int(rle_encode(mask)[0]), 0)
        self._roundtrip(mask)

    def test_roundtrip_random_blobs(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            mask = np.zeros((32, 32), dtype=bool)
            y, x = rng.integers(0, 24, 2)
            mask[y:y + rng.integers(1, 8), x:x + rng.integers(1, 8)] = True
            self._roundtrip(mask)

    def test_a_sparse_mask_encodes_far_smaller_than_it_packs(self):
        mask = np.zeros((512, 512), dtype=bool)
        mask[250:260, 250:262] = True  # a 120-pixel drone
        # 10 rows of runs, versus 32 KB packed.
        self.assertLess(rle_encode(mask).nbytes, 200)


class TestMaskStore(unittest.TestCase):
    def test_roundtrip_through_a_file(self):
        masks = {}
        for frame in (0, 5, 17):
            mask = np.zeros((64, 64), dtype=bool)
            mask[frame:frame + 6, frame:frame + 4] = True
            masks[frame] = mask

        with tempfile.TemporaryDirectory() as tmp:
            path = save_masks(Path(tmp) / "m.npz", (64, 64), masks)
            shape, loaded = load_masks(path)

        self.assertEqual(shape, (64, 64))
        self.assertEqual(sorted(loaded), [0, 5, 17])
        for frame, mask in masks.items():
            np.testing.assert_array_equal(loaded[frame], mask)

    def test_unlabelled_frames_are_simply_absent(self):
        # The training loop reads a missing key as "no mask supervision here",
        # so an empty store must not turn into masks full of zeros.
        with tempfile.TemporaryDirectory() as tmp:
            path = save_masks(Path(tmp) / "m.npz", (32, 32), {})
            shape, loaded = load_masks(path)

        self.assertEqual(shape, (32, 32))
        self.assertEqual(loaded, {})


class TestSummarise(unittest.TestCase):
    def test_reports_the_dominant_failure_gate(self):
        reports = [
            {"visible": 100, "accepted": 60, "rejected": {"area": 30, "box_iou": 10}},
            {"visible": 100, "accepted": 80, "rejected": {"area": 20}},
        ]
        text = summarise(reports)

        self.assertIn("70.0%", text)     # 140 of 200 accepted
        self.assertIn("`area` | 50", text)
        self.assertLess(text.index("`area`"), text.index("`box_iou`"))  # sorted


if __name__ == "__main__":
    unittest.main()
