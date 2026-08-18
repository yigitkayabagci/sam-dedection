"""Pseudo-label geometry, gates and the mask store.

The teacher itself is not exercised here -- it needs a GPU and a 2 GB
checkpoint. Everything around it is, which is the point of keeping the
`transformers` call down to one short method.
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

from src.training.labels import (  # noqa: E402
    Gates,
    Measurement,
    frames_to_label,
    load_masks,
    open_masks,
    reject_reason,
    rle_decode,
    rle_encode,
    save_masks,
    summarise,
    zoom_window,
)

FRAME = (640, 512)

try:
    import cv2  # noqa: F401 -- only the end-to-end labelling test needs it
    HAVE_CV2 = True
except ImportError:  # pragma: no cover - environment, not logic
    HAVE_CV2 = False


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


class TestLazyMaskStore(unittest.TestCase):
    """The store the training run actually opens: runs held, masks made to order."""

    def _store(self, tmp: Path):
        masks = {i: np.zeros((48, 64), dtype=bool) for i in (0, 3, 9)}
        for i, mask in masks.items():
            mask[i:i + 5, i:i + 7] = True
        return save_masks(Path(tmp) / "m.npz", (48, 64), masks), masks

    def test_decodes_the_same_masks_load_masks_returns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, masks = self._store(Path(tmp))
            store = open_masks(path)
            self.assertEqual(store.shape, (48, 64))
            self.assertEqual(sorted(store), [0, 3, 9])
            for frame, mask in masks.items():
                np.testing.assert_array_equal(store[frame], mask)

    def test_an_unlabelled_frame_is_absent_not_empty(self):
        # clip_masks calls .get() and reads None as "no mask supervision here",
        # which is a different instruction from an all-zero mask.
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = self._store(Path(tmp))
            store = open_masks(path)
            self.assertIsNone(store.get(4))
            self.assertNotIn(4, store)

    def test_holds_the_runs_rather_than_the_frames(self):
        # The point of the class: a 512x640 boolean frame is 328 KB and the
        # training set has tens of thousands of them.
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = self._store(Path(tmp))
            store = open_masks(path)
            held = store._runs.nbytes + store._offsets.nbytes
            self.assertLess(held, 48 * 64 * len(store))


class TestFramesToLabel(unittest.TestCase):
    class FakeSequence:
        def __init__(self, exist):
            from src.training.antiuav import SequenceLabels
            exist = np.asarray(exist, dtype=bool)
            self.labels = SequenceLabels(
                exist=exist, boxes=np.zeros((exist.size, 4), dtype=np.float32)
            )

    def test_only_annotated_frames_are_offered_to_the_teacher(self):
        sequence = self.FakeSequence([1, 0, 1, 1, 0, 1])
        np.testing.assert_array_equal(frames_to_label(sequence), [0, 2, 3, 5])

    def test_stride_thins_the_annotated_frames_not_the_raw_ones(self):
        sequence = self.FakeSequence([1, 0, 1, 1, 0, 1])
        np.testing.assert_array_equal(frames_to_label(sequence, stride=2), [0, 3])

    def test_max_frames_caps_the_cost_of_one_sequence(self):
        sequence = self.FakeSequence([1] * 100)
        self.assertEqual(len(frames_to_label(sequence, max_frames=10)), 10)


@unittest.skipUnless(HAVE_CV2, "labelling reads pixels, which needs OpenCV")
class TestLabelSequence(unittest.TestCase):
    """`label_sequence` against a teacher that records how it was called.

    The real teacher is a 2 GB checkpoint on a GPU. What is worth pinning here
    is everything around it: that a stride reaches it, that crops arrive in
    batches, that a rejected mask is counted rather than stored, and that the
    report's denominator is what was attempted.
    """

    class FakeTeacher:
        def __init__(self, accept=True):
            self.accept = accept
            self.calls: list[int] = []

        def masks_for(self, crops, boxes):
            self.calls.append(len(crops))
            out = []
            for crop, box in zip(crops, boxes):
                mask = np.zeros(crop.shape[:2], dtype=bool)
                if self.accept:
                    x0, y0, x1, y1 = (int(round(v)) for v in box)
                    mask[y0:y1, x0:x1] = True
                out.append((mask, 0.95))
            return out

    def _sequence(self, tmp: Path, frames: int = 12):
        import cv2

        from src.training.antiuav import list_sequences

        folder = Path(tmp) / "train" / "s0"
        folder.mkdir(parents=True)
        for i in range(frames):
            image = np.full((48, 64), 20, dtype=np.uint8)
            image[20:26, 30:38] = 250
            cv2.imwrite(str(folder / f"{i}.jpg"), image)
        (folder / "IR_label.json").write_text(json.dumps({
            "exist": [1] * frames,
            "gt_rect": [[30, 20, 8, 6]] * frames,
        }))
        return list_sequences(Path(tmp), "train")[0]

    def test_accepted_masks_land_in_the_store_at_frame_coordinates(self):
        from src.training.labels import label_sequence

        with tempfile.TemporaryDirectory() as tmp:
            sequence = self._sequence(Path(tmp))
            teacher = self.FakeTeacher()
            report = label_sequence(sequence, teacher, Path(tmp) / "labels",
                                    frame_size=(64, 48), min_size=32, batch_size=4)

            self.assertEqual(report["attempted"], 12)
            self.assertEqual(report["accepted"], 12)
            store = open_masks(Path(tmp) / "labels" / "s0" / "pseudo_masks.npz")
            self.assertEqual(store.shape, (48, 64))
            self.assertTrue(store[0][20:26, 30:38].all())

    def test_crops_reach_the_teacher_in_batches(self):
        from src.training.labels import label_sequence

        with tempfile.TemporaryDirectory() as tmp:
            sequence = self._sequence(Path(tmp))
            teacher = self.FakeTeacher()
            label_sequence(sequence, teacher, Path(tmp) / "labels",
                           frame_size=(64, 48), min_size=32, batch_size=5)
            self.assertEqual(teacher.calls, [5, 5, 2])

    def test_stride_is_what_the_teacher_is_asked_about(self):
        from src.training.labels import label_sequence

        with tempfile.TemporaryDirectory() as tmp:
            sequence = self._sequence(Path(tmp))
            teacher = self.FakeTeacher()
            report = label_sequence(sequence, teacher, Path(tmp) / "labels",
                                    frame_size=(64, 48), min_size=32,
                                    stride=3, batch_size=8)

            self.assertEqual(sum(teacher.calls), 4)      # 12 // 3
            self.assertEqual(report["attempted"], 4)
            self.assertEqual(report["visible"], 12)      # the flag is still there
            self.assertEqual(sorted(open_masks(
                Path(tmp) / "labels" / "s0" / "pseudo_masks.npz")), [0, 3, 6, 9])

    def test_a_rejected_mask_is_counted_and_not_stored(self):
        from src.training.labels import label_sequence

        with tempfile.TemporaryDirectory() as tmp:
            sequence = self._sequence(Path(tmp))
            report = label_sequence(sequence, self.FakeTeacher(accept=False),
                                    Path(tmp) / "labels", frame_size=(64, 48),
                                    min_size=32, batch_size=4)

            self.assertEqual(report["accepted"], 0)
            self.assertEqual(report["acceptance_rate"], 0.0)
            self.assertEqual(sum(report["rejected"].values()), 12)
            self.assertEqual(len(open_masks(
                Path(tmp) / "labels" / "s0" / "pseudo_masks.npz")), 0)


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
