"""Anti-UAV410 annotation parsing, crop geometry and clip sampling.

Runs against synthetic fixture files with nothing installed but numpy: no
OpenCV, no EdgeTAM, no GPU. The parts that touch pixels are exercised
separately in the notebook, on the real dataset.
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

from src.training.antiuav import (  # noqa: E402
    Sequence,
    SequenceLabels,
    crop_window,
    list_sequences,
    load_labels,
    map_boxes,
    sample_clips,
)

FRAME_SIZE = (640, 512)


def write_sequence(root: Path, name: str, exist, rects, suffix: str = ".jpg") -> Path:
    """A sequence folder: empty frame files plus an IR_label.json."""
    folder = root / name
    folder.mkdir(parents=True)
    for i in range(len(exist)):
        (folder / f"{i}{suffix}").write_bytes(b"")
    (folder / "IR_label.json").write_text(
        json.dumps({"exist": list(exist), "gt_rect": rects})
    )
    return folder


def moving_target(frames: int, step: float = 4.0, box: float = 20.0):
    """A target drifting right along the middle of a 640x512 frame."""
    rects = [[100.0 + step * t, 240.0, box, box] for t in range(frames)]
    return [1] * frames, rects


class TestLoadLabels(unittest.TestCase):
    def test_parses_exist_and_converts_xywh_to_xyxy(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = write_sequence(
                Path(tmp), "seq", [1, 1], [[10, 20, 30, 40], [11, 21, 31, 41]]
            )
            labels = load_labels(folder / "IR_label.json")

        self.assertEqual(len(labels), 2)
        self.assertTrue(labels.exist.all())
        np.testing.assert_allclose(labels.boxes[0], [10, 20, 40, 60])
        np.testing.assert_allclose(labels.boxes[1], [11, 21, 42, 62])

    def test_absent_frames_are_nan_however_they_are_written(self):
        # The dataset writes an absent frame as [], as zeros, or occasionally
        # as a stale box. All three have to mean the same thing.
        with tempfile.TemporaryDirectory() as tmp:
            folder = write_sequence(
                Path(tmp), "seq",
                [1, 0, 0, 1],
                [[10, 20, 30, 40], [], [0, 0, 0, 0], [12, 22, 30, 40]],
            )
            labels = load_labels(folder / "IR_label.json")

        np.testing.assert_array_equal(labels.exist, [True, False, False, True])
        self.assertTrue(np.isnan(labels.boxes[1]).all())
        self.assertTrue(np.isnan(labels.boxes[2]).all())
        np.testing.assert_array_equal(labels.visible_indices(), [0, 3])

    def test_zero_area_box_counts_as_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = write_sequence(Path(tmp), "seq", [1, 1], [[10, 20, 0, 40], [1, 2, 3, 4]])
            labels = load_labels(folder / "IR_label.json")
        np.testing.assert_array_equal(labels.exist, [False, True])

    def test_length_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "seq"
            folder.mkdir()
            (folder / "IR_label.json").write_text(
                json.dumps({"exist": [1, 1, 1], "gt_rect": [[1, 2, 3, 4]]})
            )
            with self.assertRaises(ValueError):
                load_labels(folder / "IR_label.json")


class TestListSequences(unittest.TestCase):
    def test_frames_are_ordered_numerically_not_lexically(self):
        # Anti-UAV410 names frames 0.jpg ... 10.jpg; a lexical sort would give
        # 0, 1, 10, 2 and silently shuffle the video.
        exist, rects = moving_target(12)
        with tempfile.TemporaryDirectory() as tmp:
            write_sequence(Path(tmp) / "train", "seq", exist, rects)
            sequence = list_sequences(tmp, "train")[0]

        self.assertEqual([p.stem for p in sequence.frames],
                         [str(i) for i in range(12)])

    def test_truncates_to_the_shorter_of_frames_and_labels(self):
        exist, rects = moving_target(6)
        with tempfile.TemporaryDirectory() as tmp:
            folder = write_sequence(Path(tmp) / "train", "seq", exist, rects)
            (folder / "6.jpg").write_bytes(b"")  # one frame past the annotation
            sequence = list_sequences(tmp, "train")[0]

        self.assertEqual(len(sequence), 6)
        self.assertEqual(len(sequence.labels), 6)

    def test_accepts_a_single_sequence_folder_directly(self):
        exist, rects = moving_target(4)
        with tempfile.TemporaryDirectory() as tmp:
            folder = write_sequence(Path(tmp), "seq", exist, rects)
            sequences = list_sequences(folder)

        self.assertEqual(len(sequences), 1)
        self.assertEqual(sequences[0].name, "seq")


class TestCropWindow(unittest.TestCase):
    def test_centres_on_the_target_and_stays_inside_the_frame(self):
        boxes = np.array([[300.0, 240.0, 320.0, 260.0]])
        origin = crop_window(boxes, FRAME_SIZE, 512)

        self.assertIsNotNone(origin)
        x0, y0 = origin
        self.assertEqual(y0, 0)  # a 512 window on a 512-tall frame has one place
        self.assertGreaterEqual(x0, 0)
        self.assertLessEqual(x0 + 512, 640)
        self.assertTrue(x0 <= 300 and 320 <= x0 + 512)

    def test_window_contains_every_box_in_the_clip(self):
        boxes = np.array([
            [60.0, 100.0, 80.0, 120.0],
            [500.0, 400.0, 520.0, 420.0],
        ])
        x0, y0 = crop_window(boxes, FRAME_SIZE, 512)

        self.assertTrue(x0 <= 60 and 520 <= x0 + 512)
        self.assertTrue(y0 <= 100 and 420 <= y0 + 512)

    def test_returns_none_when_the_target_outruns_the_window(self):
        boxes = np.array([
            [10.0, 10.0, 30.0, 30.0],
            [600.0, 10.0, 620.0, 30.0],  # 610 px of travel, wider than 512
        ])
        self.assertIsNone(crop_window(boxes, FRAME_SIZE, 512))

    def test_absent_frames_do_not_drag_the_window(self):
        boxes = np.array([
            [300.0, 240.0, 320.0, 260.0],
            [np.nan] * 4,
        ])
        self.assertEqual(crop_window(boxes, FRAME_SIZE, 512),
                         crop_window(boxes[:1], FRAME_SIZE, 512))

    def test_all_absent_returns_none(self):
        self.assertIsNone(crop_window(np.full((3, 4), np.nan), FRAME_SIZE, 512))

    def test_jitter_moves_the_window_but_never_off_the_target(self):
        boxes = np.array([[300.0, 240.0, 320.0, 260.0]])
        rng = np.random.default_rng(0)
        seen = set()
        for _ in range(50):
            x0, y0 = crop_window(boxes, FRAME_SIZE, 512, jitter=40, rng=rng)
            seen.add((x0, y0))
            self.assertTrue(x0 <= 300 and 320 <= x0 + 512)
            self.assertTrue(0 <= x0 <= 640 - 512)
        self.assertGreater(len(seen), 1, "jitter never moved the window")


class TestMapBoxes(unittest.TestCase):
    def test_native_crop_is_a_pure_translation(self):
        boxes = np.array([[300.0, 240.0, 320.0, 260.0]])
        out = map_boxes(boxes, (100, 0), (512, 512), 512)
        np.testing.assert_allclose(out, [[200.0, 240.0, 220.0, 260.0]])

    def test_full_frame_scales_each_axis_independently(self):
        # 640x512 into a 512 square: x by 0.8, y by 1.0. A single scale factor
        # on both axes would put the box off its target.
        boxes = np.array([[320.0, 256.0, 640.0, 512.0]])
        out = map_boxes(boxes, (0, 0), (640, 512), 512)
        np.testing.assert_allclose(out, [[256.0, 256.0, 512.0, 512.0]])


class TestSampleClips(unittest.TestCase):
    def _sequence(self, exist, rects, name="seq") -> Sequence:
        labels = SequenceLabels(
            exist=np.asarray(exist, dtype=bool),
            boxes=np.asarray(
                [[r[0], r[1], r[0] + r[2], r[1] + r[3]] if e else [np.nan] * 4
                 for e, r in zip(exist, rects)],
                dtype=np.float32,
            ),
        )
        frames = tuple(Path(f"{name}/{i}.jpg") for i in range(len(exist)))
        return Sequence(name=name, split="train", frames=frames, labels=labels)

    def test_one_clip_per_valid_start(self):
        exist, rects = moving_target(12)
        clips = sample_clips([self._sequence(exist, rects)], length=8, stride=1)

        self.assertEqual(len(clips), 5)  # starts 0..4
        self.assertEqual(clips[0].indices, tuple(range(8)))
        self.assertEqual(clips[-1].indices, tuple(range(4, 12)))

    def test_stride_widens_the_span(self):
        exist, rects = moving_target(20)
        clips = sample_clips([self._sequence(exist, rects)], length=4, stride=3)

        self.assertEqual(clips[0].indices, (0, 3, 6, 9))
        self.assertEqual(len(clips), 20 - ((4 - 1) * 3 + 1) + 1)

    def test_clip_is_dropped_when_the_first_frame_carries_no_prompt(self):
        exist, rects = moving_target(10)
        exist[0] = 0
        clips = sample_clips([self._sequence(exist, rects)], length=8, stride=1)

        self.assertTrue(all(c.indices[0] != 0 for c in clips))

    def test_clip_is_dropped_when_too_few_frames_are_annotated(self):
        frames = 10
        exist, rects = moving_target(frames)
        exist[1:] = [0] * (frames - 1)  # only frame 0 is visible
        clips = sample_clips([self._sequence(exist, rects)], length=8, min_visible=2)

        self.assertEqual(clips, [])

    def test_fast_target_falls_back_to_the_whole_frame(self):
        # 100 px per frame over 8 frames is 700 px of travel: no 512 window
        # holds it, so the clip has to be the resized full frame instead.
        exist, rects = moving_target(8, step=100.0)
        clips = sample_clips([self._sequence(exist, rects)], length=8, stride=1)

        self.assertEqual(len(clips), 1)
        self.assertFalse(clips[0].native)
        self.assertEqual(clips[0].window, FRAME_SIZE)

    def test_slow_target_stays_on_native_pixels(self):
        exist, rects = moving_target(8, step=4.0)
        clip = sample_clips([self._sequence(exist, rects)], length=8, stride=1)[0]

        self.assertTrue(clip.native)
        self.assertEqual(clip.window, (512, 512))

    def test_clip_boxes_land_inside_the_model_input(self):
        exist, rects = moving_target(8)
        clip = sample_clips([self._sequence(exist, rects)], length=8, stride=1)[0]
        boxes = clip.boxes

        self.assertEqual(boxes.shape, (8, 4))
        self.assertTrue((boxes >= 0).all() and (boxes <= 512).all())

    def test_absent_frames_stay_nan_after_mapping(self):
        exist, rects = moving_target(8)
        exist[3] = 0
        clip = sample_clips([self._sequence(exist, rects)], length=8, stride=1)[0]

        self.assertTrue(np.isnan(clip.boxes[3]).all())
        self.assertFalse(np.isnan(clip.boxes[0]).any())
        np.testing.assert_array_equal(clip.exist, np.asarray(exist, dtype=bool))


if __name__ == "__main__":
    unittest.main()
