"""Masklet chunking, gating, storage and calibration -- everything but the teacher.

The video teacher itself needs a GPU and a multi-gigabyte checkpoint, which is
exactly why `masklets.py` keeps every `transformers` call inside two short
methods: all the decisions this file exercises -- where a run breaks, which
frames a gate drops, what lands in the store, what the calibration numbers
mean -- run here against a fake that returns box-shaped masks.
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

from src.training.labels import Gates  # noqa: E402
from src.training.masklets import (  # noqa: E402
    Sam2VideoTeacher,
    Sam3VideoTeacher,
    calibrate,
    find_sequences,
    load_store,
    masklet_sequence,
    read_boxes,
    summarise_masklets,
    teacher_class_for,
    visible_runs,
)

try:
    import cv2  # noqa: F401 -- writing the synthetic sequence needs it
    HAVE_CV2 = True
except ImportError:  # pragma: no cover - environment, not logic
    HAVE_CV2 = False

WIDTH, HEIGHT = 64, 48


class FakeVideoTeacher:
    """Returns the prompt box as a mask, for every frame of the chunk.

    Deliberately drift-free: the *annotation* moves while the fake's mask does
    not, so the frames far from the chunk's prompt fail the `box_iou` gate --
    which is the shape of failure the gate exists for, produced on purpose.
    """

    model_id = "fake-video-teacher"

    def __init__(self) -> None:
        self.prompts: list[tuple[int, tuple[float, ...]]] = []

    def propagate(self, frames, box):
        self.prompts.append((len(frames), tuple(float(v) for v in box)))
        height, width = frames[0].shape[:2]
        mask = np.zeros((height, width), dtype=bool)
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        mask[y0:y1, x0:x1] = True
        return [mask.copy() for _ in frames]


class TestReadBoxes(unittest.TestCase):
    def write(self, text: str) -> Path:
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name) / "rgb.txt"
        path.write_text(text)
        return path

    def tearDown(self):
        if hasattr(self, "tmp"):
            self.tmp.cleanup()

    def test_xywh_becomes_xyxy_with_commas_or_spaces(self):
        boxes, exist = read_boxes(self.write("10,20,8,6\n12 22 8 6\n"))
        np.testing.assert_allclose(boxes[0], [10, 20, 18, 26])
        np.testing.assert_allclose(boxes[1], [12, 22, 20, 28])
        self.assertTrue(exist.all())

    def test_a_zero_or_nan_box_is_an_absent_frame_not_a_dropped_row(self):
        boxes, exist = read_boxes(self.write("10 20 8 6\n0 0 0 0\nnan nan nan nan\n"))
        self.assertEqual(list(exist), [True, False, False])
        self.assertEqual(len(boxes), 3)          # indices still line up

    def test_an_empty_file_is_an_error_not_an_empty_sequence(self):
        with self.assertRaises(ValueError):
            read_boxes(self.write("\n"))


class TestVisibleRuns(unittest.TestCase):
    def test_a_run_breaks_at_an_absence_and_at_the_chunk_bound(self):
        exist = np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1], dtype=bool)
        self.assertEqual(visible_runs(exist, chunk=3),
                         [[0, 1, 2], [3], [6, 7, 8], [9]])

    def test_max_frames_caps_the_total_not_each_run(self):
        exist = np.ones(10, dtype=bool)
        runs = visible_runs(exist, chunk=4, max_frames=6)
        self.assertEqual(sum(len(r) for r in runs), 6)


class TestTeacherDispatch(unittest.TestCase):
    def test_the_id_chooses_the_class(self):
        self.assertIs(teacher_class_for("facebook/sam2.1-hiera-large"),
                      Sam2VideoTeacher)
        self.assertIs(teacher_class_for("facebook/sam3"), Sam3VideoTeacher)
        self.assertIs(teacher_class_for("me/SAM3-finetuned"), Sam3VideoTeacher)


@unittest.skipUnless(HAVE_CV2, "writing the synthetic sequence needs OpenCV")
@unittest.skipUnless(HAVE_CV2, "writing the synthetic sequence needs OpenCV")
class TestFindSequences(unittest.TestCase):
    """Position equals frame number, or the sequence is not read at all."""

    def build(self, keep: int, frames: int = 30) -> Path:
        import cv2

        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        seq = root / "bus_017"
        picture = np.full((HEIGHT, WIDTH), 40, dtype=np.uint8)
        for index in range(frames):
            if index % keep:
                continue
            for modality in ("rgb", "ir"):
                (seq / modality).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(seq / modality / f"{index:06d}.jpg"), picture)
        rows = len(range(0, frames, keep))
        (seq / "rgb.txt").write_text(
            "".join(f"{5 + i} 10 8 8\n" for i in range(rows)))
        return root

    def tearDown(self):
        if hasattr(self, "tmp"):
            self.tmp.cleanup()

    def test_a_contiguous_tree_is_read(self):
        self.assertEqual([s.name for s in find_sequences(self.build(1))],
                         ["bus_017"])

    def test_a_strided_tree_is_refused_rather_than_read_off_by_ten(self):
        """The trap `--frames tracked_rgb` sets for this module.

        Files 0, 10, 20 against rows 0, 1, 2 pair perfectly by position and
        completely wrongly by frame: the store is keyed by frame number, so
        every mask would be filed under a frame that box does not describe,
        and `calibrate` -- which looks masks up by the number on the drawn
        PNG -- would score a mask against the wrong frame's truth.
        """
        self.assertEqual(find_sequences(self.build(10)), [])


class TestMaskletSequence(unittest.TestCase):
    """One synthetic flight, end to end: boxes drift, the fake's masks do not."""

    def setUp(self):
        import cv2

        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        seq = root / "bike_009"
        frame = np.full((HEIGHT, WIDTH), 40, dtype=np.uint8)
        lines = []
        for i in range(10):
            for modality in ("rgb", "ir"):
                (seq / modality).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(seq / modality / f"{i:06d}.jpg"), frame)
            if i in (4, 5):
                lines.append("0 0 0 0")          # the target leaves the view
            else:
                lines.append(f"{5 + 2 * i} 10 8 8")
        (seq / "rgb.txt").write_text("\n".join(lines) + "\n")

        # Drawn masks on frames 0 and 2, the way VTUAV annotates sparsely.
        (seq / "mask" / "ir").mkdir(parents=True)
        for i in (0, 2):
            gt = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
            x0 = 5 + 2 * i
            gt[10:18, x0:x0 + 8] = 255
            cv2.imwrite(str(seq / "mask" / "ir" / f"{i:06d}.png"), gt)

        self.root = root
        self.out = root / "masklets"
        [self.sequence] = find_sequences(root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_discovery_reads_frames_boxes_absences_and_drawn_masks(self):
        self.assertEqual(self.sequence.name, "bike_009")
        self.assertEqual(len(self.sequence), 10)
        self.assertEqual(list(np.flatnonzero(~self.sequence.exist)), [4, 5])
        self.assertEqual(sorted(self.sequence.gt_masks["ir"]), [0, 2])

    def test_each_run_is_prompted_with_its_own_first_box(self):
        teacher = FakeVideoTeacher()
        masklet_sequence(self.sequence, teacher, self.out, chunk=200)
        # One run either side of the absence; each prompted at its start.
        self.assertEqual([p[0] for p in teacher.prompts], [4, 4])
        self.assertEqual(teacher.prompts[0][1][0], 5.0)     # frame 0's box
        self.assertEqual(teacher.prompts[1][1][0], 17.0)    # frame 6's box

    def test_drifted_frames_fail_the_box_gate_and_stay_out_of_the_store(self):
        report = masklet_sequence(self.sequence, FakeVideoTeacher(), self.out,
                                  chunk=200)
        store = load_store(self.out / "bike_009")
        # The fake's mask stands still while the annotation walks away, so
        # each run keeps its first frames and sheds the rest on `box_iou`.
        self.assertEqual(sorted(store), [0, 1, 6, 7])
        self.assertEqual(report["rejected"], {"box_iou": 4})
        self.assertEqual(report["attempted"], 8)
        self.assertEqual(report["accepted"], 4)
        self.assertTrue(store[0][10:18, 5:13].all())

    def test_a_stricter_gate_keeps_less_and_says_why(self):
        report = masklet_sequence(self.sequence, FakeVideoTeacher(), self.out,
                                  gates=Gates(box_iou=0.99), chunk=200)
        self.assertEqual(report["accepted"], 2)              # only the prompts
        self.assertEqual(report["rejected"], {"box_iou": 6})

    def test_calibration_counts_a_dropped_frame_as_an_empty_prediction(self):
        masklet_sequence(self.sequence, FakeVideoTeacher(), self.out, chunk=2)
        # chunk=2 re-prompts at frames 0, 2, 3(run tail), 6, 8 -- frame 2 is a
        # prompt again, so both drawn frames are covered and near-perfect.
        result = calibrate(self.sequence, load_store(self.out / "bike_009"))
        self.assertEqual(result["drawn_frames"], 2)
        self.assertEqual(result["covered"], 2)
        self.assertGreater(result["iou_accepted"], 0.9)

        # With one long chunk, frame 2 drifts out of the store: accepted-IoU
        # stays high while all-frames IoU halves. The second number is the one
        # stage C lives on.
        masklet_sequence(self.sequence, FakeVideoTeacher(), self.out, chunk=200)
        result = calibrate(self.sequence, load_store(self.out / "bike_009"))
        self.assertEqual(result["covered"], 1)
        self.assertGreater(result["iou_accepted"], 0.9)
        self.assertLess(result["iou_all"], 0.6)

    def test_the_summary_carries_the_calibration_pair(self):
        report = masklet_sequence(self.sequence, FakeVideoTeacher(), self.out,
                                  chunk=200)
        report["calibration"] = calibrate(
            self.sequence, load_store(self.out / "bike_009"))
        text = summarise_masklets([report])
        self.assertIn("bike_009", text)
        self.assertIn("box_iou", text)
        self.assertIn(" / ", text.splitlines()[4])   # the calibration pair


if __name__ == "__main__":
    unittest.main()
