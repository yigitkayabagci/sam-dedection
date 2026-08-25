"""The mask pools: box readers, the labelling pass, and the new fetch routes.

Same contract as `test_masklets.py` one axis over: the teacher itself needs a
GPU and a checkpoint, so every decision this stage makes -- how a box file is
read, which gate drops what, what resume skips, what the store commits -- runs
here against a fake that draws box-shaped masks. Cases that must decode real
pixels skip themselves cleanly when OpenCV is absent, exactly as the rest of
the suite does.
"""
from __future__ import annotations

import io
import json
import sys
import tarfile
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.boxes import (  # noqa: E402
    BIRDSAI_SPECIES,
    VISDRONE_NAMES,
    BoxFrame,
    birdsai_frames,
    class_histogram,
    coco_frames,
    dronevehicle_frames,
    hituav_frames,
    rgbtdroneperson_frames,
    summarise_frames,
    vtuavdet_frames,
    yolo_frames,
)
from src.training.labels import (  # noqa: E402
    MASK_STORE,
    Gates,
    Sam2Teacher,
    Sam3Teacher,
    open_masks,
    teacher_class_for,
)
from src.training.masklets import find_rgbt_sequences, masklet_sequence  # noqa: E402
from src.training.pool import (  # noqa: E402
    RECORD_FILE,
    calibration_table,
    label_pool,
    pool_report,
    summarise_pool,
    write_index,
)
from tools.fetch_datasets import RECIPES, staged, stream_extract  # noqa: E402

try:
    import cv2  # noqa: F401 -- decoding the synthetic frames needs it
    HAVE_CV2 = True
except ImportError:  # pragma: no cover - environment, not logic
    HAVE_CV2 = False


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def write_yolo(root: Path, stems=("a", "b")) -> None:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for stem in stems:
        if HAVE_CV2:
            frame = np.full((100, 200, 3), 40, np.uint8)
            frame[40:60, 70:130] = 220
            cv2.imwrite(str(root / "images" / f"{stem}.jpg"), frame)
        else:
            (root / "images" / f"{stem}.jpg").write_bytes(b"jpg")
        (root / "labels" / f"{stem}.txt").write_text(
            "3 0.5 0.5 0.3 0.2\n0 0.1 0.1 0.05 0.05\n")


class FakeImageTeacher:
    """Draws the prompted box, shrunk by `shrink` pixels per side.

    `shrink=0` passes every gate; a large shrink fails `area` and `box_iou`.
    The score it reports is fixed, so the `teacher_iou` gate can be aimed at
    it exactly.
    """

    model_id = "fake/teacher"

    def __init__(self, shrink: int = 0, score: float = 0.95) -> None:
        self.shrink = shrink
        self.score = score
        self.calls = 0

    def masks_for(self, crops, boxes):
        out = []
        for crop, box in zip(crops, boxes):
            self.calls += 1
            mask = np.zeros(crop.shape[:2], dtype=bool)
            x0, y0, x1, y1 = (int(round(v)) for v in box)
            mask[y0 + self.shrink:max(y1 - self.shrink, 0),
                 x0 + self.shrink:max(x1 - self.shrink, 0)] = True
            out.append((mask, self.score))
        return out


class FakeVideoTeacher:
    """Propagates the *prompt* box through every frame, unchanged."""

    model_id = "fake/video-teacher"

    def propagate(self, frames, box):
        height, width = frames[0].shape[:2]
        mask = np.zeros((height, width), dtype=bool)
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        mask[y0:y1, x0:x1] = True
        return [mask.copy() for _ in frames]


# --------------------------------------------------------------------------
# Box readers
# --------------------------------------------------------------------------


class TestYoloFrames(unittest.TestCase):
    def test_boxes_stay_normalised_until_an_image_size_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_yolo(Path(tmp))
            frames = yolo_frames(tmp)
            self.assertEqual(len(frames), 2)
            self.assertTrue(frames[0].normalized)
            boxes, keep = frames[0].resolved((100, 200))
            self.assertTrue(keep.all())
            np.testing.assert_allclose(boxes[0], [70, 40, 130, 60], atol=1e-9)

    def test_the_names_table_is_the_visdrone_one_and_an_index_past_it_survives(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_yolo(Path(tmp), stems=("a",))
            (Path(tmp) / "labels" / "a.txt").write_text("12 0.5 0.5 0.1 0.1\n")
            frames = yolo_frames(tmp)
            # Not a crash: the histogram probe is where a surprise id should
            # surface, and it can only do that if the reader survives it.
            self.assertEqual(frames[0].classes, ("class 12",))
            self.assertEqual(VISDRONE_NAMES[3], "car")

    def test_images_without_a_single_label_are_a_loud_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "images").mkdir()
            (Path(tmp) / "images" / "a.jpg").write_bytes(b"jpg")
            with self.assertRaises(FileNotFoundError):
                yolo_frames(tmp)


class TestCocoFrames(unittest.TestCase):
    def coco(self, tmp: Path) -> Path:
        payload = {
            "images": [{"id": 1, "file_name": "train/i1.jpg"}],
            "categories": [{"id": 0, "name": "Person"},
                           {"id": 4, "name": "DontCare"}],
            "annotations": [
                {"image_id": 1, "category_id": 0, "bbox": [10, 20, 30, 40]},
                {"image_id": 1, "category_id": 4, "bbox": [0, 0, 5, 5]},
            ],
        }
        path = tmp / "ann.json"
        path.write_text(json.dumps(payload))
        return path

    def test_names_come_from_the_json_and_dontcare_is_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = coco_frames(self.coco(Path(tmp)), tmp)
            self.assertEqual(frames[0].classes, ("Person",))
            np.testing.assert_allclose(frames[0].boxes[0], [10, 20, 40, 60])

    def test_hituav_locator_finds_the_nested_json_and_tests_the_image_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "HIT-UAV-Infrared-Thermal-Dataset-main" / "normal_json"
            (base / "annotations").mkdir(parents=True)
            (base / "train").mkdir()
            (base / "annotations" / "train.json").write_text(json.dumps({
                "images": [{"id": 1, "file_name": "i1.jpg"}],
                "categories": [{"id": 0, "name": "Person"}],
                "annotations": [{"image_id": 1, "category_id": 0,
                                 "bbox": [1, 2, 3, 4]}]}))
            (base / "train" / "i1.jpg").write_bytes(b"jpg")
            frames = hituav_frames(tmp)
            self.assertTrue(frames[0].image.is_file(),
                            "file_name without the split prefix must be "
                            "resolved against normal_json/<split>/")


class TestDroneVehicleFrames(unittest.TestCase):
    def fixture(self, tmp: Path) -> Path:
        root = tmp / "dv"
        for folder in ("trainimg", "trainimgr", "trainlabelr"):
            (root / "train" / folder).mkdir(parents=True)
        (root / "train" / "trainimg" / "00001.jpg").write_bytes(b"jpg")
        (root / "train" / "trainimgr" / "00001.jpg").write_bytes(b"jpg")
        (root / "train" / "trainlabelr" / "00001.xml").write_text(
            "<annotation>"
            "<object><name>car</name><polygon>"
            "<x1>150</x1><y1>150</y1><x2>200</x2><y2>160</y2>"
            "<x3>190</x3><y3>210</y3><x4>140</x4><y4>200</y4>"
            "</polygon></object>"
            "<object><name>truck</name><bndbox>"
            "<xmin>300</xmin><ymin>300</ymin><xmax>350</xmax><ymax>340</ymax>"
            "</bndbox></object>"
            "</annotation>")
        return root

    def test_the_polygon_becomes_its_envelope_and_the_border_shifts_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = dronevehicle_frames(self.fixture(Path(tmp)))
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0].inset, 100)
            self.assertIsNotNone(frames[0].pair)
            boxes, keep = frames[0].resolved((712, 840))
            self.assertTrue(keep.all())
            np.testing.assert_allclose(boxes[0], [40, 50, 100, 110])

    def test_a_box_entirely_inside_the_white_band_is_dropped_not_clipped_to_nothing(self):
        frame = BoxFrame(key="k", image=Path("x.jpg"),
                         boxes=np.array([[10.0, 10.0, 60.0, 60.0]]),
                         classes=("car",), inset=100)
        _, keep = frame.resolved((712, 840))
        self.assertFalse(keep[0])

    def test_a_class_count_mismatch_is_refused_at_construction(self):
        with self.assertRaises(ValueError):
            BoxFrame(key="k", image=Path("x.jpg"),
                     boxes=np.zeros((2, 4)), classes=("car",))


class TestRgbtDronePersonFrames(unittest.TestCase):
    """The jsons ship beside the zip, and the boxes are on the thermal half."""

    def fixture(self, tmp: Path, split: str = "train",
                folder: str = "train") -> Path:
        root = tmp / "rdp"
        for side in ("thermal", "visible"):
            (root / folder / side).mkdir(parents=True)
            (root / folder / side / "00000.jpg").write_bytes(b"jpg")
        (root / f"{split}_thermal.json").write_text(json.dumps({
            "images": [{"id": 0, "file_name": "00000.jpg",
                        "height": 512, "width": 640}],
            "categories": [{"id": 0, "name": "person"},
                           {"id": 2, "name": "crowd"},
                           {"id": 3, "name": "uncertain"}],
            "annotations": [
                {"image_id": 0, "category_id": 0, "bbox": [329, 343, 16, 28]},
                {"image_id": 0, "category_id": 2, "bbox": [10, 10, 40, 40]},
                {"image_id": 0, "category_id": 3, "bbox": [1, 1, 4, 4]},
            ]}))
        return root

    def test_uncertain_is_dropped_and_crowd_is_kept(self):
        # `crowd` stays so the component gate's rejection rate on group boxes
        # shows up in the report instead of being hidden by the reader.
        with tempfile.TemporaryDirectory() as tmp:
            frames = rgbtdroneperson_frames(self.fixture(Path(tmp)))
            self.assertEqual(frames[0].classes, ("person", "crowd"))
            np.testing.assert_allclose(frames[0].boxes[0], [329, 343, 345, 371])

    def test_the_visible_twin_is_attached_as_the_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame, = rgbtdroneperson_frames(self.fixture(Path(tmp)))
            self.assertIsNotNone(frame.pair)
            self.assertEqual(frame.pair.parent.name, "visible")
            self.assertEqual(frame.image.parent.name, "thermal")

    def test_the_sub_train_subset_reads_its_images_out_of_train(self):
        # `sub_train_*.json` is the publisher's 1 013-frame subset; its frames
        # live in `train/`, not in a folder of its own.
        with tempfile.TemporaryDirectory() as tmp:
            root = self.fixture(Path(tmp), split="sub_train", folder="train")
            frame, = rgbtdroneperson_frames(root, split="sub_train")
            self.assertEqual(frame.image.parent.parent.name, "train")

    def test_a_missing_json_says_which_command_fetches_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as caught:
                rgbtdroneperson_frames(tmp)
            self.assertIn("fetch_datasets.py rgbtdroneperson",
                          str(caught.exception))


class TestVtuavDetFrames(unittest.TestCase):
    """`val_ir.json` annotates the folder called `test/` -- map, do not assume."""

    def fixture(self, tmp: Path, stem: str, folder: str) -> Path:
        root = tmp / "vd"
        for side in ("ir", "rgb"):
            (root / folder / side).mkdir(parents=True)
            (root / folder / side / "00001.jpg").write_bytes(b"jpg")
        (root / f"{stem}_ir.json").write_text(json.dumps({
            "images": [{"id": 0, "file_name": "00001.jpg",
                        "height": 1080, "width": 1920}],
            "categories": [{"id": 0, "name": "person"}],
            "annotations": [{"image_id": 0, "category_id": 0,
                             "bbox": [768, 216, 24, 41]}]}))
        return root

    def test_val_reads_the_test_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.fixture(Path(tmp), stem="val", folder="test")
            frame, = vtuavdet_frames(root, split="val")
            self.assertEqual(frame.image.parent.parent.name, "test")
            self.assertEqual(frame.image.parent.name, "ir")

    def test_test_is_accepted_as_a_name_for_the_same_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.fixture(Path(tmp), stem="val", folder="test")
            self.assertEqual(len(vtuavdet_frames(root, split="test")), 1)

    def test_the_rgb_half_is_prompted_with_the_same_boxes_and_pairs_back(self):
        # One box set serves both modalities in this dataset; the reader must
        # not pretend there is a second one.
        with tempfile.TemporaryDirectory() as tmp:
            root = self.fixture(Path(tmp), stem="train", folder="train")
            thermal, = vtuavdet_frames(root, modality="thermal")
            rgb, = vtuavdet_frames(root, modality="rgb")
            np.testing.assert_allclose(thermal.boxes, rgb.boxes)
            self.assertEqual(rgb.image.parent.name, "rgb")
            self.assertEqual(rgb.pair.parent.name, "ir")

    def test_an_unknown_split_is_refused_before_any_globbing(self):
        with self.assertRaises(ValueError):
            vtuavdet_frames("/nowhere", split="trainval")


class TestBirdsaiFrames(unittest.TestCase):
    """MOT csv per sequence, and the noise flag means interpolated."""

    SEQUENCE = "0000000060_0000000000"

    def fixture(self, tmp: Path, numbers=(278, 279)) -> Path:
        root = tmp / "birdsai"
        annotations = root / "TrainReal" / "annotations"
        images = root / "TrainReal" / "images" / self.SEQUENCE
        annotations.mkdir(parents=True)
        images.mkdir(parents=True)
        for number in numbers:
            (images / f"{self.SEQUENCE}_{number:010d}.jpg").write_bytes(b"jpg")
        # frame, id, x, y, w, h, class, species, occlusion, noise
        (annotations / f"{self.SEQUENCE}.csv").write_text(
            "278,1,354,118,9,11,1,0,0,0\n"
            "278,2,380,102,9,11,0,1,0,0\n"
            "279,1,353,116,10,13,1,-1,0,1\n")
        (annotations / "water_metadata.txt").write_text("not a sequence")
        return root

    def test_species_names_come_from_the_published_legend(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = birdsai_frames(self.fixture(Path(tmp)))
            first = {f.key: f for f in frames}[f"{self.SEQUENCE}/000278"]
            self.assertEqual(first.classes, ("human", "elephant"))
            np.testing.assert_allclose(first.boxes[0], [354, 118, 363, 129])

    def test_an_interpolated_box_is_left_out_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.fixture(Path(tmp))
            self.assertEqual([f.key for f in birdsai_frames(root)],
                             [f"{self.SEQUENCE}/000278"])
            kept = birdsai_frames(root, drop_noisy=False)
            self.assertEqual(len(kept), 2)
            # species -1 with class 1 falls back to the coarse column.
            self.assertEqual(kept[1].classes, ("human",))

    def test_the_txt_beside_the_csvs_is_not_read_as_a_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = birdsai_frames(self.fixture(Path(tmp)))
            self.assertTrue(all(f.key.startswith(self.SEQUENCE) for f in frames))

    def test_a_sequence_numbered_from_zero_still_finds_its_frames(self):
        # Filenames carry the video's own offset in some sequences; the reader
        # falls back to position in the sorted listing rather than dropping it.
        with tempfile.TemporaryDirectory() as tmp:
            root = tmp_root = Path(tmp) / "b2"
            annotations = root / "TrainReal" / "annotations"
            images = root / "TrainReal" / "images" / "seq"
            annotations.mkdir(parents=True)
            images.mkdir(parents=True)
            for name in ("frame_a.jpg", "frame_b.jpg"):
                (images / name).write_bytes(b"jpg")
            (annotations / "seq.csv").write_text("0,1,10,10,5,5,1,0,0,0\n")
            frame, = birdsai_frames(tmp_root)
            self.assertEqual(frame.image.name, "frame_a.jpg")

    def test_a_missing_split_says_which_command_fetches_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as caught:
                birdsai_frames(tmp)
            self.assertIn("fetch_datasets.py birdsai", str(caught.exception))

    def test_the_legend_matches_the_published_table(self):
        self.assertEqual(BIRDSAI_SPECIES[0], "human")
        self.assertEqual(BIRDSAI_SPECIES[8], "rhino")
        self.assertEqual(len(BIRDSAI_SPECIES), 10)


class TestProbes(unittest.TestCase):
    def test_histogram_counts_boxes_not_images(self):
        frames = [BoxFrame(key="a", image=Path("a"), boxes=np.zeros((2, 4)),
                           classes=("car", "car")),
                  BoxFrame(key="b", image=Path("b"), boxes=np.zeros((1, 4)),
                           classes=("person",))]
        self.assertEqual(class_histogram(frames), {"car": 2, "person": 1})
        self.assertIn("| car | 2 |", summarise_frames(frames, "x"))


# --------------------------------------------------------------------------
# The labelling pass
# --------------------------------------------------------------------------


@unittest.skipUnless(HAVE_CV2, "labelling decodes real image files")
class TestLabelPool(unittest.TestCase):
    def frames(self, tmp: Path, count: int = 2) -> list:
        write_yolo(tmp / "y", stems=tuple(f"img{i}" for i in range(count)))
        return yolo_frames(tmp / "y")

    def test_accepted_masks_land_in_the_store_keyed_by_box_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = self.frames(Path(tmp))
            report = label_pool(frames, FakeImageTeacher(), Path(tmp) / "pool",
                                dataset="toy", batch_size=2)
            self.assertEqual(report["images"], 2)
            self.assertEqual(report["attempted"], 4)
            # The 5 % box at 0.1,0.1 is 10x5 px: the fake draws it exactly, so
            # both boxes pass every gate.
            self.assertEqual(report["accepted"], 4)
            store = open_masks(Path(tmp) / "pool" / "toy" / "img0" / MASK_STORE)
            self.assertEqual(sorted(store), [0, 1])
            record = json.loads(
                (Path(tmp) / "pool" / "toy" / "img0" / RECORD_FILE).read_text())
            self.assertEqual([i["verdict"] for i in record["instances"]],
                             [None, None])
            self.assertEqual(record["teacher"], "fake/teacher")

    def test_a_bad_mask_is_counted_by_the_gate_that_stopped_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = self.frames(Path(tmp), count=1)
            report = label_pool(frames, FakeImageTeacher(shrink=50),
                                Path(tmp) / "pool", dataset="toy")
            self.assertEqual(report["accepted"], 0)
            self.assertEqual(sum(report["rejected"].values()),
                             report["attempted"])

    def test_resume_skips_a_finished_frame_and_keeps_its_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = self.frames(Path(tmp))
            first = FakeImageTeacher()
            label_pool(frames, first, Path(tmp) / "pool", dataset="toy")
            again = FakeImageTeacher()
            report = label_pool(frames, again, Path(tmp) / "pool", dataset="toy")
            self.assertEqual(again.calls, 0, "resume must not re-prompt")
            self.assertEqual(report["resumed"], 2)
            self.assertEqual(report["accepted"], 4,
                             "resumed frames keep their accepted count")

    def test_prompting_on_a_missing_pair_skips_loudly_not_wrongly(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = self.frames(Path(tmp), count=1)
            report = label_pool(frames, FakeImageTeacher(), Path(tmp) / "pool",
                                dataset="toy", prompt="pair")
            self.assertEqual(report["no_pair"], 1)
            self.assertEqual(report["attempted"], 0)

    def test_max_boxes_takes_the_largest_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = self.frames(Path(tmp), count=1)
            teacher = FakeImageTeacher()
            label_pool(frames, teacher, Path(tmp) / "pool", dataset="toy",
                       max_boxes=1)
            record = json.loads(
                (Path(tmp) / "pool" / "toy" / "img0" / RECORD_FILE).read_text())
            self.assertEqual(len(record["instances"]), 1)
            # Row 0 is the 30 % x 20 % box; row 1 is the 5 % one.
            self.assertEqual(record["instances"][0]["i"], 0)

    def test_the_index_is_rebuilt_without_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = self.frames(Path(tmp))
            label_pool(frames, FakeImageTeacher(), Path(tmp) / "pool",
                       dataset="toy")
            write_index(Path(tmp) / "pool")
            index = write_index(Path(tmp) / "pool")   # twice, on purpose
            lines = index.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            summary = summarise_pool(pool_report(Path(tmp) / "pool"))
            self.assertIn("| toy |", summary)


class TestCalibrationTable(unittest.TestCase):
    def test_routes_sit_side_by_side_per_class_and_size(self):
        table = calibration_table({
            "thermal": [{"class": "car", "area": 400, "iou": 0.5},
                        {"class": "car", "area": 10_000, "iou": 0.9}],
            "rgb": [{"class": "car", "area": 400, "iou": 0.7}],
        })
        self.assertIn("| car | 16–32 px | 0.500 (1) | 0.700 (1) |", table)
        self.assertIn("≥64 px", table)
        self.assertIn("| **all** |", table)


# --------------------------------------------------------------------------
# LasHeR/RGBT234 sequences and the per-modality gate
# --------------------------------------------------------------------------


class TestFindRgbtSequences(unittest.TestCase):
    def sequence(self, tmp: Path, name: str = "seq1",
                 infrared_boxes: bool = True,
                 infrared_frames: bool = True) -> Path:
        seq = tmp / name
        for folder, stem in (("visible", "v"), ("infrared", "i")):
            if folder == "infrared" and not infrared_frames:
                continue
            (seq / folder).mkdir(parents=True)
            for i in range(3):
                if HAVE_CV2:
                    frame = np.full((60, 80, 3), 30, np.uint8)
                    cv2.imwrite(str(seq / folder / f"{stem}{i:03d}.jpg"), frame)
                else:
                    (seq / folder / f"{stem}{i:03d}.jpg").write_bytes(b"jpg")
        (seq / "visible.txt").write_text("10 10 20 20\n12 10 20 20\n14 10 20 20\n")
        if infrared_boxes:
            (seq / "infrared.txt").write_text("11 11 20 20\n13 11 20 20\nnan nan nan nan\n")
        return seq

    def test_discovery_reads_both_halves_and_both_box_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.sequence(Path(tmp))
            found = find_rgbt_sequences(tmp)
            self.assertEqual(len(found), 1)
            sequence = found[0]
            self.assertEqual(set(sequence.frames), {"rgb", "ir"})
            self.assertIsNotNone(sequence.gate_boxes)
            np.testing.assert_allclose(sequence.boxes[0], [10, 10, 30, 30])
            np.testing.assert_allclose(sequence.gate_boxes[0], [11, 11, 31, 31])

    def test_a_sequence_without_thermal_frames_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.sequence(Path(tmp), infrared_frames=False)
            self.assertEqual(find_rgbt_sequences(tmp), [])

    def test_a_nan_gate_row_falls_back_to_the_prompt_box(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.sequence(Path(tmp))
            sequence = find_rgbt_sequences(tmp)[0]
            reference = sequence.gate_reference()
            np.testing.assert_allclose(reference[0], [11, 11, 31, 31])
            np.testing.assert_allclose(reference[2], sequence.boxes[2])

    @unittest.skipUnless(HAVE_CV2, "propagation decodes real frames")
    def test_the_gate_measures_against_the_thermal_annotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            seq = self.sequence(Path(tmp))
            # Thermal annotation far from the visible one: a mask that sits on
            # the visible box must now fail the box gate.
            (seq / "infrared.txt").write_text(
                "50 40 20 15\n50 40 20 15\n50 40 20 15\n")
            sequence = find_rgbt_sequences(tmp)[0]
            report = masklet_sequence(sequence, FakeVideoTeacher(), Path(tmp) / "out",
                                      gates=Gates(box_iou=0.6))
            self.assertEqual(report["accepted"], 0)
            self.assertEqual(report["gated_by"], "separate boxes")
            self.assertIn("box_iou", report["rejected"])


# --------------------------------------------------------------------------
# Teachers and fetch routes
# --------------------------------------------------------------------------


class TestTeacherDispatch(unittest.TestCase):
    def test_the_id_chooses_the_class(self):
        self.assertIs(teacher_class_for("facebook/sam3"), Sam3Teacher)
        self.assertIs(teacher_class_for("facebook/sam2.1-hiera-large"),
                      Sam2Teacher)


class TestNewRecipes(unittest.TestCase):
    def test_lasher_never_downloads_by_default(self):
        self.assertEqual(RECIPES["lasher"].chosen(None), [])
        self.assertTrue(RECIPES["lasher"].stream)

    def test_visdrone_patterns_cover_exactly_the_named_split(self):
        for part in RECIPES["visdrone"].parts:
            self.assertIn(part.name, part.into)
            self.assertTrue(part.into.endswith("/**"))
        self.assertEqual([p.name for p in RECIPES["visdrone"].chosen(None)],
                         ["train", "val"])

    def test_the_single_archive_recipes_point_at_https(self):
        for name, marker in (("hituav", "HIT-UAV"), ("rgbt234", "rgbt234")):
            (part,) = RECIPES[name].parts
            self.assertTrue(part.url.startswith("https://"), name)
            self.assertIn(marker, part.url)

    def test_every_pool_dataset_has_both_a_reader_and_a_fetch_recipe(self):
        # `--dataset` on `make_mask_pool.py` and the recipe list on
        # `fetch_datasets.py` are two hand-written lists that have to agree:
        # a dataset the CLI offers but cannot fetch is a dead end at 3 a.m.
        from tools.make_mask_pool import DATASETS, frames_for  # noqa: F401
        from src.training import boxes

        for dataset in DATASETS:
            self.assertIn(dataset, RECIPES, f"{dataset} has no fetch recipe")
        for dataset in ("rgbtdroneperson", "vtuavdet", "birdsai"):
            self.assertIn(dataset, DATASETS)
            self.assertTrue(hasattr(boxes, f"{dataset}_frames"), dataset)

    def test_a_staged_tarball_is_found_like_a_staged_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "rgbt234.tar.gz"
            archive.write_bytes(b"x" * (2 << 20))
            self.assertEqual(staged("rgbt234", (tmp,)), archive)


class TestStreamExtract(unittest.TestCase):
    """The LasHeR route: slices of one tar.gz, read as a single stream."""

    def serve(self, blobs: list[bytes]):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):                          # noqa: N802 (stdlib API)
                index = int(self.path.rsplit(".", 1)[-1])
                self.send_response(200)
                self.send_header("Content-Length", str(len(blobs[index])))
                self.end_headers()
                self.wfile.write(blobs[index])

            def log_message(self, *args):              # keep the test quiet
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, f"http://127.0.0.1:{server.server_port}"

    def test_members_of_wanted_sequences_survive_the_slice_boundary(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for name, payload in (("lasher/seq1/visible/v000.jpg", b"a" * 3000),
                                  ("lasher/seq2/visible/v000.jpg", b"b" * 3000),
                                  ("lasher/seq1/infrared/i000.jpg", b"c" * 3000)):
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        blob = buffer.getvalue()
        cut = len(blob) // 2                 # a boundary mid-member, on purpose
        server, base = self.serve([blob[:cut], blob[cut:]])
        try:
            with tempfile.TemporaryDirectory() as tmp:
                kept = stream_extract([f"{base}/part.0", f"{base}/part.1"],
                                      Path(tmp), sequences=["seq1"], quiet=True)
                self.assertEqual(kept, 2)
                self.assertTrue(
                    (Path(tmp) / "lasher/seq1/infrared/i000.jpg").is_file())
                self.assertFalse(
                    (Path(tmp) / "lasher/seq2").exists())
        finally:
            server.shutdown()


if __name__ == "__main__":
    unittest.main()
