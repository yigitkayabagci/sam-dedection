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
import zipfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.boxes import (  # noqa: E402
    BIRDSAI_SPECIES,
    DRONEVEHICLE_ALIASES,
    dronevehicle_only_frames,
    vtuav_frames,
    dronevehicle_shared_frames,
    VISDRONE_NAMES,
    BoxFrame,
    birdsai_frames,
    class_histogram,
    coco_frames,
    dronevehicle_frames,
    hituav_frames,
    rgbtdroneperson_frames,
    kust4k_frames,
    summarise_frames,
    vtuavdet_frames,
    filter_by_scale,
    scale_table,
    yolo_frames,
    yolo_label_frames,
)
from src.training.labels import (  # noqa: E402
    MASK_STORE,
    transformers_import_message,
    Gates,
    Sam2Teacher,
    Sam3Teacher,
    open_masks,
    teacher_class_for,
)
from src.training.masklets import find_rgbt_sequences, masklet_sequence  # noqa: E402
from src.training.aerial import SPECS, read_mask  # noqa: E402
from src.training.pool import (  # noqa: E402
    RECORD_FILE,
    agreement_table,
    boundary_agreement,
    intact_modalities,
    label_boxes,
    label_many,
    calibration_table,
    label_pool,
    modality_agreement,
    pool_report,
    summarise_pool,
    write_index,
)
from tools.fetch_datasets import (  # noqa: E402
    RECIPES,
    staged,
    stream_extract,
    tracked_members,
)

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
        self.batches = 0          # forward passes, as opposed to crops

    def masks_for(self, crops, boxes):
        self.batches += 1
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


class TestYoloLabelFrames(unittest.TestCase):
    """The label-side reader: the survey runs before the images are unpacked."""

    def test_the_frames_are_built_with_not_one_image_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_yolo(Path(tmp))
            for image in (Path(tmp) / "images").iterdir():
                image.unlink()                      # still inside the archive
            frames = yolo_label_frames(tmp)
            self.assertEqual([f.key for f in frames], ["a", "b"])
            self.assertEqual(frames[0].image,
                             Path(tmp) / "images" / "a.jpg")
            self.assertTrue(frames[0].normalized)
            self.assertEqual(frames[0].classes, ("car", "pedestrian"))

    def test_an_extracted_image_names_its_own_suffix_and_the_rest_follow_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_yolo(Path(tmp), stems=("a", "b"))
            (Path(tmp) / "images" / "a.jpg").rename(Path(tmp) / "images" / "a.png")
            (Path(tmp) / "images" / "b.jpg").unlink()
            frames = {f.key: f.image for f in yolo_label_frames(tmp)}
            self.assertEqual(frames["a"].suffix, ".png")
            self.assertEqual(frames["b"].suffix, ".png")

    def test_the_suffix_can_be_stated_when_nothing_is_unpacked(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_yolo(Path(tmp))
            for image in (Path(tmp) / "images").iterdir():
                image.unlink()
            frames = yolo_label_frames(tmp, suffix=".tif")
            self.assertEqual(frames[0].image.suffix, ".tif")

    def test_a_background_image_is_kept_where_the_teacher_side_drops_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_yolo(Path(tmp))
            (Path(tmp) / "labels" / "b.txt").write_text("")
            frames = yolo_label_frames(tmp)
            self.assertEqual(len(frames), 2)        # the export said two
            self.assertEqual(len(frames[1].boxes), 0)
            self.assertEqual(len(yolo_frames(tmp)), 1)

    def test_the_archive_listing_names_the_extension_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_yolo(Path(tmp))
            for image in (Path(tmp) / "images").iterdir():
                image.unlink()
            frames = yolo_label_frames(tmp, members=[
                "merged/images/a.png", "merged/images/b.tif",
                "merged/labels/a.txt", "merged/data.yaml"])
            self.assertEqual([f.image.suffix for f in frames],
                             [".png", ".tif"])       # per frame, not a guess

    def test_a_listing_from_another_tree_fails_before_the_gpu_is_rented(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_yolo(Path(tmp))
            for image in (Path(tmp) / "images").iterdir():
                image.unlink()
            with self.assertRaises(FileNotFoundError) as caught:
                yolo_label_frames(tmp, members=["other/images/z.jpg"])
            self.assertIn("two different", str(caught.exception))
            self.assertIn("other/images/z.jpg", str(caught.exception))

    def test_a_split_folder_is_matched_and_a_sibling_split_is_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "labels/train").mkdir(parents=True)
            (root / "labels/train/a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
            frames = yolo_label_frames(
                root, images="images/train", labels="labels/train",
                members=["m/images/val/a.png", "m/images/train/a.bmp"])
            self.assertEqual(frames[0].image,
                             root / "images/train" / "a.bmp")

    def test_a_classes_txt_beside_the_labels_is_not_read_as_a_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_yolo(Path(tmp))
            (Path(tmp) / "labels" / "classes.txt").write_text("car\nperson\n")
            self.assertEqual([f.key for f in yolo_label_frames(tmp)], ["a", "b"])

    def test_a_split_that_indexes_nothing_is_a_loud_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_yolo(Path(tmp))
            with self.assertRaises(FileNotFoundError) as caught:
                yolo_label_frames(tmp, labels="labels/train")
            self.assertIn("labels/train", str(caught.exception))


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

    def test_the_two_spellings_of_freight_car_become_one_class(self):
        # Measured over all 35 980 train XMLs: `feright_car` 7 419 boxes and
        # `feright car` 3 773. Left alone, a class histogram reports the same
        # vehicle twice and any name filter loses a third of them.
        with tempfile.TemporaryDirectory() as tmp:
            root = self.fixture(Path(tmp))
            (root / "train" / "trainlabelr" / "00002.xml").write_text(
                "<annotation>"
                "<object><name>feright car</name><bndbox>"
                "<xmin>10</xmin><ymin>10</ymin><xmax>60</xmax><ymax>50</ymax>"
                "</bndbox></object>"
                "<object><name>feright_car</name><bndbox>"
                "<xmin>70</xmin><ymin>10</ymin><xmax>120</xmax><ymax>50</ymax>"
                "</bndbox></object>"
                "<object><name>truvk</name><bndbox>"
                "<xmin>10</xmin><ymin>70</ymin><xmax>60</xmax><ymax>110</ymax>"
                "</bndbox></object></annotation>")
            (root / "train" / "trainimgr" / "00002.jpg").write_bytes(b"jpg")
            counts = class_histogram(dronevehicle_frames(root))
            self.assertEqual(counts.get("feright_car"), 2)
            self.assertNotIn("feright car", counts)
            self.assertEqual(counts.get("truck"), 2)   # one real, one typo

    def test_an_unrecognised_name_is_left_for_the_probe_to_show(self):
        # `*` appears once in the archive. The reader must not guess at it.
        self.assertNotIn("*", DRONEVEHICLE_ALIASES)
        self.assertEqual(set(DRONEVEHICLE_ALIASES.values()),
                         {"feright_car", "truck"})

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

    def test_the_scale_table_measures_the_long_side_as_a_fraction(self):
        frames = [BoxFrame(key="a", image=Path("a"), normalized=True,
                           boxes=np.array([[0.5, 0.5, 0.02, 0.04],
                                           [0.5, 0.5, 0.90, 0.10]]),
                           classes=("car", "car"))]
        table = scale_table(frames, max_rel=0.5)
        self.assertIn("| all | 1 | 2 |", table)
        self.assertIn("| 0.900 |", table)           # the long side, not the area
        self.assertIn("| 50.0 % |", table)          # one of two boxes >= max_rel

    def test_groups_come_from_the_key_and_an_unmapped_frame_says_so(self):
        frames = [BoxFrame(key="a", image=Path("a"), normalized=True,
                           boxes=np.array([[0.5, 0.5, 0.02, 0.02]]),
                           classes=("car",)),
                  BoxFrame(key="b", image=Path("b"), normalized=True,
                           boxes=np.array([[0.5, 0.5, 0.60, 0.60]] * 2),
                           classes=("car", "car"))]
        table = scale_table(frames, groups={"a": "visdrone"}, max_rel=0.5)
        self.assertIn("| ? | 1 | 2 |", table)       # densest group first
        self.assertIn("| visdrone | 1 | 1 |", table)
        self.assertIn("| all | 2 | 3 |", table)     # and a total under them

    def test_the_scale_filter_drops_both_ends_and_the_report_adds_up(self):
        frames = [BoxFrame(key="a", image=Path("a"), normalized=True,
                           boxes=np.array([[0.5, 0.5, 0.005, 0.005],
                                           [0.5, 0.5, 0.100, 0.050],
                                           [0.5, 0.5, 0.900, 0.900]]),
                           classes=("car", "car", "car"))]
        kept, report = filter_by_scale(frames, min_rel=0.01, max_rel=0.5)
        self.assertEqual(kept[0].classes, ("car",))       # only the middle one
        np.testing.assert_allclose(kept[0].boxes[0, 2], 0.100)
        self.assertEqual(report["below_min"], 1)
        self.assertEqual(report["above_max"], 1)
        self.assertEqual(report["boxes_kept"] + report["below_min"]
                         + report["above_max"] + report["quota_boxes"],
                         report["boxes"])

    def test_a_frame_left_with_nothing_to_prompt_is_dropped_and_counted(self):
        frames = [BoxFrame(key="a", image=Path("a"), normalized=True,
                           boxes=np.array([[0.5, 0.5, 0.90, 0.90]]),
                           classes=("car",)),
                  BoxFrame(key="b", image=Path("b"), normalized=True,
                           boxes=np.array([[0.5, 0.5, 0.10, 0.10]]),
                           classes=("car",))]
        kept, report = filter_by_scale(frames, max_rel=0.5)
        self.assertEqual([f.key for f in kept], ["b"])
        self.assertEqual(report["emptied"], 1)
        self.assertEqual(report["kept"], 1)

    def test_the_quota_is_a_share_of_the_kept_set_and_drops_whole_frames(self):
        small = [BoxFrame(key=f"s{i}", image=Path("s"), normalized=True,
                          boxes=np.array([[0.5, 0.5, 0.05, 0.05]]),
                          classes=("car",)) for i in range(8)]
        large = [BoxFrame(key=f"l{i}", image=Path("l"), normalized=True,
                          boxes=np.array([[0.5, 0.5, 0.30, 0.30],
                                          [0.5, 0.5, 0.05, 0.05]]),
                          classes=("car", "car")) for i in range(6)]
        kept, report = filter_by_scale(small + large, large_rel=0.25,
                                       large_quota=0.2, max_rel=0.5)
        # 8 small, quota 0.2 of the final set: floor(0.2 * 8 / 0.8) = 2.
        close_ups = [f for f in kept if f.key.startswith("l")]
        self.assertEqual(len(close_ups), 2)
        self.assertEqual(len(kept), 10)
        self.assertEqual(report["large"], 6)
        self.assertEqual(report["over_quota"], 4)
        self.assertEqual(report["quota_boxes"], 8)     # two boxes each
        # A close-up keeps all its boxes -- the quota is not a box filter.
        self.assertEqual(len(close_ups[0].classes), 2)

    def test_the_same_seed_selects_the_same_close_ups(self):
        frames = [BoxFrame(key=f"k{i}", image=Path("k"), normalized=True,
                           boxes=np.array([[0.5, 0.5, 0.30, 0.30]]),
                           classes=("car",)) for i in range(12)]
        frames += [BoxFrame(key="s", image=Path("s"), normalized=True,
                            boxes=np.array([[0.5, 0.5, 0.05, 0.05]]),
                            classes=("car",))] * 4
        first = filter_by_scale(frames, large_quota=0.5, seed=7)[0]
        again = filter_by_scale(frames, large_quota=0.5, seed=7)[0]
        other = filter_by_scale(frames, large_quota=0.5, seed=8)[0]
        self.assertEqual([f.key for f in first], [f.key for f in again])
        self.assertNotEqual([f.key for f in first], [f.key for f in other])
        # ...and the input order survives the draw.
        self.assertEqual([f.key for f in first],
                         sorted([f.key for f in first],
                                key=lambda k: [f.key for f in frames].index(k)))

    def test_pixel_boxes_are_refused_rather_than_scaled_by_guess(self):
        frames = [BoxFrame(key="a", image=Path("a"),
                           boxes=np.array([[0, 0, 10, 10]]), classes=("car",))]
        with self.assertRaises(ValueError) as caught:
            filter_by_scale(frames)
        self.assertIn("normalised boxes", str(caught.exception))

    def test_pixel_boxes_are_counted_but_not_measured(self):
        frames = [BoxFrame(key="a", image=Path("a"),
                           boxes=np.array([[0, 0, 10, 10]]), classes=("car",))]
        table = scale_table(frames)
        self.assertIn("| all | 1 | 1 | -- |", table)
        self.assertIn("carry no fraction", table)


class TestTeacherImportMessage(unittest.TestCase):
    """A broken neighbour package must not be reported as a missing class."""

    def test_transformers_itself_is_the_version_message(self):
        message = transformers_import_message(
            ImportError("cannot import name 'Sam3TrackerModel'",
                        name="transformers"), "Sam3Tracker")
        self.assertIn("transformers>=5", message)

    def test_a_dependency_is_named_with_its_pip_name_and_a_restart(self):
        # What Colab actually raises: transformers imports torchvision, which
        # imports a half-upgraded Pillow, and the ImportError names PIL.
        message = transformers_import_message(
            ImportError("cannot import name '_Ink' from 'PIL._typing'",
                        name="PIL._typing"), "Sam3Tracker")
        self.assertIn("pillow", message)
        self.assertIn("--force-reinstall", message)
        self.assertIn("restart the runtime", message)
        self.assertNotIn("transformers>=5", message)   # the wrong fix

    def test_an_exception_with_no_module_name_still_blames_transformers(self):
        message = transformers_import_message(ImportError("boom"), "X")
        self.assertIn("transformers has no X classes", message)



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


class TestSharedFrames(unittest.TestCase):
    """The boxes both halves annotate identically, prompted on RGB."""

    SAME = ("<object><name>car</name><polygon>"
            "<x1>150</x1><y1>150</y1><x2>200</x2><y2>160</y2>"
            "<x3>190</x3><y3>210</y3><x4>140</x4><y4>200</y4>"
            "</polygon></object>")
    # Same envelope as SAME, different corners: a rotated box the two files
    # drew differently. The strict key has to reject it.
    ROTATED = ("<object><name>car</name><polygon>"
               "<x1>140</x1><y1>150</y1><x2>200</x2><y2>150</y2>"
               "<x3>200</x3><y3>210</y3><x4>140</x4><y4>210</y4>"
               "</polygon></object>")
    OTHER = ("<object><name>bus</name><bndbox>"
             "<xmin>300</xmin><ymin>300</ymin><xmax>380</xmax><ymax>360</ymax>"
             "</bndbox></object>")

    def fixture(self, tmp: Path, thermal_xml: str, rgb_xml: str) -> Path:
        root = tmp / "dv"
        for folder in ("trainimg", "trainimgr", "trainlabel", "trainlabelr"):
            (root / "train" / folder).mkdir(parents=True)
        for folder in ("trainimg", "trainimgr"):
            (root / "train" / folder / "00001.jpg").write_bytes(b"jpg")
        (root / "train" / "trainlabelr" / "00001.xml").write_text(
            f"<annotation>{thermal_xml}</annotation>")
        (root / "train" / "trainlabel" / "00001.xml").write_text(
            f"<annotation>{rgb_xml}</annotation>")
        return root

    def test_only_the_identical_geometry_survives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.fixture(Path(tmp),
                                self.SAME + self.OTHER,
                                self.SAME)
            frame, = dronevehicle_shared_frames(root)
            self.assertEqual(frame.classes, ("car",))
            np.testing.assert_allclose(frame.boxes[0], [140, 150, 200, 210])

    def test_the_same_envelope_drawn_differently_is_not_shared(self):
        # This is why the key is all eight coordinates and not `_envelope`.
        with tempfile.TemporaryDirectory() as tmp:
            root = self.fixture(Path(tmp), self.SAME, self.ROTATED)
            with self.assertRaises(FileNotFoundError):
                dronevehicle_shared_frames(root)

    def test_the_rgb_half_is_the_image_and_the_thermal_half_the_pair(self):
        # The opposite of `dronevehicle_frames`, because the teacher reads the
        # RGB pixels here and the mask is mirrored onto the thermal frame.
        with tempfile.TemporaryDirectory() as tmp:
            root = self.fixture(Path(tmp), self.SAME, self.SAME)
            frame, = dronevehicle_shared_frames(root)
            self.assertEqual(frame.image.parent.name, "trainimg")
            self.assertEqual(frame.pair.parent.name, "trainimgr")
            self.assertEqual(frame.inset, 100)

    def test_the_two_spellings_still_match_each_other(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.fixture(
                Path(tmp),
                self.OTHER.replace("bus", "feright car"),
                self.OTHER.replace("bus", "feright_car"))
            frame, = dronevehicle_shared_frames(root)
            self.assertEqual(frame.classes, ("feright_car",))

    def test_a_box_only_one_half_draws_is_left_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.fixture(Path(tmp), self.SAME + self.OTHER, self.OTHER)
            frame, = dronevehicle_shared_frames(root)
            self.assertEqual(frame.classes, ("bus",))


class TestOnlyFrames(unittest.TestCase):
    """The other half of the split: targets one modality never annotates."""

    def boxed(self, name: str, x: int, y: int) -> str:
        return (f"<object><name>{name}</name><bndbox>"
                f"<xmin>{x}</xmin><ymin>{y}</ymin>"
                f"<xmax>{x + 50}</xmax><ymax>{y + 40}</ymax>"
                f"</bndbox></object>")

    def fixture(self, tmp: Path, thermal: str, rgb: str) -> Path:
        root = tmp / "dv"
        for folder in ("trainimg", "trainimgr", "trainlabel", "trainlabelr"):
            (root / "train" / folder).mkdir(parents=True)
        for folder in ("trainimg", "trainimgr"):
            (root / "train" / folder / "00001.jpg").write_bytes(b"jpg")
        (root / "train" / "trainlabelr" / "00001.xml").write_text(
            f"<annotation>{thermal}</annotation>")
        (root / "train" / "trainlabel" / "00001.xml").write_text(
            f"<annotation>{rgb}</annotation>")
        return root

    def test_a_box_only_the_thermal_half_draws_is_thermal_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.fixture(
                Path(tmp),
                self.boxed("car", 200, 200) + self.boxed("bus", 500, 400),
                self.boxed("car", 200, 200))
            frame, = dronevehicle_only_frames(root, modality="thermal")
            self.assertEqual(frame.classes, ("bus",))
            self.assertEqual(frame.image.parent.name, "trainimgr")
            self.assertEqual(frame.pair.parent.name, "trainimg")
            self.assertEqual(frame.inset, 100)

    def test_the_rgb_side_is_the_mirror_image_of_that(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.fixture(
                Path(tmp),
                self.boxed("car", 200, 200),
                self.boxed("car", 200, 200) + self.boxed("van", 500, 400))
            frame, = dronevehicle_only_frames(root, modality="rgb")
            self.assertEqual(frame.classes, ("van",))
            self.assertEqual(frame.image.parent.name, "trainimg")
            self.assertEqual(frame.pair.parent.name, "trainimgr")

    def test_a_counterpart_drawn_slightly_differently_is_not_only(self):
        # ~36 % of thermal boxes are this case: same vehicle, different
        # rectangle. Harvesting them here would put two masks on one object.
        with tempfile.TemporaryDirectory() as tmp:
            root = self.fixture(Path(tmp),
                                self.boxed("car", 200, 200),
                                self.boxed("car", 210, 208))
            with self.assertRaises(FileNotFoundError):
                dronevehicle_only_frames(root, modality="thermal")

    def test_the_distance_gate_is_what_decides_that(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.fixture(Path(tmp),
                                self.boxed("car", 200, 200),
                                self.boxed("car", 210, 208))
            frame, = dronevehicle_only_frames(root, modality="thermal",
                                              distance=5.0)
            self.assertEqual(frame.classes, ("car",))

    def test_a_counterpart_of_another_class_does_not_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.fixture(Path(tmp),
                                self.boxed("car", 200, 200),
                                self.boxed("bus", 200, 200))
            frame, = dronevehicle_only_frames(root, modality="thermal")
            self.assertEqual(frame.classes, ("car",))

    def test_the_shared_and_only_subsets_do_not_overlap(self):
        # An identical box has distance 0, so it is matched at any gate.
        with tempfile.TemporaryDirectory() as tmp:
            root = self.fixture(
                Path(tmp),
                self.boxed("car", 200, 200) + self.boxed("bus", 500, 400),
                self.boxed("car", 200, 200))
            shared, = dronevehicle_shared_frames(root)
            only, = dronevehicle_only_frames(root, modality="thermal")
            self.assertEqual(shared.classes, ("car",))
            self.assertEqual(only.classes, ("bus",))

    def test_an_unknown_modality_is_refused(self):
        with self.assertRaises(ValueError):
            dronevehicle_only_frames("/nowhere", modality="infrared")


class TestVtuavFrames(unittest.TestCase):
    """One target per sequence, one box every tenth frame, per modality."""

    def fixture(self, tmp: Path, rgb_lines: str, ir_lines: str,
                frames: int = 31, both: bool = True) -> Path:
        root = tmp / "VTUAV" / "bus_017"
        (root / "rgb").mkdir(parents=True)
        if both:
            (root / "ir").mkdir()
        for index in range(frames):
            (root / "rgb" / f"{index:06d}.jpg").write_bytes(b"jpg")
            if both:
                (root / "ir" / f"{index:06d}.jpg").write_bytes(b"jpg")
        (root / "rgb.txt").write_text(rgb_lines)
        (root / "ir.txt").write_text(ir_lines)
        return tmp / "VTUAV"

    def test_line_k_is_frame_ten_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.fixture(Path(tmp),
                                "10 20 30 40\n11 21 31 41\n12 22 32 42\n",
                                "10 20 30 40\n11 21 31 41\n12 22 32 42\n")
            frames = vtuav_frames(root)
            self.assertEqual([f.key for f in frames],
                             ["bus_017/000000", "bus_017/000010",
                              "bus_017/000020"])
            np.testing.assert_allclose(frames[0].boxes[0], [10, 20, 40, 60])

    def test_an_absent_target_is_dropped_not_boxed_at_zero(self):
        # Long-term sequences mark the target out of view with a null box; a
        # zero-area prompt is the least useful thing to hand a teacher.
        with tempfile.TemporaryDirectory() as tmp:
            root = self.fixture(Path(tmp),
                                "10 20 30 40\n0 0 0 0\n12 22 32 42\n",
                                "10 20 30 40\n0 0 0 0\n12 22 32 42\n")
            self.assertEqual([f.key for f in vtuav_frames(root)],
                             ["bus_017/000000", "bus_017/000020"])

    def test_the_class_is_the_sequence_name_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.fixture(Path(tmp), "10 20 30 40\n", "10 20 30 40\n")
            self.assertEqual(vtuav_frames(root)[0].classes, ("bus",))

    def test_each_modality_reads_its_own_box_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.fixture(Path(tmp), "10 20 30 40\n", "15 25 30 40\n")
            rgb, = vtuav_frames(root, modality="rgb")
            ir, = vtuav_frames(root, modality="ir")
            np.testing.assert_allclose(rgb.boxes[0], [10, 20, 40, 60])
            np.testing.assert_allclose(ir.boxes[0], [15, 25, 45, 65])
            self.assertEqual(rgb.image.parent.name, "rgb")
            self.assertEqual(ir.image.parent.name, "ir")
            self.assertEqual(rgb.pair.parent.name, "ir")

    def test_a_half_extracted_tree_still_reads(self):
        # 16 unzips only `rgb/`, so the twin is simply not there.
        with tempfile.TemporaryDirectory() as tmp:
            root = self.fixture(Path(tmp), "10 20 30 40\n", "10 20 30 40\n",
                                both=False)
            frame, = vtuav_frames(root, modality="rgb")
            self.assertIsNone(frame.pair)

    def test_a_tree_extracted_for_the_other_modality_says_exactly_that(self):
        """The failure that cost a whole RGB run.

        `tracked_members` keeps both box files but only one modality's frames.
        Two arms sharing an extraction tree means the second one skips staging
        (the markers are there), then finds every `rgb.txt` present and every
        `rgb/` empty. Silently indexing nothing is the worst way to say that.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = self.fixture(Path(tmp), "10 20 30 40\n", "10 20 30 40\n")
            for stale in (root / "bus_017" / "rgb").iterdir():
                stale.unlink()
            with self.assertRaises(FileNotFoundError) as caught:
                vtuav_frames(root, modality="rgb")
            message = str(caught.exception)
            self.assertIn("not one rgb/ frame", message)
            self.assertIn("own root", message)

    def test_an_unknown_modality_is_refused(self):
        with self.assertRaises(ValueError):
            vtuav_frames("/nowhere", modality="thermal")


class TestTrackedMembers(unittest.TestCase):
    """Nine frames in ten carry no label; unzipping them is disk for nothing."""

    def archive(self, tmp: Path) -> zipfile.ZipFile:
        path = tmp / "part.zip"
        with zipfile.ZipFile(path, "w") as handle:
            for index in range(25):
                handle.writestr(f"bus_017/rgb/{index:06d}.jpg", b"x")
                handle.writestr(f"bus_017/ir/{index:06d}.jpg", b"x")
            handle.writestr("bus_017/rgb.txt", "1 2 3 4\n5 6 7 8\n9 1 2 3\n")
            handle.writestr("bus_017/ir.txt", "1 2 3 4\n5 6 7 8\n9 1 2 3\n")
        return zipfile.ZipFile(path)

    def test_only_the_annotated_frames_of_one_modality_are_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            keep = set(tracked_members(self.archive(Path(tmp)), "rgb"))
            self.assertEqual(
                {n for n in keep if n.endswith(".jpg")},
                {"bus_017/rgb/000000.jpg", "bus_017/rgb/000010.jpg",
                 "bus_017/rgb/000020.jpg"})

    def test_both_box_files_survive_either_way(self):
        with tempfile.TemporaryDirectory() as tmp:
            keep = set(tracked_members(self.archive(Path(tmp)), "ir"))
            self.assertIn("bus_017/rgb.txt", keep)
            self.assertIn("bus_017/ir.txt", keep)
            self.assertEqual({n for n in keep if n.endswith(".jpg")},
                             {"bus_017/ir/000000.jpg", "bus_017/ir/000010.jpg",
                              "bus_017/ir/000020.jpg"})

    def test_it_saves_most_of_the_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = self.archive(Path(tmp))
            self.assertLess(len(tracked_members(archive, "rgb")),
                            len(archive.namelist()) // 5)


class TestLabelMany(unittest.TestCase):
    """Crops pool across frames, or `batch_size` does nothing on sparse data."""

    def items(self, count: int, boxes_each: int = 2):
        pixels = np.zeros((100, 200, 3), np.uint8)
        boxes = np.array([[10.0, 10.0, 40.0, 40.0]] * boxes_each)
        return [(pixels, boxes) for _ in range(count)]

    def test_one_batch_can_span_several_frames(self):
        teacher = FakeImageTeacher()
        out = label_many(self.items(4), teacher, batch_size=8)
        self.assertEqual(teacher.batches, 1, "8 crops from 4 frames, one pass")
        self.assertEqual(teacher.calls, 8)
        self.assertEqual(len(out), 4)

    def test_results_come_back_split_by_frame_in_order(self):
        out = label_many(self.items(3), FakeImageTeacher(), batch_size=4)
        for masks, rows in out:
            self.assertEqual([r["i"] for r in rows], [0, 1])
            self.assertEqual(sorted(masks), [0, 1])

    def test_one_frame_gives_exactly_what_label_boxes_gave(self):
        pixels, boxes = self.items(1)[0]
        a_masks, a_rows = label_boxes(pixels, boxes, FakeImageTeacher())
        (b_masks, b_rows), = label_many([(pixels, boxes)], FakeImageTeacher())
        self.assertEqual(a_rows, b_rows)
        self.assertEqual(sorted(a_masks), sorted(b_masks))


class TestMirroredPool(unittest.TestCase):
    """One teacher pass, two pools -- the point of the shared-box subset."""

    def frames(self, tmp: Path, twin_shape=(100, 200)) -> list:
        write_yolo(tmp / "y", stems=("img0",))
        twin_dir = tmp / "t"
        twin_dir.mkdir()
        twin = twin_dir / "img0.jpg"
        if HAVE_CV2:
            cv2.imwrite(str(twin), np.full((*twin_shape, 3), 90, np.uint8))
        else:
            twin.write_bytes(b"jpg")
        base = yolo_frames(tmp / "y")[0]
        return [replace(base, pair=twin)]

    @unittest.skipUnless(HAVE_CV2, "needs OpenCV to decode the frames")
    def test_the_same_masks_are_filed_under_both_pools(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            teacher = FakeImageTeacher()
            report = label_pool(self.frames(tmp), teacher, tmp / "pool",
                                dataset="rgb", mirror="tir")
            self.assertEqual(report["mirrored"], 1)
            self.assertEqual(teacher.calls, 2, "one pass, not two")
            rgb = tmp / "pool" / "rgb" / "img0"
            tir = tmp / "pool" / "tir" / "img0"
            self.assertEqual((rgb / MASK_STORE).read_bytes(),
                             (tir / MASK_STORE).read_bytes())
            record = json.loads((tir / RECORD_FILE).read_text())
            self.assertEqual(record["prompt"], "mirror")
            self.assertEqual(record["dataset"], "tir")
            self.assertTrue(record["image"].endswith("t/img0.jpg"))
            self.assertIn("y/images/img0.jpg", record["mirror_of"])

    @unittest.skipUnless(HAVE_CV2, "needs OpenCV to decode the frames")
    def test_a_twin_of_another_shape_is_refused_not_stamped(self):
        # The one way this could be silently wrong.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            report = label_pool(self.frames(tmp, twin_shape=(64, 64)),
                                FakeImageTeacher(), tmp / "pool",
                                dataset="rgb", mirror="tir")
            self.assertEqual(report["mirror_mismatch"], 1)
            self.assertEqual(report["mirrored"], 0)
            self.assertFalse((tmp / "pool" / "tir").exists())

    @unittest.skipUnless(HAVE_CV2, "needs OpenCV to decode the frames")
    def test_a_mirror_asked_for_later_is_copied_not_re_labelled(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            frames = self.frames(tmp)
            label_pool(frames, FakeImageTeacher(), tmp / "pool", dataset="rgb")
            teacher = FakeImageTeacher()
            report = label_pool(frames, teacher, tmp / "pool", dataset="rgb",
                                mirror="tir")
            self.assertEqual(teacher.calls, 0, "resume must not re-prompt")
            self.assertEqual(report["mirrored"], 1)
            self.assertTrue((tmp / "pool" / "tir" / "img0" / MASK_STORE).is_file())

    @unittest.skipUnless(HAVE_CV2, "needs OpenCV to decode the frames")
    def test_both_pools_show_up_in_the_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            label_pool(self.frames(tmp), FakeImageTeacher(), tmp / "pool",
                       dataset="rgb", mirror="tir")
            summary = summarise_pool(pool_report(tmp / "pool"))
            self.assertIn("| rgb |", summary)
            self.assertIn("| tir |", summary)

    @unittest.skipUnless(HAVE_CV2, "needs OpenCV to decode the frames")
    def test_grouping_frames_shrinks_the_number_of_forward_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_yolo(Path(tmp) / "y", stems=("a", "b", "c", "d"))
            frames = yolo_frames(Path(tmp) / "y")
            one = FakeImageTeacher()
            label_pool(frames, one, tmp / "p1", dataset="toy", batch_size=32)
            grouped = FakeImageTeacher()
            label_pool(frames, grouped, tmp / "p2", dataset="toy",
                       batch_size=32, frame_group=4)
            self.assertEqual(one.batches, 4)
            self.assertEqual(grouped.batches, 1)
            self.assertEqual(one.calls, grouped.calls)


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
        from tools.make_mask_pool import DATASETS, RECIPE_FOR, frames_for  # noqa: F401
        from src.training import boxes

        for dataset in DATASETS:
            recipe = RECIPE_FOR.get(dataset, dataset)
            self.assertIn(recipe, RECIPES,
                          f"{dataset} has no fetch recipe ({recipe!r})")
        for dataset in ("rgbtdroneperson", "vtuavdet", "birdsai",
                        "dronevehicle_shared"):
            self.assertIn(dataset, DATASETS)
            self.assertTrue(hasattr(boxes, f"{dataset}_frames"), dataset)

    def test_a_shared_dataset_names_the_archive_it_actually_comes_from(self):
        # dronevehicle_shared is a view of the dronevehicle archive, not its
        # own download; the mapping is what stops the CLI offering a dataset
        # nothing can fetch.
        from tools.make_mask_pool import RECIPE_FOR

        self.assertEqual(RECIPE_FOR["dronevehicle_shared"], "dronevehicle")

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


# --------------------------------------------------------------------------
# Kust4K: boxes drawn from a semantic map, and which half the map describes
# --------------------------------------------------------------------------


def write_kust4k(root: Path, stems=("00001D", "00002D", "00003N"),
                 broken=(), corrupt="rgb"):
    """A miniature Kust4K: `tir/`, `rgb/`, `label/`, and its own manifests.

    Every frame holds one 12x12 car (class 4) on road (class 1). The half
    named by `corrupt` is replaced with unrelated noise in the frames listed
    in `broken`, which is what the real archive does to 1 160 of its 4 024.
    """
    import cv2

    rng = np.random.default_rng(0)
    for folder in ("tir", "rgb", "label"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    for stem in stems:
        semantic = np.ones((64, 64), dtype=np.uint8)
        semantic[20:32, 24:36] = 4                      # car
        cv2.imwrite(str(root / "label" / f"{stem}.png"), semantic)
        scene = rng.integers(0, 40, (64, 64), dtype=np.uint8)
        scene[20:32, 24:36] = 220                       # the car, as pixels
        noise = rng.integers(0, 255, (64, 64), dtype=np.uint8)
        for modality in ("rgb", "tir"):
            spoiled = stem in broken and modality == (
                "rgb" if corrupt == "rgb" else "tir")
            cv2.imwrite(str(root / modality / f"{stem}.png"),
                        noise if spoiled else scene)
    if broken:
        (root / f"broken_in_train_day_{len(broken)}.txt").write_text(
            "\n".join(f"{s}.png" for s in broken) + "\n")


@unittest.skipUnless(HAVE_CV2, "OpenCV is not installed")
class TestSemanticFrames(unittest.TestCase):
    """Kust4K annotates pixels, so its boxes are its components' envelopes."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_the_box_is_the_component_envelope_and_carries_its_class_name(self):
        write_kust4k(self.root)
        frames = kust4k_frames(self.root)
        self.assertEqual(len(frames), 3)
        self.assertEqual(frames[0].classes, ("car",))
        np.testing.assert_allclose(frames[0].boxes[0], [24, 20, 36, 32])

    def test_the_named_modality_is_the_image_and_the_other_is_the_pair(self):
        write_kust4k(self.root)
        rgb = kust4k_frames(self.root, modality="rgb")[0]
        self.assertEqual(rgb.image.parent.name, "rgb")
        self.assertEqual(rgb.pair.parent.name, "tir")
        thermal = kust4k_frames(self.root, modality="thermal")[0]
        self.assertEqual(thermal.image.parent.name, "tir")
        self.assertEqual(thermal.pair.parent.name, "rgb")

    def test_the_groups_split_on_the_datasets_own_manifests(self):
        write_kust4k(self.root, broken=("00002D",))
        self.assertEqual([f.key for f in kust4k_frames(self.root, group="clean")],
                         ["00001D", "00003N"])
        self.assertEqual([f.key for f in kust4k_frames(self.root, group="broken")],
                         ["00002D"])
        self.assertEqual(len(kust4k_frames(self.root, group="all")), 3)

    def test_asking_for_broken_frames_with_no_manifest_says_why(self):
        # `excluded_keys` globs the root and does not recurse, so a manifest
        # one folder down is invisible and every frame reads as clean. The
        # message has to name that, or a silently empty broken group looks
        # like a download with nothing wrong in it.
        write_kust4k(self.root)
        with self.assertRaises(FileNotFoundError) as caught:
            kust4k_frames(self.root, group="broken")
        self.assertIn("not recursive", str(caught.exception))

    def test_an_unknown_group_is_refused_before_any_globbing(self):
        with self.assertRaises(ValueError):
            kust4k_frames(self.root, group="daytime")


@unittest.skipUnless(HAVE_CV2, "OpenCV is not installed")
class TestModalityAgreement(unittest.TestCase):
    """Which half of a broken pair still shows the scene the map annotates."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_a_matching_image_scores_far_above_an_unrelated_one(self):
        write_kust4k(self.root, stems=("00001D",))
        records = modality_agreement(self.root, SPECS["kust4k"])
        self.assertGreater(records[0]["rgb"], 3.0)

        noise = np.random.default_rng(1).integers(
            0, 255, (64, 64), dtype=np.uint8)
        semantic = read_mask(self.root / "label" / "00001D.png")
        self.assertLess(boundary_agreement(
            np.dstack([noise] * 3), semantic), 2.0)

    def test_a_map_that_is_a_different_size_is_nan_not_a_number(self):
        write_kust4k(self.root, stems=("00001D",))
        semantic = read_mask(self.root / "label" / "00001D.png")
        self.assertTrue(np.isnan(boundary_agreement(
            np.zeros((32, 32, 3), np.uint8), semantic)))

    def test_the_corrupt_half_is_the_one_dropped(self):
        write_kust4k(self.root, stems=tuple(f"{i:05d}D" for i in range(12)),
                     broken=("00009D", "00010D"), corrupt="rgb")
        records = modality_agreement(self.root, SPECS["kust4k"])
        verdicts = intact_modalities(records)
        self.assertEqual(set(verdicts), {"00009D", "00010D"})
        for kept in verdicts.values():
            self.assertEqual(kept, ("thermal",))

    def test_the_threshold_comes_from_the_clean_frames_not_a_constant(self):
        write_kust4k(self.root, stems=tuple(f"{i:05d}D" for i in range(12)),
                     broken=("00009D",), corrupt="tir")
        records = modality_agreement(self.root, SPECS["kust4k"])
        self.assertEqual(intact_modalities(records)["00009D"], ("rgb",))
        with self.assertRaises(ValueError):
            intact_modalities([r for r in records if r["broken"]])

    def test_the_table_reports_what_each_broken_frame_kept(self):
        write_kust4k(self.root, stems=tuple(f"{i:05d}D" for i in range(12)),
                     broken=("00009D",), corrupt="rgb")
        records = modality_agreement(self.root, SPECS["kust4k"])
        table = agreement_table(records, intact_modalities(records))
        self.assertIn("| clean | 11 |", table)
        self.assertIn("| thermal | 1 |", table)


if __name__ == "__main__":
    unittest.main()
