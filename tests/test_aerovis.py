"""AeroVIS translated into the pool store, without a teacher.

Every other pool source needs a segmenter; this one arrives with its masks
already made, so the risk moves from "is the mask any good" to "did the
translation keep the mask, the box and the frame pointing at the same object".
These cases are that question, plus the four schema quirks
`tools/inspect_aerovis.py` measured against the real 12.6 GiB release: the
`iscrowd=1` ignore regions with null boxes, `counts` arriving as `str`, boxes
in XYWH that are *not* the mask envelope, and 149 boxes with no mask.
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

from src.training.aerovis import (  # noqa: E402
    SOURCE_PREFIXES,
    decode_rle,
    frame_instances,
    load_index,
    sequence_split,
    source_of,
    summarise,
    write_pool,
)
from src.training.labels import MASK_STORE, open_masks  # noqa: E402
from src.training.pool import RECORD_FILE  # noqa: E402

try:
    from pycocotools import mask as mask_utils
    HAVE_COCO = True
except ImportError:                              # pragma: no cover
    HAVE_COCO = False

try:
    import cv2
    HAVE_CV2 = True
except ImportError:                              # pragma: no cover
    HAVE_CV2 = False


def encode(array: np.ndarray) -> dict:
    """A COCO compressed RLE with `counts` as `str`, the way JSON delivers it."""
    if HAVE_COCO:
        raw = mask_utils.encode(np.asfortranarray(array.astype(np.uint8)))
        return {"size": [int(array.shape[0]), int(array.shape[1])],
                "counts": raw["counts"].decode()}
    raise unittest.SkipTest("pycocotools is needed to build the fixture")


def block(height, width, y0, x0, h, w):
    array = np.zeros((height, width), np.uint8)
    array[y0:y0 + h, x0:x0 + w] = 1
    return array


class TestDecodeRle(unittest.TestCase):
    @unittest.skipUnless(HAVE_COCO, "pycocotools is not installed")
    def test_it_agrees_with_pycocotools_on_random_masks(self):
        rng = np.random.default_rng(0)
        for trial in range(24):
            height, width = int(rng.integers(4, 48)), int(rng.integers(4, 48))
            array = ((rng.random((height, width)) > 0.5).astype(np.uint8)
                     if trial % 5 == 0
                     else block(height, width, 1, 1, height // 2, width // 2))
            raw = mask_utils.encode(np.asfortranarray(array))
            reference = mask_utils.decode(raw).astype(bool)
            np.testing.assert_array_equal(
                decode_rle({"size": [height, width],
                            "counts": raw["counts"].decode()}), reference)

    @unittest.skipUnless(HAVE_COCO, "pycocotools is not installed")
    def test_the_fallback_matches_the_reference_implementation(self):
        # The fallback is what runs where pycocotools is not installed, so it
        # cannot be trusted on the grounds that pycocotools was tested.
        array = block(20, 30, 3, 4, 9, 11)
        segmentation = encode(array)
        real_import = __builtins__["__import__"] if isinstance(
            __builtins__, dict) else __builtins__.__import__

        def blocked(name, *args, **kwargs):
            if name.startswith("pycocotools"):
                raise ImportError("hidden for this test")
            return real_import(name, *args, **kwargs)

        import builtins
        builtins.__import__ = blocked
        try:
            mine = decode_rle(segmentation)
        finally:
            builtins.__import__ = real_import
        np.testing.assert_array_equal(mine, array.astype(bool))

    @unittest.skipUnless(HAVE_COCO, "pycocotools is not installed")
    def test_counts_already_bytes_is_accepted_too(self):
        array = block(12, 14, 2, 2, 5, 5)
        raw = mask_utils.encode(np.asfortranarray(array))
        np.testing.assert_array_equal(
            decode_rle({"size": [12, 14], "counts": raw["counts"]}),
            array.astype(bool))


@unittest.skipUnless(HAVE_COCO, "pycocotools is not installed")
class TestIndex(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, payload):
        path = self.root / "aero_vis.json"
        path.write_text(json.dumps(payload))
        return path

    def test_ignore_regions_are_dropped_not_tripped_over(self):
        # The release has 80 of these and their bboxes are null throughout.
        path = self.write({
            "videos": [{"id": 1, "file_names": ["vd_1/0.jpg"]}],
            "categories": [{"id": 1, "name": "car"}],
            "annotations": [
                {"video_id": 1, "category_id": 1, "track_id": 1, "iscrowd": 0,
                 "bboxes": [[0, 0, 4, 4]], "segmentations": [None]},
                {"video_id": 1, "category_id": 1, "track_id": 2, "iscrowd": 1,
                 "bboxes": [None], "segmentations": [None]}]})
        data = load_index(path)
        self.assertEqual(len(data["annotations"]), 1)
        self.assertEqual(data["annotations"][0]["track_id"], 1)

    def test_a_file_that_is_not_ytvis_says_so_with_its_keys(self):
        path = self.write({"images": [], "annotations": []})
        with self.assertRaises(ValueError) as caught:
            load_index(path)
        self.assertIn("videos", str(caught.exception))


class TestSequenceSplit(unittest.TestCase):
    """Whole sequences, stratified, because the sources differ in class."""

    def videos(self):
        return ([{"id": i, "name": f"vd_{i:03d}"} for i in range(20)]
                + [{"id": 100 + i, "name": f"ud_{i:03d}"} for i in range(10)]
                + [{"id": 200 + i, "name": f"sd_{i:03d}"} for i in range(4)])

    def test_every_source_appears_on_both_sides(self):
        # vehicle exists only in UAVDT and boat only in SeaDronesSee, so a
        # split that empties one source of one side removes a whole class.
        train, held = sequence_split(self.videos(), holdout=0.2, seed=0)
        for group in (train, held):
            self.assertEqual(
                {source_of(v["name"]) for v in group},
                {"visdrone", "uavdt", "seadronessee"})

    def test_a_small_source_still_gives_up_one_sequence(self):
        train, held = sequence_split(self.videos(), holdout=0.05, seed=0)
        self.assertEqual(
            sum(1 for v in held if source_of(v["name"]) == "seadronessee"), 1)

    def test_no_sequence_is_on_both_sides(self):
        train, held = sequence_split(self.videos(), holdout=0.3, seed=1)
        self.assertFalse({v["id"] for v in train} & {v["id"] for v in held})
        self.assertEqual(len(train) + len(held), len(self.videos()))

    def test_zero_holdout_is_the_decision_stated_out_loud(self):
        train, held = sequence_split(self.videos(), holdout=0.0)
        self.assertEqual(held, [])
        self.assertEqual(len(train), len(self.videos()))

    def test_the_same_seed_splits_the_same_way(self):
        first = sequence_split(self.videos(), holdout=0.25, seed=7)[1]
        again = sequence_split(self.videos(), holdout=0.25, seed=7)[1]
        self.assertEqual([v["id"] for v in first], [v["id"] for v in again])

    def test_a_holdout_of_one_is_refused(self):
        with self.assertRaises(ValueError):
            sequence_split(self.videos(), holdout=1.0)

    def test_an_unknown_prefix_is_its_own_stratum_not_a_crash(self):
        train, held = sequence_split(
            [{"id": i, "name": f"xx_{i}"} for i in range(4)], holdout=0.5)
        self.assertEqual(len(held), 2)
        self.assertEqual(source_of("xx_1"), "other")

    def test_the_prefixes_are_the_three_the_archive_ships(self):
        self.assertEqual(set(SOURCE_PREFIXES), {"vd_", "ud_", "sd_"})


@unittest.skipUnless(HAVE_COCO and HAVE_CV2, "pycocotools and OpenCV needed")
class TestWritePool(unittest.TestCase):
    """The translated frames have to be indistinguishable from harvested ones."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.sequences = self.root / "sequences"
        (self.sequences / "vd_001").mkdir(parents=True)
        (self.sequences / "ud_001").mkdir(parents=True)
        for folder in ("vd_001", "ud_001"):
            for frame in range(2):
                cv2.imwrite(str(self.sequences / folder / f"{frame:05d}.jpg"),
                            np.full((20, 30, 3), 40, np.uint8))
        self.data = {
            "videos": [
                {"id": 1, "name": "vd_001",
                 "file_names": ["vd_001/00000.jpg", "vd_001/00001.jpg"]},
                {"id": 2, "name": "ud_001",
                 "file_names": ["ud_001/00000.jpg", "ud_001/00001.jpg"]}],
            "categories": [{"id": 1, "name": "car"}, {"id": 2, "name": "person"}],
            "annotations": [
                {"video_id": 1, "category_id": 1, "track_id": 11, "iscrowd": 0,
                 "bboxes": [[3.0, 4.0, 11.0, 9.0], [4.0, 4.0, 11.0, 9.0]],
                 "segmentations": [encode(block(20, 30, 4, 3, 9, 11)),
                                   encode(block(20, 30, 4, 4, 9, 11))]},
                {"video_id": 1, "category_id": 2, "track_id": 12, "iscrowd": 0,
                 # A box with no mask on frame 1: 149 of these exist and they
                 # are not a triple.
                 "bboxes": [[15.0, 2.0, 5.0, 5.0], [15.0, 2.0, 5.0, 5.0]],
                 "segmentations": [encode(block(20, 30, 2, 15, 5, 5)), None]},
                {"video_id": 2, "category_id": 1, "track_id": 21, "iscrowd": 0,
                 "bboxes": [[1.0, 1.0, 6.0, 6.0], None],
                 "segmentations": [encode(block(20, 30, 1, 1, 6, 6)), None]}]}

    def test_a_frame_becomes_the_same_files_a_harvest_would_write(self):
        report = write_pool(self.data, self.sequences, self.root / "pool")
        folder = self.root / "pool" / "aerovis" / "vd_001" / "00000"
        self.assertTrue((folder / RECORD_FILE).is_file())
        store = open_masks(folder / MASK_STORE)
        self.assertEqual(sorted(store), [0, 1])
        self.assertEqual(store.shape, (20, 30))
        self.assertEqual(report["frames"], 3)
        self.assertEqual(report["instances"], 4)

    def test_the_box_is_converted_from_xywh_and_left_where_it_was_drawn(self):
        write_pool(self.data, self.sequences, self.root / "pool")
        record = json.loads((self.root / "pool" / "aerovis" / "vd_001"
                             / "00000" / RECORD_FILE).read_text())
        car = next(i for i in record["instances"] if i["class"] == "car")
        self.assertEqual(car["box"], [3.0, 4.0, 14.0, 13.0])
        # ...and it is deliberately not the mask envelope, which is 3..13/4..12.
        mask = open_masks(self.root / "pool" / "aerovis" / "vd_001" / "00000"
                          / MASK_STORE)[car["i"]]
        rows, cols = np.where(mask)
        self.assertEqual((int(cols.max()) + 1, int(rows.max()) + 1), (14, 13))

    def test_a_box_without_a_mask_is_not_written_as_an_instance(self):
        write_pool(self.data, self.sequences, self.root / "pool")
        record = json.loads((self.root / "pool" / "aerovis" / "vd_001"
                             / "00001" / RECORD_FILE).read_text())
        self.assertEqual([i["class"] for i in record["instances"]], ["car"])

    def test_the_relative_path_travels_with_the_pool(self):
        # The store goes to Drive and the 12.6 GiB of frames does not, so a
        # training run has to be able to re-point at its own copy.
        write_pool(self.data, self.sequences, self.root / "pool")
        record = json.loads((self.root / "pool" / "aerovis" / "vd_001"
                             / "00000" / RECORD_FILE).read_text())
        self.assertEqual(record["image_rel"], "vd_001/00000.jpg")
        self.assertTrue(record["image"].endswith("vd_001/00000.jpg"))

    def test_the_track_id_and_source_survive_for_stage_c(self):
        write_pool(self.data, self.sequences, self.root / "pool")
        record = json.loads((self.root / "pool" / "aerovis" / "ud_001"
                             / "00000" / RECORD_FILE).read_text())
        self.assertEqual(record["source"], "uavdt")
        self.assertEqual(record["instances"][0]["track_id"], 21)
        self.assertEqual(record["teacher"], "aerovis:ytvis")

    def test_no_gate_runs_because_these_are_not_a_teachers_masks(self):
        write_pool(self.data, self.sequences, self.root / "pool")
        for record_path in (self.root / "pool").rglob(RECORD_FILE):
            for instance in json.loads(record_path.read_text())["instances"]:
                self.assertIsNone(instance["verdict"])

    def test_resume_skips_a_finished_frame(self):
        write_pool(self.data, self.sequences, self.root / "pool")
        again = write_pool(self.data, self.sequences, self.root / "pool")
        self.assertEqual(again["frames"], 0)
        self.assertEqual(again["resumed"], 3)

    def test_only_the_chosen_sequences_are_written(self):
        train, held = sequence_split(self.data["videos"], holdout=0.5, seed=0)
        write_pool(self.data, self.sequences, self.root / "pool",
                   dataset="aerovis_train", videos=train)
        written = {p.parts[-3] for p in
                   (self.root / "pool" / "aerovis_train").rglob(MASK_STORE)}
        self.assertEqual(written, {v["name"] for v in train})
        self.assertFalse(written & {v["name"] for v in held})

    def test_a_missing_image_is_counted_rather_than_crashing(self):
        (self.sequences / "vd_001" / "00000.jpg").unlink()
        report = write_pool(self.data, self.sequences, self.root / "pool")
        self.assertEqual(report["missing_image"], 1)
        self.assertEqual(report["frames"], 2)

    def test_the_census_counts_triples_not_entries(self):
        table = summarise(self.data)
        self.assertIn("| visdrone | 1 | 2 | 3 |", table)
        self.assertIn("| uavdt | 1 | 1 | 1 |", table)


class TestFrameInstances(unittest.TestCase):
    def test_a_mask_without_a_box_is_not_a_triple_either(self):
        data = {"videos": [], "categories": [],
                "annotations": [{"video_id": 1, "category_id": 1, "track_id": 1,
                                 "bboxes": [None],
                                 "segmentations": [{"size": [4, 4],
                                                    "counts": "0"}]}]}
        self.assertEqual(frame_instances(data), {})


if __name__ == "__main__":
    unittest.main()
