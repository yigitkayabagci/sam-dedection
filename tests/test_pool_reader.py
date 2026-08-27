"""Reading a mask pool back as stage B training data, with no GPU.

The pool writer is covered by `tests/test_mask_pool.py`; this is the other
half of the same contract, and the cases here are the ways the two halves can
silently disagree rather than the ways either can crash:

* a mask filed under instance key `k` has to come back as instance `k`'s
  target, because that pairing is the whole store and nothing downstream
  re-checks it;
* an inset the harvest cut has to be added back when the *image* is read and
  nowhere else, because the boxes, the masks and the frame size are all in
  the inset frame and only the JPEG is not;
* a frame whose pixels are somewhere else now has to be found, and a frame
  whose pixels are a different size has to be refused.
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

from src.training.aerial import (InstanceGates, Sample, pool_masks,  # noqa: E402
                                 image_origin, sample_masks)
from src.training.labels import MASK_STORE, save_masks  # noqa: E402
from src.training.pool import RECORD_FILE  # noqa: E402
from src.training.pool_reader import (Relocator, group_records,  # noqa: E402
                                      index_pool, link_pool, load_pool_index,
                                      parse_pool, pool_datasets,
                                      save_pool_index, store_areas,
                                      why_no_image)

try:
    import cv2
except ImportError:                                      # pragma: no cover
    cv2 = None


def write_frame(pool: Path, key: str, image: Path, shape, boxes_masks,
                dataset: str = "demo", border: int = 0) -> Path:
    """One pool frame on disk: `record.json` beside its run-length store.

    `boxes_masks` is `[(class, box, mask_or_None)]` in the *inset* frame's
    coordinates -- a `None` mask is a box the gates rejected, which the store
    must not hold and the index must not carry.
    """
    target = pool / dataset
    for part in Path(key).parts:
        target = target / part
    target.mkdir(parents=True, exist_ok=True)
    record = {"key": key, "dataset": dataset, "prompt": "self",
              "image": str(image), "shape": [shape[0], shape[1]],
              "teacher": "test/teacher", "instances": []}
    masks = {}
    for i, (name, box, mask) in enumerate(boxes_masks):
        record["instances"].append({
            "i": i, "class": name, "box": [float(v) for v in box],
            "teacher_iou": 0.9, "verdict": None if mask is not None else "box_iou"})
        if mask is not None:
            masks[i] = mask
    (target / RECORD_FILE).write_text(json.dumps(record) + "\n")
    save_masks(target / MASK_STORE, shape, masks)
    return target


def blob(shape, box) -> np.ndarray:
    """A filled rectangle mask, so `box_iou` and `fill` are both 1.0."""
    mask = np.zeros(shape, dtype=bool)
    x0, y0, x1, y1 = (int(v) for v in box)
    mask[y0:y1, x0:x1] = True
    return mask


def write_image(path: Path, height: int, width: int, value: int = 90) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((height, width), value, np.uint8))
    return path


@unittest.skipIf(cv2 is None, "OpenCV is needed to write the frames")
class PoolIndexTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.pool = self.root / "pool"
        self.images = self.root / "data"
        self.shape = (200, 240)                       # (height, width)
        self.boxes = [("car", (20, 30, 60, 70)), ("truck", (120, 100, 190, 160))]

    def build(self, keys=("seq_a/000000", "seq_a/000010", "seq_b/000000"),
              border: int = 0, image_root: Path | None = None):
        image_root = image_root or self.images
        for key in keys:
            image = write_image(image_root / f"{key}.png",
                                self.shape[0] + 2 * border,
                                self.shape[1] + 2 * border)
            write_frame(self.pool, key, image, self.shape,
                        [(name, box, blob(self.shape, box))
                         for name, box in self.boxes])
        return self.pool / "demo"

    def test_every_accepted_box_becomes_an_indexed_instance(self):
        index = index_pool(self.build(), self.images, workers=1)
        self.assertEqual(len(index), 3)
        self.assertEqual(sum(len(e.instances) for e in index), 6)
        self.assertEqual(index[0].size, (self.shape[1], self.shape[0]))

    def test_a_rejected_box_is_not_in_the_index(self):
        image = write_image(self.images / "only.png", *self.shape)
        write_frame(self.pool, "only", image, self.shape, [
            ("car", self.boxes[0][1], blob(self.shape, self.boxes[0][1])),
            ("truck", self.boxes[1][1], None)])
        index = index_pool(self.pool / "demo", self.images, workers=1)
        self.assertEqual([i.label for e in index for i in e.instances], [0])

    def _strip_verdicts(self, target: Path) -> None:
        """Rewrite one frame's record the way a harvest with no verdict field
        would have written it, store untouched."""
        record_file = target / RECORD_FILE
        record = json.loads(record_file.read_text())
        for instance in record["instances"]:
            instance.pop("verdict", None)
        record_file.write_text(json.dumps(record) + "\n")

    def test_a_record_with_no_verdict_reads_acceptance_off_the_store(self):
        image = write_image(self.images / "only.png", *self.shape)
        target = write_frame(self.pool, "only", image, self.shape, [
            ("car", self.boxes[0][1], blob(self.shape, self.boxes[0][1])),
            ("truck", self.boxes[1][1], None)])
        self._strip_verdicts(target)
        index = index_pool(self.pool / "demo", self.images, workers=1)
        # The rejected box has no mask in the store, so it stays out of the
        # index even though the record no longer says it was rejected.
        self.assertEqual([i.label for e in index for i in e.instances], [0])

    def test_a_record_shaped_by_another_harvest_is_named_not_raised(self):
        image = write_image(self.images / "only.png", *self.shape)
        target = write_frame(self.pool, "only", image, self.shape, [
            ("car", self.boxes[0][1], blob(self.shape, self.boxes[0][1]))])
        record_file = target / RECORD_FILE
        record = json.loads(record_file.read_text())
        for instance in record["instances"]:
            instance["index"] = instance.pop("i")
        record_file.write_text(json.dumps(record) + "\n")
        with self.assertRaises(ValueError) as caught:
            index_pool(self.pool / "demo", self.images, workers=1)
        self.assertIn("record_schema", str(caught.exception))

    def test_a_missing_mask_still_disagrees_when_the_record_claims_it(self):
        image = write_image(self.images / "only.png", *self.shape)
        target = write_frame(self.pool, "only", image, self.shape, [
            ("car", self.boxes[0][1], blob(self.shape, self.boxes[0][1])),
            ("truck", self.boxes[1][1], None)])
        record_file = target / RECORD_FILE
        record = json.loads(record_file.read_text())
        record["instances"][1]["verdict"] = None      # accepted, but not stored
        record_file.write_text(json.dumps(record) + "\n")
        with self.assertRaises(ValueError) as caught:
            index_pool(self.pool / "demo", self.images, workers=1)
        self.assertIn("store_disagrees", str(caught.exception))

    def test_class_names_survive_the_round_trip(self):
        index = index_pool(self.build(), self.images, workers=1)
        spec = index[0].source.spec
        names = {spec.name_of(i.class_id) for e in index for i in e.instances}
        self.assertEqual(names, {"car", "truck"})
        self.assertEqual(spec.name, "pool/demo")

    def test_the_mask_that_comes_back_is_the_instance_it_was_filed_under(self):
        """The pairing the whole store rests on, checked rather than assumed.

        A mask store keyed by instance index and an index carrying the same
        indices agree only by construction; nothing downstream re-derives the
        pairing from the boxes, so an off-by-one here would train every
        instance against its neighbour's mask and the loss would stay finite
        the entire time.
        """
        index = index_pool(self.build(keys=("solo",)), self.images, workers=1)
        entry = index[0]
        sample = Sample(frame=entry.frame, origin=(0, 0),
                        window=(self.shape[1], self.shape[0]), size=self.shape[0],
                        instances=entry.instances, source=entry.source)
        masks = sample_masks(sample)
        for mask, instance in zip(masks, entry.instances):
            rows, cols = np.nonzero(mask)
            # The window is the whole frame but not square, so `sample_masks`
            # resized it; compare the box the mask spans in *relative* terms.
            self.assertAlmostEqual(cols.min() / mask.shape[1],
                                   instance.box[0] / self.shape[1], places=1)
            self.assertAlmostEqual(rows.min() / mask.shape[0],
                                   instance.box[1] / self.shape[0], places=1)

    def test_a_window_crops_the_stored_mask_the_same_way_it_crops_the_image(self):
        index = index_pool(self.build(keys=("solo",)), self.images, workers=1)
        entry = index[0]
        instance = entry.instances[0]
        origin, side = (10, 20), 128
        inside = [i for i in entry.instances if i.inside(origin, (side, side))]
        self.assertIn(instance, inside)
        sample = Sample(frame=entry.frame, origin=origin, window=(side, side),
                        size=side, instances=(instance,), source=entry.source)
        mask = sample_masks(sample)[0]
        rows, cols = np.nonzero(mask)
        self.assertEqual((cols.min(), rows.min()),
                         (instance.box[0] - origin[0], instance.box[1] - origin[1]))

    def test_an_inset_is_added_back_only_when_the_image_is_read(self):
        """DroneVehicle's white band: cut at harvest, still on disk at training.

        The boxes, the store and the frame size are all the inset frame's, so
        the one thing that has to move is where `load_image` starts reading.
        """
        index = index_pool(self.build(keys=("solo",), border=100), self.images,
                           workers=1)
        entry = index[0]
        self.assertEqual(entry.source.spec.border, 100)
        self.assertEqual(entry.size, (self.shape[1], self.shape[0]))
        sample = Sample(frame=entry.frame, origin=(10, 20), window=(64, 64),
                        size=64, instances=entry.instances[:1],
                        source=entry.source)
        self.assertEqual(image_origin(sample), (110, 120))
        # The mask is *not* offset -- it lives in the inset frame already.
        self.assertTrue(pool_masks(sample).shape == (1, 64, 64))

    def test_a_frame_of_the_wrong_size_is_refused_rather_than_stamped(self):
        image = write_image(self.images / "odd.png", 111, 222)
        write_frame(self.pool, "odd", image, self.shape,
                    [(n, b, blob(self.shape, b)) for n, b in self.boxes])
        with self.assertRaises(ValueError) as caught:
            index_pool(self.pool / "demo", self.images, workers=1)
        self.assertIn("shape_mismatch", str(caught.exception))

    def test_gates_drop_a_target_too_small_to_prompt(self):
        image = write_image(self.images / "tiny.png", *self.shape)
        write_frame(self.pool, "tiny", image, self.shape, [
            ("car", (10, 10, 13, 13), blob(self.shape, (10, 10, 13, 13))),
            ("truck", self.boxes[1][1], blob(self.shape, self.boxes[1][1]))])
        index = index_pool(self.pool / "demo", self.images, workers=1,
                           gates=InstanceGates(min_area=48))
        self.assertEqual([i.label for e in index for i in e.instances], [1])
        self.assertEqual(index[0].rejects.get("min_area"), 1)

    def test_a_cached_index_carries_its_class_table(self):
        """`aerial.save_index` stamps a spec *name*; a pool's spec has no repo
        entry to look the names back up in, so the cache has to hold them."""
        index = index_pool(self.build(), self.images, workers=1)
        cache = save_pool_index(self.root / "cache.json", index)
        back = load_pool_index(cache)
        self.assertEqual(len(back), len(index))
        self.assertEqual(back[0].source.spec.classes,
                         index[0].source.spec.classes)
        self.assertEqual(back[0].source.mode, "pool")
        self.assertEqual([i.label for i in back[0].instances],
                         [i.label for i in index[0].instances])

    def test_the_pool_names_under_a_staging_root_are_listed(self):
        self.build()
        write_frame(self.pool, "solo", write_image(self.images / "solo.png",
                                                   *self.shape),
                    self.shape, [(self.boxes[0][0], self.boxes[0][1],
                                  blob(self.shape, self.boxes[0][1]))],
                    dataset="other")
        self.assertEqual(pool_datasets(self.pool), ["demo", "other"])

    def test_pools_are_discovered_by_the_name_their_records_state(self):
        """An archive that carried its parent folder makes every pool under it
        read as one directory called `pool`; the records say otherwise."""
        from src.training.pool_reader import discover_pools

        nested = self.root / "staged" / "pool"
        for name in ("demo", "other"):
            image = write_image(self.images / f"{name}.png", *self.shape)
            write_frame(nested, f"{name}_key", image, self.shape,
                        [(self.boxes[0][0], self.boxes[0][1],
                          blob(self.shape, self.boxes[0][1]))], dataset=name)
        self.assertEqual(pool_datasets(self.root / "staged"), ["pool"])
        found = discover_pools(self.root / "staged")
        self.assertEqual(sorted(found), ["demo", "other"])
        self.assertEqual(found["demo"], nested / "demo")

    def test_discovery_stops_at_the_shallowest_depth_that_finds_anything(self):
        from src.training.pool_reader import discover_pools

        self.build()
        self.assertEqual(sorted(discover_pools(self.pool)), ["demo"])
        self.assertEqual(discover_pools(self.pool)["demo"], self.pool / "demo")

    def test_records_group_by_pool_however_the_archives_were_cut(self):
        """The layout question, settled by the records rather than the folders.

        A real staging folder does not agree with itself: one harvest zipped
        per split (`train.zip`), another per chunk (`000000.zip`), a third per
        pool. All three land somewhere different under the extraction root and
        all three carry the pool's name in every record.
        """
        for split in ("train", "val"):
            for n in range(2):
                key = f"{split}/{n:03d}"
                image = write_image(self.images / f"{key}.png", *self.shape)
                write_frame(self.root / "unzipped" / split, "hituav_thermal",
                            image, self.shape,
                            [(self.boxes[0][0], self.boxes[0][1],
                              blob(self.shape, self.boxes[0][1]))],
                            dataset="hituav_thermal")
        grouped = group_records(self.root / "unzipped")
        self.assertEqual(sorted(grouped), ["hituav_thermal"])
        self.assertEqual(len(grouped["hituav_thermal"]), 2)

    def test_linking_rebuilds_a_pool_directory_the_index_can_read(self):
        for n in range(3):
            image = write_image(self.images / f"scattered_{n}.png", *self.shape)
            write_frame(self.root / "chunks" / f"{n:06d}", f"seq/{n:04d}",
                        image, self.shape,
                        [(name, box, blob(self.shape, box))
                         for name, box in self.boxes], dataset="segfly_thermal")
        grouped = group_records(self.root / "chunks")
        target = link_pool(grouped["segfly_thermal"], self.root / "by_name"
                           / "segfly_thermal")
        index = index_pool(target, self.images, workers=1)
        self.assertEqual(len(index), 3)
        self.assertEqual(index[0].source.spec.name, "pool/segfly_thermal")
        # A hard link, not a copy: same inode, so the tree costs no disk.
        linked = sorted(target.rglob(MASK_STORE))[0]
        original = sorted((self.root / "chunks").rglob(MASK_STORE))[0]
        self.assertEqual(linked.stat().st_ino, original.stat().st_ino)

    def test_linking_twice_finishes_a_partial_run_rather_than_failing(self):
        self.build(keys=("solo",))
        grouped = group_records(self.pool)
        target = self.root / "by_name" / "demo"
        link_pool(grouped["demo"], target)
        link_pool(grouped["demo"], target)
        self.assertEqual(len(list(target.rglob(RECORD_FILE))), 1)

    def test_the_graded_frames_are_removed_from_a_pool_of_the_same_set(self):
        """The leak `split_index` cannot see, and the filter that closes it.

        Kust4K's pool is teacher masks on Kust4K's own frames, and Kust4K's
        drawn maps are the run's grade. They are separate sources with separate
        permutations, so a frame can land in the pool's training half and the
        grade's test half at once. Matching on the stem is what both readers
        always share.
        """
        from src.training.pool_reader import exclude_frames, frame_keys

        index = index_pool(self.build(
            keys=("seq_a/000000", "seq_a/000010", "seq_b/000000")),
            self.images, workers=1)
        # What a semantic reader would call the same three frames.
        held = {"000010"}
        kept = exclude_frames(index, held)
        self.assertEqual(len(kept), 2)
        self.assertNotIn("000010", frame_keys(kept))
        self.assertEqual(frame_keys(index) - frame_keys(kept), held)

    def test_frame_keys_ignore_the_directory_each_reader_prefixes(self):
        from src.training.pool_reader import frame_keys

        index = index_pool(self.build(keys=("train/00042",)), self.images,
                           workers=1)
        self.assertEqual(frame_keys(index), {"00042"})

    def test_acceptance_counts_what_the_harvest_kept_and_what_stopped_it(self):
        """The number that says whether a pool is worth its download.

        A rejected box writes nothing to the store, so a pool whose teacher
        refused nine boxes in ten looks, from the directory tree alone,
        exactly like one that kept them all.
        """
        from src.training.pool_reader import acceptance

        image = write_image(self.images / "counted.png", *self.shape)
        write_frame(self.pool, "counted", image, self.shape, [
            ("car", self.boxes[0][1], blob(self.shape, self.boxes[0][1])),
            ("truck", self.boxes[1][1], None),
            ("car", (10, 10, 40, 40), None)])
        report = acceptance(sorted(self.pool.rglob(RECORD_FILE)))
        self.assertEqual(report["frames"], 1)
        self.assertEqual((report["attempted"], report["accepted"]), (3, 1))
        self.assertAlmostEqual(report["rate"], 1 / 3)
        self.assertEqual(report["rejected"], {"box_iou": 2})
        self.assertEqual(report["accepted_by_class"], {"car": 1})
        self.assertEqual(report["teachers"], ["test/teacher"])

    def test_acceptance_of_nothing_is_zero_rather_than_a_division(self):
        from src.training.pool_reader import acceptance

        self.assertEqual(acceptance([])["rate"], 0.0)

    def test_an_archive_relative_path_beats_the_suffix_search(self):
        """`aerovis.write_pool` writes `image_rel` so a pool that travelled to
        Drive without its 12.6 GiB of frames can be re-pointed exactly."""
        image = write_image(self.images / "seq" / "0001.png", *self.shape)
        target = write_frame(self.pool, "seq/0001", Path("/gone/seq/0001.png"),
                             self.shape,
                             [(self.boxes[0][0], self.boxes[0][1],
                               blob(self.shape, self.boxes[0][1]))])
        record = json.loads((target / RECORD_FILE).read_text())
        record["image_rel"] = "seq/0001.png"
        (target / RECORD_FILE).write_text(json.dumps(record))
        index = index_pool(self.pool / "demo", self.images, workers=1)
        self.assertEqual(index[0].frame.image, image)

    def test_a_record_with_no_usable_path_at_all_is_counted(self):
        write_frame(self.pool, "gone", Path("/gone/x.png"), self.shape,
                    [(self.boxes[0][0], self.boxes[0][1],
                      blob(self.shape, self.boxes[0][1]))])
        with self.assertRaises(ValueError) as caught:
            index_pool(self.pool / "demo", self.images, workers=1)
        self.assertIn("no_image", str(caught.exception))

    def test_an_ungated_pool_reads_as_fully_accepted(self):
        """AeroVIS ships its own masks, so `write_pool` sets every `verdict` to
        None and no gate ever ran. 100 % here is provenance, not quality."""
        from src.training.pool_reader import acceptance

        image = write_image(self.images / "ungated.png", *self.shape)
        target = write_frame(self.pool, "ungated", image, self.shape,
                             [(n, b, blob(self.shape, b)) for n, b in self.boxes])
        record = json.loads((target / RECORD_FILE).read_text())
        record["teacher"] = "aerovis:ytvis"
        for row in record["instances"]:
            row["teacher_iou"] = None
        (target / RECORD_FILE).write_text(json.dumps(record))
        report = acceptance([target / RECORD_FILE])
        self.assertEqual(report["rate"], 1.0)
        self.assertEqual(report["rejected"], {})
        self.assertEqual(report["teachers"], ["aerovis:ytvis"])

    def test_a_cap_is_spread_over_sequences_not_drawn_uniformly(self):
        """The reason a cap needs a sampler at all.

        AeroVIS is 99 sequences with a median track of 117 frames, so a
        uniform draw keeps roughly a quarter of every sequence -- and a
        quarter of a sequence is mostly neighbouring frames, which are the
        same picture. The batch would be large and its variety would not.
        """
        from src.training.pool_reader import spread

        keys = [f"seq_{s:02d}/{f:04d}" for s in range(20) for f in range(100)]
        for key in keys[:1]:                      # one real frame is enough
            write_image(self.images / f"{key}.png", *self.shape)
        index = index_pool(self.build(keys=keys), self.images, workers=4)
        self.assertEqual(len(index), 2000)

        kept = spread(index, 200, seed=0)
        self.assertEqual(len(kept), 200)
        per_sequence = {}
        for entry in kept:
            group = entry.frame.name.rsplit("/", 1)[0]
            per_sequence[group] = per_sequence.get(group, 0) + 1
        # Every sequence survives, and none dominates.
        self.assertEqual(len(per_sequence), 20)
        self.assertEqual(set(per_sequence.values()), {10})

        # Within a sequence the frames are evenly spaced, not clustered.
        first = sorted(int(e.frame.name.split("/")[1]) for e in kept
                       if e.frame.name.startswith("seq_00/"))
        gaps = {b - a for a, b in zip(first, first[1:])}
        self.assertEqual(gaps, {10})

    def test_spreading_is_deterministic_and_bounded(self):
        from src.training.pool_reader import spread

        keys = [f"seq_{s:02d}/{f:04d}" for s in range(4) for f in range(9)]
        index = index_pool(self.build(keys=keys), self.images, workers=4)
        self.assertEqual([e.frame.name for e in spread(index, 10, seed=0)],
                         [e.frame.name for e in spread(index, 10, seed=0)])
        self.assertEqual(len(spread(index, 10, seed=0)), 10)
        # A cap at or above the pool changes nothing; a cap of zero empties it.
        self.assertEqual(len(spread(index, 999, seed=0)), len(index))
        self.assertEqual(spread(index, 0, seed=0), [])

    def test_spreading_an_ungrouped_pool_is_a_stride_over_the_whole_thing(self):
        from src.training.pool_reader import spread

        index = index_pool(self.build(keys=[f"{n:04d}" for n in range(40)]),
                           self.images, workers=4)
        kept = spread(index, 10, seed=0)
        self.assertEqual(len(kept), 10)
        self.assertEqual(len({e.frame.name for e in kept}), 10)

    def test_a_pool_with_no_records_says_where_to_look(self):
        self.build()
        with self.assertRaises(FileNotFoundError) as caught:
            index_pool(self.pool / "missing", self.images, workers=1)
        self.assertIn("demo", str(caught.exception))


@unittest.skipIf(cv2 is None, "OpenCV is needed to write the frames")
class WhyNoImageTest(unittest.TestCase):
    """`no_image` on every record has three different fixes, and the report has
    to say which one rather than leaving a re-index to find out."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pool = self.root / "pool"
        self.shape = (40, 60)
        self.image = write_image(self.root / "elsewhere" / "seq" / "000000.png",
                                 *self.shape)
        write_frame(self.pool, "seq/000000", self.image, self.shape,
                    [("car", (2, 2, 20, 20), blob(self.shape, (2, 2, 20, 20)))])

    def test_a_root_that_is_not_there_says_so(self):
        report = why_no_image(self.pool / "demo", self.root / "never_fetched")
        self.assertFalse(report["root_exists"])
        self.assertIn("never fetched", report["verdict"])

    def test_a_root_holding_a_different_part_of_the_set_says_so(self):
        other = self.root / "other"
        write_image(other / "seq" / "999999.png", *self.shape)
        report = why_no_image(self.pool / "demo", other)
        self.assertTrue(report["root_exists"])
        self.assertEqual(report["files_under_root"], 1)
        self.assertIn("different part of the set", report["verdict"])
        self.assertEqual(report["extensions"], {".png": 1})

    def test_the_frame_being_there_under_another_prefix_says_so(self):
        # The same file, but under a directory chain no suffix of the recorded
        # path ends with -- so `Relocator` misses it and an rglob finds it.
        moved = self.root / "moved" / "train" / "images" / "000000.png"
        write_image(moved, *self.shape)
        # The harvest runtime's own copy is gone, which is the situation this
        # runs in: at depth 0 the recorded path is absolute and joinpath keeps
        # it, so while it exists the relocator resolves to it and never misses.
        self.image.unlink()
        self.assertIsNone(Relocator(self.root / "moved")(self.image))
        report = why_no_image(self.pool / "demo", self.root / "moved")
        self.assertIn("000000.png", report["same_name_here"][0])
        self.assertIn("one level off", report["verdict"])

    def test_a_pool_with_no_records_is_named_as_that(self):
        report = why_no_image(self.root / "empty", self.root)
        self.assertEqual(report["records"], 0)
        self.assertIn("pool zip is missing", report["verdict"])


class StoreAreasTest(unittest.TestCase):
    def test_areas_match_a_decoded_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            shape = (40, 50)
            masks = {0: blob(shape, (5, 5, 15, 25)), 3: blob(shape, (0, 0, 50, 1))}
            path = save_masks(Path(tmp) / MASK_STORE, shape, masks)
            self.assertEqual(store_areas(path),
                             {key: int(mask.sum()) for key, mask in masks.items()})

    def test_an_empty_store_has_no_areas(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = save_masks(Path(tmp) / MASK_STORE, (8, 8), {})
            self.assertEqual(store_areas(path), {})


class RelocatorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_an_absolute_path_from_another_machine_is_re_rooted(self):
        local = self.root / "HIT_UAV" / "train"
        local.mkdir(parents=True)
        (local / "0_01.jpg").write_bytes(b"")
        relocate = Relocator(self.root / "HIT_UAV")
        found = relocate("/content/data/HIT_UAV/train/0_01.jpg")
        self.assertEqual(found, local / "0_01.jpg")

    def test_the_depth_is_learned_once_and_reused(self):
        local = self.root / "d" / "seq" / "rgb"
        local.mkdir(parents=True)
        for name in ("a.jpg", "b.jpg"):
            (local / name).write_bytes(b"")
        relocate = Relocator(self.root / "d")
        relocate("/content/data/d/seq/rgb/a.jpg")
        depth = relocate.depth
        self.assertEqual(relocate("/content/data/d/seq/rgb/b.jpg"),
                         local / "b.jpg")
        self.assertEqual(relocate.depth, depth)

    def test_a_frame_that_is_not_there_is_counted_not_guessed(self):
        relocate = Relocator(self.root)
        self.assertIsNone(relocate("/content/data/nope/x.jpg"))
        self.assertEqual(relocate.misses, 1)

    def test_no_root_means_the_recorded_path_is_used_as_it_is(self):
        here = self.root / "x.jpg"
        here.write_bytes(b"")
        self.assertEqual(Relocator(None)(here), here)


class ParsePoolTest(unittest.TestCase):
    def test_the_pool_alone_is_enough(self):
        request = parse_pool("/p/hituav_thermal")
        self.assertEqual(request.pool, Path("/p/hituav_thermal"))
        self.assertIsNone(request.images)
        self.assertEqual((request.modality, request.role), ("thermal", "train"))

    def test_every_field_in_order(self):
        request = parse_pool("/p/vtuav_rgb:/d/VTUAV:rgb:all")
        self.assertEqual(request.images, Path("/d/VTUAV"))
        self.assertEqual((request.modality, request.role), ("rgb", "all"))

    def test_a_pool_defaults_to_train_because_its_masks_are_a_guess(self):
        # The one default worth a test of its own: scoring on teacher output
        # measures the teacher, exactly as `split_index` argues for
        # reconstructed semantic sets.
        self.assertEqual(parse_pool("/p/x").role, "train")

    def test_an_unknown_modality_is_refused(self):
        with self.assertRaises(ValueError):
            parse_pool("/p/x:/d:infrared")

    def test_an_unknown_role_is_refused(self):
        with self.assertRaises(ValueError):
            parse_pool("/p/x:/d:thermal:sometimes")

    def test_the_cache_name_separates_two_modalities_of_one_pool(self):
        self.assertNotEqual(parse_pool("/p/x:/d:thermal").cache_name,
                            parse_pool("/p/x:/d:rgb").cache_name)


class DatasetFlagTest(unittest.TestCase):
    def test_pool_is_not_a_dataset_mode(self):
        from src.training.datasets import parse

        with self.assertRaises(ValueError) as caught:
            parse("kust4k:/d/Kust4K:thermal:pool")
        self.assertIn("--pool", str(caught.exception))

    def test_decompose_refuses_the_pool_mode(self):
        from src.training.aerial import SPECS, decompose

        with self.assertRaises(ValueError):
            decompose(np.zeros((4, 4), np.uint8), SPECS["kust4k"], mode="pool")


if __name__ == "__main__":
    unittest.main()
