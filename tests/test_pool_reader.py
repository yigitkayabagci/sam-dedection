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
import shutil
import sys
import tempfile
import zipfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.aerial import (InstanceGates, Sample, pool_masks,  # noqa: E402
                                 image_origin, sample_masks)
from src.training.labels import (MASK_STORE, load_masks,  # noqa: E402
                                 save_masks)
from src.training.pool import RECORD_FILE  # noqa: E402
from src.training.pool_reader import (Relocator,  # noqa: E402
                                      _root_of_archive_paths, group_records,
                                      index_pool, link_pool, load_pool_index,
                                      parse_pool, pool_datasets,
                                      extract_frames, resolve_images_root,
                                      save_pool_index, store_areas,
                                      wanted_frames, why_no_image)

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

    def _set_box_iou(self, target: Path, values) -> None:
        record_file = target / RECORD_FILE
        record = json.loads(record_file.read_text())
        for instance, value in zip(record["instances"], values):
            instance["box_iou"] = value
        record_file.write_text(json.dumps(record) + "\n")

    def test_a_stricter_box_iou_can_be_applied_without_re_harvesting(self):
        """The point of storing the reading and not just the verdict.

        A pool of a hundred thousand frames costs GPU-days to make. Deciding
        afterwards that only masks agreeing with the annotation at 0.7 should
        train has to be a pass over the records, or it is not a decision anyone
        will actually take.
        """
        image = write_image(self.images / "only.png", *self.shape)
        target = write_frame(self.pool, "only", image, self.shape, [
            (name, box, blob(self.shape, box)) for name, box in self.boxes])
        self._set_box_iou(target, [0.92, 0.61])

        keep_all = index_pool(self.pool / "demo", self.images, workers=1)
        self.assertEqual([i.label for e in keep_all for i in e.instances],
                         [0, 1])
        strict = index_pool(self.pool / "demo", self.images, workers=1,
                            min_box_iou=0.7)
        self.assertEqual([i.label for e in strict for i in e.instances], [0])

    def test_a_frame_the_stricter_cut_empties_is_dropped_as_no_accepted(self):
        """And a cut that empties the whole pool says so with that count.

        The alternative -- an empty index and a training run that starts on
        nothing -- is the failure this error already exists to prevent.
        """
        image = write_image(self.images / "only.png", *self.shape)
        target = write_frame(self.pool, "only", image, self.shape, [
            (name, box, blob(self.shape, box)) for name, box in self.boxes])
        self._set_box_iou(target, [0.55, 0.61])
        with self.assertRaises(ValueError) as caught:
            index_pool(self.pool / "demo", self.images, workers=1,
                       min_box_iou=0.7)
        self.assertIn("no_accepted", str(caught.exception))

    def test_an_older_pool_with_no_reading_is_kept_not_silently_emptied(self):
        """Dropping a whole pool because its records are a version behind is
        the worse failure of the two."""
        image = write_image(self.images / "only.png", *self.shape)
        write_frame(self.pool, "only", image, self.shape, [
            (name, box, blob(self.shape, box)) for name, box in self.boxes])
        index = index_pool(self.pool / "demo", self.images, workers=1,
                           min_box_iou=0.95)
        self.assertEqual([i.label for e in index for i in e.instances], [0, 1])

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

    def test_the_frame_being_there_under_another_prefix_is_named(self):
        # A tree that gained a level: no suffix of the recorded path matches,
        # which is what the by-name fallback exists for. Suffix matching alone
        # still misses it, and that is the case the report has to describe.
        moved = self.root / "moved" / "train" / "images" / "000000.png"
        write_image(moved, *self.shape)
        # The harvest runtime's own copy is gone, which is the situation this
        # runs in: at depth 0 the recorded path is absolute and joinpath keeps
        # it, so while it exists the relocator resolves to it and never misses.
        self.image.unlink()
        self.assertIsNone(
            Relocator(self.root / "moved", by_name=False)(self.image))
        self.assertEqual(Relocator(self.root / "moved")(self.image), moved)
        report = why_no_image(self.pool / "demo", self.root / "moved")
        self.assertIn("000000.png", report["same_name_here"][0])
        self.assertIn("one level off", report["verdict"])

    def test_a_pool_with_no_records_is_named_as_that(self):
        report = why_no_image(self.root / "empty", self.root)
        self.assertEqual(report["records"], 0)
        self.assertIn("pool zip is missing", report["verdict"])


@unittest.skipIf(cv2 is None, "OpenCV is needed to write the frames")
class ExtractFramesTest(unittest.TestCase):
    """A pool is a manifest: 214 GiB of archives to reach a few percent of it
    is a bill nobody has to pay."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pool = self.root / "pool"
        self.shape = (24, 32)
        self.box = (2, 2, 12, 12)
        # Two modalities keeping the same frame name, which is what makes a
        # basename match unsafe and a tail match necessary.
        self.wanted = ["/content/data/VTUAV/train_ST_008/car_01/ir/000000.jpg",
                       "/content/data/VTUAV/train_ST_008/car_01/ir/000010.jpg"]
        for index, recorded in enumerate(self.wanted):
            write_frame(self.pool, f"car_01/{index:06d}", Path(recorded),
                        self.shape,
                        [("car", self.box, blob(self.shape, self.box))])

    def archive(self, name, members):
        path = self.root / name
        with zipfile.ZipFile(path, "w") as handle:
            for member in members:
                handle.writestr(member, b"x" * 16)
        return path

    def records(self):
        return sorted((self.pool / "demo").rglob(RECORD_FILE))

    def test_only_the_frames_the_pool_names_come_out(self):
        archive = self.archive("train_ST_008.zip", [
            "train_ST_008/car_01/ir/000000.jpg",
            "train_ST_008/car_01/ir/000010.jpg",
            "train_ST_008/car_01/ir/000020.jpg",     # not in the pool
            "train_ST_008/car_01/rgb/000000.jpg",    # other modality, same name
        ])
        out = self.root / "out"
        report = extract_frames(self.records(), [archive], out)
        self.assertEqual(report["asked"], 2)
        self.assertEqual(report["taken"], 2)
        self.assertEqual(report["missing"], 0)
        taken = sorted(p.relative_to(out).as_posix()
                       for p in out.rglob("*") if p.is_file())
        self.assertEqual(taken, ["train_ST_008/car_01/ir/000000.jpg",
                                 "train_ST_008/car_01/ir/000010.jpg"])

    def test_a_frame_no_archive_holds_is_counted_as_missing(self):
        archive = self.archive("part.zip", ["train_ST_008/car_01/ir/000000.jpg"])
        report = extract_frames(self.records(), [archive], self.root / "out")
        self.assertEqual(report["taken"], 1)
        self.assertEqual(report["missing"], 1)

    def test_a_second_run_takes_nothing_and_loses_nothing(self):
        archive = self.archive("part.zip", [
            "train_ST_008/car_01/ir/000000.jpg",
            "train_ST_008/car_01/ir/000010.jpg"])
        out = self.root / "out"
        extract_frames(self.records(), [archive], out)
        again = extract_frames(self.records(), [archive], out)
        self.assertEqual(again["taken"], 0)
        self.assertEqual(again["already"], 2)
        self.assertEqual(again["missing"], 0)

    def test_the_report_says_which_archive_carried_what(self):
        first = self.archive("a.zip", ["train_ST_008/car_01/ir/000000.jpg"])
        second = self.archive("b.zip", ["train_ST_008/car_01/ir/000010.jpg"])
        report = extract_frames(self.records(), [first, second], self.root / "out")
        self.assertEqual(report["by_archive"], {"a.zip": 1, "b.zip": 1})

    def test_a_frame_named_the_archives_own_way_is_still_extracted(self):
        """AeroVIS is the case. Its records name a frame `vd_001/0000001.jpg`
        -- relative to the directory inside the archive -- while the archive
        stores it at `AeroVIS/sequences/vd_001/0000001.jpg`. Matching only the
        recorded absolute path ties the extraction to whatever directory the
        harvest happened to run in, and a pool harvested elsewhere reads as
        every frame missing."""
        import zipfile

        archive = self.root / "aerovis.zip"
        members = [f"AeroVIS/sequences/{sequence}/{name}"
                   for sequence in ("vd_001", "ud_001")
                   for name in ("0000001.jpg", "0000002.jpg")]
        with zipfile.ZipFile(archive, "w") as handle:
            for member in members:
                handle.writestr(member, b"\xff\xd8\xff\xdb" + b"0" * 64)

        pool = self.root / "aerovis_train"
        for member in members:
            relative = member.split("AeroVIS/sequences/")[1]
            target = pool.joinpath(*Path(relative[:-4]).parts)
            target.mkdir(parents=True, exist_ok=True)
            (target / RECORD_FILE).write_text(json.dumps({
                "key": relative[:-4], "dataset": "aerovis_train",
                # A harvest directory sharing nothing with the archive's own
                # layout: only `image_rel` can answer this.
                "image": f"/kaggle/input/aerovis/frames/{relative}",
                "image_rel": relative, "shape": [64, 64],
                "teacher": "aerovis:ytvis", "instances": []}))
            save_masks(target / MASK_STORE, (64, 64), {})

        records = sorted(pool.rglob(RECORD_FILE))
        report = extract_frames(records, [archive], self.root / "frames")
        self.assertEqual((report["asked"], report["taken"], report["missing"]),
                         (4, 4, 0))
        self.assertTrue(
            (self.root / "frames" / "AeroVIS" / "sequences"
             / "vd_001" / "0000001.jpg").is_file())

    def test_two_names_for_one_frame_are_counted_once(self):
        """`asked` and `missing` are frames, not the strings naming them."""
        pool = self.root / "twice"
        target = pool / "a"
        target.mkdir(parents=True, exist_ok=True)
        (target / RECORD_FILE).write_text(json.dumps({
            "key": "a", "dataset": "twice", "image": "/somewhere/else/a.jpg",
            "image_rel": "deep/a.jpg", "shape": [8, 8], "teacher": "x",
            "instances": []}))
        save_masks(target / MASK_STORE, (8, 8), {})
        report = extract_frames(sorted(pool.rglob(RECORD_FILE)), [],
                                self.root / "nothing")
        self.assertEqual((report["asked"], report["missing"]), (1, 1))

    def test_wanted_frames_keys_by_basename(self):
        wanted = wanted_frames(self.records())
        self.assertEqual(sorted(wanted), ["000000.jpg", "000010.jpg"])
        self.assertEqual(wanted["000000.jpg"], {self.wanted[0]})

    def test_one_archive_that_will_not_open_does_not_take_the_shelf_with_it(self):
        """VTUAV is fifteen files on a Drive mount. A part that synced short is
        one archive's worth of frames, not the run: raising here loses the
        fourteen that are fine, and the frames already taken out of the ones
        walked before it, because the notebook's loop dies with the call."""
        good = self.archive("part_a.zip", [
            "train_ST_008/car_01/ir/000000.jpg"])
        broken = self.root / "part_b.zip"
        broken.write_bytes(b"PK\x03\x04 and then the sync stopped")
        later = self.archive("part_c.zip", [
            "train_ST_008/car_01/ir/000010.jpg"])

        report = extract_frames(self.records(), [good, broken, later],
                                self.root / "out")

        self.assertEqual((report["taken"], report["missing"]), (2, 0))
        self.assertEqual([name for name, _ in report["unopened"]],
                         ["part_b.zip"])
        self.assertIn("BadZipFile", report["unopened"][0][1])
        self.assertEqual(report["by_archive"]["part_b.zip"], 0)

    def test_a_record_that_will_not_parse_is_skipped_not_raised_on(self):
        """A pool zip that unpacked short leaves a truncated `record.json`
        behind. One of those is not a reason for the other forty thousand
        frames to stay inside the archives -- and it read as a crash in the
        cell that extracts them, with nothing in the log naming the file."""
        stray = self.pool / "demo" / "car_01" / "999999"
        stray.mkdir(parents=True, exist_ok=True)
        (stray / RECORD_FILE).write_text('{"image": "/half/a/rec')
        archive = self.archive("part.zip", [
            "train_ST_008/car_01/ir/000000.jpg",
            "train_ST_008/car_01/ir/000010.jpg"])

        self.assertEqual(sorted(wanted_frames(self.records())),
                         ["000000.jpg", "000010.jpg"])
        report = extract_frames(self.records(), [archive], self.root / "out")
        self.assertEqual((report["asked"], report["taken"], report["missing"]),
                         (2, 2, 0))


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

    def test_a_tree_that_gained_a_level_is_found_by_name(self):
        # HIT-UAV's archive nests under a branch-named folder; a pool
        # harvested from a copy without it records the shallower path, and
        # stripping leading components can never put the level back.
        local = self.root / "HIT-UAV-main" / "normal_json" / "train"
        local.mkdir(parents=True)
        (local / "0_01.jpg").write_bytes(b"")
        relocate = Relocator(self.root)
        found = relocate("/content/data/HIT_UAV/normal_json/train/0_01.jpg")
        self.assertEqual(found, local / "0_01.jpg")
        self.assertEqual((relocate.found_by_name, relocate.misses), (1, 0))

    def test_the_longest_matching_tail_picks_between_two_of_a_name(self):
        # DroneVehicle keeps one file name per frame in each modality.
        for folder in ("trainimg", "trainimgr"):
            (self.root / "train" / folder).mkdir(parents=True)
            (self.root / "train" / folder / "04991.jpg").write_bytes(b"")
        relocate = Relocator(self.root)
        found = relocate("/data/DroneVehicle/train/trainimgr/04991.jpg")
        self.assertEqual(found, self.root / "train" / "trainimgr" / "04991.jpg")

    def test_a_name_two_files_share_equally_is_refused_not_guessed(self):
        for folder in ("a", "b"):
            (self.root / folder).mkdir(parents=True)
            (self.root / folder / "0.jpg").write_bytes(b"")
        relocate = Relocator(self.root)
        self.assertIsNone(relocate("/data/somewhere/else/0.jpg"))
        self.assertEqual((relocate.ambiguous, relocate.misses), (1, 1))

    def test_the_fallback_can_be_turned_off(self):
        deep = self.root / "wrapper" / "train"
        deep.mkdir(parents=True)
        (deep / "0_01.jpg").write_bytes(b"")
        relocate = Relocator(self.root, by_name=False)
        self.assertIsNone(relocate("/data/HIT_UAV/train/0_01.jpg"))
        self.assertEqual(relocate.misses, 1)


class ResolveImagesRootTest(unittest.TestCase):
    """Naming the tree the recorded paths were written against, once per pool.

    `Relocator`'s by-name fallback settles one frame at a time and refuses a
    name that is ambiguous under its root -- correct per frame, and no answer at
    all for a pool whose every frame is ambiguous the same way, which is what an
    archive that unpacks a copy of itself produces. These are the three answers
    the caller has to be able to tell apart: the configured root is right, the
    frames are under a different prefix, or they are not on this disk.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.records = 0

    def record_naming(self, image: str) -> Path:
        self.records += 1
        path = self.root / "pool" / str(self.records) / RECORD_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"key": str(self.records), "image": image}))
        return path

    def touch(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        return path

    def test_a_wrapper_directory_the_harvest_did_not_have_is_found(self):
        inner = (self.root / "HIT_UAV" / "HIT-UAV-Infrared-Thermal-Dataset-main"
                 / "normal_json")
        self.touch(inner / "train" / "0_01.jpg")
        record = self.record_naming("/content/data/HIT_UAV/normal_json/train/"
                                    "0_01.jpg")
        self.assertEqual(resolve_images_root([record], self.root / "HIT_UAV"),
                         inner.parent)

    def test_the_tree_holding_the_frames_beats_the_wrapper_holding_one(self):
        # The archive re-packs itself, so the name is ambiguous under the root
        # and every frame of the pool ties -- the case the by-name fallback
        # refuses. The root that resolves the *most* frames is the answer.
        outer = self.root / "HIT_UAV" / "main" / "normal_json"
        inner = outer / "main" / "normal_json"
        self.touch(outer / "train" / "0_01.jpg")
        for name in ("0_01.jpg", "0_02.jpg", "0_03.jpg"):
            self.touch(inner / "train" / name)
        records = [self.record_naming(f"/content/data/HIT_UAV/normal_json/train/"
                                      f"{name}")
                   for name in ("0_01.jpg", "0_02.jpg", "0_03.jpg")]
        self.assertIsNone(Relocator(self.root / "HIT_UAV")(
            "/content/data/HIT_UAV/normal_json/train/0_01.jpg"))
        self.assertEqual(resolve_images_root(records, self.root / "HIT_UAV"),
                         inner.parent)

    def test_a_mirror_that_renamed_a_component_is_still_found(self):
        # The pool was harvested from the kagglehub copy (`hit-uav/images/`)
        # and the run staged the GitHub archive (`normal_json/`): the two
        # disagree on a component, so no suffix of one is a suffix of the other.
        frames = self.root / "HIT_UAV" / "wrapper" / "normal_json"
        self.touch(frames / "test" / "0_01.jpg")
        record = self.record_naming("/content/data/HIT_UAV/hit-uav/images/test/"
                                    "0_01.jpg")
        found = resolve_images_root([record], self.root / "HIT_UAV")
        self.assertEqual(found, frames)
        self.assertEqual(Relocator(found, by_name=False)(
            "/content/data/HIT_UAV/hit-uav/images/test/0_01.jpg"),
            frames / "test" / "0_01.jpg")

    def test_four_copies_of_one_archive_are_not_an_ambiguity_to_refuse(self):
        # HIT-UAV ships each frame under `normal_json/<split>/`,
        # `rotate_json/<split>/` and two `JPEGImages/` trees because its
        # annotations come in four formats. The first two tie on shared tail
        # against a record written from the Kaggle mirror
        # (`hit-uav/images/<split>/`), which is what makes a per-frame reader
        # refuse every one of the 2 866 frames.
        base = self.root / "HIT_UAV" / "HIT-UAV-Infrared-Thermal-Dataset-main"
        pixels = b"\xff\xd8the same frame"
        for folder in ("normal_json/train", "rotate_json/train"):
            self.touch(base / folder / "0_100_30_0_03280.jpg").write_bytes(pixels)
        for folder in ("normal_xml/JPEGImages", "rotate_xml/JPEGImages"):
            self.touch(base / folder / "0_100_30_0_03280.jpg").write_bytes(pixels)
        record = self.record_naming("/root/.cache/kagglehub/datasets/pandrii000/"
                                    "hituav/versions/1/hit-uav/images/train/"
                                    "0_100_30_0_03280.jpg")
        found = resolve_images_root([record], self.root / "HIT_UAV")
        self.assertEqual(found, base / "normal_json")
        self.assertEqual(Relocator(found, by_name=False)(
            json.loads(record.read_text())["image"]),
            base / "normal_json" / "train" / "0_100_30_0_03280.jpg")

    def test_two_roots_holding_different_pixels_are_still_refused(self):
        # DroneVehicle's two modalities under one root: picking either would
        # train one modality's masks on the other's pixels, silently.
        base = self.root / "DroneVehicle"
        self.touch(base / "trainimg" / "04991.jpg").write_bytes(b"rgb")
        self.touch(base / "trainimgr" / "04991.jpg").write_bytes(b"thermal")
        record = self.record_naming("/content/data/DroneVehicle/train/"
                                    "04991.jpg")
        self.assertIsNone(resolve_images_root([record], base))

    def test_a_root_that_already_works_is_returned_unchanged(self):
        self.touch(self.root / "HIT_UAV" / "normal_json" / "train" / "0_01.jpg")
        record = self.record_naming("/content/data/HIT_UAV/normal_json/train/"
                                    "0_01.jpg")
        self.assertEqual(resolve_images_root([record], self.root / "HIT_UAV"),
                         self.root / "HIT_UAV")

    def test_a_half_downloaded_pool_keeps_the_root_it_was_given(self):
        # Four of five probes resolve and the fifth is simply not there yet.
        # Re-rooting one level up would "find" them all and hide the missing
        # download behind a wider search -- and one level up is where another
        # dataset's `04991.jpg` lives.
        base = self.root / "DroneVehicle"
        for index in range(4):
            self.touch(base / "train" / "trainimgr" / f"0499{index}.jpg")
        records = [self.record_naming(f"/content/data/DroneVehicle/train/"
                                      f"trainimgr/0499{index}.jpg")
                   for index in range(5)]
        self.assertEqual(resolve_images_root(records, base), base)

    def test_frames_that_are_not_on_this_disk_are_not_guessed_at(self):
        (self.root / "HIT_UAV").mkdir()
        record = self.record_naming("/content/data/HIT_UAV/train/0_01.jpg")
        self.assertIsNone(resolve_images_root([record], self.root / "HIT_UAV"))

    def test_an_unresolved_glob_searches_the_prefix_it_is_sure_of(self):
        # `images_for` leaves the pattern in place when nothing matched it, and
        # nothing matches a pattern globbed before the download. The part of it
        # in front of the first `*` is still a real directory.
        frames = self.root / "HIT_UAV" / "wrapper" / "normal_json"
        self.touch(frames / "train" / "0_01.jpg")
        record = self.record_naming("/content/data/HIT_UAV/normal_json/train/"
                                    "0_01.jpg")
        pattern = str(self.root / "HIT_UAV" / "**" / "normal_json")
        self.assertEqual(resolve_images_root([record], pattern), frames.parent)

    def test_a_wider_search_root_finds_a_sibling_of_the_configured_one(self):
        frames = self.root / "HIT-UAV-Infrared-Thermal-Dataset-main" / "train"
        self.touch(frames / "0_01.jpg")
        record = self.record_naming("/content/data/HIT_UAV/train/0_01.jpg")
        (self.root / "HIT_UAV").mkdir()
        self.assertIsNone(resolve_images_root([record], self.root / "HIT_UAV"))
        self.assertEqual(resolve_images_root([record], self.root / "HIT_UAV",
                                             search_root=self.root),
                         frames.parent)



class AeroVISPathsTest(unittest.TestCase):
    """AeroVIS end to end: the frame it names and the store beside its record.

    The layout here is not invented. It was read out of `AeroVIS.zip` on the
    Drive this project trains from, by range-requesting the archive's central
    directory rather than the 13.5 GiB in front of it:

        AeroVIS/aero_vis.json
        AeroVIS/data.md
        AeroVIS/sequences/{sd_001..vd_052}/{frame}.jpg   117 dirs, 49 204 files

    and `aero_vis.json` opens `"videos": [{"id": 1, "name": "vd_001",
    "file_names": ["vd_001/0000001.jpg", ...`. So `file_names` -- which is what
    `aerovis.write_pool` stores as `image_rel` -- is relative to
    **`AeroVIS/sequences`**, one level below the `AeroVIS` directory the
    archive unpacks as and two below the `IMAGE_ROOTS` entry pointing at the
    extract. Frame names are per-source and repeat: `0000001.jpg` exists in 91
    of the 117 sequences, which is what makes the by-name fallback the wrong
    thing to be leaning on here.
    """

    SEQUENCES = ("vd_001", "vd_002", "ud_001")
    FRAMES = ("0000001.jpg", "0000002.jpg")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # What `IMAGE_ROOTS` names, and where the archive really lands under it.
        self.configured = self.root / "data" / "AeroVIS"
        self.sequences = self.configured / "AeroVIS" / "sequences"
        for order, sequence in enumerate(self.SEQUENCES):
            for number, name in enumerate(self.FRAMES):
                frame = self.sequences / sequence / name
                frame.parent.mkdir(parents=True, exist_ok=True)
                # Distinct pixels per frame: a reader that resolved the right
                # name in the wrong sequence has to be able to fail.
                shade = 20 + 40 * order + 7 * number
                cv2.imwrite(str(frame), np.full((64, 64, 3), shade, np.uint8))

    def write_pool(self, harvest: str) -> Path:
        """The pool `aerovis.write_pool` writes, with `harvest` as its root."""
        pool = self.root / "pool" / "aerovis_train"
        for sequence in self.SEQUENCES:
            for name in self.FRAMES:
                relative = f"{sequence}/{name}"
                key = relative[:-len(".jpg")]
                target = pool.joinpath(*Path(key).parts)
                target.mkdir(parents=True, exist_ok=True)
                mask = np.zeros((64, 64), bool)
                mask[20:32, 20:34] = True
                (target / RECORD_FILE).write_text(json.dumps({
                    "key": key, "dataset": "aerovis_train", "prompt": "dataset",
                    "image": f"{harvest}/{relative}", "image_rel": relative,
                    "shape": [64, 64], "teacher": "aerovis:ytvis",
                    "source": "visdrone", "video_id": 1,
                    "instances": [{"i": 0, "class": "car", "track_id": 1,
                                   "box": [20.0, 20.0, 34.0, 32.0],
                                   "area": int(mask.sum()),
                                   "teacher_iou": None, "verdict": None}]}))
                save_masks(target / MASK_STORE, (64, 64), {0: mask})
        return pool

    def test_a_frame_name_that_repeats_across_sequences_is_the_whole_problem(self):
        """The premise, checked rather than asserted in a comment."""
        self.assertEqual(len(list(self.sequences.rglob("0000001.jpg"))),
                         len(self.SEQUENCES))

    def test_the_sequences_directory_is_named_from_the_archive_relative_path(self):
        """`image_rel` joins onto exactly one root, and that root is the answer.

        The configured root is two levels above it and resolves nothing by
        joining -- which is the state a run is in on a fresh runtime.
        """
        pool = self.write_pool("/content/data/AeroVIS/AeroVIS/sequences")
        records = sorted(pool.rglob(RECORD_FILE))
        self.assertIsNone(Relocator(self.configured).direct("vd_001/0000001.jpg"))
        self.assertEqual(resolve_images_root(records, self.configured),
                         self.sequences)

    def test_it_is_named_the_same_way_when_the_harvest_staged_it_elsewhere(self):
        """`image_rel` is the archive's own statement, so where the harvest
        happened to put the frames does not enter into it. This is the case the
        absolute path cannot answer: a recorded root sharing no component with
        the local tree leaves the file name, and the file name matches every
        sequence."""
        pool = self.write_pool("/kaggle/input/aerovis/frames")
        self.assertEqual(resolve_images_root(sorted(pool.rglob(RECORD_FILE)),
                                             self.configured),
                         self.sequences)

    def test_both_paths_resolve_and_the_masks_belong_to_the_frames(self):
        """The two halves together: every record finds its frame *and* the
        store written beside it, and no frame is answered with another
        sequence's copy of the same file name."""
        pool = self.write_pool("/content/data/AeroVIS/AeroVIS/sequences")
        found = resolve_images_root(sorted(pool.rglob(RECORD_FILE)),
                                    self.configured)
        index = index_pool(pool, str(found), modality="rgb")
        self.assertEqual(len(index), len(self.SEQUENCES) * len(self.FRAMES))
        for entry in index:
            sequence, name = entry.frame.name.split("/")
            self.assertEqual(entry.frame.image,
                             self.sequences / sequence / f"{name}.jpg")
            shape, masks = load_masks(entry.frame.mask)
            self.assertEqual(shape, (64, 64))
            self.assertEqual(set(masks), {0})
            self.assertEqual(int(masks[0].sum()), 12 * 14)

    def test_resolving_the_root_means_no_frame_is_matched_by_name(self):
        """A join is not a search. Under the root `image_rel` names, `direct`
        answers every frame and the by-name index -- the part that would have
        to refuse 91 candidates -- is never built."""
        pool = self.write_pool("/content/data/AeroVIS/AeroVIS/sequences")
        found = resolve_images_root(sorted(pool.rglob(RECORD_FILE)),
                                    self.configured)
        relocate = Relocator(str(found))
        for sequence in self.SEQUENCES:
            for name in self.FRAMES:
                self.assertEqual(relocate.direct(f"{sequence}/{name}"),
                                 self.sequences / sequence / name)
        self.assertEqual((relocate.found_by_name, relocate.ambiguous,
                          relocate.misses), (0, 0, 0))

    def release_archive(self) -> Path:
        """`AeroVIS.zip` as the Drive holds it: everything under `AeroVIS/`."""
        archive = self.root / "AeroVIS.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("AeroVIS/aero_vis.json", "{}")
            for sequence in self.SEQUENCES:
                for name in self.FRAMES:
                    handle.write(self.sequences / sequence / name,
                                 f"AeroVIS/sequences/{sequence}/{name}")
        return archive

    def test_the_release_archive_lands_where_the_resolver_then_names(self):
        """`POOL_ARCHIVES` and `resolve_images_root`, joined up.

        Both are already right on their own and neither is any use alone: the
        pool travels without its 12.6 GiB, so until an archive is named there is
        nothing under the images root for `image_rel` to join onto, and every
        arm that plans these pools reports them unusable -- an assert, in the
        three that require them.

        `extract_frames` takes the members whose path the records end with and
        writes them at `target / member`, so the `AeroVIS/sequences/` the
        archive nests them under is reproduced under the configured root. That
        is the tree `resolve_images_root` names, which is why the two compose
        rather than needing a third thing to agree with both.
        """
        archive = self.release_archive()
        pool = self.write_pool("/content/data/AeroVIS/AeroVIS/sequences")
        records = sorted(pool.rglob(RECORD_FILE))
        fresh = self.root / "fresh" / "AeroVIS"          # a runtime with nothing
        fresh.mkdir(parents=True)
        self.assertIsNone(resolve_images_root(records, fresh))

        report = extract_frames(records, [archive], fresh)
        self.assertEqual((report["taken"], report["missing"]),
                         (len(self.SEQUENCES) * len(self.FRAMES), 0))
        # aero_vis.json is in the archive and in no record, so it stays there.
        self.assertFalse((fresh / "AeroVIS" / "aero_vis.json").exists())

        found = resolve_images_root(records, fresh)
        self.assertEqual(found, fresh / "AeroVIS" / "sequences")
        index = index_pool(pool, str(found), modality="rgb")
        self.assertEqual(len(index), len(self.SEQUENCES) * len(self.FRAMES))

    def test_a_second_run_re_extracts_nothing(self):
        """The archive is 12.6 GiB on a mounted Drive, so the resume path is
        not a nicety -- a run that reconnects must not read it again."""
        archive = self.release_archive()
        records = sorted(self.write_pool(
            "/content/data/AeroVIS/AeroVIS/sequences").rglob(RECORD_FILE))
        fresh = self.root / "fresh" / "AeroVIS"
        fresh.mkdir(parents=True)
        extract_frames(records, [archive], fresh)
        again = extract_frames(records, [archive], fresh)
        self.assertEqual((again["taken"], again["missing"]), (0, 0))
        self.assertEqual(again["already"],
                         len(self.SEQUENCES) * len(self.FRAMES))

    def test_a_pool_that_is_not_on_this_disk_is_still_reported_as_missing(self):
        """`image_rel` must not turn a missing download into a search that
        wanders off and finds something. Nothing joins, nothing is proposed."""
        pool = self.write_pool("/content/data/AeroVIS/AeroVIS/sequences")
        empty = self.root / "data" / "nothing"
        empty.mkdir(parents=True, exist_ok=True)
        self.assertIsNone(resolve_images_root(sorted(pool.rglob(RECORD_FILE)),
                                              empty))

    def test_two_copies_of_the_tree_hand_the_question_back(self):
        """A second extraction under the same root joins onto `image_rel` just
        as well as the first, and there is nothing in an archive-relative path
        to tell them apart -- so the exact pass declines and the recorded
        absolute paths, which *can* tell them apart, decide. The frames still
        come from the tree the harvest saw and never from the copy."""
        pool = self.write_pool("/content/data/AeroVIS/AeroVIS/sequences")
        twin = self.configured / "copy" / "sequences"
        for sequence in self.SEQUENCES:
            for name in self.FRAMES:
                target = twin / sequence / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.sequences / sequence / name, target)
        records = sorted(pool.rglob(RECORD_FILE))
        self.assertIsNone(_root_of_archive_paths(
            [Path(f"{s}/{self.FRAMES[0]}") for s in self.SEQUENCES],
            self.configured, self.configured))
        found = resolve_images_root(records, self.configured)
        self.assertEqual(found, self.configured)
        for entry in index_pool(pool, str(found), modality="rgb"):
            sequence, name = entry.frame.name.split("/")
            self.assertEqual(entry.frame.image,
                             self.sequences / sequence / f"{name}.jpg")

    def test_a_pool_without_image_rel_still_reads_the_recorded_paths(self):
        """Most harvests write no `image_rel`. The exact pass has to stand
        aside for them rather than answer `None`."""
        pool = self.write_pool("/content/data/AeroVIS/AeroVIS/sequences")
        for record in pool.rglob(RECORD_FILE):
            body = json.loads(record.read_text())
            body.pop("image_rel")
            record.write_text(json.dumps(body))
        self.assertEqual(resolve_images_root(sorted(pool.rglob(RECORD_FILE)),
                                             self.configured),
                         self.configured)


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

    def test_a_gate_change_is_a_different_index_cache(self):
        """The gates decide which components exist, and `decompose` numbers
        only the ones it keeps -- so an index cached under looser gates pairs
        every instance after the first newly-rejected one with its neighbour's
        mask. A cache key that ignores the gates makes that silent."""
        from src.training.aerial import InstanceGates
        from src.training.datasets import parse

        loose = parse("segfly:/d/S:thermal:components", InstanceGates(min_area=8))
        tight = parse("segfly:/d/S:thermal:components", InstanceGates(min_area=48))
        self.assertNotEqual(loose.label, tight.label)
        one = parse_pool("/p/hituav:/d/H:thermal:all", InstanceGates(max_area=0.9))
        two = parse_pool("/p/hituav:/d/H:thermal:all", InstanceGates(max_area=0.25))
        self.assertNotEqual(one.cache_name, two.cache_name)

    def test_the_role_is_still_absent_from_the_cache_key(self):
        """Re-indexing tens of thousands of frames because a pool moved from
        train to eval would be a bill for nothing: the role picks the split, it
        does not change a mask."""
        from src.training.datasets import parse

        self.assertEqual(parse("segfly:/d/S:thermal:components:train").label,
                         parse("segfly:/d/S:thermal:components:all").label)

    def test_a_stricter_gate_renumbers_the_components_it_keeps(self):
        """The reason the two tests above matter, in the function itself."""
        from src.training.aerial import SPECS, InstanceGates, decompose

        semantic = np.zeros((200, 200), np.int32)
        semantic[10:14, 10:14] = 13          # 16 px: dies under min_area 48
        semantic[50:70, 50:80] = 13          # 600 px
        loose = decompose(semantic, SPECS["segfly"], InstanceGates(min_area=8))[1]
        tight = decompose(semantic, SPECS["segfly"], InstanceGates(min_area=48))[1]
        self.assertEqual([(i.label, i.area) for i in loose], [(1, 16), (2, 600)])
        self.assertEqual([(i.label, i.area) for i in tight], [(1, 600)])

    def test_decompose_refuses_the_pool_mode(self):
        from src.training.aerial import SPECS, decompose

        with self.assertRaises(ValueError):
            decompose(np.zeros((4, 4), np.uint8), SPECS["kust4k"], mode="pool")


if __name__ == "__main__":
    unittest.main()
