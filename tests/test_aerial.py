"""Semantic maps into prompted instances, and the gates that decide what counts.

The property under test is the one the whole static stage rests on: a semantic
map says "car", and what training needs is "*this* car". Everything here is
about whether the decomposition produces that, and about saying so out loud
when it does not -- the adjacent-vehicle fusion case has its own test, and it
asserts the *failure*, because pretending otherwise would be the expensive
mistake.

No EdgeTAM and no GPU. OpenCV is needed to write and read the synthetic maps,
so the suite skips itself cleanly when it is absent.
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

try:
    import cv2
except ImportError:                                     # pragma: no cover
    cv2 = None

from src.training.aerial import (  # noqa: E402
    DatasetSpec,
    Frame,
    FrameIndex,
    Instance,
    InstanceGates,
    Sample,
    Source,
    decompose,
    index_frames,
    list_frames,
    load_index,
    probe_classes,
    read_mask,
    reject_reason,
    sample_masks,
    sample_windows,
    save_index,
    split_frames,
    summarise,
    windows_for,
)

SPEC = DatasetSpec(
    name="toy",
    thermal="**/tir/*.png",
    rgb="**/rgb/*.png",
    masks="**/label/*.png",
    classes={"background": 0, "road": 1, "car": 2, "person": 3},
    things=("car", "person"),
)
GATES = InstanceGates(min_area=4, min_side=2, max_area=0.9, fill=0.25)
SOURCE = Source(spec=SPEC, gates=GATES)


def semantic(width: int = 64, height: int = 64) -> np.ndarray:
    return np.zeros((height, width), dtype=np.uint8)


def put(mask: np.ndarray, value: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    mask[y0:y1, x0:x1] = value
    return mask


@unittest.skipIf(cv2 is None, "OpenCV is not installed")
class TestDecompose(unittest.TestCase):
    def test_two_separated_cars_are_two_instances(self):
        mask = put(put(semantic(), 2, 4, 4, 12, 12), 2, 40, 40, 48, 48)
        components, instances, _ = decompose(mask, SPEC, GATES)

        self.assertEqual(len(instances), 2)
        self.assertEqual(len({i.label for i in instances}), 2)
        # Every instance's label is present in the component image, which is
        # the contract `sample_masks` recovers a target through.
        for instance in instances:
            self.assertTrue((components == instance.label).any())

    def test_two_touching_cars_of_the_same_class_fuse_into_one(self):
        # The known limit of connected components, asserted rather than hoped
        # against: two cars sharing an edge are one component and there is no
        # gate that can separate them. This is the measurement `summarise` and
        # notebook 07 exist to put a number on -- if a dataset's vehicles
        # routinely touch, its instances are not usable as targets.
        mask = put(put(semantic(), 2, 10, 10, 20, 20), 2, 20, 10, 30, 20)
        _, instances, _ = decompose(mask, SPEC, GATES)

        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0].box, (10.0, 10.0, 30.0, 20.0))

    def test_watershed_rescues_two_cars_joined_by_a_thin_bridge(self):
        # The case the `fill` gate detects and throws away: two blobs and a few
        # pixels of annotation joining them. Distance-transform watershed has
        # two peaks and a valley here, so it splits them.
        mask = put(put(semantic(), 2, 8, 8, 24, 24), 2, 32, 8, 48, 24)
        mask[15:17, 24:32] = 2                       # the bridge

        _, fused, _ = decompose(mask, SPEC, GATES)
        _, split, _ = decompose(mask, SPEC, GATES, mode="watershed")

        self.assertEqual(len(fused), 1)
        self.assertEqual(len(split), 2)
        for instance in split:
            self.assertLess(instance.width, 24)      # neither spans both blobs

    def test_watershed_cannot_split_two_perfectly_abutting_rectangles(self):
        # The honest limit, asserted rather than left to be discovered: two
        # rectangles sharing a full edge tile into a bigger rectangle whose
        # distance transform has no valley. No mask geometry can separate
        # these -- only something that looks at the pixels can.
        mask = put(put(semantic(), 2, 10, 10, 20, 20), 2, 20, 10, 30, 20)
        _, split, _ = decompose(mask, SPEC, GATES, mode="watershed")
        self.assertEqual(len(split), 1)

    def test_a_single_object_survives_watershed_unchanged(self):
        # Over-splitting is watershed's own failure mode; one convex blob must
        # come through as one instance or the repair costs more than it fixes.
        mask = put(semantic(), 2, 12, 12, 30, 28)
        _, plain, _ = decompose(mask, SPEC, GATES)
        _, split, _ = decompose(mask, SPEC, GATES, mode="watershed")

        self.assertEqual(len(plain), len(split), 1)
        self.assertEqual(plain[0].box, split[0].box)

    def test_an_unknown_split_strategy_is_rejected(self):
        with self.assertRaises(ValueError):
            decompose(semantic(), SPEC, GATES, mode="magic")

    def test_a_car_touching_a_person_stays_two_instances(self):
        # Why components are found per class rather than over the union of
        # things: the union would fuse these two, which is strictly worse than
        # the same-class case because the classes are known to differ.
        mask = put(put(semantic(), 2, 10, 10, 20, 20), 3, 20, 10, 26, 20)
        _, instances, _ = decompose(mask, SPEC, GATES)

        self.assertEqual(len(instances), 2)
        self.assertEqual({i.class_id for i in instances}, {2, 3})

    def test_stuff_classes_are_not_instances(self):
        mask = put(semantic(), 1, 0, 0, 64, 32)          # road, half the frame
        components, instances, _ = decompose(mask, SPEC, GATES)

        self.assertEqual(instances, [])
        self.assertEqual(int(components.max()), 0)

    def test_an_rgb_map_of_one_grey_value_reads_as_that_value(self):
        grey = np.stack([put(semantic(), 2, 4, 4, 12, 12)] * 3, axis=-1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.png"
            cv2.imwrite(str(path), grey)
            self.assertEqual(int(read_mask(path).max()), 2)


@unittest.skipIf(cv2 is None, "OpenCV is not installed")
class TestGates(unittest.TestCase):
    def make(self, x0, y0, x1, y1, area):
        return Instance(label=1, class_id=2, box=(x0, y0, x1, y1), area=area)

    def test_each_gate_names_itself(self):
        cases = {
            "min_area": self.make(0, 0, 10, 10, 2),
            "min_side": self.make(0, 0, 40, 1, 40),
            "max_area": self.make(0, 0, 64, 64, 4000),
            "fill": self.make(0, 0, 40, 40, 100),
        }
        for expected, instance in cases.items():
            self.assertEqual(reject_reason(instance, 64 * 64, GATES), expected,
                             f"expected {expected}")

    def test_a_solid_square_passes_every_gate(self):
        self.assertIsNone(reject_reason(self.make(0, 0, 10, 10, 100), 64 * 64, GATES))

    def test_the_rejects_are_counted_by_reason(self):
        mask = put(semantic(), 2, 4, 4, 6, 5)            # 2 px tall, 2 px area
        _, instances, rejects = decompose(mask, SPEC, InstanceGates(min_area=64))

        self.assertEqual(instances, [])
        self.assertEqual(rejects, {"min_area": 1})


@unittest.skipIf(cv2 is None, "OpenCV is not installed")
class TestWindows(unittest.TestCase):
    def entry(self, size=(64, 64), boxes=((10, 10, 20, 20),)):
        instances = tuple(
            Instance(label=n + 1, class_id=2, box=tuple(float(v) for v in box),
                     area=int((box[2] - box[0]) * (box[3] - box[1])))
            for n, box in enumerate(boxes))
        frame = Frame(name="f", image=Path("f.png"), mask=Path("f_label.png"))
        return FrameIndex(frame=frame, size=size, instances=instances, rejects={},
                          source=SOURCE)

    def test_a_window_that_fits_is_native_pixels(self):
        # The property the whole 512 story depends on: no resampling, so the
        # model sees the sensor's own pixels, exactly as the Orin's crop mode
        # feeds it.
        samples = windows_for(self.entry(size=(640, 512)), size=512,
                              rng=np.random.default_rng(0))
        self.assertTrue(samples[0].native)
        self.assertEqual(samples[0].window, (512, 512))

    def test_an_instance_wider_than_the_window_falls_back_to_the_whole_frame(self):
        entry = self.entry(size=(640, 512), boxes=((0, 0, 600, 400),))
        sample = windows_for(entry, size=512, rng=np.random.default_rng(0))[0]

        self.assertFalse(sample.native)
        self.assertEqual((sample.origin, sample.window), ((0, 0), (640, 512)))

    def test_other_instances_inside_the_window_come_along(self):
        entry = self.entry(size=(64, 64),
                           boxes=((10, 10, 20, 20), (30, 30, 40, 40)))
        sample = windows_for(entry, size=64, rng=np.random.default_rng(0))[0]
        self.assertEqual(len(sample.instances), 2)

    def test_an_instance_straddling_the_edge_is_dropped_not_truncated(self):
        # A box that claims an extent its pixels do not have is a worse target
        # than no target.
        entry = self.entry(size=(128, 128),
                           boxes=((10, 10, 20, 20), (60, 60, 100, 100)))
        sample = windows_for(entry, size=64, jitter=0,
                             rng=np.random.default_rng(0))[0]
        for instance in sample.instances:
            self.assertTrue(instance.inside(sample.origin, sample.window))

    def test_the_anchor_survives_the_instance_cap(self):
        # A window is placed *because of* its anchor. Shuffling it out would
        # leave a window centred on an object that is not in the batch.
        boxes = [(4 + 20 * i, 4, 12 + 20 * i, 12) for i in range(6)]
        entry = self.entry(size=(128, 128), boxes=tuple(boxes))
        for seed in range(12):
            rng = np.random.default_rng(seed)
            anchor = entry.instances[int(rng.permutation(len(entry.instances))[0])]
            sample = windows_for(entry, size=128, max_instances=2,
                                 rng=np.random.default_rng(seed))[0]
            self.assertEqual(len(sample.instances), 2)
            self.assertIn(anchor.label, [i.label for i in sample.instances])

    def test_a_frame_smaller_than_the_window_resizes_rather_than_running_off_it(self):
        # crop_window clamps an origin to 0 on an axis it cannot satisfy, which
        # would hand back a rectangle extending past the image. The whole frame
        # is the only window a 400x300 image has.
        entry = self.entry(size=(400, 300), boxes=((10, 10, 30, 30),))
        sample = windows_for(entry, size=512, rng=np.random.default_rng(0))[0]

        self.assertEqual((sample.origin, sample.window), ((0, 0), (400, 300)))
        self.assertFalse(sample.native)

    def test_boxes_are_mapped_into_model_input_coordinates(self):
        entry = self.entry(size=(128, 128), boxes=((32, 32, 64, 64),))
        sample = Sample(frame=entry.frame, origin=(0, 0), window=(128, 128),
                        size=64, instances=entry.instances)
        np.testing.assert_allclose(sample.boxes[0], [16, 16, 32, 32])

    def test_a_frame_with_no_instances_produces_no_windows(self):
        entry = FrameIndex(frame=Frame("f", Path("a"), Path("b")),
                           size=(64, 64), instances=(), rejects={}, source=SOURCE)
        self.assertEqual(windows_for(entry), [])

    def test_the_same_seed_gives_the_same_pool(self):
        entries = [self.entry(size=(128, 128),
                              boxes=((10, 10, 20, 20), (60, 60, 70, 70)))]
        first = sample_windows(entries, size=64, per_image=2, seed=3)
        again = sample_windows(entries, size=64, per_image=2, seed=3)
        self.assertEqual([s.origin for s in first], [s.origin for s in again])


@unittest.skipIf(cv2 is None, "OpenCV is not installed")
class TestOnDisk(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ("a", "b"):
            for sub in ("tir", "rgb", "label"):
                (self.root / "scene" / sub).mkdir(parents=True, exist_ok=True)
            mask = put(put(semantic(), 2, 8, 8, 24, 24), 3, 40, 40, 50, 52)
            cv2.imwrite(str(self.root / "scene" / "label" / f"{name}_label.png"), mask)
            cv2.imwrite(str(self.root / "scene" / "tir" / f"{name}.png"),
                        (mask * 40).astype(np.uint8))
            cv2.imwrite(str(self.root / "scene" / "rgb" / f"{name}.png"),
                        np.stack([mask * 40] * 3, axis=-1))
        self.spec = DatasetSpec(**{**SPEC.__dict__, "strip": ("_label",)})
        self.source = Source(spec=self.spec, gates=GATES)

    def test_frames_pair_by_stem_after_stripping_the_mask_suffix(self):
        frames = list_frames(self.root, self.spec, "thermal")
        self.assertEqual([f.name for f in frames], ["a", "b"])
        self.assertTrue(all(f.pair is not None for f in frames))

    def test_an_unpaired_layout_is_an_error_not_an_empty_run(self):
        with self.assertRaises(FileNotFoundError):
            list_frames(self.root, DatasetSpec(**{**self.spec.__dict__,
                                                  "masks": "**/nope/*.png"}))

    def test_probe_classes_reports_what_is_really_in_the_maps(self):
        frames = list_frames(self.root, self.spec)
        self.assertEqual(sorted(probe_classes(frames)), [0, 2, 3])

    def test_the_index_survives_a_round_trip(self):
        frames = list_frames(self.root, self.spec)
        index = index_frames(frames, self.source, workers=2)
        self.assertEqual(len(index), 2)
        self.assertEqual(len(index[0].instances), 2)

        path = save_index(self.root / "index.json", index)
        again = load_index(path, self.source)
        self.assertEqual([e.frame.name for e in again], [e.frame.name for e in index])
        self.assertEqual([i.box for i in again[0].instances],
                         [i.box for i in index[0].instances])

    def test_a_target_mask_is_one_instance_not_its_whole_class(self):
        # Two cars in one window must give two different targets. If this ever
        # returned the class map instead, training would pull adjacent objects
        # together -- the exact failure the decomposition exists to prevent.
        mask = put(put(semantic(), 2, 4, 4, 14, 14), 2, 40, 40, 54, 54)
        cv2.imwrite(str(self.root / "scene" / "label" / "c_label.png"), mask)
        cv2.imwrite(str(self.root / "scene" / "tir" / "c.png"), (mask * 40).astype(np.uint8))

        frame = [f for f in list_frames(self.root, self.spec) if f.name == "c"][0]
        entry = index_frames([frame], self.source, workers=1)[0]
        sample = windows_for(entry, size=64, max_instances=4,
                             rng=np.random.default_rng(0))[0]
        masks = sample_masks(sample)

        self.assertEqual(masks.shape, (len(sample.instances), 64, 64))
        self.assertEqual(len(sample.instances), 2)
        self.assertFalse((masks[0] & masks[1]).any())
        for row, instance in zip(masks, sample.instances):
            self.assertEqual(int(row.sum()), instance.area)

    def test_summarise_counts_instances_and_names_the_rejects(self):
        frames = list_frames(self.root, self.spec)
        index = index_frames(frames, Source(self.spec, InstanceGates(min_area=200)),
                             workers=2)
        text = summarise(index)

        self.assertIn("min_area", text)
        self.assertIn("hold no instance", text)


class TestSplit(unittest.TestCase):
    def frames(self, n=100):
        return [Frame(name=f"{i:03d}", image=Path(f"{i}.png"), mask=Path(f"{i}_m.png"))
                for i in range(n)]

    def test_the_three_splits_partition_the_frames(self):
        parts = split_frames(self.frames(), seed=0)
        names = [f.name for part in parts.values() for f in part]
        self.assertEqual(len(names), 100)
        self.assertEqual(len(set(names)), 100)

    def test_the_split_is_reproducible_from_its_seed(self):
        first = split_frames(self.frames(), seed=7)
        again = split_frames(self.frames(), seed=7)
        self.assertEqual([f.name for f in first["test"]],
                         [f.name for f in again["test"]])

    def test_a_different_seed_moves_frames(self):
        first = split_frames(self.frames(), seed=1)["test"]
        other = split_frames(self.frames(), seed=2)["test"]
        self.assertNotEqual([f.name for f in first], [f.name for f in other])

    def test_fractions_that_do_not_sum_to_one_are_rejected(self):
        with self.assertRaises(ValueError):
            split_frames(self.frames(), fractions=(0.8, 0.8, 0.1))


if __name__ == "__main__":
    unittest.main()
