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

import json
import sys
import tempfile
import unittest
from dataclasses import replace
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
    list_pairs,
    replace,
    DatasetSpec,
    Frame,
    FrameIndex,
    Instance,
    InstanceGates,
    SPECS,
    Sample,
    Source,
    _allocate,
    decompose,
    group_of,
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
    apply_splits,
    drop_merge_profile,
    keep_only,
    rebalance,
    resolve_quantiles,
    size_bands,
    save_splits,
    split_index,
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

    def test_a_frame_smaller_than_the_window_gives_its_largest_square(self):
        # crop_window clamps an origin to 0 on an axis it cannot satisfy, which
        # would hand back a rectangle extending past the image. So the crop is
        # taken down to the largest square the frame has -- upsampled to `size`
        # afterwards, but never stretched unevenly, which the whole 400x300
        # frame would be at 1.28x across and 1.71x down.
        entry = self.entry(size=(400, 300), boxes=((10, 10, 30, 30),))
        sample = windows_for(entry, size=512, rng=np.random.default_rng(0))[0]

        self.assertEqual(sample.window, (300, 300))
        self.assertTrue(sample.instances[0].inside(sample.origin, sample.window))
        self.assertFalse(sample.native)

    def test_a_768_window_on_a_640x512_frame_stays_square(self):
        """The case that decides what a 768 stage B trains on.

        Most of the thermal sets are 640x512, so at `size=768` none of them can
        supply a native window. A 512 square upsampled to 768 pays the
        interpolation and keeps a target's shape; the whole frame would deform
        every one of them by 1.2x against 1.5x, and shape is what the mask head
        is being taught.
        """
        entry = self.entry(size=(640, 512), boxes=((300, 200, 340, 260),))
        sample = windows_for(entry, size=768, rng=np.random.default_rng(0))[0]

        self.assertEqual(sample.window, (512, 512))
        self.assertEqual(sample.size, 768)
        self.assertFalse(sample.native)
        # 40x60 source pixels magnified by 768/512, and the two axes agree.
        box = sample.boxes[0]
        self.assertAlmostEqual(float(box[2] - box[0]), 40 * 1.5, places=4)
        self.assertAlmostEqual(float(box[3] - box[1]), 60 * 1.5, places=4)

    def test_boxes_are_mapped_into_model_input_coordinates(self):
        entry = self.entry(size=(128, 128), boxes=((32, 32, 64, 64),))
        sample = Sample(frame=entry.frame, origin=(0, 0), window=(128, 128),
                        size=64, instances=entry.instances)
        np.testing.assert_allclose(sample.boxes[0], [16, 16, 32, 32])

    def test_a_window_larger_than_the_input_shrinks_the_targets(self):
        # The small-object regime, taken out of a set that has none: a wider
        # crop resized down divides every target's apparent side by
        # window/size, and the annotation is never touched.
        entry = self.entry(size=(640, 512), boxes=((300, 200, 340, 260),))
        sample = windows_for(entry, size=128, window=384,
                             rng=np.random.default_rng(0))[0]

        self.assertEqual(sample.window, (384, 384))
        self.assertEqual(sample.size, 128)
        self.assertFalse(sample.native)
        # 40x60 source pixels at a 3x reduction.
        box = sample.boxes[0]
        self.assertAlmostEqual(float(box[2] - box[0]), 40 / 3, places=4)
        self.assertAlmostEqual(float(box[3] - box[1]), 60 / 3, places=4)

    def test_window_defaults_to_size_so_the_training_runs_are_unchanged(self):
        # Every number recorded before `window` existed was taken with the crop
        # equal to the input. Passing neither has to still mean exactly that.
        entry = self.entry(size=(640, 512), boxes=((100, 100, 140, 140),))
        default = windows_for(entry, size=256, rng=np.random.default_rng(7))[0]
        spelled = windows_for(entry, size=256, window=256,
                              rng=np.random.default_rng(7))[0]

        self.assertEqual(default.window, (256, 256))
        self.assertTrue(default.native)
        self.assertEqual((default.origin, default.window),
                         (spelled.origin, spelled.window))

    def test_a_window_wider_than_the_frame_is_taken_down_to_what_it_has(self):
        # A 1536 window on a 640x512 frame would ask crop_window for a
        # rectangle running off the sensor. The source cannot deliver the
        # reduction asked for, so it delivers the largest square it has and
        # `Sample.window` records what was actually taken -- 512, not 1536.
        entry = self.entry(size=(640, 512), boxes=((10, 10, 30, 30),))
        sample = windows_for(entry, size=512, window=1536,
                             rng=np.random.default_rng(0))[0]

        self.assertEqual(sample.window, (512, 512))
        self.assertTrue(sample.instances[0].inside(sample.origin, sample.window))

    def test_sample_windows_passes_the_window_through(self):
        entries = [self.entry(size=(640, 512), boxes=((100, 100, 140, 140),))]
        samples = sample_windows(entries, size=128, window=384, seed=1)
        self.assertEqual([s.window for s in samples], [(384, 384)])

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
        # Qualified by the directory the frame sits in, not the bare stem --
        # see `test_sequences_that_number_frames_the_same_stay_apart`.
        self.assertEqual([f.name for f in frames], ["scene/a", "scene/b"])
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

        frame = [f for f in list_frames(self.root, self.spec) if f.name == "scene/c"][0]
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


@unittest.skipIf(cv2 is None, "OpenCV is not installed")
class TestPerSequenceLayout(unittest.TestCase):
    """The VTUAV layout: frames numbered from zero inside every sequence.

    Real numbers from `training/train_001.zip`: fourteen sequences, each
    starting at `000000.jpg`, and masks in a directory per modality.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.spec = DatasetSpec(
            name="toy_vtuav", thermal="**/ir/*.png", rgb="**/rgb/*.png",
            masks="**/mask/{modality}/*.png",
            classes={"background": 0, "target": 255}, things=("target",))

        for sequence in ("bike_009", "car_083"):
            for sub in ("ir", "rgb", "mask/ir", "mask/rgb"):
                (self.root / sequence / sub).mkdir(parents=True, exist_ok=True)
        # The same stem in both sequences, and a mask that differs between
        # them so a mixed-up pairing is visible rather than merely wrong.
        for sequence, column in (("bike_009", 10), ("car_083", 40)):
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[20:30, column:column + 10] = 255
            for modality in ("ir", "rgb"):
                cv2.imwrite(str(self.root / sequence / modality / "000000.png"),
                            (mask // 3).astype(np.uint8))
                cv2.imwrite(str(self.root / sequence / "mask" / modality /
                                "000000.png"), mask)

    def test_sequences_that_number_frames_the_same_stay_apart(self):
        # Keyed on the bare stem both sequences are `000000`, one overwrites
        # the other in the dict, and half the download disappears silently.
        frames = list_frames(self.root, self.spec, "thermal")
        self.assertEqual([f.name for f in frames],
                         ["bike_009/000000", "car_083/000000"])

    def test_the_image_and_its_mask_come_from_the_same_sequence(self):
        # The worse half of the same bug: the image dict and the mask dict are
        # built from different globs, so the survivor on each side can be a
        # different sequence and the model trains on one flight's picture
        # against another flight's mask.
        for frame in list_frames(self.root, self.spec, "thermal"):
            sequence = frame.name.split("/")[0]
            self.assertEqual(frame.image.parts[-3], sequence)
            self.assertEqual(frame.mask.parts[-4], sequence)
            self.assertEqual(frame.pair.parts[-3], sequence)

    def test_each_modality_reads_its_own_mask_directory(self):
        self.assertEqual(self.spec.mask_glob("thermal"), "**/mask/ir/*.png")
        self.assertEqual(self.spec.mask_glob("rgb"), "**/mask/rgb/*.png")
        for modality, folder in (("thermal", "ir"), ("rgb", "rgb")):
            for frame in list_frames(self.root, self.spec, modality):
                self.assertEqual(frame.mask.parent.name, folder)

    def test_a_spec_with_one_mask_directory_is_left_alone(self):
        flat = DatasetSpec(name="flat", thermal="**/tir/*.png",
                           masks="**/label/*.png", classes={"background": 0},
                           things=())
        self.assertEqual(flat.mask_glob("thermal"), "**/label/*.png")


@unittest.skipIf(cv2 is None, "OpenCV is not installed")
class TestExcludedFrames(unittest.TestCase):
    """Frames a dataset itself marks unusable, dropped before anything reads them.

    Kust4K ships five `broken_in_*.txt` manifests naming **1 160 of its 4 024
    frames** -- 29 % -- where one modality is deliberately corrupted to
    simulate a sensor failure. Nothing about the download hints at it: the
    images are present and decode fine, they are simply not pictures of the
    scene. Trained on, they are noise carrying a confident label.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.spec = DatasetSpec(
            name="toy", thermal="**/tir/*.png", rgb="**/rgb/*.png",
            masks="**/label/*.png", classes={"background": 0, "car": 2},
            things=("car",), exclude="broken_*.txt")
        for sub in ("tir", "rgb", "label"):
            (self.root / sub).mkdir(parents=True)
        for stem in ("00001D", "00002D", "00003D"):
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[8:16, 8:16] = 2
            cv2.imwrite(str(self.root / "label" / f"{stem}.png"), mask)
            for sub in ("tir", "rgb"):
                cv2.imwrite(str(self.root / sub / f"{stem}.png"), mask * 40)

    def write(self, name, lines):
        (self.root / name).write_text("\n".join(lines) + "\n")

    def test_a_listed_frame_is_dropped_from_frames_and_pairs(self):
        self.write("broken_a.txt", ["00002D"])
        self.assertEqual([f.name for f in list_frames(self.root, self.spec)],
                         ["00001D", "00003D"])
        self.assertEqual(len(list_pairs(self.root, self.spec)), 2)

    def test_the_suffix_is_optional_because_the_manifests_disagree(self):
        # Kust4K's own five files are inconsistent: the test lists write
        # `01275D`, the train lists write `00261D.png`.
        self.write("broken_a.txt", ["00002D.png"])
        self.write("broken_b.txt", ["00003D"])
        self.assertEqual([f.name for f in list_frames(self.root, self.spec)],
                         ["00001D"])

    def test_every_manifest_is_read_not_just_the_first(self):
        self.write("broken_a.txt", ["00001D"])
        self.write("broken_b.txt", ["00002D"])
        self.assertEqual(len(list_frames(self.root, self.spec)), 1)

    def test_blank_lines_are_ignored(self):
        self.write("broken_a.txt", ["", "00002D", "   ", ""])
        self.assertEqual(len(list_frames(self.root, self.spec)), 2)

    def test_a_spec_with_no_exclude_reads_everything(self):
        self.write("broken_a.txt", ["00002D"])
        plain = replace(self.spec, exclude="")
        self.assertEqual(len(list_frames(self.root, plain)), 3)

    def test_no_manifest_present_is_not_an_error(self):
        # The manifests are fetched beside the archives; an older download will
        # not have them and must still read.
        self.assertEqual(len(list_frames(self.root, self.spec)), 3)

    def test_kust4k_is_the_spec_that_declares_them(self):
        self.assertEqual(SPECS["kust4k"].exclude, "broken_in_*.txt")


class TestPaletteProvenance(unittest.TestCase):
    """Every spec has to say where its class ids came from.

    Three palettes in this file were guessed and all three were wrong, each in
    the same direction: `things` ended up selecting scenery. Kust4K picked
    **tree** and missed motorcycle; SegFly made grass, vegetation, tree and
    ground obstacle into targets; Caltech was wrong in all nine of its entries
    and pointed at **Trees** and **Sky**. None of them raised anything -- the
    model would just have learned to answer a prompt with a hedge.

    A citation cannot make a palette right. What it does is force the author to
    state, out loud, whether the numbers were read off the archive or off a
    paper -- and those are very different claims.
    """

    def test_every_spec_says_where_its_palette_came_from(self):
        for name, spec in sorted(SPECS.items()):
            with self.subTest(spec=name):
                self.assertTrue(
                    spec.palette_source.strip(),
                    f"{name}: set palette_source to where `classes` was read "
                    f"from. If it was guessed, verify it first -- three "
                    f"guesses here have already been wrong.")

    def test_no_spec_makes_scenery_a_tracking_target(self):
        # The shape all three failures took, as one assertion. These are names
        # that cannot be a single trackable object, whatever the dataset.
        scenery = {"tree", "trees", "sky", "grass", "vegetation", "shrubs",
                   "water", "road", "terrain", "bare_ground", "rocky_terrain",
                   "building", "road_marking", "background", "unlabelled",
                   "unlabeled", "unknown"}
        for name, spec in sorted(SPECS.items()):
            with self.subTest(spec=name):
                wrong = sorted(set(spec.things) & scenery)
                self.assertEqual(wrong, [], f"{name}: {wrong} cannot be tracked")

    def test_a_things_entry_must_exist_in_the_palette(self):
        # `thing_ids` raises on a name that is not in `classes`, which is what
        # would have caught a rename. Assert it stays that way.
        for name, spec in sorted(SPECS.items()):
            with self.subTest(spec=name):
                self.assertEqual(len(spec.thing_ids), len(spec.things))


class TestCaltechPalette(unittest.TestCase):
    """Read off the archive's own colour table; the guess was wrong throughout."""

    def test_the_two_archives_share_one_palette(self):
        self.assertEqual(SPECS["caltech"].classes, SPECS["caltech_rgbt"].classes)

    def test_trees_and_sky_are_not_tracking_targets(self):
        # The guess called id 7 "vehicle" and id 8 "person"; they are Trees and
        # Sky, so `things` selected every tree and the whole sky.
        spec = SPECS["caltech"]
        self.assertEqual(spec.name_of(7), "trees")
        self.assertEqual(spec.name_of(8), "sky")
        self.assertEqual(sorted(spec.thing_ids), [10, 11])

    def test_the_singles_archive_declares_no_rgb_twin(self):
        # It ships a `color/` directory that is 819x512 against the thermal's
        # 640x512, and about 5 % of it is a pure black frame. Globbing it would
        # give the teacher a different picture from the student's.
        self.assertIsNone(SPECS["caltech"].rgb)
        with self.assertRaises(ValueError):
            SPECS["caltech"].glob("rgb")

    def test_the_paired_archive_does_have_one(self):
        self.assertIsNotNone(SPECS["caltech_rgbt"].rgb)

    def test_both_ignore_the_two_non_classes(self):
        for name in ("caltech", "caltech_rgbt"):
            self.assertEqual(SPECS[name].ignore, (0, 1))


class TestKust4kPalette(unittest.TestCase):
    """Read off the archive's own `visual.py`, pinned so it cannot drift back.

    The bug this replaces: the palette assumed a `vegetation` class at id 3.
    There is none, so every id above it was shifted by one and there was no
    room for id 8 at all. `things` therefore selected id 6 -- **tree** -- and
    missed id 3, **motorcycle**.
    """

    def test_the_ids_are_the_ones_in_get_palette(self):
        # palette = [unlabelled, road, building, motorcycle, car, truck, tree,
        #            human, traffic_facilities], indexed by class id.
        self.assertEqual(
            [SPECS["kust4k"].name_of(i) for i in range(9)],
            ["unlabelled", "road", "building", "motorcycle", "car", "truck",
             "tree", "human", "traffic_facilities"])

    def test_trees_are_not_tracking_targets_and_motorcycles_are(self):
        spec = SPECS["kust4k"]
        self.assertEqual(sorted(spec.thing_ids), [3, 4, 5, 7])
        self.assertNotIn(spec.classes["tree"], spec.thing_ids)
        self.assertIn(spec.classes["motorcycle"], spec.thing_ids)

    def test_the_ninth_class_the_old_palette_had_no_room_for_exists(self):
        self.assertEqual(SPECS["kust4k"].classes["traffic_facilities"], 8)


class TestGroupedSplit(unittest.TestCase):
    """Whole sequences on one side of the line, not frames.

    VTUAV masks every 30th frame, so two masks from one flight are a second
    apart and show the same target against the same background. Splitting
    frames puts them on opposite sides and the test score measures memory.
    """

    def frames(self, sequences=5, per=20):
        return [Frame(name=f"seq_{s:02d}/{i:06d}", image=Path(f"{s}_{i}.jpg"),
                      mask=Path(f"{s}_{i}.png"))
                for s in range(sequences) for i in range(per)]

    def test_no_sequence_appears_in_two_splits(self):
        parts = split_frames(self.frames(), seed=0)
        seen = {name: {group_of(f) for f in part} for name, part in parts.items()}
        self.assertTrue(seen["train"])
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            self.assertFalse(seen[a] & seen[b], f"{a} and {b} share a sequence")

    def test_every_frame_still_lands_somewhere_exactly_once(self):
        frames = self.frames()
        parts = split_frames(frames, seed=0)
        names = [f.name for part in parts.values() for f in part]
        self.assertEqual(sorted(names), sorted(f.name for f in frames))

    def test_a_sequence_is_kept_whole(self):
        parts = split_frames(self.frames(), seed=0)
        for part in parts.values():
            counts: dict[str, int] = {}
            for frame in part:
                counts[group_of(frame)] = counts.get(group_of(frame), 0) + 1
            self.assertTrue(all(n == 20 for n in counts.values()), counts)

    def test_a_flat_set_is_split_frame_by_frame_as_before(self):
        # Kust4K stems carry no sequence, so each is its own group and the
        # result has to be the plain permutation it always was.
        flat = [Frame(name=f"{i:05d}D", image=Path(f"{i}.png"),
                      mask=Path(f"{i}_m.png")) for i in range(100)]
        parts = split_frames(flat, seed=0)
        self.assertEqual([len(p) for p in parts.values()], [80, 10, 10])

    def test_too_few_sequences_to_split_warns_instead_of_emptying_val(self):
        with self.assertWarns(UserWarning):
            parts = split_frames(self.frames(sequences=2), seed=0)
        self.assertTrue(parts["val"])
        self.assertTrue(parts["test"])

    def test_group_none_asks_for_the_ungrouped_split(self):
        parts = split_frames(self.frames(), seed=0, group=None)
        overlap = {group_of(f) for f in parts["train"]} & \
                  {group_of(f) for f in parts["test"]}
        self.assertTrue(overlap, "group=None should not hold sequences out")


class TestAllocation(unittest.TestCase):
    """Rounding must not empty a split when there are few sequences to divide.

    Independent rounding is fine on thousands of frames and wrong on a handful
    of sequences: 6 at (0.8, 0.1, 0.1) rounds to 5, 1, 0 and the test set is
    gone. That surfaces much later as a `nan`, not as an error here.
    """

    def test_nothing_is_starved_once_there_are_enough_groups(self):
        for total in range(3, 40):
            counts = _allocate(total, (0.8, 0.1, 0.1))
            self.assertEqual(sum(counts), total, total)
            self.assertTrue(all(c > 0 for c in counts), f"{total} -> {counts}")

    def test_large_counts_are_still_the_plain_proportions(self):
        self.assertEqual(_allocate(100, (0.8, 0.1, 0.1)), [80, 10, 10])
        self.assertEqual(_allocate(1000, (0.8, 0.1, 0.1)), [800, 100, 100])

    def test_a_zero_fraction_stays_zero(self):
        self.assertEqual(_allocate(10, (0.5, 0.5, 0.0))[2], 0)


class TestSplitsAreComparableAcrossRuns(unittest.TestCase):
    """The property the two notebooks' comparison rests on.

    Notebook 07 trains on three datasets and 08 on VTUAV alone. Comparing their
    test scores is only meaningful if *the same VTUAV sequences are held out in
    both*. The per-source seed used to be the source's position in the sorted
    grouping, so VTUAV was source 3 of 3 in one run and 1 of 1 in the other --
    different seed, different sequences, and the two numbers would have been
    measured on different test sets.
    """

    def entries(self, source, sequences=8, per=5):
        return [FrameIndex(frame=Frame(name=f"seq_{s:02d}/{i:06d}",
                                       image=Path("i"), mask=Path("m")),
                           instances=(), size=(64, 64), rejects={}, source=source)
                for s in range(sequences) for i in range(per)]

    def vtuav_sequences(self, part):
        return sorted({e.frame.name.split("/")[0] for e in part
                       if e.source.spec.name == "vtuav_vis"})

    def test_adding_datasets_does_not_move_another_ones_split(self):
        gates = InstanceGates()
        vtuav = Source(SPECS["vtuav_vis"], gates, mode="labels", role="all")
        alone = split_index(self.entries(vtuav), seed=0)
        mixed = split_index(
            self.entries(vtuav)
            + self.entries(Source(SPECS["kust4k"], gates, role="train"))
            + self.entries(Source(SPECS["segfly"], gates, role="train")),
            seed=0)

        for part in ("train", "val", "test"):
            self.assertEqual(self.vtuav_sequences(alone[part]),
                             self.vtuav_sequences(mixed[part]), part)
        self.assertTrue(self.vtuav_sequences(alone["test"]),
                        "a test set that is empty would pass this vacuously")

    def test_two_sources_still_get_different_permutations(self):
        gates = InstanceGates()
        index = (self.entries(Source(SPECS["kust4k"], gates))
                 + self.entries(Source(SPECS["segfly"], gates)))
        parts = split_index(index, seed=0)
        by_spec = {}
        for entry in parts["test"]:
            by_spec.setdefault(entry.source.spec.name, set()).add(
                entry.frame.name.split("/")[0])
        self.assertNotEqual(by_spec.get("kust4k"), by_spec.get("segfly"))


class DropMergeProfile(unittest.TestCase):
    """The shape cut for merges `fill` cannot see."""

    def index(self, boxes, spec="segfly", cls="vehicle"):
        source = Source(SPECS[spec], InstanceGates(), role="train")
        out = []
        for f, box in enumerate(boxes):
            w, h = box
            out.append(FrameIndex(
                frame=Frame(name=f"f/{f:06d}", image=Path("i"), mask=Path("m")),
                instances=(Instance(label=1, class_id=SPECS[spec].classes[cls],
                                    box=(0.0, 0.0, float(w), float(h)),
                                    area=int(w * h)),),
                size=(640, 512), rejects={}, source=source))
        return out

    def population(self, n=300):
        # A single vehicle: 70x40, the pool's own median shape.
        return self.index([(70, 40)] * n)

    def test_two_abreast_go_and_the_singles_stay(self):
        # Width doubles, so the blob turns square -- the case `fill` misses.
        index = self.population() + self.index([(70, 80)] * 20)
        kept, report = drop_merge_profile(index, min_sample=100)
        row = report["by_class"]["segfly:vehicle"]
        self.assertEqual(row["abreast"], 20)
        self.assertEqual(row["after"], 300, "the singles must survive intact")

    def test_two_nose_to_tail_go_as_well(self):
        index = self.population() + self.index([(140, 40)] * 20)
        _, report = drop_merge_profile(index, min_sample=100)
        row = report["by_class"]["segfly:vehicle"]
        self.assertEqual(row["end_to_end"], 20)

    def test_a_large_vehicle_that_keeps_its_proportions_stays(self):
        # A lorry is longer *and* wider; a merge doubles one dimension only.
        index = self.population() + self.index([(105, 60)] * 20)
        _, report = drop_merge_profile(index, min_sample=100)
        row = report["by_class"]["segfly:vehicle"]
        self.assertEqual(row["dropped"], 0,
                         "growing in both directions is not a merge signature")

    def test_a_source_not_named_is_never_touched(self):
        index = self.population() + self.index([(70, 80)] * 20)
        kept, report = drop_merge_profile(index, sources=("kust4k",))
        self.assertEqual(len(kept), len(index))
        self.assertEqual(report["by_class"], {})

    def test_too_few_instances_to_estimate_a_median_are_left_alone(self):
        index = self.index([(70, 40)] * 10 + [(70, 80)] * 4)
        kept, report = drop_merge_profile(index, min_sample=200)
        self.assertEqual(len(kept), len(index))
        self.assertEqual(report["unmeasured"], ["segfly:vehicle"],
                         "a median over a handful of boxes is not a prior")


class TestThinningAnOverRepresentedClass(unittest.TestCase):
    """A vehicle-heavy pool teaches "a target is a car" as surely as it teaches
    thermal, and the classes that make the set general are outvoted."""

    def index(self, per_frame, frames=40):
        gates = InstanceGates()
        source = Source(SPECS["kust4k"], gates, role="train")
        spec = source.spec
        out = []
        for f in range(frames):
            instances = tuple(
                Instance(label=i, class_id=spec.classes[name],
                         box=(0.0, 0.0, 8.0, 8.0), area=64)
                for i, name in enumerate(per_frame))
            out.append(FrameIndex(
                frame=Frame(name=f"seq/{f:06d}", image=Path("i"), mask=Path("m")),
                instances=instances, size=(64, 64), rejects={}, source=source))
        return out

    def names(self):
        return list(SPECS["kust4k"].classes)[:2]

    def test_a_weight_below_one_thins_that_class_and_leaves_the_rest(self):
        common, rare = self.names()
        index = self.index([common] * 8 + [rare])
        kept, report = rebalance(index, {common: 0.5}, seed=0)
        was, now = report["by_class"][common]
        self.assertEqual(was, 320)
        self.assertLess(now, was)
        self.assertGreater(now, was // 4)         # thinned, not deleted
        self.assertEqual(report["by_class"][rare], (40, 40))

    def test_the_same_weights_give_the_same_set_every_time(self):
        common, _ = self.names()
        index = self.index([common] * 4)
        first, _ = rebalance(index, {common: 0.5}, seed=0)
        second, _ = rebalance(list(reversed(index)), {common: 0.5}, seed=0)
        self.assertEqual(
            sorted((e.frame.name, i.label) for e in first for i in e.instances),
            sorted((e.frame.name, i.label) for e in second for i in e.instances),
            "the keep decision must not depend on order")

    def test_a_frame_left_with_nothing_is_dropped(self):
        common, _ = self.names()
        index = self.index([common])
        kept, report = rebalance(index, {common: 0.0}, seed=0)
        self.assertEqual(kept, [])
        self.assertEqual(report["frames"]["after"], 0)

    def test_a_frame_keeps_its_rare_instance_when_its_common_ones_go(self):
        common, rare = self.names()
        index = self.index([common] * 6 + [rare])
        kept, _ = rebalance(index, {common: 0.0}, seed=0)
        self.assertEqual(len(kept), len(index),
                         "dropping the frame would throw the rare class away too")
        spec = kept[0].source.spec
        self.assertEqual({spec.name_of(i.class_id)
                          for e in kept for i in e.instances}, {rare})

    def test_a_source_weight_thins_only_that_source(self):
        common, _ = self.names()
        gates = InstanceGates()
        other = Source(SPECS["segfly"], gates, role="train")
        mine = self.index([common] * 4)
        theirs = [replace(e, source=other) for e in self.index([common] * 4)]
        kept, report = rebalance(mine + theirs, {"kust4k": 0.0}, seed=0)
        left = {e.source.spec.name for e in kept}
        self.assertEqual(left, {"segfly"},
                         "a source weight must not reach another source")
        self.assertEqual(report["by_source"]["kust4k"][1], 0)
        self.assertEqual(*report["by_source"]["segfly"])

    def test_a_source_scoped_class_beats_the_bare_class(self):
        common, rare = self.names()
        kept, report = rebalance(self.index([common, rare]),
                                 {common: 0.0, f"kust4k:{common}": 1.0}, seed=0)
        self.assertEqual(report["by_class"][common][1],
                         report["by_class"][common][0],
                         "the specific weight wins, and nothing compounds")

    def test_the_report_counts_instances_per_source(self):
        common, _ = self.names()
        _, report = rebalance(self.index([common] * 3, frames=10), {}, seed=0)
        self.assertEqual(report["by_source"], {"kust4k": (30, 30)})

    def test_a_weight_naming_no_class_here_is_reported(self):
        common, _ = self.names()
        _, report = rebalance(self.index([common]), {"unicorn": 0.5}, seed=0)
        self.assertEqual(report["unmatched"], ["unicorn"])

    def test_no_weights_changes_nothing(self):
        index = self.index(self.names())
        kept, report = rebalance(index, {}, seed=0)
        self.assertEqual(len(kept), len(index))
        self.assertEqual(report["instances"]["before"],
                         report["instances"]["after"])


class TestSizeBands(unittest.TestCase):
    """Dropping the large cars in one source without touching another.

    `InstanceGates.max_area` cannot express this: it is one fraction of the
    frame for every source at once, and a frame is 640x512 in HIT-UAV against
    1920x1080 in VTUAV, so the same fraction is two very different pixel sizes.
    """

    def index(self, sides, name=None, frames=4):
        gates = InstanceGates()
        source = Source(SPECS["kust4k"], gates, role="train")
        spec = source.spec
        name = name or list(spec.classes)[0]
        out = []
        for f in range(frames):
            instances = tuple(
                Instance(label=i, class_id=spec.classes[name],
                         box=(0.0, 0.0, float(side), float(side)),
                         area=int(side * side))
                for i, side in enumerate(sides))
            out.append(FrameIndex(
                frame=Frame(name=f"seq/{f:06d}", image=Path("i"), mask=Path("m")),
                instances=instances, size=(2048, 2048), rejects={}, source=source))
        return out

    def test_an_empty_band_table_drops_nothing_and_still_measures(self):
        # The report is what a threshold is chosen from, so it has to exist
        # before anyone has picked one.
        kept, report = size_bands(self.index([10, 40, 200]), {})
        self.assertEqual(report["instances"], {"before": 12, "after": 12})
        self.assertEqual(len(kept), 4)
        row = next(iter(report["sides"].values()))
        self.assertEqual(row["n"], 12)
        self.assertEqual(row["max"], 200.0)

    def test_the_band_keeps_what_is_inside_it(self):
        name = list(SPECS["kust4k"].classes)[0]
        kept, report = size_bands(self.index([10, 40, 200], name),
                                  {f"kust4k:{name}": (0, 96)})
        self.assertEqual(report["instances"]["after"], 8)   # 10 and 40 survive
        self.assertEqual(report["dropped"][f"kust4k:{name}"], 4)
        for entry in kept:
            for instance in entry.instances:
                self.assertLessEqual(max(instance.width, instance.height), 96)

    def test_a_floor_drops_the_ones_below_it_too(self):
        name = list(SPECS["kust4k"].classes)[0]
        _, report = size_bands(self.index([10, 40, 200], name),
                               {f"kust4k:{name}": (32, 96)})
        self.assertEqual(report["instances"]["after"], 4)   # only 40 survives

    def test_the_longer_side_decides_not_the_area(self):
        # A thin sliver and a square of the same area are different targets,
        # and the grade's own small-object row splits on the side.
        gates = InstanceGates()
        source = Source(SPECS["kust4k"], gates, role="train")
        name = list(SPECS["kust4k"].classes)[0]
        entry = FrameIndex(
            frame=Frame(name="s/0", image=Path("i"), mask=Path("m")),
            instances=(Instance(label=1, class_id=source.spec.classes[name],
                                box=(0.0, 0.0, 200.0, 4.0), area=800),),
            size=(2048, 2048), rejects={}, source=source)
        _, report = size_bands([entry], {f"kust4k:{name}": (0, 96)})
        self.assertEqual(report["instances"]["after"], 0)

    def test_a_band_on_one_source_leaves_another_alone(self):
        name = list(SPECS["kust4k"].classes)[0]
        mine = self.index([200], name)
        theirs = [replace(e, source=Source(SPECS["dronevehicle"], InstanceGates(),
                                           role="train"))
                  for e in self.index([200], name)]
        # The other source's class ids belong to another spec, so key it by
        # source alone -- which is the form "leave that set alone" takes.
        _, report = size_bands(mine + theirs, {"kust4k": (0, 96)})
        self.assertEqual(report["dropped"]["kust4k"], 4)
        self.assertEqual(report["instances"]["after"], 4)

    def test_a_frame_left_with_nothing_is_dropped(self):
        name = list(SPECS["kust4k"].classes)[0]
        kept, report = size_bands(self.index([200], name),
                                  {f"kust4k:{name}": (0, 96)})
        self.assertEqual(kept, [])
        self.assertEqual(report["frames"], {"before": 4, "after": 0})

    def test_a_key_nothing_matched_is_named(self):
        # A misspelt source looks exactly like a source that is already rare.
        _, report = size_bands(self.index([10]), {"unicorn:car": (0, 96)})
        self.assertEqual(report["unmatched"], ["unicorn:car"])

    def test_the_most_specific_key_wins_and_nothing_compounds(self):
        name = list(SPECS["kust4k"].classes)[0]
        _, report = size_bands(
            self.index([200], name),
            {"kust4k": (0, 96), f"kust4k:{name}": (0, 1000)})
        self.assertEqual(report["instances"]["after"], 4)   # the class band won
        self.assertEqual(report["dropped"], {})


class TestKeepOnly(unittest.TestCase):
    """"This run sees these three things and nothing else."

    A weight of 0.0 can exclude, but saying it that way means enumerating
    everything unwanted, and a pool that grows a class later starts training on
    it silently. An allowlist cannot widen by accident.
    """

    def frame(self, spec, names, key="s/000000"):
        source = Source(SPECS[spec], InstanceGates(), role="train")
        return FrameIndex(
            frame=Frame(name=key, image=Path("i"), mask=Path("m")),
            instances=tuple(
                Instance(label=i, class_id=source.spec.classes[n],
                         box=(0.0, 0.0, 8.0, 8.0), area=64)
                for i, n in enumerate(names)),
            size=(64, 64), rejects={}, source=source)

    def test_a_source_and_class_key_keeps_only_that_pair(self):
        index = [self.frame("kust4k", ["car", "truck", "human"])]
        kept, report = keep_only(index, ["kust4k:car"])
        self.assertEqual(report["instances"], {"before": 3, "after": 1})
        self.assertEqual([i.class_id for i in kept[0].instances],
                         [SPECS["kust4k"].classes["car"]])

    def test_a_source_alone_keeps_all_of_its_classes(self):
        # How "all of HIT-UAV" is written.
        index = [self.frame("kust4k", ["car", "truck", "human"])]
        _, report = keep_only(index, ["kust4k"])
        self.assertEqual(report["instances"], {"before": 3, "after": 3})

    def test_what_is_not_named_is_dropped(self):
        index = [self.frame("kust4k", ["car"]),
                 self.frame("segfly_temiz", ["vehicle"])]
        kept, report = keep_only(index, ["kust4k:car"])
        self.assertEqual(report["instances"], {"before": 2, "after": 1})
        self.assertEqual([e.source.spec.name for e in kept], ["kust4k"])

    def test_a_frame_left_with_nothing_is_dropped(self):
        index = [self.frame("kust4k", ["truck"])]
        kept, report = keep_only(index, ["kust4k:car"])
        self.assertEqual(kept, [])
        self.assertEqual(report["frames"], {"before": 1, "after": 0})

    def test_a_misspelt_key_is_named_rather_than_silently_emptying(self):
        # The failure this has to be loud about: a typo does not thin slightly
        # too much here, it trains on nothing.
        index = [self.frame("kust4k", ["car"])]
        kept, report = keep_only(index, ["kust4k:Car"])
        self.assertEqual(kept, [])
        self.assertEqual(report["unmatched"], ["kust4k:Car"])

    def test_the_report_says_what_each_class_lost(self):
        index = [self.frame("kust4k", ["car", "car", "truck"])]
        _, report = keep_only(index, ["kust4k:car"])
        self.assertEqual(report["by_class"]["car"], (2, 2))
        self.assertEqual(report["by_class"]["truck"], (1, 0))

    def test_an_empty_allowlist_keeps_nothing(self):
        # Not "keeps everything": an allowlist that names nothing allows
        # nothing, and the caller asserts on the count rather than discovering
        # it after an epoch.
        index = [self.frame("kust4k", ["car"])]
        kept, _ = keep_only(index, [])
        self.assertEqual(kept, [])


class TestQuantileBands(unittest.TestCase):
    """A cut that names the number it came from, without a second run.

    `(0, 96)` typed by hand is a guess until the distribution is measured.
    The 90th percentile of what is actually there is the measurement, and it
    resolves to a pixel band that `save_splits` can record.
    """

    def index(self, sides, name=None, spec="kust4k", frames=1):
        source = Source(SPECS[spec], InstanceGates(), role="train")
        name = name or list(source.spec.classes)[0]
        return [FrameIndex(
            frame=Frame(name=f"s/{f:06d}", image=Path("i"), mask=Path("m")),
            instances=tuple(
                Instance(label=i, class_id=source.spec.classes[name],
                         box=(0.0, 0.0, float(s), float(s)), area=int(s * s))
                for i, s in enumerate(sides)),
            size=(2048, 2048), rejects={}, source=source) for f in range(frames)]

    def test_a_quantile_becomes_the_pixel_band_it_measured(self):
        name = list(SPECS["kust4k"].classes)[0]
        index = self.index(list(range(1, 101)), name)      # sides 1..100
        got = resolve_quantiles(index, {f"kust4k:{name}": (0.0, 0.90)})
        low, high = got[f"kust4k:{name}"]
        self.assertEqual(low, 1.0)
        self.assertAlmostEqual(high, 90.1, places=6)

    def test_the_resolved_band_is_what_size_bands_then_applies(self):
        name = list(SPECS["kust4k"].classes)[0]
        index = self.index(list(range(1, 101)), name)
        bands = resolve_quantiles(index, {f"kust4k:{name}": (0.0, 0.90)})
        _, report = size_bands(index, bands)
        self.assertEqual(report["instances"], {"before": 100, "after": 90})

    def test_a_source_key_sees_every_class_it_speaks_for(self):
        # A quantile of "kust4k" must be taken over the source's instances,
        # not over one class's column, or the cut is not the one asked for.
        first, second = list(SPECS["kust4k"].classes)[:2]
        index = (self.index([10] * 90, first) + self.index([500] * 10, second))
        source_p90 = resolve_quantiles(index, {"kust4k": (0.0, 0.90)})["kust4k"][1]
        class_p90 = resolve_quantiles(
            index, {f"kust4k:{first}": (0.0, 0.90)})[f"kust4k:{first}"][1]
        self.assertEqual(class_p90, 10.0, "one class's column is only its own")
        self.assertGreater(source_p90, 10.0,
                           "the source key must have seen the other class too")
        self.assertLess(source_p90, 500.0)

    def test_a_key_nothing_matched_is_left_out_not_defaulted(self):
        # Left out so `size_bands` reports it as unmatched, rather than
        # quietly banding nothing under a key that looks applied.
        self.assertEqual(resolve_quantiles(self.index([10]), {"unicorn": (0, 0.9)}),
                         {})

    def test_the_two_keys_do_not_pool_their_sizes(self):
        # Each source names its own classes, so let each index use its own.
        index = self.index([10] * 10) + self.index([500] * 10, spec="segfly")
        got = resolve_quantiles(index, {"kust4k": (0.0, 1.0),
                                        "segfly": (0.0, 1.0)})
        self.assertEqual(got["kust4k"][1], 10.0)
        self.assertEqual(got["segfly"][1], 500.0)


class TestASavedSplitIsTheSplitTheRunUses(unittest.TestCase):
    """A caller that caps a pool and drops overlapping frames has decided the
    run's data, and handing the CLI the same *flags* re-derives all of it. The
    split file is how that decision travels."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "splits.json"
        gates = InstanceGates()
        self.kust = Source(SPECS["kust4k"], gates, role="train")
        self.segfly = Source(SPECS["segfly"], gates, role="train")

    def entries(self, source, count=10, offset=0):
        return [FrameIndex(frame=Frame(name=f"{i + offset:06d}",
                                       image=Path("i"), mask=Path("m")),
                           instances=(), size=(64, 64), rejects={}, source=source)
                for i in range(count)]

    def test_a_capped_source_stays_capped_through_the_file(self):
        full = self.entries(self.kust, 10) + self.entries(self.segfly, 10)
        chosen = {"train": full[:3] + full[10:12], "val": [full[3]],
                  "test": [full[4]]}
        save_splits(self.path, chosen)
        # The index the CLI builds is the *uncapped* one, as it would be.
        back = apply_splits(full, self.path)
        self.assertEqual(len(back["train"]), 5)
        self.assertEqual([e.frame.name for e in back["val"]], ["000003"])
        self.assertEqual(sum(len(v) for v in back.values()), 7,
                         "the five frames no split names must be dropped")

    def test_a_name_two_sources_share_does_not_cross_over(self):
        both = self.entries(self.kust, 4) + self.entries(self.segfly, 4)
        save_splits(self.path, {"train": both[:4], "val": [], "test": both[4:]})
        back = apply_splits(both, self.path)
        self.assertEqual({e.source.spec.name for e in back["train"]}, {"kust4k"})
        self.assertEqual({e.source.spec.name for e in back["test"]}, {"segfly"})

    def sized(self, source, sides, count=6):
        name = list(source.spec.classes)[0]
        return [FrameIndex(
            frame=Frame(name=f"{i:06d}", image=Path("i"), mask=Path("m")),
            instances=tuple(
                Instance(label=k, class_id=source.spec.classes[name],
                         box=(0.0, 0.0, float(s), float(s)), area=int(s * s))
                for k, s in enumerate(sides)),
            size=(2048, 2048), rejects={}, source=source) for i in range(count)]

    def test_a_size_band_survives_the_trip_to_the_cli(self):
        """The half a frame list cannot keep, for the second reason.

        `size_bands` drops per instance like `rebalance` does, so a file that
        recorded only the surviving frames would bring every large one back --
        the run would train on exactly what the notebook said it had excluded.
        """
        full = self.sized(self.kust, [10, 40, 200])
        bands = {"kust4k": (0.0, 96.0)}
        kept, _ = size_bands(full, bands)
        save_splits(self.path, {"train": kept, "val": [], "test": []},
                    bands=bands)
        back = apply_splits(full, self.path)
        sides = [max(i.width, i.height)
                 for e in back["train"] for i in e.instances]
        self.assertEqual(len(sides), 12, "two of three per frame survive")
        self.assertTrue(all(s <= 96 for s in sides), sides)

    def test_a_band_and_a_weight_both_travel(self):
        full = self.sized(self.kust, [10, 40, 200])
        save_splits(self.path, {"train": full, "val": [], "test": []},
                    weights={"kust4k": 0.5}, seed=0, bands={"kust4k": (0.0, 96.0)})
        body = json.loads(self.path.read_text())
        self.assertIn("rebalance", body)
        self.assertIn("size_bands", body)
        back = apply_splits(full, self.path)
        sides = [max(i.width, i.height)
                 for e in back["train"] for i in e.instances]
        self.assertTrue(all(s <= 96 for s in sides), sides)
        self.assertLess(len(sides), 12, "the weight thinned as well")

    def test_the_allowlist_narrows_train_and_leaves_the_grade_whole(self):
        """The grade has to stay whole or the run cannot see what it cost.

        Narrowing the val split with the train split makes the loss fall
        because the hard sources left, and the run reports an improvement it
        did not make.
        """
        source = Source(SPECS["kust4k"], InstanceGates(), role="train")
        def frame(key):
            return FrameIndex(
                frame=Frame(name=key, image=Path("i"), mask=Path("m")),
                instances=tuple(
                    Instance(label=i, class_id=source.spec.classes[n],
                             box=(0.0, 0.0, 8.0, 8.0), area=64)
                    for i, n in enumerate(["car", "truck"])),
                size=(64, 64), rejects={}, source=source)
        full = [frame(f"{i:06d}") for i in range(4)]
        save_splits(self.path, {"train": full[:2], "val": full[2:], "test": []},
                    allow=["kust4k:car"])
        back = apply_splits(full, self.path)
        self.assertEqual([len(e.instances) for e in back["train"]], [1, 1])
        self.assertEqual([len(e.instances) for e in back["val"]], [2, 2])

    def test_a_file_with_neither_keeps_the_flat_shape(self):
        # Every file written before either could travel is this shape, and it
        # still has to read.
        full = self.entries(self.kust, 4)
        save_splits(self.path, {"train": full, "val": [], "test": []})
        self.assertEqual(sorted(json.loads(self.path.read_text())),
                         ["test", "train", "val"])

    def test_an_index_missing_a_named_frame_refuses(self):
        full = self.entries(self.kust, 6)
        save_splits(self.path, {"train": full[:4], "val": [], "test": full[4:]})
        with self.assertRaises(ValueError) as caught:
            apply_splits(full[:5], self.path)
        self.assertIn("resolved 5", str(caught.exception))

    def test_entries_keep_the_source_the_index_built(self):
        full = self.entries(self.kust, 4)
        save_splits(self.path, {"train": full, "val": [], "test": []})
        back = apply_splits(full, self.path)
        self.assertIs(back["train"][0].source, self.kust)


class TestBothModalitiesOfAFlightTravelTogether(unittest.TestCase):
    """The property `11_encoder_rgb_mixed.ipynb`'s no-leak argument rests on.

    That notebook lists VTUAV twice -- once per modality, both `role=all` --
    and scores the RGB entries. That is only sound if `bike_009`'s thermal and
    RGB frames land on the *same side* of the split: the group key is the
    sequence name and both entries share `Source.name`, so they are permuted
    as one pool. If either half of that ever changes, RGB frames of a
    held-out thermal flight silently enter training, and the thermal score
    starts measuring memory across modalities.
    """

    def entries(self, source, sequences=8, per=5):
        return [FrameIndex(frame=Frame(name=f"seq_{s:02d}/{i:06d}",
                                       image=Path("i"), mask=Path("m")),
                           instances=(), size=(64, 64), rejects={}, source=source)
                for s in range(sequences) for i in range(per)]

    def test_the_same_flights_are_held_out_in_both_modalities(self):
        gates = InstanceGates()
        thermal = Source(SPECS["vtuav_vis"], gates, mode="labels", role="all",
                         gray=True)
        rgb = Source(SPECS["vtuav_vis"], gates, mode="labels", role="all",
                     gray=False)
        parts = split_index(self.entries(thermal) + self.entries(rgb), seed=0)

        for name, part in parts.items():
            of = lambda gray: {e.frame.name.split("/")[0] for e in part
                               if e.source.gray is gray}
            self.assertEqual(of(True), of(False), name)

    def test_adding_the_rgb_entries_leaves_the_thermal_split_alone(self):
        # The number 11's guard is read against: its thermal test windows must
        # be 07's thermal test windows, or "any drop is the price of RGB"
        # stops being true.
        gates = InstanceGates()
        thermal = Source(SPECS["vtuav_vis"], gates, mode="labels", role="all",
                         gray=True)
        rgb = Source(SPECS["vtuav_vis"], gates, mode="labels", role="all",
                     gray=False)
        alone = split_index(self.entries(thermal), seed=0)
        mixed = split_index(self.entries(thermal) + self.entries(rgb), seed=0)

        for name in ("train", "val", "test"):
            keep = lambda part: sorted(e.frame.name for e in part
                                       if e.source.gray)
            self.assertEqual(keep(alone[name]), keep(mixed[name]), name)


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
