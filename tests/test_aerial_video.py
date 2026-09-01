from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.training.aerial_video import (
    birdsai_sequences,
    flight_name,
    pool_sequence_stores,
    source_name,
    split_flights,
    vtuav_sequences,
    vtuav_vis_sequences,
    weighted_clip_sample,
)
from src.training.antiuav import Sequence, SequenceLabels
from src.training.labels import save_masks


def image(path: Path, shape=(32, 48)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # The adapters tested here index paths and annotations; pixel decoding is
    # exercised by the existing clip-loader tests. Keeping this fixture as an
    # opaque file makes the dataset contract testable without OpenCV installed.
    path.write_bytes(b"fixture")


def fake_sequence(name: str, frames: int) -> Sequence:
    """A sequence with paths that are never opened -- split tests read names."""
    return Sequence(
        name=name, split="unsplit",
        frames=tuple(Path(f"/nowhere/{name}/{i}.jpg") for i in range(frames)),
        labels=SequenceLabels(exist=np.ones(frames, bool),
                              boxes=np.zeros((frames, 4), np.float32)))


class AerialVideoTest(unittest.TestCase):
    def test_vtuav_strided_extraction_aligns_rows_and_keeps_absence(self):
        with tempfile.TemporaryDirectory() as tmp:
            seq = Path(tmp) / "car_001"
            image(seq / "ir/0.jpg")
            image(seq / "ir/10.jpg")
            image(seq / "ir/20.jpg")
            (seq / "ir.txt").write_text("1,2,10,8\n0,0,0,0\n3,4,9,7\n")

            loaded, = vtuav_sequences(tmp)
            self.assertEqual(loaded.name, "vtuav__car_001")
            self.assertEqual([p.stem for p in loaded.frames], ["0", "10", "20"])
            np.testing.assert_array_equal(loaded.labels.exist, [True, False, True])
            np.testing.assert_allclose(loaded.labels.boxes[0], [1, 2, 11, 10])
            self.assertTrue(np.isnan(loaded.labels.boxes[1]).all())

    def birdsai_fixture(self, root: Path) -> None:
        base = root / "TrainReal"
        for number in range(8):
            image(base / "images/flight_a" / f"camera_{number:010d}.jpg")
        rows = [
            f"{number},1,1,2,8,6,1,0,0,{1 if number == 3 else 0}"
            for number in range(8)
        ]
        rows += [f"{number},2,4,5,7,5,0,3,1,0" for number in range(8)]
        annotations = base / "annotations"
        annotations.mkdir(parents=True)
        (annotations / "flight_a.csv").write_text("\n".join(rows) + "\n")

    def test_birdsai_noise_splits_track_instead_of_becoming_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.birdsai_fixture(Path(tmp))
            sequences = birdsai_sequences(tmp, min_run=2)
            track1 = [s for s in sequences if "track1" in s.name]
            self.assertEqual([len(s) for s in track1], [3, 4])
            self.assertTrue(all(s.labels.exist.all() for s in track1))
            track2, = [s for s in sequences if "track2" in s.name]
            self.assertEqual(len(track2), 8)  # occlusion is retained by default

    def test_all_tracks_from_one_flight_share_a_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for flight in ("a", "b", "c", "d", "e"):
                for track in (1, 2):
                    seq = root / f"{flight}_{track}"
                    image(seq / "ir/0.jpg")
                    image(seq / "ir/10.jpg")
                    (seq / "ir.txt").write_text("1,2,3,4\n1,2,3,4\n")
            loaded = []
            for flight in ("a", "b", "c", "d", "e"):
                for track in (1, 2):
                    loaded.extend(vtuav_sequences(
                        root / f"{flight}_{track}", prefix=f"birdsai__{flight}"))
            parts = split_flights(loaded, val_fraction=0.2, test_fraction=0.2)
            where = {}
            for part, sequences in parts.items():
                for sequence in sequences:
                    where.setdefault(flight_name(sequence), set()).add(part)
            self.assertTrue(all(len(value) == 1 for value in where.values()))

    def test_weighted_sample_obeys_source_counts(self):
        class Clip:
            def __init__(self, name):
                self.sequence = type("SequenceStub", (), {"name": name})()

        clips = [Clip("vtuav__a"), Clip("birdsai__b")]
        sampled = weighted_clip_sample(
            clips, {"vtuav": 0.75, "birdsai": 0.25}, total=20, seed=4)
        counts = {name: 0 for name in ("vtuav", "birdsai")}
        for clip in sampled:
            counts[source_name(clip.sequence.name)] += 1
        self.assertEqual(counts, {"vtuav": 15, "birdsai": 5})

    def test_vtuav_vis_uses_mask_stems_and_derives_boxes(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            seq = Path(tmp) / "car_001"
            for stem, box in (("000000", (2, 3, 8, 9)),
                              ("000030", (5, 4, 11, 10))):
                frame = np.full((16, 20), 80, np.uint8)
                mask = np.zeros((16, 20), np.uint8)
                x0, y0, x1, y1 = box
                mask[y0:y1, x0:x1] = 255
                (seq / "ir").mkdir(parents=True, exist_ok=True)
                (seq / "mask/ir").mkdir(parents=True, exist_ok=True)
                Image.fromarray(frame).save(seq / "ir" / f"{stem}.jpg")
                Image.fromarray(mask).save(seq / "mask/ir" / f"{stem}.png")
            # Full-rate rows would be unsafe to zip against the sparse files;
            # the adapter must ignore them and derive boxes from masks.
            (seq / "ir.txt").write_text("0,0,1,1\n" * 31)

            sequences, stores = vtuav_vis_sequences(tmp)
            loaded, = sequences
            self.assertEqual(loaded.name, "vtuav_vis__car_001")
            np.testing.assert_allclose(
                loaded.labels.boxes, [[2, 3, 8, 9], [5, 4, 11, 10]])
            self.assertEqual(list(stores[loaded.name]), [0, 1])
            self.assertEqual(int(stores[loaded.name][1].sum()), 36)

    def test_a_given_hold_out_replaces_the_sampled_test_split(self):
        """A dataset that ships its own held-out split has already decided.

        Sampling on top of it would grade on a mixture of the authors' choice
        and a hash, and which half a number came from would not be recoverable
        from the number.
        """
        train = [fake_sequence(f"vtuav_vis__train_001_car_{i}", 8)
                 for i in range(6)]
        held = [fake_sequence("vtuav_vis__test_001_train_003", 8)]
        splits = split_flights(train, seed=0, val_fraction=0.2,
                               test_fraction=0.0, hold_out=held)
        self.assertEqual([row.name for row in splits["test"]],
                         ["vtuav_vis__test_001_train_003"])
        self.assertTrue(splits["train"] and splits["val"])
        self.assertTrue(all(row.name.startswith("vtuav_vis__train_001")
                            for row in splits["train"] + splits["val"]))

    def test_without_a_hold_out_nothing_about_the_split_changes(self):
        rows = [fake_sequence(f"vtuav_vis__train_001_car_{i}", 8)
                for i in range(10)]
        before = split_flights(rows, seed=3)
        after = split_flights(rows, seed=3, hold_out=None)
        self.assertEqual({k: [r.name for r in v] for k, v in before.items()},
                         {k: [r.name for r in v] for k, v in after.items()})

    def test_two_archives_sharing_a_sequence_name_stay_apart(self):
        """The collision this project would otherwise walk into.

        VTUAV-VIS names a sequence by target kind and a counter that restarts
        per archive, so `test_001.zip` contains a sequence called `train_003`
        while `train_003.zip` is a different archive. Unpacked into folders of
        their own (`Part.into`), the part goes into the sequence name -- and it
        has to, because `STORES` is keyed by that name and one would otherwise
        replace the other with nothing said.
        """
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for part in ("train_003", "test_001"):
                seq = root / part / "train_003"
                (seq / "rgb").mkdir(parents=True)
                (seq / "mask/rgb").mkdir(parents=True)
                for stem in ("000000", "000030"):
                    mask = np.zeros((16, 20), np.uint8)
                    mask[3:9, 2:8] = 255
                    Image.fromarray(np.full((16, 20), 80, np.uint8)).save(
                        seq / "rgb" / f"{stem}.jpg")
                    Image.fromarray(mask).save(seq / "mask/rgb" / f"{stem}.png")

            sequences, stores = vtuav_vis_sequences(root, modality="rgb")
            names = sorted(row.name for row in sequences)
            self.assertEqual(names, ["vtuav_vis__test_001_train_003",
                                     "vtuav_vis__train_003_train_003"])
            self.assertEqual(len(stores), 2, "one store replaced the other")

    def test_a_flat_extraction_keeps_the_old_sequence_names(self):
        """Nothing already measured moves: the part only enters the name when
        the archives were unpacked into directories of their own."""
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seq = root / "car_001"
            (seq / "rgb").mkdir(parents=True)
            (seq / "mask/rgb").mkdir(parents=True)
            for stem in ("000000", "000030"):
                mask = np.zeros((16, 20), np.uint8)
                mask[3:9, 2:8] = 255
                Image.fromarray(np.full((16, 20), 80, np.uint8)).save(
                    seq / "rgb" / f"{stem}.jpg")
                Image.fromarray(mask).save(seq / "mask/rgb" / f"{stem}.png")

            sequences, _ = vtuav_vis_sequences(root, modality="rgb")
            self.assertEqual(sequences[0].name, "vtuav_vis__car_001")

    def test_teacher_pool_masks_join_by_sequence_and_frame_stem(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seq = root / "frames/car_001"
            image(seq / "ir/000000.jpg")
            image(seq / "ir/000010.jpg")
            (seq / "ir.txt").write_text("1,2,6,5\n2,3,6,5\n")
            loaded, = vtuav_sequences(seq)

            target = root / "pool/vtuav_lt_thermal/car_001/000010"
            target.mkdir(parents=True)
            record = {
                "key": "car_001/000010", "dataset": "vtuav_lt_thermal",
                "image": "/content/data/VTUAV_lt_ir/car_001/ir/000010.jpg",
                "instances": [{"i": 0, "verdict": None, "box_iou": 0.85}],
            }
            (target / "record.json").write_text(json.dumps(record))
            mask = np.zeros((32, 48), bool)
            mask[3:8, 2:8] = True
            save_masks(target / "pseudo_masks.npz", mask.shape, {0: mask})

            stores = pool_sequence_stores(
                root / "pool", [loaded], {"vtuav_lt_thermal"})
            self.assertEqual(list(stores[loaded.name]), [1])
            np.testing.assert_array_equal(stores[loaded.name][1], mask)
            strict = pool_sequence_stores(
                root / "pool", [loaded], {"vtuav_lt_thermal"}, min_box_iou=0.9)
            self.assertEqual(strict, {})


if __name__ == "__main__":
    unittest.main()
