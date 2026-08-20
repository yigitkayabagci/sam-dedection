"""Downloading the aerial sets: the parts registry, and where files land.

The network half is not tested -- these are gigabyte archives on third-party
hosts. What is tested is everything that decides whether a download is
*usable* once it arrives: that each archive is extracted into the folder its
spec globs, that the "only annotated frames" filter keeps an image with its
mask and drops the rest, and that asking for a part that does not exist fails
by name instead of quietly downloading the defaults.

The folder/glob agreement is the one worth having. Kust4K's three archives are
**flat** -- all three contain the same 4 024 filenames at the top level -- so
extracting them side by side would have them overwrite each other, and
extracting them into the wrong folders would pair a thermal image with a
thermal image. `into` is what prevents both, and it has to keep matching
`SPECS["kust4k"]`.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.aerial import SPECS  # noqa: E402
from tools.fetch_datasets import (  # noqa: E402
    KUST4K,
    STAGING,
    RECIPES,
    VTUAV_VIS,
    drive_download,
    extract,
    human,
    masked_members,
    staged,
)


class TestRegistry(unittest.TestCase):
    def test_every_archive_names_exactly_one_source(self):
        for recipe in RECIPES.values():
            for part in recipe.parts:
                self.assertNotEqual(
                    bool(part.url), bool(part.drive),
                    f"{recipe.name}/{part.name} needs a url or a drive id, not "
                    f"both and not neither")

    def test_a_hub_dataset_has_no_archives(self):
        segfly = RECIPES["segfly"]
        self.assertTrue(segfly.hub)
        self.assertEqual(segfly.parts, ())

    def test_the_defaults_are_the_cheap_useful_subset(self):
        # Kust4K's RGB half is 1.66 GB and only stage A reads it; VTUAV's other
        # seven archives are 14-17 GB each.
        self.assertEqual([p.name for p in KUST4K.chosen(None)], ["tir", "labels"])
        self.assertEqual([p.name for p in VTUAV_VIS.chosen(None)], ["train_001"])

    def test_asking_for_a_part_that_does_not_exist_says_so(self):
        with self.assertRaises(SystemExit) as caught:
            VTUAV_VIS.chosen(("train_009",))
        self.assertIn("train_009", str(caught.exception))

    def test_a_named_part_is_returned_in_the_order_asked_for(self):
        names = [p.name for p in VTUAV_VIS.chosen(("test_001", "train_001"))]
        self.assertEqual(names, ["test_001", "train_001"])

    def test_the_publisher_checksums_are_kept_for_the_direct_downloads(self):
        # figshare gives an md5 per file; a truncated 1 GB download is
        # otherwise indistinguishable from a complete one.
        for part in KUST4K.parts:
            self.assertTrue(part.md5, part.name)
            self.assertEqual(len(part.md5), 32)


class TestLayoutAgreesWithTheSpec(unittest.TestCase):
    def test_each_kust4k_archive_lands_where_the_spec_globs(self):
        spec = SPECS["kust4k"]
        wanted = {"tir": spec.thermal, "labels": spec.masks, "rgb": spec.rgb}
        for part in KUST4K.parts:
            self.assertIn(f"/{part.into}/", wanted[part.name],
                          f"{part.name} extracts into {part.into!r} but the "
                          f"spec globs {wanted[part.name]!r}")

    def test_the_three_archives_do_not_share_a_folder(self):
        folders = [p.into for p in KUST4K.parts]
        self.assertEqual(len(folders), len(set(folders)))
        self.assertTrue(all(folders), "a flat archive needs a folder of its own")

    def test_vtuav_keeps_the_archives_own_sequence_layout(self):
        # Its zips already contain `bike_009/rgb/000000.jpg`, so imposing a
        # folder would bury the sequence names one level deeper than the globs.
        for part in VTUAV_VIS.parts:
            self.assertEqual(part.into, "")


class TestMaskedMembers(unittest.TestCase):
    """`--frames masked`: the annotated frames and their twins, nothing else."""

    def archive(self, names) -> zipfile.ZipFile:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name) / "a.zip"
        with zipfile.ZipFile(path, "w") as handle:
            for name in names:
                handle.writestr(name, b"x")
        return zipfile.ZipFile(path)

    def vtuav_like(self):
        names = []
        for sequence in ("bike_009", "car_083"):
            for index in range(0, 90, 30):
                names.append(f"{sequence}/mask/rgb/{index:06d}.png")
            for index in range(90):                      # every frame
                names.append(f"{sequence}/rgb/{index:06d}.jpg")
                names.append(f"{sequence}/ir/{index:06d}.jpg")
            names.append(f"{sequence}/rgb.txt")
        return names

    def test_an_annotated_frame_keeps_both_of_its_modalities(self):
        keep = set(masked_members(self.archive(self.vtuav_like())))
        for sequence in ("bike_009", "car_083"):
            for index in (0, 30, 60):
                self.assertIn(f"{sequence}/rgb/{index:06d}.jpg", keep)
                self.assertIn(f"{sequence}/ir/{index:06d}.jpg", keep)
                self.assertIn(f"{sequence}/mask/rgb/{index:06d}.png", keep)

    def test_frames_with_no_mask_are_dropped(self):
        keep = set(masked_members(self.archive(self.vtuav_like())))
        for index in (1, 15, 29, 31, 89):
            self.assertNotIn(f"bike_009/rgb/{index:06d}.jpg", keep)

    def test_a_frame_number_annotated_in_one_sequence_is_not_kept_in_another(self):
        # Both sequences number from zero, so a filename-only rule would keep
        # `car_083/rgb/000000.jpg` because `bike_009` annotates frame 0.
        names = [f"bike_009/mask/rgb/{i:06d}.png" for i in (0, 30)]
        names += [f"{s}/rgb/{i:06d}.jpg" for s in ("bike_009", "car_083")
                  for i in (0, 30)]
        keep = set(masked_members(self.archive(names)))
        self.assertIn("bike_009/rgb/000000.jpg", keep)
        self.assertNotIn("car_083/rgb/000000.jpg", keep)

    def test_it_saves_most_of_the_disk(self):
        names = self.vtuav_like()
        self.assertLess(len(masked_members(self.archive(names))), len(names) / 4)


class TestExtract(unittest.TestCase):
    def test_a_flat_archive_lands_inside_its_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "tir.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("00001D.png", b"x")
                handle.writestr("00002D.png", b"x")

            written = extract(archive, root / "out", into="tir", quiet=True)
            self.assertEqual(written, 2)
            self.assertTrue((root / "out" / "tir" / "00001D.png").exists())
            # And the spec's glob finds it from the dataset root.
            self.assertEqual(
                len(list((root / "out").glob(SPECS["kust4k"].thermal))), 2)


class TestStagedCopy(unittest.TestCase):
    """The escape hatch from Drive's download quota.

    "Too many users have viewed or downloaded this file recently" is about who
    is asking, not about the file: Colab shares its egress addresses with a
    great many people, and the same archive served fine from elsewhere while a
    Colab session was being refused it. Copying the file into your own Drive
    turns a widely-shared file into your own, and reading your own has no
    shared-file quota -- so the fetcher looks for a staged copy before it
    touches the network.
    """

    def test_a_staged_archive_is_found_by_part_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "train_001.zip").write_bytes(b"x" * (2 << 20))
            self.assertIsNotNone(staged("train_001", [tmp]))
            self.assertIsNone(staged("train_002", [tmp]))

    def test_a_stub_too_small_to_be_the_archive_is_ignored(self):
        # A half-finished copy, or Drive's HTML refusal saved under the right
        # name, would otherwise be extracted as if it were the dataset.
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "train_001.zip").write_bytes(b"<html>quota</html>")
            self.assertIsNone(staged("train_001", [tmp]))

    def test_the_first_folder_that_has_it_wins(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            (Path(b) / "train_001.zip").write_bytes(b"x" * (2 << 20))
            self.assertEqual(staged("train_001", [a, b]), Path(b) / "train_001.zip")

    def test_a_missing_folder_is_not_an_error(self):
        self.assertIsNone(staged("train_001", ["/nowhere/at/all"]))

    def test_the_default_search_starts_in_the_colab_drive_mount(self):
        self.assertTrue(STAGING[0].startswith("/content/drive/"))


class TestQuotaMessage(unittest.TestCase):
    def test_giving_up_explains_all_three_ways_out(self):
        # The failure the user actually hits, and the one place they will read
        # about it. An id that cannot resolve exercises the same path.
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError) as caught:
                drive_download("0" * 25, Path(tmp) / "x.zip", quiet=True,
                               attempts=1)
        text = str(caught.exception)
        self.assertIn("Make a copy", text)          # 1: your own Drive
        self.assertIn("few hours", text)            # 2: wait
        self.assertIn("PRETRAIN", text)             # 3: what dropping it costs
        self.assertIn("gdown", text)                # both routes reported
        self.assertIn("direct", text)

    def test_both_routes_are_tried_before_giving_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError) as caught:
                drive_download("0" * 25, Path(tmp) / "x.zip", quiet=True,
                               attempts=2)
        text = str(caught.exception)
        for attempt in ("gdown #1", "direct #1", "gdown #2", "direct #2"):
            self.assertIn(attempt, text)


class TestHuman(unittest.TestCase):
    def test_sizes_read_the_way_a_person_would_say_them(self):
        self.assertEqual(human(512), "512 B")
        self.assertEqual(human(1536), "1.5 KB")
        self.assertEqual(human(9_080_000_000), "8.5 GB")


if __name__ == "__main__":
    unittest.main()
