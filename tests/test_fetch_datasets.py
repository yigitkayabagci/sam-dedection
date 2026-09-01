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

import io
import contextlib
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.aerial import SPECS  # noqa: E402
from tools import fetch_datasets  # noqa: E402
from tools.fetch_datasets import (  # noqa: E402
    AIRESQ,
    BIRDSAI,
    KUST4K,
    RGBTDRONEPERSON,
    STAGING,
    RECIPES,
    VTUAVDET,
    VTUAV_VIS,
    PartsFailed,
    drive_download,
    extract,
    fetch,
    fetch_extra,
    human,
    main,
    masked_members,
    staged,
)


class TestRegistry(unittest.TestCase):
    def test_every_archive_names_exactly_one_source(self):
        for recipe in RECIPES.values():
            if recipe.snapshot:
                # A snapshot recipe's parts are allow_patterns on one Hub
                # repo; the repo id is the source and the parts carry none.
                continue
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

    def test_the_refusal_lists_the_known_parts_and_what_to_do(self):
        """A wrong `--parts` name is refused before anything downloads, so the
        message is the only thing the caller gets. It has to carry three
        facts: which name is wrong, which names are right, and where the list
        it came from lives -- notebook 31 builds it from VTUAV_VIS_PARTS, and
        a reader who is not told that goes looking for a missing Drive id
        instead of a typo. All three ids under `train_*` were checked live in
        2026-09 and resolve, so the id is never the thing to suspect first.
        """
        with self.assertRaises(SystemExit) as caught:
            VTUAV_VIS.chosen(("train_001", "train_00b"))
        text = str(caught.exception)
        self.assertIn("train_00b", text)
        self.assertIn("train_001", text)            # the known names
        self.assertIn("test_005", text)
        self.assertIn("VTUAV_VIS_PARTS", text)     # where the list lives
        self.assertIn("Nothing was downloaded", text)

    def test_the_three_training_parts_all_name_a_drive_id(self):
        """The hypothesis this test exists to close.

        When `VTUAV_VIS_PARTS` went from one part to three and the fetch
        started exiting 1, the obvious explanation was that train_002 and
        train_003 were half-registered -- no id, or an id copied wrong. They
        are not: each of the three carries a 33-character Drive id, and all
        three were resolved against Drive in 2026-09 (the interstitial names
        `train_001.zip` 8.5G, `train_002.zip` 15G, `train_003.zip` 17G). A
        part that loses its id would make the fetch fail for a reason that has
        nothing to do with the network, so the registry is pinned here.
        """
        parts = {p.name: p for p in VTUAV_VIS.parts}
        for name in ("train_001", "train_002", "train_003"):
            self.assertIn(name, parts)
            self.assertTrue(parts[name].drive, name)
            self.assertEqual(len(parts[name].drive), 33, name)
            self.assertIsNone(parts[name].url, name)

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

    def test_every_vtuav_vis_part_gets_a_folder_of_its_own(self):
        """This used to pin `into == ""`, on the reading that a folder would
        bury the sequence names below the globs. It does not: both readers use
        `**`/`rglob`, so depth is free -- and flat extraction is not, because
        VTUAV-VIS names a sequence by target kind and a counter that restarts
        per archive. `test_001.zip` holds a sequence called `train_003` while
        `train_003.zip` is a different archive, and unpacked into one directory
        their frames interleave."""
        folders = [part.into for part in VTUAV_VIS.parts]
        self.assertEqual(folders, [part.name for part in VTUAV_VIS.parts])
        self.assertEqual(len(folders), len(set(folders)))


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

    def test_one_bad_member_does_not_cost_the_rest_of_the_archive(self):
        # A Drive copy of DroneVehicle's train.zip failed its CRC on one JPEG
        # and `extractall` abandoned the other 66 974 members, which reached
        # the pools as `no_image` on three quarters of their frames.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "train.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                for name in ("a.jpg", "bad.jpg", "c.jpg"):
                    handle.writestr(f"train/{name}", b"x" * 32)
            raw = bytearray(archive.read_bytes())
            spot = raw.index(b"x" * 32, raw.index(b"bad.jpg"))
            raw[spot] = ord("y")             # the stored bytes, not the CRC
            archive.write_bytes(bytes(raw))

            written = extract(archive, root / "out", quiet=True)
            self.assertEqual(written, 2)
            for name in ("a.jpg", "c.jpg"):
                self.assertTrue((root / "out" / "train" / name).is_file())

    def test_an_archive_that_gives_nothing_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "train.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("train/only.jpg", b"x" * 32)
            raw = bytearray(archive.read_bytes())
            raw[raw.index(b"x" * 32)] = ord("y")
            archive.write_bytes(bytes(raw))

            with self.assertRaises(RuntimeError):
                extract(archive, root / "out", quiet=True)


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


class TestBoxLabelledSets(unittest.TestCase):
    """The four sets added for the mask pool: box labels, no dense masks.

    Every number here was read off the live host in 2026-08 -- the Drive
    folders' central directories over a range request, LILA's blobs, Zenodo's
    API -- so a test that drifts from the recipe means the recipe drifted from
    what was measured, not that the numbers were rounded.
    """

    def test_each_archive_keeps_its_own_top_level_layout(self):
        # All four zips already carry their split at the top (`train/`, `val/`,
        # `TrainReal/`, `Benchmark/`), so an `into` would bury it a level down
        # -- the same reason VTUAV's parts have none.
        for recipe in (RGBTDRONEPERSON, VTUAVDET, BIRDSAI, AIRESQ):
            for part in recipe.parts:
                self.assertEqual(part.into, "", f"{recipe.name}/{part.name}")

    def test_the_coco_jsons_are_sidecars_not_archive_members(self):
        # They sit beside the zip as separate Drive files; fetching them as
        # parts would send them through `extract`.
        for recipe, count in ((RGBTDRONEPERSON, 4), (VTUAVDET, 2)):
            self.assertEqual(len(recipe.extras), count, recipe.name)
            self.assertEqual(len(recipe.parts), 1, recipe.name)
            for name, source in recipe.extras:
                self.assertTrue(name.endswith(".json"), name)
                self.assertTrue(source.startswith("drive:"), source)

    def test_rgbtdroneperson_carries_both_modalities_annotations(self):
        # The visible json is the whole point of keeping it: it is what the
        # 11.7 px modality disagreement was measured from.
        names = {name for name, _ in RGBTDRONEPERSON.extras}
        self.assertIn("sub_train_thermal.json", names)
        self.assertIn("sub_train_visible.json", names)

    def test_the_42_gb_synthetic_half_of_birdsai_is_off_by_default(self):
        chosen = [p.name for p in BIRDSAI.chosen(None)]
        self.assertEqual(chosen, ["train_real", "test_real"])
        simulation = {p.name: p for p in BIRDSAI.parts}["train_simulation"]
        self.assertGreater(simulation.size, 39 << 30)   # 39.3 GiB

    def test_zenodo_gives_a_checksum_so_it_is_kept(self):
        # Same rule as Kust4K: where the publisher hands out an md5, a
        # truncated download must not pass for a complete one.
        part, = AIRESQ.parts
        self.assertEqual(part.md5, "6a7e8920ee62c6c8234e0f08e631410a")
        self.assertEqual(len(part.md5), 32)

    def test_every_new_part_says_how_big_it_is(self):
        for recipe in (RGBTDRONEPERSON, VTUAVDET, BIRDSAI, AIRESQ):
            for part in recipe.parts:
                self.assertGreater(part.size, 1 << 20,
                                   f"{recipe.name}/{part.name}")

    def test_the_notes_do_not_promise_masks(self):
        # These sets are prompts for the teacher, not ground truth for J&F.
        for recipe in (RGBTDRONEPERSON, VTUAVDET, BIRDSAI, AIRESQ):
            self.assertIn("box", recipe.note.lower(), recipe.name)


class TestOneRefusedPartDoesNotCostTheOthers(unittest.TestCase):
    """Fetching three archives when the second one is refused.

    Notebook 31 runs the fetch through `subprocess.run([...], check=True)`.
    Before this, one archive failing let the exception out of `main`, so the
    command died with a traceback on stderr the moment the *second* of three
    parts was refused: the third was never attempted, the report of what had
    landed never ran, and the caller was left with a bare
    `CalledProcessError` naming no part and no reason. The parts that had
    already extracted were still on disk, but nothing said so.

    What is pinned here is the shape of the recovery: every requested part is
    attempted, whatever extracted stays, the summary names each part that
    failed and the reason, and the exit code is 1 only after all of that has
    been printed -- on **stdout**, so it lands in the notebook cell rather
    than in whichever stream a Jupyter kernel happens to forward.

    No network: `drive_download` is replaced with one that writes a small
    VTUAV-shaped zip, or raises for the parts a test wants refused.
    """

    REFUSED = "Drive would not serve this file (test)"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dest = Path(self.tmp.name) / "VTUAV_VIS"
        self.original = fetch_datasets.drive_download
        self.addCleanup(setattr, fetch_datasets, "drive_download",
                        self.original)

    def refuse(self, *names):
        """Every part but `names`; those raise the way a quota refusal does."""
        sequences = {"train_001": "bike_009", "train_002": "car_010",
                     "train_003": "pedestrian_2"}

        def fake_drive(file_id, path, **kwargs):
            part = Path(path).stem
            if part in names:
                raise RuntimeError(self.REFUSED)
            path.parent.mkdir(parents=True, exist_ok=True)
            sequence = sequences.get(part, part)
            with zipfile.ZipFile(path, "w") as handle:
                for index in range(0, 60, 30):
                    handle.writestr(
                        f"{sequence}/mask/rgb/{index:06d}.png", b"x")
                for index in range(60):
                    handle.writestr(f"{sequence}/rgb/{index:06d}.jpg", b"x")
                    handle.writestr(f"{sequence}/ir/{index:06d}.jpg", b"x")
            return path

        fetch_datasets.drive_download = fake_drive

    def fetch_three(self):
        return fetch("vtuav_vis", self.dest,
                     ("train_001", "train_002", "train_003"),
                     frames="masked", quiet=True, staging=())

    def test_the_part_after_the_refused_one_is_still_fetched(self):
        self.refuse("train_002")
        with self.assertRaises(PartsFailed):
            self.fetch_three()
        self.assertTrue(
            (self.dest / "train_003" / "pedestrian_2" / "rgb").is_dir(),
            "train_003 was never attempted after train_002 failed")

    def test_the_parts_that_landed_before_it_are_kept(self):
        self.refuse("train_002")
        with self.assertRaises(PartsFailed):
            self.fetch_three()
        self.assertTrue(
            (self.dest / "train_001" / "bike_009" / "mask" / "rgb").is_dir())
        self.assertFalse((self.dest / "car_010").exists())

    def test_the_failure_names_the_part_and_carries_both_lists(self):
        self.refuse("train_002")
        with self.assertRaises(PartsFailed) as caught:
            self.fetch_three()
        self.assertEqual(caught.exception.done, ["train_001", "train_003"])
        self.assertEqual([name for name, _ in caught.exception.failed],
                         ["train_002"])
        text = caught.exception.summary()
        self.assertIn("train_002", text)
        self.assertIn(self.REFUSED, text)
        self.assertIn("train_001", text)                # what was kept
        self.assertIn("VTUAV_VIS_PARTS", text)          # how to drop it

    def test_every_refused_part_is_named_not_just_the_first(self):
        self.refuse("train_002", "train_003")
        with self.assertRaises(PartsFailed) as caught:
            self.fetch_three()
        text = caught.exception.summary()
        self.assertIn("train_002", text)
        self.assertIn("train_003", text)
        self.assertIn("--parts train_002 train_003", text)   # a runnable line

    def test_a_whole_set_that_landed_raises_nothing(self):
        self.refuse()
        self.assertEqual(self.fetch_three(), self.dest)

    def test_main_says_which_part_failed_on_stdout_then_exits_one(self):
        # The notebook reads the child's streams, not its exception. If the
        # only account of the failure were a traceback, `check=True` would
        # surface a CalledProcessError with no part name in it -- which is
        # exactly the report this fix exists to prevent.
        self.refuse("train_002")
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            code = main(["vtuav_vis", "--dest", str(self.dest), "--parts",
                         "train_001", "train_002", "train_003",
                         "--frames", "masked", "--quiet"])
        text = printed.getvalue()
        self.assertEqual(code, 1)
        self.assertIn("train_002", text)
        self.assertIn(self.REFUSED, text)
        self.assertIn("exiting 1", text)

    def test_main_still_reports_what_landed_before_it_gives_up(self):
        self.refuse("train_002")
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            main(["vtuav_vis", "--dest", str(self.dest), "--parts",
                  "train_001", "train_002", "train_003",
                  "--frames", "masked", "--quiet"])
        self.assertIn("labelled rgb frames", printed.getvalue())


class TestTheCliKnowsEveryRecipe(unittest.TestCase):
    def test_every_recipe_has_a_default_folder(self):
        # `--dest` is optional, and without it main() looks the dataset up in a
        # hand-written table. A recipe missing from it fails with a KeyError
        # halfway through `fetch_datasets.py all`, after the earlier sets have
        # already been downloaded.
        source = (ROOT / "tools" / "fetch_datasets.py").read_text()
        table = source.split("folders = {", 1)[1].split("}", 1)[0]
        for name in RECIPES:
            self.assertIn(f'"{name}":', table, f"{name} has no default folder")


class TestFetchExtra(unittest.TestCase):
    """Three hosts behind one field, because the sets do not share one."""

    def dispatch(self, source):
        calls = {}
        original = (fetch_datasets.drive_download, fetch_datasets.http_download)

        def fake_drive(file_id, path, **kwargs):
            calls["drive"] = file_id
            return path

        def fake_http(url, path, **kwargs):
            calls["http"] = url
            return path

        fetch_datasets.drive_download = fake_drive
        fetch_datasets.http_download = fake_http
        try:
            fetch_extra(source, Path("/tmp/never-written.json"))
        finally:
            fetch_datasets.drive_download, fetch_datasets.http_download = original
        return calls

    def test_a_drive_id_goes_through_the_virus_scan_form(self):
        self.assertEqual(self.dispatch("drive:abc123"), {"drive": "abc123"})

    def test_a_full_url_is_fetched_as_it_stands(self):
        calls = self.dispatch("https://example.org/train.json")
        self.assertEqual(calls, {"http": "https://example.org/train.json"})

    def test_a_bare_id_is_still_read_as_figshare(self):
        # Kust4K's palette script predates the other two forms.
        calls = self.dispatch("57140510")
        self.assertEqual(
            calls, {"http": "https://ndownloader.figshare.com/files/57140510"})


class TestHuman(unittest.TestCase):
    def test_sizes_read_the_way_a_person_would_say_them(self):
        self.assertEqual(human(512), "512 B")
        self.assertEqual(human(1536), "1.5 KB")
        self.assertEqual(human(9_080_000_000), "8.5 GB")


if __name__ == "__main__":
    unittest.main()


class TestStagedCopies(unittest.TestCase):
    """Drive renames a file when you copy it, and the copy is the escape hatch.

    "Make a copy" is what `drive_download` tells a user to do when the quota
    refuses a shared archive. Drive hands back `test_001.zip adlı dosyanın
    kopyası`, `Copy of test_001.zip` or `test_001 (1).zip` -- never the name
    the lookup was built from. Missing them meant the escape hatch did not
    exist for the person who took it: the run went back to the network and hit
    the same quota again.
    """

    def folder(self, *names):
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        root = Path(holder.name)
        for name in names:
            (root / name).write_bytes(b"\0" * (2 << 20))
        return root

    def test_the_plain_name_is_found(self):
        root = self.folder("test_001.zip")
        self.assertEqual(staged("test_001", [str(root)]).name, "test_001.zip")

    def test_drives_turkish_copy_is_found(self):
        root = self.folder("test_001.zip adlı dosyanın kopyası")
        found = staged("test_001", [str(root)])
        self.assertIsNotNone(found)
        self.assertTrue(found.name.startswith("test_001.zip"))

    def test_drives_english_copy_is_found(self):
        root = self.folder("Copy of test_001.zip")
        self.assertIsNotNone(staged("test_001", [str(root)]))

    def test_a_numbered_copy_is_found(self):
        root = self.folder("test_001 (1).zip")
        self.assertIsNotNone(staged("test_001", [str(root)]))

    def test_the_exact_name_wins_over_a_copy(self):
        """A deliberate copy must never shadow the real archive."""
        root = self.folder("test_001.zip", "Copy of test_001.zip")
        self.assertEqual(staged("test_001", [str(root)]).name, "test_001.zip")

    def test_a_longer_name_is_not_mistaken_for_this_one(self):
        """`test_0010.zip` starts with `test_001`; matching on the name alone
        rather than on name-plus-suffix would answer the wrong lookup."""
        root = self.folder("test_0010.zip")
        self.assertIsNone(staged("test_001", [str(root)]))

    def test_a_tiny_file_is_still_ignored(self):
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        root = Path(holder.name)
        (root / "test_001.zip adlı dosyanın kopyası").write_bytes(b"x")
        self.assertIsNone(staged("test_001", [str(root)]))

    def test_a_missing_folder_is_skipped_not_raised(self):
        self.assertIsNone(staged("test_001", ["/nowhere/at/all"]))


class TestVtuavVisParts(unittest.TestCase):
    def test_every_part_unpacks_into_a_directory_of_its_own(self):
        """The sequence counters restart per archive: `test_001.zip` holds a
        sequence called `train_003` and `train_003.zip` is a different archive.
        Extracted flat they would merge into one directory."""
        for part in VTUAV_VIS.parts:
            self.assertEqual(part.into, part.name, part.name)
