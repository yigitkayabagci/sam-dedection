"""Exporting a parquet dataset into the layout the specs glob.

`datasets` is not installed here and the download is tens of gigabytes, so the
network half is not tested. What is tested is everything that decides whether
the export is *usable*: that the three directories pair by stem, that scene and
altitude survive into the filename -- once rows become files there is nowhere
else for them to live -- and that the label check tells the two failure modes
apart, because "a few strays" and "a completely different palette" have
different causes and different fixes.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.aerial import SPECS  # noqa: E402
from tools import export_hf_dataset  # noqa: E402
from tools.export_hf_dataset import (  # noqa: E402
    COLUMNS,
    folders_for,
    stem,
    verify,
)

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:                       # pragma: no cover
    pa = pq = None


class TestStem(unittest.TestCase):
    def test_scene_and_altitude_reach_the_filename(self):
        # They are columns in the parquet and nothing else carries them once
        # the rows are files. SegFly spans three altitudes and that axis is an
        # experimental variable here, not decoration.
        self.assertEqual(stem({"scene": "scene_05", "altitude": "40m"}, 123),
                         "scene_05_40m_000123")

    def test_the_index_is_padded_so_a_lexical_sort_is_numeric(self):
        names = [stem({"scene": "s"}, i) for i in (2, 10, 100)]
        self.assertEqual(sorted(names), names)

    def test_missing_metadata_degrades_rather_than_crashing(self):
        self.assertEqual(stem({}, 7), "scene_000007")
        self.assertEqual(stem({"scene": "s3", "altitude": None}, 7), "s3_000007")

    def test_separators_that_would_make_a_subdirectory_are_removed(self):
        self.assertNotIn("/", stem({"scene": "a/b", "altitude": "40 m"}, 1))

    def test_the_three_columns_land_where_the_spec_globs(self):
        # The exporter and SPECS["segfly"] have to agree, and this is the join.
        spec = SPECS["segfly"]
        for column, folder in COLUMNS.items():
            self.assertIn(f"/{folder}/", {"images": spec.thermal, "labels": spec.masks,
                                          "rgb": spec.rgb}[folder],
                          f"{column} -> {folder}")


class TestVerify(unittest.TestCase):
    def palette(self) -> dict[int, int]:
        return {value: 10 for value in SPECS["segfly"].classes.values()}

    def test_a_clean_export_says_so(self):
        text = verify(self.palette(), "segfly")
        self.assertIn("every value is a class id", text)
        self.assertNotIn("!!", text)

    def test_a_few_strays_are_flagged_as_a_lossy_re_encode(self):
        # A segmentation map saved through a lossy codec comes back with values
        # *near* the ids rather than on them, and every count downstream is
        # then quietly wrong.
        values = self.palette() | {12: 3, 14: 900, 35: 2, 37: 1}
        text = verify(values, "segfly")
        self.assertIn("!!", text)
        self.assertIn("12", text)
        self.assertIn("lossily", text)

    def test_a_disjoint_set_is_flagged_too(self):
        text = verify({200: 5, 201: 5}, "segfly")
        self.assertIn("!!", text)
        self.assertIn("palette is wrong", text)

    def test_absent_palette_values_are_reported_without_alarm(self):
        values = self.palette()
        values.pop(SPECS["segfly"].classes["truck"])
        text = verify(values, "segfly")
        self.assertIn("absent from this", text)
        self.assertNotIn("!!", text)

    def test_leftover_ids_the_spec_ignores_are_not_an_alarm(self):
        # Measured on the real release: SegFly's own remapping (Rock into
        # Ground Obstacle, dynamic classes into Unlabeled) left stray pixels
        # with raw ids 5, 11, 12, 21. They are in `ignore`, so the check
        # names them without the "!!" that means a broken export.
        values = self.palette() | {5: 40, 11: 3, 12: 7, 21: 2}
        text = verify(values, "segfly")
        self.assertNotIn("!!", text)
        self.assertIn("leftover", text)
        self.assertIn("[5, 11, 12, 21]", text)

    def test_a_stray_outside_both_palette_and_ignore_still_alarms(self):
        values = self.palette() | {5: 40, 200: 3}
        text = verify(values, "segfly")
        self.assertIn("!!", text)
        self.assertIn("200", text)
        self.assertNotIn("5,", text.split("!!")[1])   # 5 is not in the alarm

    def test_with_no_spec_it_only_lists_what_it_saw(self):
        text = verify({0: 1, 13: 2}, None)
        self.assertIn("[0, 13]", text)
        self.assertNotIn("palette", text)


class TestSegflyPalette(unittest.TestCase):
    """The correction that mattered most, pinned so it cannot drift back."""

    def test_the_thing_classes_are_the_two_the_authors_list(self):
        spec = SPECS["segfly"]
        self.assertEqual(sorted(spec.thing_ids), [13, 36])
        self.assertEqual({spec.name_of(i) for i in spec.thing_ids},
                         {"vehicle", "truck"})

    def test_vegetation_and_grass_are_not_tracking_targets(self):
        # The bug this replaces: a guessed contiguous palette made ids 6-9
        # things, which are Grass, Vegetation, Tree and Ground Obstacle. That
        # would have trained the model to answer a prompt with a hedge.
        spec = SPECS["segfly"]
        for name in ("grass", "vegetation", "tree", "ground_obstacle"):
            self.assertNotIn(spec.classes[name], spec.thing_ids, name)

    def test_the_ids_are_the_published_ones_gaps_included(self):
        self.assertEqual(sorted(SPECS["segfly"].classes.values()),
                         [0, 1, 2, 3, 4, 6, 7, 8, 9, 13, 14, 16, 17, 33, 34, 36])

    def test_the_remapping_leftovers_are_ignored_not_classes(self):
        # The release maps OccuFly's raw ontology onto the benchmark palette
        # and missed some pixels; those raw ids are ignored, never named as
        # classes and never things.
        spec = SPECS["segfly"]
        self.assertEqual(sorted(spec.ignore), [0, 5, 11, 12, 21])
        self.assertFalse(set(spec.ignore) & set(spec.thing_ids))
        self.assertFalse({5, 11, 12, 21} & set(spec.classes.values()))


@unittest.skipUnless(pq is not None, "pyarrow is not installed")
class TestExportParquet(unittest.TestCase):
    """The shard-planning path, against real parquet on local disk.

    The Hub half is not tested -- it is 26 GB over the network. What is tested
    is the part that decides whether skipping shards is *safe*: that a row's
    name comes from its position in the split rather than in its shard, and
    that an unselected shard is never opened.
    """

    MODALITY = ("RGB", "RGB", "thermal", "RGB", "thermal")

    def shard(self, directory: Path, name: str, kinds: tuple[str, ...]) -> Path:
        import io

        import numpy as np
        import PIL.Image

        def cell(value: int) -> dict:
            buf = io.BytesIO()
            PIL.Image.fromarray(np.full((4, 4), value, np.uint8)).save(
                buf, format="PNG")
            return {"bytes": buf.getvalue(), "path": f"{value}.png"}

        table = pa.table({
            "image": [cell(1) for _ in kinds],
            "label": [cell(13) for _ in kinds],
            "RGB_aligned": [cell(2) for _ in kinds],
            "scene": ["scene_05"] * len(kinds),
            "altitude": ["40m"] * len(kinds),
            "modality": list(kinds),
        })
        path = directory / name
        pq.write_table(table, path)
        return path

    def run_export(self, dest: Path, shards: list[dict], columns,
                   **kwargs) -> dict:
        pulled = []

        def fake_pull(dataset_id, path, into):
            pulled.append(path)
            copy = into / Path(path).name
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes(Path(path).read_bytes())
            return copy

        original_plan, original_pull = export_hf_dataset.plan, export_hf_dataset.pull
        export_hf_dataset.plan = lambda *a, **k: shards
        export_hf_dataset.pull = fake_pull
        try:
            result = export_hf_dataset.export_parquet(
                "owner/name", dest,
                kwargs.pop("modality", "thermal"), "train",
                kwargs.pop("limit", None), columns, quiet=True, **kwargs)
        finally:
            export_hf_dataset.plan, export_hf_dataset.pull = original_plan, original_pull
        return result | {"pulled": pulled}

    def setUp(self):
        import tempfile

        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)
        # Two shards; the first holds no thermal row at all.
        self.cold = self.shard(self.tmp, "a.parquet", ("RGB", "RGB"))
        self.hot = self.shard(self.tmp, "b.parquet", self.MODALITY)
        self.plan = [
            {"path": str(self.cold), "offset": 0, "rows": 2, "keep": []},
            {"path": str(self.hot), "offset": 2, "rows": 5, "keep": [2, 4]},
        ]

    def test_a_shard_with_no_wanted_row_is_never_downloaded(self):
        # This is the whole point of planning: on SegFly it is 434 shards and
        # 156 GiB that a streaming filter would fetch and throw away.
        result = self.run_export(self.tmp / "out", self.plan, ("image", "label"))
        self.assertEqual(result["pulled"], [str(self.hot)])

    def test_names_come_from_the_row_index_in_the_split_not_the_shard(self):
        # Rows 2 and 4 of the second shard, which starts at offset 2, are rows
        # 4 and 6 of the split. Naming them 000002/000004 would mean an export
        # narrowed to one modality and one widened later disagree about which
        # file is which frame.
        dest = self.tmp / "out"
        self.run_export(dest, self.plan, ("image", "label"))
        self.assertEqual(sorted(p.name for p in (dest / "images").iterdir()),
                         ["scene_05_40m_000004.png", "scene_05_40m_000006.png"])

    def test_only_the_wanted_rows_and_columns_are_written(self):
        dest = self.tmp / "out"
        result = self.run_export(dest, self.plan, ("image", "label"))
        self.assertEqual(result["written"], 2)
        self.assertEqual(result["skipped"], 5)          # 2 cold + 3 unwanted
        self.assertEqual(result["values"], {13: 2})   # once per map, not per pixel
        self.assertFalse((dest / "rgb").exists())       # RGB_aligned was dropped

    def test_the_downloaded_shard_does_not_outlive_its_export(self):
        # Peak disk is one shard plus the output, not the 22 GiB of shards.
        dest = self.tmp / "out"
        self.run_export(dest, self.plan, ("image", "label"))
        self.assertFalse((dest / "_parquet").exists())

    def test_an_rgb_export_lands_where_the_spec_rgb_glob_reads(self):
        # On an RGB export the `image` column *is* the RGB frame. Writing it
        # to `images/` would leave `--dataset segfly:<dest>:rgb:...` globbing
        # an empty `rgb/` -- the export and the spec must agree on the
        # directory, and this is the join.
        self.assertEqual(folders_for("RGB")["image"], "rgb")
        self.assertEqual(folders_for("thermal")["image"], "images")
        self.assertEqual(folders_for(None)["image"], "images")

        dest = self.tmp / "out"
        plan = [{"path": str(self.hot), "offset": 0, "rows": 5,
                 "keep": [2, 4]}]
        self.run_export(dest, plan, ("image", "label"), modality="thermal")
        self.assertTrue((dest / "images").exists())
        self.assertFalse((dest / "rgb").exists())

    def test_passthrough_keeps_the_publishers_bytes_and_the_labels_png(self):
        # SegFly's RGB frames re-encode from 11 MB of JPEG to 27 MB of PNG,
        # so the passthrough export writes the original bytes untouched. The
        # label map is the exception both ways: it must be decoded to count
        # its values, and must stay PNG to stay lossless.
        dest = self.tmp / "out"
        plan = [{"path": str(self.hot), "offset": 2, "rows": 5,
                 "keep": [2, 4]}]
        result = self.run_export(dest, plan, ("image", "label"),
                                 passthrough=True)

        images = sorted((dest / "images").iterdir())
        self.assertEqual([p.suffix for p in images], [".png", ".png"])
        # The synthetic cells store PNG bytes under a `.png` path, so a
        # byte-for-byte copy is checkable directly.
        table = pq.read_table(self.hot, columns=["image"])
        self.assertEqual(images[0].read_bytes(),
                         table.column("image")[2].as_py()["bytes"])
        self.assertEqual(result["values"], {13: 2})   # labels still counted

    def test_spread_takes_evenly_spaced_shards_not_the_first_ones(self):
        # `--limit` alone exports the first N matching rows -- one scene.
        # With `spread`, the N rows come from shards spaced across the whole
        # match, and shards outside the slice are never downloaded.
        shards = [self.shard(self.tmp, f"s{i}.parquet", self.MODALITY)
                  for i in range(6)]
        plan = [{"path": str(p), "offset": 5 * i, "rows": 5, "keep": [2, 4]}
                for i, p in enumerate(shards)]

        dest = self.tmp / "out"
        result = self.run_export(dest, plan, ("image", "label"),
                                 limit=4, spread=True)
        self.assertEqual(result["written"], 4)
        self.assertEqual([Path(p).name for p in result["pulled"]],
                         ["s0.parquet", "s3.parquet"])

        # Without spread: the same cap comes from the first shards only.
        contiguous = self.run_export(self.tmp / "out2", plan,
                                     ("image", "label"), limit=4)
        self.assertEqual([Path(p).name for p in contiguous["pulled"]],
                         ["s0.parquet", "s1.parquet"])


if __name__ == "__main__":
    unittest.main()
