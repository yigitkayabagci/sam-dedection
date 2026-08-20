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
from tools.export_hf_dataset import COLUMNS, stem, verify  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
