"""The cleaned SegFly: reading the audit's manifest, and the spec that reads its masks.

No OpenCV and no download -- the manifest is JSON and the spec is a dataclass,
so both are checkable here. What matters is that the two agree: the tool writes
frame stems into a file whose name the spec's `exclude` glob matches, and
`excluded_keys` reads exactly those stems back.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.aerial import SPECS, excluded_keys  # noqa: E402
from src.training.datasets import parse  # noqa: E402
from tools.segfly_clean_manifest import MANIFESTS, census, main, stems  # noqa: E402


def entry(name, instances, shifted=False, measured=False, offset=(0, 0)):
    return {"kare": name, "goruntu": f"images/{name}.png",
            "maske": f"masks/{name}.png", "instance_sayisi": len(instances),
            "instances": [{"id": i + 1, "box": [0, 0, 10, 10], "alan": 100,
                           "incele": review}
                          for i, review in enumerate(instances)],
            "kayma_suphesi": shifted, "kayma_olcum": list(offset),
            "kayma_olculebildi": measured}


INDEX = {
    "1": entry("scene_03_30m_000001", [False]),
    "2": entry("scene_03_30m_000002", [False, True]),
    "3": entry("scene_03_30m_000003", [False], shifted=True, measured=True,
               offset=(22, -36)),
    "4": entry("scene_03_30m_000004", [True, True], shifted=True,
               measured=True, offset=(-8, 4)),
}


class TestSpec(unittest.TestCase):
    def test_the_masks_are_read_as_instances_not_as_a_palette(self):
        """The audit already decomposed, with watershed. Re-decomposing here
        would undo the one thing the cleaning bought."""
        request = parse("segfly_temiz:/data/SegFly_temiz:thermal:labels")
        self.assertEqual(request.source.mode, "labels")
        self.assertTrue(request.source.gray)

    def test_the_one_thing_class_is_vehicle(self):
        spec = SPECS["segfly_temiz"]
        self.assertEqual(spec.things, ("vehicle",))
        self.assertEqual(spec.thing_ids, (13,))
        self.assertIn(0, spec.ignore)

    def test_it_reads_no_rgb(self):
        """The audit could not verify that thermal and RGB are aligned -- its
        mutual-information probe reported an injected 6 px shift in the wrong
        direction -- so no measurement in the set comes from RGB."""
        self.assertIsNone(SPECS["segfly_temiz"].rgb)

    def test_every_manifest_this_tool_writes_is_one_the_spec_reads(self):
        """A file the glob misses is a drop that silently does not happen."""
        spec = SPECS["segfly_temiz"]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in MANIFESTS.values():
                (root / name).write_text("frame_a\n")
            found = {p.name for p in root.glob(spec.exclude)}
        self.assertEqual(found, set(MANIFESTS.values()))

    def test_the_glob_reaches_a_manifest_one_level_down(self):
        """An archive that unpacks into a folder of its own puts index.json --
        and the manifest written beside it -- below the root the --dataset flag
        names."""
        spec = SPECS["segfly_temiz"]
        with tempfile.TemporaryDirectory() as folder:
            nested = Path(folder) / "SegFly_temiz"
            nested.mkdir()
            (nested / "atlanan_kayma.txt").write_text("frame_a\nframe_b\n")
            self.assertEqual(excluded_keys(Path(folder), spec),
                             {"frame_a", "frame_b"})


class TestCensus(unittest.TestCase):
    def test_it_counts_frames_instances_and_both_kinds_of_doubt(self):
        counts = census(INDEX)
        self.assertEqual(counts["frames"], 4)
        self.assertEqual(counts["instances"], 6)
        self.assertEqual(counts["shifted"], 2)
        self.assertEqual(counts["measured"], 2)
        self.assertEqual(counts["review_instances"], 3)
        self.assertEqual(counts["review_frames"], 2)

    def test_the_worst_offset_is_the_largest_absolute_component(self):
        """A shift of (22, -36) is 36 px off, not 22 and not -36."""
        self.assertEqual(census(INDEX)["max_offset"], 36)

    def test_stems_are_frame_names_not_index_keys(self):
        """`excluded_keys` matches on the mask's stem, and the index is keyed
        by a row number that appears nowhere on disk."""
        self.assertEqual(stems(INDEX, ["3"]), ["scene_03_30m_000003"])


class TestManifests(unittest.TestCase):
    def setUp(self):
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.root = Path(holder.name)
        (self.root / "index.json").write_text(json.dumps(INDEX))

    def run_tool(self, drop):
        argv = sys.argv
        sys.argv = ["segfly_clean_manifest", "--set", str(self.root),
                    "--drop", drop]
        try:
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()) as out:
                code = main()
            return code, out.getvalue()
        finally:
            sys.argv = argv

    def test_dropping_the_shift_frames_writes_what_the_spec_reads_back(self):
        code, _ = self.run_tool("shift")
        self.assertEqual(code, 0)
        self.assertEqual(excluded_keys(self.root, SPECS["segfly_temiz"]),
                         {"scene_03_30m_000003", "scene_03_30m_000004"})

    def test_both_unions_the_two_reasons(self):
        self.run_tool("both")
        self.assertEqual(
            excluded_keys(self.root, SPECS["segfly_temiz"]),
            {"scene_03_30m_000002", "scene_03_30m_000003",
             "scene_03_30m_000004"})

    def test_none_puts_the_set_back(self):
        """Re-running with a narrower `--drop` has to remove the wider run's
        file, or an exclusion outlives the decision that made it."""
        self.run_tool("both")
        self.run_tool("none")
        self.assertEqual(excluded_keys(self.root, SPECS["segfly_temiz"]), set())

    def test_the_report_says_what_dropping_costs_before_it_drops(self):
        _, text = self.run_tool("none")
        self.assertIn("50.0%", text)          # measurable share
        self.assertIn("floor", text)
        self.assertIn("would leave 2 frames", text)

    def test_a_missing_index_is_an_error_that_names_the_directory(self):
        empty = self.root / "nothing"
        empty.mkdir()
        argv = sys.argv
        sys.argv = ["segfly_clean_manifest", "--set", str(empty)]
        try:
            with self.assertRaises(SystemExit) as caught:
                main()
        finally:
            sys.argv = argv
        self.assertIn("index.json", str(caught.exception))


if __name__ == "__main__":
    unittest.main()


class TestRole(unittest.TestCase):
    def test_the_cleaned_set_still_trains_only(self):
        """The audit is not a reason to let SegFly score.

        `role="train"` was there because a reconstructed target cannot grade a
        model. The audit fixed that where it could measure, and left 706 frames
        whose masks sit 25-50 px off the vehicle plus 50.5 % it could not
        measure -- and a label in the wrong *place* is the worst kind of
        validation error when validation selects the checkpoint.
        """
        request = parse("segfly_temiz:/data/SegFly_temiz:thermal:labels:train")
        self.assertEqual(request.source.role, "train")
