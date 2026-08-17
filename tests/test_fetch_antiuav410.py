"""Archive layout, split discovery and the extractor's refusals.

No download happens here. What is tested is everything that decides *where the
8.7 GB goes* -- which is exactly the part you do not want to debug by
downloading it again.
"""
from __future__ import annotations

import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.fetch_antiuav410 import (  # noqa: E402
    check_archive,
    count_sequences,
    dataset_root,
    describe,
    extract,
    find_splits,
    safe_target,
    split_of,
)

LABEL = '{"exist": [1], "gt_rect": [[1, 2, 3, 4]]}'


def build_archive(path: Path, layout: dict[str, list[str]], prefix: str = "Anti-UAV410") -> Path:
    """A miniature Anti-UAV410: `<prefix>/<split>/<sequence>/{0.jpg, IR_label.json}`."""
    with zipfile.ZipFile(path, "w") as zf:
        for split, sequences in layout.items():
            for name in sequences:
                base = "/".join(p for p in (prefix, split, name) if p)
                zf.writestr(f"{base}/0.jpg", b"\xff\xd8\xff")
                zf.writestr(f"{base}/1.jpg", b"\xff\xd8\xff")
                zf.writestr(f"{base}/IR_label.json", LABEL)
    return path


class TestSplitOf(unittest.TestCase):
    def test_reads_the_split_from_the_path(self):
        self.assertEqual(split_of("Anti-UAV410/train/20190925_101846_1_8/0.jpg"), "train")
        self.assertEqual(split_of("Anti-UAV410/val/seq/IR_label.json"), "val")
        self.assertEqual(split_of("AntiUAV410/test/seq/0.jpg"), "test")

    def test_a_flat_archive_with_no_wrapper_folder_still_works(self):
        self.assertEqual(split_of("train/seq/0.jpg"), "train")

    def test_a_filename_is_not_a_split(self):
        # Only directory components count -- otherwise a frame named test.jpg
        # would drag its whole sequence into the test split.
        self.assertIsNone(split_of("Anti-UAV410/train_notes/test.jpg"))

    def test_members_outside_any_split_are_unclaimed(self):
        self.assertIsNone(split_of("Anti-UAV410/README.md"))
        self.assertIsNone(split_of("Anti-UAV410/annos/att/seq.txt"))


class TestSafeTarget(unittest.TestCase):
    def test_a_normal_member_lands_under_dest(self):
        dest = Path("/data")
        self.assertEqual(safe_target("a/b/0.jpg", dest), Path("/data/a/b/0.jpg"))

    def test_traversal_and_absolute_members_are_refused(self):
        dest = Path("/data")
        for member in ("../../etc/passwd", "a/../../b.jpg", "/etc/passwd"):
            self.assertIsNone(safe_target(member, dest), member)

    def test_directory_entries_are_not_files_to_write(self):
        self.assertIsNone(safe_target("a/b/", Path("/data")))


class TestFindSplits(unittest.TestCase):
    def _tree(self, tmp: Path, prefix: str, splits=("train", "val", "test")) -> Path:
        for split in splits:
            seq = tmp / prefix / split / "seq0" if prefix else tmp / split / "seq0"
            seq.mkdir(parents=True)
            (seq / "IR_label.json").write_text(LABEL)
            (seq / "0.jpg").write_bytes(b"\xff\xd8\xff")
        return tmp

    def test_finds_splits_inside_the_archives_wrapper_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree(Path(tmp), "Anti-UAV410")
            found = find_splits(root)
            self.assertEqual(sorted(found), ["test", "train", "val"])
            self.assertEqual(found["train"], root / "Anti-UAV410" / "train")

    def test_finds_splits_extracted_flat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree(Path(tmp), "")
            self.assertEqual(sorted(find_splits(root)), ["test", "train", "val"])

    def test_a_split_directory_with_no_annotated_sequence_does_not_count(self):
        # An interrupted extraction leaves the folder there and the sequences
        # half-written; reporting that as present is how a training run ends up
        # silently using a third of the data.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train" / "seq0").mkdir(parents=True)
            (root / "train" / "seq0" / "0.jpg").write_bytes(b"\xff\xd8\xff")
            self.assertEqual(find_splits(root), {})

    def test_missing_root_is_empty_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(find_splits(Path(tmp) / "nope"), {})

    def test_dataset_root_is_what_list_sequences_wants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree(Path(tmp), "Anti-UAV410")
            found = find_splits(root)
            self.assertEqual(dataset_root(found), root / "Anti-UAV410")

    def test_dataset_root_refuses_splits_that_do_not_share_a_parent(self):
        with self.assertRaises(ValueError):
            dataset_root({"train": Path("/a/train"), "val": Path("/b/val")})

    def test_counts_and_describes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._tree(Path(tmp), "Anti-UAV410", splits=("train",))
            found = find_splits(root)
            self.assertEqual(count_sequences(found["train"]), 1)
            self.assertIn("train", describe(found))


class TestExtract(unittest.TestCase):
    def test_extracts_only_the_requested_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            archive = build_archive(tmp / "a.zip", {
                "train": ["s0", "s1"], "val": ["s2"], "test": ["s3"],
            })
            dest = tmp / "out"
            extract(archive, dest, ("train", "val"), workers=2, quiet=True)

            found = find_splits(dest)
            self.assertEqual(sorted(found), ["train", "val"])
            self.assertEqual(count_sequences(found["train"]), 2)
            self.assertFalse((dest / "Anti-UAV410" / "test").exists())

    def test_the_extracted_tree_is_what_list_sequences_reads(self):
        from src.training import list_sequences

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            archive = build_archive(tmp / "a.zip", {"train": ["s0", "s1"]})
            dest = tmp / "out"
            extract(archive, dest, ("train",), workers=2, quiet=True)

            sequences = list_sequences(dataset_root(find_splits(dest)), "train")
            self.assertEqual([s.name for s in sequences], ["s0", "s1"])
            self.assertEqual(len(sequences[0].frames), 1)  # labels truncate to 1

    def test_an_archive_with_no_matching_member_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            archive = tmp / "a.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("something/else.txt", "x")
            with self.assertRaises(SystemExit):
                extract(archive, tmp / "out", ("train",), quiet=True)

    def test_a_traversal_member_is_never_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            archive = tmp / "a.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("train/seq/0.jpg", b"\xff\xd8\xff")
                zf.writestr("train/seq/../../../escaped.jpg", b"nope")
            dest = tmp / "out"
            extract(archive, dest, ("train",), workers=2, quiet=True)
            written = sorted(p.relative_to(dest).as_posix()
                             for p in dest.rglob("*") if p.is_file())
            self.assertEqual(written, ["train/seq/0.jpg"])


class TestCheckArchive(unittest.TestCase):
    def test_googles_quota_page_is_named_for_what_it_is(self):
        with tempfile.TemporaryDirectory() as tmp:
            page = Path(tmp) / "Anti-UAV410.zip"
            page.write_text("<html><title>Google Drive - Quota exceeded</title></html>")
            with self.assertRaises(SystemExit) as caught:
                check_archive(page)
            self.assertIn("quota", str(caught.exception).lower())

    def test_a_missing_file_is_reported_before_anything_else(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                check_archive(Path(tmp) / "absent.zip")

    def test_a_large_non_zip_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            blob = Path(tmp) / "a.zip"
            blob.write_bytes(b"\x00" * 2_000_000)
            with self.assertRaises(SystemExit):
                check_archive(blob)

    def test_a_real_archive_of_unexpected_size_warns_but_proceeds(self):
        # A small *valid* zip is a hand-made or partial copy, not an error page.
        # Refusing it would block the --zip path this leaves open for people
        # whose Drive quota is spent.
        with tempfile.TemporaryDirectory() as tmp:
            archive = build_archive(Path(tmp) / "a.zip", {"train": ["s0"]})
            buffer, real = io.StringIO(), sys.stdout
            sys.stdout = buffer
            try:
                check_archive(archive, expect_bytes=9_361_681_896)
            finally:
                sys.stdout = real
            self.assertIn("expected", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
