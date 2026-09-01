"""The resolution a checkpoint was trained at, and the warning nobody got.

The bug this guards is a silent one. EdgeTAM keeps no resolution in any
parameter, so a 512-trained checkpoint loads into a 768 build under
`strict=True` with identical keys and shapes and no complaint from anything.
`meta["image_size"]` has been written by every fine-tune this repository ever
ran, and until `src/checkpoint_meta.py` only the training path read it -- the
tracker and the ONNX exporter, the two places a mismatch actually reaches a
measurement, loaded the file and said nothing.

Most of this runs without torch: `trained_size` imports it inside the function,
so a stub in `sys.modules` exercises every branch. The round trip against a
real `torch.save` is what proves the stub is not testing a fiction, and it is
skipped where torch is not installed.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
except ImportError:                          # pragma: no cover - CPU-only CI
    torch = None

from src.checkpoint_meta import (recorded_meta, size_mismatch_note,  # noqa: E402
                                 trained_size, warn_size_mismatch)


class FakeTorch:
    """Just enough torch for `trained_size`: one `load` that returns a blob."""

    def __init__(self, blob):
        self._blob = blob

    def load(self, path, map_location=None, weights_only=False):
        if isinstance(self._blob, Exception):
            raise self._blob
        return self._blob


def read(blob, checkpoint="ckpt.pt"):
    with mock.patch.dict(sys.modules, {"torch": FakeTorch(blob)}):
        return trained_size(checkpoint)


def note(blob, size, checkpoint="ckpt.pt"):
    with mock.patch.dict(sys.modules, {"torch": FakeTorch(blob)}):
        return size_mismatch_note(checkpoint, size)


def meta(blob, checkpoint="ckpt.pt"):
    with mock.patch.dict(sys.modules, {"torch": FakeTorch(blob)}):
        return recorded_meta(checkpoint)


class RecordedMeta(unittest.TestCase):
    """What produced a checkpoint, which strict loading will never tell you.

    A stage that continues another one loads its base with `strict=True` and
    scores normally whichever run wrote it, so continuing the wrong arm is
    invisible until the run is over. `meta["base"]` is the only thing on disk
    that answers it.
    """

    def test_reads_the_base_the_run_started_from(self):
        self.assertEqual(
            meta({"model": {}, "meta": {"base": "/x/edgetam.pt",
                                        "train_frames": 67199}}),
            {"base": "/x/edgetam.pt", "train_frames": 67199})

    def test_stock_edgetam_gives_an_empty_dict_not_a_guess(self):
        # No `meta` is a different answer from "stock was its base", and the
        # caller has to be able to tell those apart.
        self.assertEqual(meta({"model": {"trunk.weight": 1}}), {})

    def test_a_file_that_is_not_there_is_empty_not_an_error(self):
        self.assertEqual(meta(FileNotFoundError("nope")), {})

    def test_a_blob_that_is_not_a_dict_is_empty(self):
        self.assertEqual(meta(["not", "a", "checkpoint"]), {})

    def test_a_meta_that_is_not_a_dict_is_empty(self):
        self.assertEqual(meta({"model": {}, "meta": "corrupted"}), {})

    def test_no_path_never_touches_torch(self):
        self.assertEqual(recorded_meta(None), {})
        self.assertEqual(recorded_meta(""), {})

    def test_the_caller_cannot_mutate_what_the_file_holds(self):
        blob = {"meta": {"base": "/x/edgetam.pt"}}
        got = meta(blob)
        got["base"] = "/tampered"
        self.assertEqual(blob["meta"]["base"], "/x/edgetam.pt")


class TrainedSize(unittest.TestCase):
    def test_reads_what_the_fine_tune_wrote(self):
        self.assertEqual(read({"model": {}, "meta": {"image_size": 512}}), 512)

    def test_a_float_is_still_a_size(self):
        """`json`-round-tripped meta can hand back 512.0, which is the size."""
        self.assertEqual(read({"meta": {"image_size": 512.0}}), 512)

    def test_stock_edgetam_records_nothing(self):
        """The shipped checkpoint has no `meta`, and that is not an error.

        It was trained at 1024 long before this repository existed, so the
        honest answer is "unknown" and the caller must not be warned about a
        mismatch it cannot substantiate.
        """
        self.assertIsNone(read({"model": {"trunk.weight": 1}}))

    def test_meta_without_a_size(self):
        self.assertIsNone(read({"meta": {"method": "finetune"}}))

    def test_a_blob_that_is_not_a_dict(self):
        self.assertIsNone(read(["not", "a", "checkpoint"]))

    def test_meta_that_is_not_a_dict(self):
        self.assertIsNone(read({"meta": "512"}))

    def test_a_bool_is_not_a_size(self):
        """`isinstance(True, int)` is the trap; True is not an image size."""
        self.assertIsNone(read({"meta": {"image_size": True}}))

    def test_an_unreadable_file_is_unknown_not_fatal(self):
        """Reading the meta must never be the thing that ends a run."""
        self.assertIsNone(read(RuntimeError("truncated file")))

    def test_no_checkpoint_at_all(self):
        self.assertIsNone(trained_size(None))
        self.assertIsNone(trained_size(""))


class MismatchNote(unittest.TestCase):
    def test_silent_when_the_sizes_agree(self):
        self.assertIsNone(note({"meta": {"image_size": 512}}, 512))

    def test_silent_when_the_size_was_not_overridden(self):
        """`image_size: null` leaves the Hydra config's own 1024 standing."""
        self.assertIsNone(note({"meta": {"image_size": 512}}, None))

    def test_silent_when_the_checkpoint_never_said(self):
        self.assertIsNone(note({"model": {}}, 768))

    def test_names_both_sizes_when_they_differ(self):
        text = note({"meta": {"image_size": 512}}, 768)
        self.assertIsNotNone(text)
        self.assertIn("512", text)
        self.assertIn("768", text)

    def test_the_512_checkpoint_at_768_is_the_case_this_exists_for(self):
        """22_thermal_deep.ipynb trains at 512; configs/edgetam_768_*.yaml runs 768."""
        self.assertIsNotNone(
            note({"meta": {"image_size": 512}}, 768,
                 "checkpoints/edgetam_pool_thermal_deep_512.pt"))

    def test_warn_prints_to_stderr_and_returns_it(self):
        import io
        import contextlib

        captured = io.StringIO()
        with mock.patch.dict(sys.modules,
                             {"torch": FakeTorch({"meta": {"image_size": 512}})}):
            with contextlib.redirect_stderr(captured):
                returned = warn_size_mismatch("ckpt.pt", 768)
        self.assertIsNotNone(returned)
        self.assertIn(returned, captured.getvalue())

    def test_warn_says_nothing_when_there_is_nothing_to_say(self):
        import io
        import contextlib

        captured = io.StringIO()
        with mock.patch.dict(sys.modules,
                             {"torch": FakeTorch({"meta": {"image_size": 512}})}):
            with contextlib.redirect_stderr(captured):
                returned = warn_size_mismatch("ckpt.pt", 512)
        self.assertIsNone(returned)
        self.assertEqual(captured.getvalue(), "")


class Wiring(unittest.TestCase):
    """The two paths where a mismatch reaches a measurement must both warn."""

    def test_the_tracker_holds_the_warning(self):
        source = (ROOT / "src/trackers/edgetam_tracker.py").read_text()
        self.assertIn("warn_size_mismatch(self.checkpoint, self.image_size)",
                      source)
        try:
            from src import checkpoint_meta
            from src.trackers import edgetam_tracker
        except ImportError as missing:       # numpy-less environment
            self.skipTest(f"tracker imports unavailable: {missing}")
        self.assertIs(edgetam_tracker.warn_size_mismatch,
                      checkpoint_meta.warn_size_mismatch)

    def test_the_exporter_holds_the_warning(self):
        """Export is the point of no return -- weights bake into the engines."""
        source = (ROOT / "tools/export_edgetam_onnx.py").read_text()
        self.assertIn("warn_size_mismatch(checkpoint, args.image_size)", source)


@unittest.skipIf(torch is None, "torch is not installed")
class RoundTrip(unittest.TestCase):
    """What `finetune.save_checkpoint` writes is what `trained_size` reads."""

    def test_reads_a_real_saved_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "edgetam_pool_thermal_deep_512.pt"
            torch.save({"model": {"a": torch.zeros(1)},
                        "meta": {"image_size": 512, "method": "finetune"}}, path)
            self.assertEqual(trained_size(path), 512)
            self.assertIsNone(size_mismatch_note(path, 512))
            self.assertIsNotNone(size_mismatch_note(path, 768))

    def test_a_checkpoint_without_meta_is_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stock.pt"
            torch.save({"model": {"a": torch.zeros(1)}}, path)
            self.assertIsNone(trained_size(path))


if __name__ == "__main__":
    unittest.main()
