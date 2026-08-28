"""The two axes run_records.py varies, and the ways they must not be confused.

A mode is an input configuration -- a resolution and whether the frame is
cropped or resized into it. `--weights` is a different question entirely: which
trained checkpoint runs inside that configuration. The comparison is only worth
anything while each row of the summary holds one of them fixed, so what is
pinned here is that no combination can silently produce a row labelled one
thing and measured on another:

  * every (weights, size) pair resolves to a config that exists, declares that
    size, and takes its engines from a directory of its own;
  * a pair with no config raises instead of falling back to stock;
  * the output folder separates weights and skip runs from the baseline.

None of it needs torch, EdgeTAM or an engine -- it is table arithmetic, which
is why it can guard every combination rather than the one being worked on.
"""
from __future__ import annotations

import sys
import unittest
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from tools.run_records import (MODES, TRAINED_AT, WEIGHTS, config_for,  # noqa: E402
                               engines_missing, folder)


def body(config: str) -> dict:
    return yaml.safe_load((ROOT / config).read_text()) or {}


class Tables(unittest.TestCase):
    def test_every_mode_names_a_size_a_weights_set_could_serve(self):
        for mode, (size, crop) in MODES.items():
            self.assertIn(size, WEIGHTS["stock"], mode)
            self.assertIn(crop, (None, size),
                          f"{mode}: a crop is either off or the model input")

    def test_every_config_in_the_table_exists(self):
        for weights, table in WEIGHTS.items():
            for size, config in table.items():
                self.assertTrue((ROOT / config).is_file(),
                                f"{weights}/{size} -> {config} is not there")

    def test_every_config_declares_the_size_it_is_filed_under(self):
        """The table's key and the YAML's own `image_size` cannot disagree.

        They are what makes `--modes full768` mean 768: the mode picks the key,
        the tracker obeys the YAML, and a mismatch would run one and report the
        other. An absent `image_size` is not a mismatch -- it means the Hydra
        config's own 1024 stands, which is what configs/edgetam_trt.yaml relies
        on and why it is filed under 1024 while declaring nothing.
        """
        for weights, table in WEIGHTS.items():
            for size, config in table.items():
                declared = body(config).get("image_size", 1024)
                self.assertEqual(int(declared), size,
                                 f"{weights}/{size} -> {config}")

    def test_every_config_is_a_tensorrt_one(self):
        """run_records always runs `--tracker edgetam_trt`."""
        for weights, table in WEIGHTS.items():
            for size, config in table.items():
                self.assertIn("image_encoder_engine", body(config),
                              f"{weights}/{size} -> {config} has no engines")

    def test_each_weights_set_takes_its_engines_from_its_own_directory(self):
        """Engines are weight-specific: sharing a directory overwrites one set."""
        seen: dict[str, tuple[str, int]] = {}
        for weights, table in WEIGHTS.items():
            for size, config in table.items():
                directory = Path(body(config)["image_encoder_engine"]).parent.name
                self.assertNotIn(directory, seen,
                                 f"{weights}/{size} shares {directory}/ with "
                                 f"{seen.get(directory)}")
                seen[directory] = (weights, size)

    def test_each_weights_set_names_one_checkpoint(self):
        """A set that mixed checkpoints across sizes would not be one model."""
        for weights, table in WEIGHTS.items():
            checkpoints = {body(c)["checkpoint"] for c in table.values()}
            self.assertEqual(len(checkpoints), 1,
                             f"{weights} spans {checkpoints}")

    def test_pool_deep_is_the_thermal_stage_b_checkpoint_at_its_own_size(self):
        self.assertEqual(TRAINED_AT["pool_deep"], 512)
        self.assertIn("thermal_deep", body(WEIGHTS["pool_deep"][512])["checkpoint"])
        self.assertNotIn(1024, WEIGHTS["pool_deep"],
                         "a 512-trained checkpoint at 1024 measures the "
                         "resolution, not the training")


class ConfigFor(unittest.TestCase):
    def test_resolves_a_pair_that_exists(self):
        self.assertEqual(config_for("pool_deep", 512, "full512"),
                         WEIGHTS["pool_deep"][512])

    def test_refuses_a_pair_that_does_not_rather_than_falling_back(self):
        """Falling back to stock would label the row with the wrong weights."""
        with self.assertRaises(SystemExit) as raised:
            config_for("pool_deep", 1024, "full1024")
        message = str(raised.exception)
        self.assertIn("full1024", message)
        self.assertIn("stock", message)


class Folder(unittest.TestCase):
    @staticmethod
    def args(weights="stock", frame_skip=1):
        return Namespace(weights=weights, frame_skip=frame_skip)

    def test_a_plain_stock_run_keeps_the_bare_name(self):
        """Runs written before there was a second axis stay where they are."""
        self.assertEqual(folder("full512", self.args()), "full512")

    def test_other_weights_land_beside_the_baseline(self):
        self.assertEqual(folder("full512", self.args(weights="pool_deep")),
                         "full512_pool_deep")

    def test_a_skip_run_still_gets_its_own_folder(self):
        self.assertEqual(folder("full512", self.args(frame_skip=2)),
                         "full512_skip2")

    def test_both_axes_at_once_stay_distinct(self):
        names = {folder("full512", self.args(w, s))
                 for w in WEIGHTS for s in (1, 2)}
        self.assertEqual(len(names), 2 * len(WEIGHTS))


class Preflight(unittest.TestCase):
    def test_reports_the_directory_an_unbuilt_config_wants(self):
        """Engines are gitignored, so nothing in a clean checkout has them."""
        directory = engines_missing(WEIGHTS["pool_deep"][512])
        self.assertEqual(directory, "models512_pool_deep")

    def test_says_nothing_about_a_config_with_no_engines_at_all(self):
        self.assertIsNone(engines_missing("configs/edgetam_512.yaml"))


if __name__ == "__main__":
    unittest.main()
