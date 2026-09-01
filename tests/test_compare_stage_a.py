"""Ranking the stage-A arms by what stage B did afterwards.

The two pretrains minimise different losses on different tasks, so their own
numbers cannot be compared -- only the downstream grade can. What this has to
get right is therefore not arithmetic but refusal: no control, or arms that did
not run the same way, and a ranking would be measuring the settings as well as
the base.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.compare_stage_a import control_of, load, report, unfair  # noqa: E402

FAIR = dict(seed=0, epochs=[2, 12], steps=500, batch=128, min_box_iou=0.8,
            prompt="mix", image_size=512, method="finetune")


class CompareTest(unittest.TestCase):
    def folder(self):
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        return Path(holder.name)

    def arm(self, folder: Path, run: str, mean: float, **over):
        body = {"run": run, **FAIR, **over,
                "frames": {"train": 100, "val": 10, "test": 10},
                "before": {"box": {"mean_iou": 0.6, "small_mean_iou": 0.4,
                                   "iou_50": 0.7}},
                "after": {"box": {"mean_iou": mean, "small_mean_iou": mean - 0.2,
                                  "iou_50": mean + 0.08}}}
        path = folder / f"{run}.json"
        path.write_text(json.dumps(body))
        return path

    def run_on(self, paths):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = report(list(paths))
        return code, out.getvalue()

    def four(self, folder: Path, **over):
        return [
            self.arm(folder, "aerial_thermal_stable", 0.70),
            self.arm(folder, "aerial_thermal_stable_from_pretrain_thermal_aerial", 0.724),
            self.arm(folder, "aerial_thermal_stable_from_edgetam_fcmae_grn", 0.741, **over),
            self.arm(folder, "aerial_thermal_stable_from_edgetam_fcmae_plain", 0.702),
        ]

    def test_the_control_is_the_arm_with_no_base(self):
        """32 names an arm that was given no BASE_CHECKPOINT without a
        `_from_`, which is how the control identifies itself."""
        folder = self.folder()
        arms = [load(p) for p in self.four(folder)]
        self.assertEqual(control_of(arms)["run"], "aerial_thermal_stable")

    def test_it_ranks_and_shows_the_delta_against_stock(self):
        code, text = self.run_on(self.four(self.folder()))
        self.assertEqual(code, 0)
        first = [l for l in text.splitlines() if "fcmae_grn" in l][0]
        self.assertIn("+0.0410", first)
        self.assertIn("(control)", text)
        self.assertIn("is ahead of stock", text)

    def test_it_refuses_to_rank_without_a_control(self):
        """A pretrain is measured against stage B from stock. Without that arm
        the table would be four numbers with nothing to be better than."""
        folder = self.folder()
        paths = [p for p in self.four(folder) if "_from_" in p.name]
        code, text = self.run_on(paths)
        self.assertEqual(code, 1)
        self.assertIn("no single arm started from stock", text)

    def test_an_arm_run_differently_is_named_rather_than_ranked_silently(self):
        code, text = self.run_on(self.four(self.folder(), seed=7))
        self.assertEqual(code, 1)
        self.assertIn("did not run the same way", text)
        self.assertIn("seed is 7", text)

    def test_a_tighter_area_gate_is_caught_too(self):
        """The gate that drops big targets changes what the arm trained on.

        `MAX_AREA` decides which instances the index keeps, so an arm run with
        a tighter one saw a different training set. Ranking it against the
        control measures the gate as well as the base, which is the thing this
        tool exists to refuse.
        """
        folder = self.folder()
        loose = {"min_area": 64, "min_side": 6, "max_area": 0.2, "fill": 0.25}
        paths = self.four(folder, gates=dict(loose, max_area=0.06))
        for path in paths:
            body = json.loads(path.read_text())
            body.setdefault("gates", loose)
            path.write_text(json.dumps(body))
        code, text = self.run_on(paths)
        self.assertEqual(code, 1)
        self.assertIn("did not run the same way", text)
        self.assertIn("gates is", text)

    def test_a_different_split_is_caught_too(self):
        folder = self.folder()
        paths = self.four(folder)
        body = json.loads(paths[2].read_text())
        body["frames"] = {"train": 90, "val": 10, "test": 10}
        paths[2].write_text(json.dumps(body))
        code, text = self.run_on(paths)
        self.assertEqual(code, 1)
        self.assertIn("split sizes", text)

    def test_no_pretrain_winning_is_reported_as_the_result_it_is(self):
        """The outcome the paper's own ablation predicts for one of these arms,
        and the one it would be easiest to dress up as a ranking."""
        folder = self.folder()
        paths = [
            self.arm(folder, "aerial_thermal_stable", 0.70),
            self.arm(folder, "aerial_thermal_stable_from_edgetam_fcmae_plain", 0.698),
        ]
        code, text = self.run_on(paths)
        self.assertEqual(code, 0)
        self.assertIn("No pretrain beat stage B from stock", text)
        self.assertIn("bought nothing", text)

    def test_a_close_margin_asks_for_a_second_seed(self):
        """One seed each is not an ordering, and the tool says so rather than
        letting a 0.002 gap become a claim."""
        _, text = self.run_on(self.four(self.folder()))
        self.assertIn("another SEED", text)

    def test_an_unscored_arm_stops_the_ranking(self):
        folder = self.folder()
        paths = self.four(folder)
        body = json.loads(paths[1].read_text())
        body["after"] = {}
        paths[1].write_text(json.dumps(body))
        code, text = self.run_on(paths)
        self.assertEqual(code, 1)
        self.assertIn("did not finish scoring", text)

    def test_unfair_reports_nothing_when_the_arms_agree(self):
        folder = self.folder()
        arms = [load(p) for p in self.four(folder)]
        self.assertEqual(unfair(arms, control_of(arms)), [])


if __name__ == "__main__":
    unittest.main()
