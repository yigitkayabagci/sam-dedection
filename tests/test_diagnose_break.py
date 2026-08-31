"""Telling the encoder's failure apart from the memory bank's.

Both look the same in the video -- the box leaves the target and settles on
something else -- and they need opposite fixes: one is a problem with the
pixels going *into* the encoder, the other with what the bank is holding from
before the exposure changed. The two are separable by measurement, and these
tests hold that the measurement says the right thing on cases where the answer
is known by construction.
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

from src.pipeline import photometric_drift  # noqa: E402
from tools.diagnose_break import (  # noqa: E402
    BIG_DRIFT,
    DARK_SPAN,
    breaks,
    episodes,
    report,
    verdict,
)


class DriftTest(unittest.TestCase):
    """`photometric_drift`: how far a frame has moved from the ones the bank
    is holding, in grey levels."""

    def rows(self, spans):
        return [{"frame": i, "p1": low, "p99": high, "span": high - low}
                for i, (low, high) in enumerate(spans)]

    def test_a_steady_clip_does_not_drift(self):
        drift = photometric_drift(self.rows([(20, 80)] * 10))
        self.assertIsNone(drift[0])
        self.assertEqual(set(drift[1:]), {0.0})

    def test_an_exposure_step_shows_up_for_the_bank_s_depth(self):
        """And then stops: once the bank has filled with the new exposure the
        frames match it again, which is the point -- the number describes the
        mismatch, not the change."""
        drift = photometric_drift(self.rows([(20, 80)] * 8 + [(120, 180)] * 8),
                                  depth=7)
        self.assertEqual(drift[8], 100.0)
        self.assertEqual(drift[-1], 0.0)

    def test_a_slow_ramp_the_bank_follows_is_not_a_mismatch(self):
        """A gain that creeps up over a minute is not what breaks a track: the
        bank's contents move with it."""
        rows = self.rows([(20 + i, 80 + i) for i in range(40)])
        drift = [d for d in photometric_drift(rows) if d is not None]
        self.assertLess(max(drift), 6.0)

    def test_a_frame_with_no_reading_is_not_invented(self):
        rows = self.rows([(20, 80)] * 4)
        rows[2] = {"frame": 2, "p1": None, "p99": None, "span": None}
        self.assertIsNone(photometric_drift(rows)[2])


class VerdictTest(unittest.TestCase):
    def test_a_dark_frame_is_the_encoder_s(self):
        self.assertTrue(verdict(DARK_SPAN - 10, 0.0).startswith("ENCODER"))

    def test_a_readable_frame_that_moved_is_the_memory_s(self):
        self.assertTrue(verdict(DARK_SPAN + 60, BIG_DRIFT + 10).startswith("MEMORY"))

    def test_both_at_once_says_both(self):
        self.assertTrue(verdict(DARK_SPAN - 10, BIG_DRIFT + 10).startswith("BOTH"))

    def test_neither_is_reported_rather_than_forced(self):
        """The finding that matters most: a break where the pixels did not
        change is a break neither fix addresses."""
        self.assertTrue(verdict(DARK_SPAN + 60, 0.0).startswith("NEITHER"))

    def test_a_missing_reading_is_not_evidence(self):
        self.assertTrue(verdict(None, None).startswith("NEITHER"))


class BreakTest(unittest.TestCase):
    def rows(self, **overrides):
        base = [{"frame": i, "held": True, "share": 0.01, "jump": 0.1,
                 "grew": 1.0} for i in range(20)]
        for frame, patch in overrides.items():
            base[int(frame[1:])].update(patch)
        return {"rows": base}

    def test_a_clean_run_has_nothing_to_explain(self):
        self.assertEqual(breaks(self.rows()), [])

    def test_a_lost_frame_a_jump_and_a_balloon_are_all_breaks(self):
        found = breaks(self.rows(f5={"held": False, "share": 0.0},
                                 f9={"jump": 6.0}, f14={"grew": 4.0}))
        self.assertEqual([b["frame"] for b in found], [5, 9, 14])
        self.assertIn("lost", found[0]["why"])
        self.assertIn("jumped", found[1]["why"])
        self.assertIn("grew", found[2]["why"])

    def test_consecutive_bad_frames_are_one_event(self):
        rows = [{"frame": i, "why": "lost"} for i in (4, 5, 6, 7, 30)]
        self.assertEqual(episodes(rows), [(4, 7, "lost"), (30, 30, "lost")])


class ReportTest(unittest.TestCase):
    """The whole join, on two clips whose answer is known by construction."""

    def build(self, folder: Path, spans, break_at):
        rows = [{"frame": i, "held": True, "share": 0.01,
                 "jump": 6.0 if i == break_at else 0.1, "grew": 1.0}
                for i in range(len(spans))]
        (folder / "track.json").write_text(json.dumps({
            "frames": len(rows), "held": len(rows), "lost": 0,
            "longest_gap": 0, "share_median": 0.01, "share_max": 0.01,
            "jumps": 1, "rows": rows}))
        photo = [{"frame": i, "p1": low, "p99": high, "span": high - low}
                 for i, (low, high) in enumerate(spans)]
        drift = photometric_drift(photo)
        for row, value in zip(photo, drift):
            row["drift"] = value
        moved = [d for d in drift if d is not None]
        (folder / "photometry.json").write_text(json.dumps({
            "floor": 0, "frames": len(photo), "stretched": 0,
            "span_median": photo[0]["span"], "span_min": photo[0]["span"],
            "dark_frames": sum(1 for r in photo if r["span"] < DARK_SPAN),
            "drift_median": 0.0, "drift_max": max(moved), "rows": photo}))

    def run_on(self, spans, break_at):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.build(folder, spans, break_at)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = report(folder)
            self.assertEqual(code, 0)
            return out.getvalue()

    def test_a_dark_clip_that_breaks_is_called_the_encoder_s(self):
        text = self.run_on([(30, 60)] * 30, break_at=20)
        self.assertIn("ENCODER", text)
        self.assertNotIn("MEMORY —", text)

    def test_a_bright_clip_whose_exposure_steps_is_called_the_memory_s(self):
        spans = [(20, 200)] * 15 + [(120, 300)] * 15
        text = self.run_on(spans, break_at=17)
        self.assertIn("MEMORY", text)

    def test_a_clip_where_nothing_moved_is_called_neither(self):
        text = self.run_on([(20, 200)] * 30, break_at=20)
        self.assertIn("NEITHER", text)

    def test_a_run_without_the_measurement_says_what_to_re_run_with(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.build(folder, [(30, 60)] * 12, break_at=6)
            (folder / "photometry.json").unlink()
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(report(folder), 1)
            self.assertIn("--photometry", out.getvalue())


if __name__ == "__main__":
    unittest.main()
