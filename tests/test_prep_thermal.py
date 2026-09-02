"""Raw thermal frames into 8-bit, and the claims the mapping has to hold.

What is pinned: the 1st and 99th percentile of a frame land on 0 and 255 and
nothing outside them survives; the running window really smooths from frame
to frame and really snaps on a jump, because a window that fades over a scene
cut is ten frames of a saturated target; one global window is one window; a
headerless dump is read at the shape it was given; and a record laid out as
`<record>/etiketlenecek/` gets its `prep/` beside the raw folder, not inside
it -- also when the raw folder itself is what was given -- with a report that
says what every frame was mapped through.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import cv2
except ImportError:                                      # pragma: no cover
    cv2 = None

from tools.prep_thermal import (Histogram, Params, RawSpec, Window,  # noqa: E402
                                find_records, infer_raw_spec, main, percentiles_of,
                                prep_record, read_frame, source_files, to_uint8, widen)


def frame(level: float, spread: float = 200.0, shape=(96, 128), seed: int = 0,
          dtype=np.uint16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(level, spread, size=shape)
    top = np.iinfo(dtype).max
    return np.clip(values, 0, top).astype(dtype)


@unittest.skipIf(cv2 is None, "OpenCV is not installed")
class MappingTest(unittest.TestCase):
    def test_histogram_percentiles_agree_with_numpy_within_a_level(self):
        img = frame(3000, 400)
        lo, hi = percentiles_of(img, (0.01, 0.99))
        ref_lo, ref_hi = np.percentile(img, [1, 99], method="lower")
        self.assertLessEqual(abs(lo - ref_lo), 1)
        self.assertLessEqual(abs(hi - ref_hi), 1)
        # uint8 and float take different paths and must give the same answer.
        small = (img // 256).astype(np.uint8)
        self.assertEqual(percentiles_of(small, (0.5,)), percentiles_of(small.astype(np.float32), (0.5,)))

    def test_window_edges_land_on_0_and_255_and_outside_saturates(self):
        img = np.arange(0, 65536, 64, dtype=np.uint16).reshape(32, 32)
        params = Params(scope="frame")
        out = to_uint8(img, 1000.0, 2000.0, params)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(int(out[img < 1000].max()), 0)
        self.assertEqual(int(out[img > 2000].min()), 255)
        inside = img[(img >= 1000) & (img <= 2000)]
        expected = np.rint((inside.astype(np.float64) - 1000) / 1000 * 255).astype(np.uint8)
        np.testing.assert_array_equal(out[(img >= 1000) & (img <= 2000)], expected)
        # The 8-bit and float paths give the same numbers as the 16-bit take.
        as8 = (img // 256).astype(np.uint8)
        self.assertTrue(np.array_equal(to_uint8(as8, 4.0, 8.0, params),
                                       to_uint8(as8.astype(np.float32), 4.0, 8.0, params)))

    def test_invert_and_gamma_change_the_curve_only(self):
        img = np.linspace(0, 1000, 1000, dtype=np.float32).reshape(20, 50)
        plain = to_uint8(img, 0.0, 1000.0, Params())
        flipped = to_uint8(img, 0.0, 1000.0, Params(invert=True))
        np.testing.assert_array_equal(flipped, 255 - plain)
        bright = to_uint8(img, 0.0, 1000.0, Params(gamma=0.5))
        self.assertGreater(int(bright[10, 25]), int(plain[10, 25]))
        self.assertEqual(int(bright[0, 0]), 0)
        self.assertEqual(int(bright[-1, -1]), 255)

    def test_widen_keeps_a_flat_frame_from_becoming_noise(self):
        self.assertEqual(widen(100.0, 110.0, 0.0), (100.0, 110.0))
        self.assertEqual(widen(100.0, 110.0, 50.0), (80.0, 130.0))

    def test_params_reject_nonsense(self):
        with self.assertRaises(ValueError):
            Params(lo=99, hi=1)
        with self.assertRaises(ValueError):
            Params(ema=1.0)
        with self.assertRaises(ValueError):
            Params(median=4)
        with self.assertRaises(ValueError):
            Params(scope="video")


class WindowTest(unittest.TestCase):
    def test_per_frame_when_ema_is_zero(self):
        w = Window(0.0, 0.5)
        self.assertEqual(w.update(10, 20), (10, 20, False))
        self.assertEqual(w.update(30, 40), (30, 40, False))

    def test_ema_follows_slowly_and_snaps_on_a_jump(self):
        w = Window(0.9, 0.5, slack=0.2)
        w.update(1000.0, 2000.0)
        lo, hi, reset = w.update(1100.0, 2100.0)          # drift of a tenth of the width
        self.assertFalse(reset)
        self.assertAlmostEqual(lo, 1010.0)
        self.assertAlmostEqual(hi, 2010.0)
        lo, hi, reset = w.update(5000.0, 6000.0)          # a scene cut
        self.assertTrue(reset)
        self.assertEqual((lo, hi), (5000.0, 6000.0))

    def test_slack_bounds_the_lag_behind_a_steady_drift(self):
        loose, tight = Window(0.9, 0.5, slack=10.0), Window(0.9, 0.5, slack=0.1)
        for step in range(40):                             # 4 % of the width per frame
            lo = 1000.0 + 40.0 * step
            loose.update(lo, lo + 1000.0)
            hi_used = tight.update(lo, lo + 1000.0)[1]
            self.assertLessEqual(lo + 1000.0 - hi_used, 0.1 * 1000.0 + 1e-6)
        self.assertGreater(lo + 1000.0 - loose.hi, 300.0)  # the unbounded EMA lags ~9 frames
        # Under the bound a stable scene is still a plain EMA: nothing clamps.
        w = Window(0.9, 0.5, slack=0.1)
        w.update(1000.0, 2000.0)
        self.assertAlmostEqual(w.update(1020.0, 2020.0)[0], 1002.0)

    def test_global_histogram_merges_frames(self):
        h = Histogram()
        h.add(np.full((10, 10), 100, dtype=np.uint16))
        h.add(np.full((10, 10), 300, dtype=np.uint16))
        self.assertEqual(h.percentiles((0.25, 0.75)), [100.0, 300.0])
        f = Histogram()
        f.add(np.full((10, 10), 1.5, dtype=np.float32))
        f.add(np.full((10, 10), 3.5, dtype=np.float32))
        self.assertEqual(f.percentiles((0.25, 0.75)), [1.5, 3.5])


@unittest.skipIf(cv2 is None, "OpenCV is not installed")
class RecordTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def record(self, name: str, levels, src="etiketlenecek", suffix=".tiff", dtype=np.uint16) -> Path:
        folder = self.root / name / src
        folder.mkdir(parents=True)
        for index, level in enumerate(levels):
            cv2.imwrite(str(folder / f"frame_{index:06d}{suffix}"), frame(level, seed=index, dtype=dtype))
        return self.root / name

    def test_layouts_resolve_to_record_and_source(self):
        rec = self.record("record_a", [1000, 1000])
        raw = rec / "etiketlenecek"
        self.assertEqual(find_records(rec), [(rec, raw)])
        self.assertEqual(find_records(raw), [(rec, raw)])
        # An older record with the raws under vis/ is still found, after the default name.
        other = self.record("record_b", [1000], src="vis")
        self.assertEqual(find_records(self.root), [(rec, raw), (other, other / "vis")])
        self.assertEqual(find_records(self.root, "vis"), [(other, other / "vis")])
        self.assertEqual(find_records(other, "etiketlenecek,vis"), [(other, other / "vis")])
        # A raw folder under any name, given directly: prep goes one level up.
        odd = self.record("record_c", [1000], src="ham_kareler")
        self.assertEqual(find_records(odd / "ham_kareler"), [(odd, odd / "ham_kareler")])
        with self.assertRaises(SystemExit):
            find_records(odd)                          # no known raw folder name in it
        with self.assertRaises(SystemExit):
            find_records(self.root / "record_a" / "nothing_here")

    def test_source_files_fall_back_to_the_suffix_that_is_there(self):
        rec = self.record("pngs", [1000, 1000], suffix=".png")
        files, pattern = source_files(rec / "etiketlenecek", "*.tif*")
        self.assertEqual(len(files), 2)
        self.assertEqual(pattern, "*.png")
        (rec / "etiketlenecek" / "frame_000000.png").unlink()
        (rec / "etiketlenecek" / "frame_000001.png").unlink()
        with self.assertRaises(SystemExit):
            source_files(rec / "etiketlenecek", "*.tif*")

    def test_main_writes_prep_beside_the_raw_folder_with_a_report_and_a_preview(self):
        rec = self.record("record_s40", [1000, 1010, 1020, 1030])
        self.assertEqual(main([str(rec), "--quiet"]), 0)
        prep = rec / "prep"
        self.assertTrue(prep.is_dir())
        outputs = sorted(p.name for p in prep.glob("frame_*.png"))
        self.assertEqual(outputs, [f"frame_{i:06d}.png" for i in range(4)])
        self.assertFalse((rec / "etiketlenecek" / "prep").exists())
        written = cv2.imread(str(prep / "frame_000000.png"), cv2.IMREAD_UNCHANGED)
        self.assertEqual(written.dtype, np.uint8)
        self.assertEqual(written.shape, (96, 128))
        self.assertTrue((prep / "preview.png").is_file())
        body = json.loads((prep / "prep.json").read_text())
        self.assertEqual(body["frames"], 4)
        self.assertEqual(body["dtype"], "uint16")
        self.assertEqual(body["shape"], [96, 128])
        self.assertEqual(body["params"]["scope"], "ema")
        self.assertEqual(body["reader"], "image")
        self.assertEqual(len(body["rows"]), 4)
        first = body["rows"][0]
        for key in ("frame", "raw_lo", "raw_hi", "lo", "hi", "reset", "saturated", "out_span"):
            self.assertIn(key, first)
        # Every frame uses close to its full range and gives up ~2 % of itself.
        self.assertGreater(body["summary"]["out_span_median"], 240)
        self.assertLess(body["summary"]["saturated_max"], 0.05)
        self.assertGreater(body["summary"]["saturated_median"], 0.005)

    def test_the_raw_folder_given_directly_puts_prep_one_level_up(self):
        rec = self.record("direct", [1000, 1000])
        self.assertEqual(main([str(rec / "etiketlenecek"), "--quiet", "--preview", "0"]), 0)
        self.assertEqual(len(list((rec / "prep").glob("*.png"))), 2)
        self.assertFalse((rec / "etiketlenecek" / "prep").exists())

    def test_a_second_run_is_a_no_op_unless_overwritten_or_changed(self):
        rec = self.record("again", [1000, 1000])
        main([str(rec), "--quiet", "--preview", "0"])
        stamp = (rec / "prep" / "frame_000000.png").stat().st_mtime_ns
        main([str(rec), "--quiet", "--preview", "0"])
        self.assertEqual((rec / "prep" / "frame_000000.png").stat().st_mtime_ns, stamp)
        main([str(rec), "--quiet", "--preview", "0", "--scope", "frame"])
        body = json.loads((rec / "prep" / "prep.json").read_text())
        self.assertEqual(body["params"]["scope"], "frame")

    def test_ema_smooths_the_window_and_snaps_on_a_step(self):
        # Levels drift a little each frame, then jump: the window must follow
        # the drift with less motion than the frames have, and reset once.
        levels = [1000 + 20 * i for i in range(10)] + [4000 + 20 * i for i in range(10)]
        rec = self.record("drift", levels)
        report = prep_record(rec, rec / "etiketlenecek", rec / "prep", Params(scope="ema", ema=0.9),
                             preview=0, progress=False)
        rows = report["rows"]
        raw = np.diff([r["raw_lo"] for r in rows[:10]])
        used = np.diff([r["lo"] for r in rows[:10]])
        self.assertLess(np.abs(used).mean(), np.abs(raw).mean())
        for r in rows:                                     # and never further than the slack
            self.assertLessEqual(abs(r["hi"] - r["raw_hi"]), 0.1 * r["raw_span"] + 0.01)
            self.assertLessEqual(abs(r["lo"] - r["raw_lo"]), 0.1 * r["raw_span"] + 0.01)
        self.assertEqual([r["frame"] for r in rows if r["reset"]], ["frame_000010.tiff"])
        self.assertEqual(report["summary"]["resets"], 1)
        # After the snap the window is the frame's own.
        self.assertEqual(rows[10]["lo"], round(rows[10]["raw_lo"], 2))
        self.assertLess(rows[10]["saturated"], 0.05)

    def test_frame_scope_uses_each_frames_own_percentiles(self):
        rec = self.record("frame", [1000, 3000, 5000])
        report = prep_record(rec, rec / "etiketlenecek", rec / "prep", Params(scope="frame"),
                             preview=0, progress=False)
        for row in report["rows"]:
            self.assertEqual(row["lo"], round(row["raw_lo"], 2))
            self.assertEqual(row["hi"], round(row["raw_hi"], 2))
            self.assertFalse(row["reset"])

    def test_global_scope_maps_every_frame_through_one_window(self):
        rec = self.record("global", [1000, 3000, 5000])
        report = prep_record(rec, rec / "etiketlenecek", rec / "prep", Params(scope="global"),
                             preview=0, progress=False)
        windows = {(r["lo"], r["hi"]) for r in report["rows"]}
        self.assertEqual(len(windows), 1)
        lo, hi = report["summary"]["window"]
        self.assertLess(lo, 1500)
        self.assertGreater(hi, 4500)
        # The cold frame is dark and the hot one bright, under the same window.
        cold = cv2.imread(str(rec / "prep" / "frame_000000.png"), cv2.IMREAD_UNCHANGED)
        hot = cv2.imread(str(rec / "prep" / "frame_000002.png"), cv2.IMREAD_UNCHANGED)
        self.assertLess(np.median(cold), 100)
        self.assertGreater(np.median(hot), 155)

    def test_measure_writes_nothing(self):
        rec = self.record("measure", [1000, 1000])
        self.assertEqual(main([str(rec), "--quiet", "--measure"]), 0)
        self.assertFalse((rec / "prep").exists())

    def test_clahe_median_and_jpg_paths_run(self):
        rec = self.record("options", [1000, 1000, 1000])
        self.assertEqual(main([str(rec), "--quiet", "--clahe", "2.0", "--median", "3", "--ext", "jpg",
                               "--gamma", "0.8", "--invert", "--preview", "2"]), 0)
        self.assertEqual(len(list((rec / "prep").glob("*.jpg"))), 3)
        body = json.loads((rec / "prep" / "prep.json").read_text())
        self.assertEqual(body["ext"], "jpg")
        self.assertTrue(body["params"]["invert"])

    def test_eight_bit_three_channel_sources_are_taken_as_grey(self):
        folder = self.root / "rgb" / "etiketlenecek"
        folder.mkdir(parents=True)
        grey = frame(120, 30, dtype=np.uint8)
        cv2.imwrite(str(folder / "frame_000000.png"), cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR))
        img = read_frame(folder / "frame_000000.png")
        self.assertEqual(img.ndim, 2)
        self.assertEqual(img.dtype, np.uint8)
        np.testing.assert_array_equal(img, grey)
        report = prep_record(self.root / "rgb", folder, self.root / "rgb" / "prep", Params(),
                             preview=0, progress=False)
        self.assertEqual(report["dtype"], "uint8")

    def test_limit_takes_the_first_frames_only(self):
        rec = self.record("limit", [1000] * 5)
        self.assertEqual(main([str(rec), "--quiet", "--limit", "2", "--preview", "0"]), 0)
        self.assertEqual(len(list((rec / "prep").glob("*.png"))), 2)

    def test_explicit_out_folder(self):
        rec = self.record("explicit", [1000])
        out = self.root / "elsewhere"
        self.assertEqual(main([str(rec), "--quiet", "--preview", "0", "--out", str(out)]), 0)
        self.assertTrue((out / "frame_000000.png").is_file())
        self.assertFalse((rec / "prep").exists())

    def test_fractions_are_read_as_percent(self):
        rec = self.record("fractions", [1000])
        self.assertEqual(main([str(rec), "--quiet", "--preview", "0", "--lo", "0.01", "--hi", "0.99"]), 0)
        body = json.loads((rec / "prep" / "prep.json").read_text())
        self.assertEqual((body["params"]["lo"], body["params"]["hi"]), (1.0, 99.0))


@unittest.skipIf(cv2 is None, "OpenCV is not installed")
class DumpTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_a_dump_is_read_at_its_shape_with_the_header_skipped(self):
        img = frame(2000, shape=(48, 64))
        path = self.root / "frame.raw"
        path.write_bytes(b"HDR!" * 8 + img.astype("<u2").tobytes())
        back = read_frame(path, RawSpec(64, 48, "<u2"))
        np.testing.assert_array_equal(back, img)
        big = read_frame(self.root / "be.raw", RawSpec(64, 48, ">u2")) if (
            (self.root / "be.raw").write_bytes(img.astype(">u2").tobytes()) or True) else None
        np.testing.assert_array_equal(big, img)
        self.assertTrue(big.dtype.isnative)
        with self.assertRaises(ValueError):
            read_frame(path, RawSpec(64, 480, "<u2"))
        with self.assertRaises(SystemExit):
            read_frame(path)

    def test_a_known_size_is_inferred(self):
        path = self.root / "frame.raw"
        path.write_bytes(np.zeros(1280 * 768, dtype="<u2").tobytes())
        self.assertEqual(infer_raw_spec(path), RawSpec(1280, 768, "<u2", 0))
        (self.root / "odd.raw").write_bytes(b"\0" * 1234)
        self.assertIsNone(infer_raw_spec(self.root / "odd.raw"))

    def test_a_record_of_dumps_end_to_end(self):
        folder = self.root / "rec" / "etiketlenecek"
        folder.mkdir(parents=True)
        for index in range(3):
            (folder / f"frame_{index:06d}.raw").write_bytes(
                frame(3000, shape=(48, 64), seed=index).astype("<u2").tobytes())
        self.assertEqual(main([str(self.root / "rec"), "--quiet", "--preview", "0",
                               "--raw-shape", "64", "48"]), 0)
        body = json.loads((self.root / "rec" / "prep" / "prep.json").read_text())
        self.assertEqual(body["reader"], "dump")
        self.assertEqual(body["raw"], {"width": 64, "height": 48, "dtype": "<u2", "offset": None})
        self.assertEqual(body["shape"], [48, 64])
        self.assertEqual(len(list((self.root / "rec" / "prep").glob("*.png"))), 3)


if __name__ == "__main__":
    unittest.main()
