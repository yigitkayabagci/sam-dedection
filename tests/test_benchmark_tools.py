"""Tests for the Orin benchmark tooling.

None of these need CUDA, TensorRT or the EdgeTAM weights — they cover the
plumbing that would otherwise only break once you are already sitting in
front of the Jetson.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.jetson_doctor import _version_tuple, run_probe  # noqa: E402
from tools.benchmark import (  # noqa: E402
    AttrTimer,
    ModuleTimer,
    build_frame_cache,
    build_synthetic_cache,
    csv_table,
    latex_table,
    load_matrix,
    markdown_table,
    variant_kwargs,
    variant_runnable,
    _percentile,
)
from tools.calibration import IMG_MEAN, IMG_STD, calibration_batches, preprocess_rgb  # noqa: E402
from tools.compare_masks import compare, iou, load_dump, unpack  # noqa: E402


def _write_video(path: Path, frames: int = 6, size: tuple[int, int] = (64, 48)) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    for i in range(frames):
        img = np.full((size[1], size[0], 3), i * 20 % 256, dtype=np.uint8)
        writer.write(img)
    writer.release()


def _write_dump(path: Path, variant: str, shift: int = 0,
                height: int = 24, width: int = 61, frames: int = 6) -> None:
    """Mirror what benchmark.py writes. Width is odd on purpose: packbits pads
    to a byte boundary and unpacking must not leak those padding bits."""
    packed = []
    for _ in range(frames):
        stack = np.zeros((2, height, width), dtype=bool)
        for o in range(2):
            stack[o, 2 + o * 8:12 + o * 8, 5 + o * 10 + shift:25 + o * 10 + shift] = True
        packed.append(np.packbits(stack, axis=-1))
    np.savez_compressed(
        path, masks=np.stack(packed), frames=np.arange(frames, dtype=np.int32),
        obj_ids=np.array([1, 2], dtype=np.int32),
        width=np.int32(width), height=np.int32(height), variant=variant,
    )


class TestBenchMatrix(unittest.TestCase):
    def setUp(self):
        self.defaults, self.matrix = load_matrix(ROOT / "configs" / "bench_matrix.yaml")

    def test_matrix_is_not_empty(self):
        self.assertGreater(len(self.matrix), 5)
        self.assertIn("pytorch_fp32", self.matrix)
        self.assertIn("trt_fp16", self.matrix)

    def test_every_variant_builds_a_tracker(self):
        from src.trackers import available_trackers, build_tracker

        known = available_trackers()
        for name, spec in self.matrix.items():
            with self.subTest(variant=name):
                tracker, kwargs = variant_kwargs(self.defaults, spec)
                self.assertIn(tracker, known)
                # `note` is documentation, it must never reach the constructor.
                self.assertNotIn("note", kwargs)
                build_tracker(tracker, **{**kwargs, "device": "cpu",
                                          "precision": "float32"})

    def test_trt_variants_are_strict(self):
        # A non-strict TRT variant silently falls back to PyTorch and would be
        # reported as a TensorRT result. That must not be possible from the matrix.
        for name, spec in self.matrix.items():
            if spec.get("tracker") == "edgetam_trt":
                with self.subTest(variant=name):
                    self.assertTrue(spec.get("strict"), f"{name} must set strict: true")

    def test_missing_engine_is_reported_not_crashed(self):
        tracker, kwargs = variant_kwargs(self.defaults, self.matrix["trt_fp16"])
        kwargs["checkpoint"] = __file__  # pretend the checkpoint exists
        kwargs["engine_path"] = "/tmp/definitely_missing.engine"
        ok, reason = variant_runnable(tracker, kwargs)
        self.assertFalse(ok)
        self.assertIn("engine missing", reason)

    def test_paths_are_resolved_to_absolute(self):
        _, kwargs = variant_kwargs(self.defaults, self.matrix["trt_fp16"])
        self.assertTrue(Path(kwargs["checkpoint"]).is_absolute())
        self.assertTrue(Path(kwargs["engine_path"]).is_absolute())
        # model_cfg is a Hydra config NAME resolved by the sam2 package, so it
        # must stay relative.
        self.assertFalse(Path(kwargs["model_cfg"]).is_absolute())


class TestReportRendering(unittest.TestCase):
    ROWS = [
        {"variant": "trt_fp16", "fps_median": 41.2, "encoder_ms": 9.0,
         "rest_ms": 15.0, "encoder_share_pct": 37.5, "peak_torch_mb": 1800.0},
        {"variant": "pytorch_bf16", "fps_median": 22.0, "encoder_ms": None,
         "rest_ms": None, "encoder_share_pct": None, "peak_torch_mb": 2100.0},
        {"variant": "trt_int8", "error": "engine build failed"},
    ]

    def test_markdown_orders_by_fps_and_tolerates_none(self):
        table = markdown_table(self.ROWS).splitlines()
        self.assertIn("| variant |", table[0])
        self.assertTrue(table[2].startswith("| trt_fp16 |"))
        self.assertTrue(table[3].startswith("| pytorch_bf16 |"))
        self.assertIn("| - |", table[3])          # encoder_ms is None
        self.assertTrue(table[4].startswith("| trt_int8 |"))  # failed run, no fps

    def test_csv_has_variant_first(self):
        self.assertTrue(csv_table(self.ROWS).splitlines()[0].startswith("variant,"))

    def test_percentile(self):
        vals = [float(v) for v in range(1, 101)]
        self.assertAlmostEqual(_percentile(vals, 50), 50.5)
        self.assertAlmostEqual(_percentile(vals, 100), 100.0)
        self.assertEqual(_percentile([], 95), 0.0)

    def test_latex_escapes_and_renders_missing_values(self):
        tex = latex_table(self.ROWS)
        self.assertIn(r"\begin{tabular}", tex)
        self.assertIn(r"\toprule", tex)
        self.assertIn(r"trt\_fp16", tex)        # underscore escaped
        self.assertIn(r"enc \%", tex)           # percent escaped
        self.assertIn("--", tex)                # None rendered as an em-dash cell
        # One column spec letter per column, left-aligned label then numbers.
        self.assertRegex(tex, r"\\begin\{tabular\}\{lr+\}")


class TestVersionComparison(unittest.TestCase):
    """jetson_doctor compares dependency versions against requirements.txt.
    String comparison gets 4.11 vs 4.8 backwards, which would fire a bogus
    'too old' warning on a perfectly current OpenCV."""

    def test_numeric_not_lexical(self):
        self.assertLess(_version_tuple("4.8"), _version_tuple("4.11.0"))
        self.assertLess(_version_tuple("3.5.1"), _version_tuple("3.7"))
        self.assertLess(_version_tuple("5.4.1"), _version_tuple("6.0"))
        self.assertGreater(_version_tuple("1.26.4"), _version_tuple("1.24"))

    def test_stops_at_the_first_non_numeric_chunk(self):
        # Real versions seen on Jetson: "10.3.0.30-1+cuda12.5", "2.8.0a0+git1234"
        self.assertEqual(_version_tuple("10.3.0.30-1+cuda12.5"), (10, 3, 0, 30, 1))
        self.assertEqual(_version_tuple("2.8.0a0"), (2, 8))
        self.assertEqual(_version_tuple("?"), ())


class TestDoctorProbe(unittest.TestCase):
    """Importing torch or timm prints warnings on both streams, before and
    after whatever the probe wanted to say. Reading the answer by line index
    picked up a warning instead -- on a real Orin the doctor reported timm's
    version as '(Triggered internally at .../tensor_numpy.cpp:82.)'."""

    def test_answer_survives_noise_on_both_streams(self):
        result = run_probe(
            "import sys\n"
            "print('UserWarning: Failed to initialize NumPy: _ARRAY_API not found')\n"
            "sys.stderr.write('a warning on stderr\\n')\n"
            "emit(dict(version='1.0.28', file='/x/timm/__init__.py'))\n"
            "print('(Triggered internally at tensor_numpy.cpp:82.)')\n"
            "sys.stderr.write('trailing stderr noise\\n')\n"
        )
        self.assertEqual(result, {"version": "1.0.28", "file": "/x/timm/__init__.py"})

    def test_returns_none_when_the_snippet_raises(self):
        self.assertIsNone(run_probe("import definitely_not_a_real_module\n"))

    def test_returns_none_when_nothing_is_emitted(self):
        self.assertIsNone(run_probe("print('chatty but useless')\n"))


class _FakeBlock:
    def __init__(self, ms: float) -> None:
        self.ms = ms

    def forward(self, *args, **kwargs):
        import time
        time.sleep(self.ms / 1000.0)
        return "out"

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class _FakePredictor:
    def __init__(self) -> None:
        self.image_encoder = _FakeBlock(4.0)
        self.memory_attention = _FakeBlock(2.0)
        self.sam_mask_decoder = _FakeBlock(1.0)
        # memory_encoder / sam_prompt_encoder absent on purpose: the timer must
        # tolerate a predictor that does not expose every block.


class TestModuleTimer(unittest.TestCase):
    def _run(self, frames: int = 8, warmup: int = 3) -> tuple[dict, _FakePredictor]:
        predictor = _FakePredictor()
        timer = ModuleTimer(predictor, device="cpu")
        for i in range(frames):
            if i == warmup:
                timer.mark()
            predictor.image_encoder(1)
            predictor.memory_attention(1)
            predictor.sam_mask_decoder(1)
            if i % 2 == 0:
                predictor.sam_mask_decoder(1)  # some frames decode twice
        return timer.stop(), predictor

    def test_only_post_warmup_calls_are_reported(self):
        result, _ = self._run(frames=8, warmup=3)
        self.assertEqual(len(result["image_encoder"]), 5)
        self.assertEqual(len(result["memory_attention"]), 5)

    def test_blocks_called_twice_a_frame_are_all_counted(self):
        result, _ = self._run(frames=8, warmup=3)
        self.assertEqual(len(result["sam_mask_decoder"]), 7)

    def test_absent_blocks_are_skipped(self):
        result, _ = self._run()
        self.assertNotIn("sam_prompt_encoder", result)
        self.assertNotIn("memory_encoder", result)

    def test_forwards_are_restored(self):
        _, predictor = self._run()
        for name in ("image_encoder", "memory_attention", "sam_mask_decoder"):
            forward = getattr(predictor, name).forward
            self.assertTrue(forward.__qualname__.startswith("_FakeBlock.forward"),
                            f"{name}.forward was not restored: {forward}")

    def test_relative_costs_are_preserved(self):
        result, _ = self._run()
        self.assertGreater(sum(result["image_encoder"]),
                           sum(result["memory_attention"]))


class TestAttrTimer(unittest.TestCase):
    def test_restores_a_class_method_without_leaving_a_shadow(self):
        class Thing:
            def work(self):
                return "done"

        thing = Thing()
        self.assertNotIn("work", thing.__dict__)
        timer = AttrTimer(thing, "work")
        thing.work()
        calls = timer.stop()
        self.assertEqual(len(calls), 1)
        self.assertEqual(thing.work(), "done")
        # The instance must be left exactly as found, not carrying a bound copy.
        self.assertNotIn("work", thing.__dict__)

    def test_restores_a_module_level_function(self):
        import types

        module = types.ModuleType("fake_mod")
        module.load = lambda x: x * 2
        original = module.load
        timer = AttrTimer(module, "load")
        self.assertEqual(module.load(3), 6)
        self.assertEqual(len(timer.stop()), 1)
        self.assertIs(module.load, original)

    def test_mark_drops_earlier_calls(self):
        class Thing:
            def work(self):
                return 1

        thing = Thing()
        timer = AttrTimer(thing, "work")
        for _ in range(3):
            thing.work()
        timer.mark()
        for _ in range(5):
            thing.work()
        self.assertEqual(len(timer.stop()), 5)


class TestSweepReporting(unittest.TestCase):
    """The sweep varies the INPUT with the model held fixed. Its headline claim
    is that inference stays flat while preprocess/postprocess grow, so the
    rendering has to survive a failed condition without dropping the rest."""

    ROWS = [
        {"condition": "640x480", "fps": 14.8, "frame_mean_ms": 67.6,
         "preprocess_ms": 3.1, "inference_ms": 58.0, "postprocess_ms": 1.2,
         "unaccounted_ms": 5.3, "encoder_ms": 31.0, "frames_measured": 275},
        {"condition": "3840x2160", "fps": 7.2, "frame_mean_ms": 138.9,
         "preprocess_ms": 48.0, "inference_ms": 58.3, "postprocess_ms": 21.4,
         "unaccounted_ms": 11.2, "encoder_ms": 31.2, "frames_measured": 275},
        {"condition": "broken", "error": "CUDA out of memory"},
    ]

    def test_markdown_keeps_a_failed_condition_visible(self):
        from tools.sweep import markdown

        lines = markdown(self.ROWS).splitlines()
        self.assertTrue(lines[2].startswith("| 640x480 |"))
        self.assertTrue(lines[-1].startswith("| broken |"))
        self.assertIn("| - |", lines[-1])

    def test_latex_uses_em_dash_and_escapes_only_the_label(self):
        from tools.sweep import latex

        tex = latex(self.ROWS)
        self.assertIn(r"\begin{tabular}", tex)
        self.assertRegex(tex, r"\\begin\{tabular\}\{lr+\}")
        self.assertIn("--", tex)                       # missing cells
        self.assertIn("fps min", tex)

    def test_csv_leads_with_condition(self):
        from tools.sweep import csv

        self.assertTrue(csv(self.ROWS).splitlines()[0].startswith("condition,"))

    def test_chart_skips_conditions_with_no_timing(self):
        from tools.sweep import stage_chart

        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")
        with tempfile.TemporaryDirectory() as tmp:
            out = stage_chart(self.ROWS, Path(tmp) / "stages.png", "resolution")
            self.assertIsNotNone(out)
            self.assertGreater(Path(out).stat().st_size, 0)

    def test_condition_settings_map_each_axis_to_the_right_knob(self):
        from tools.sweep import condition_settings

        class Args:
            width, height, box, frames = 1920, 1080, 96, 300

        res = condition_settings("resolution", "640x480", Args)
        self.assertEqual((res["width"], res["height"]), (640, 480))
        self.assertIsNone(res["image_size"])

        img = condition_settings("image-size", "512", Args)
        self.assertEqual(img["image_size"], 512)
        # image_size must not disturb the source resolution: they are separate
        # knobs driving separate stages.
        self.assertEqual((img["width"], img["height"]), (1920, 1080))

        box = condition_settings("box", "400", Args)
        self.assertEqual(box["box"], 400)
        self.assertEqual((box["width"], box["height"]), (1920, 1080))


class TestSyntheticFrames(unittest.TestCase):
    def test_frames_are_unique_and_prompt_lands_on_the_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, prompts = build_synthetic_cache(
                Path(tmp) / "syn", frames=10, width=320, height=240, box=48)
            files = sorted(out.glob("*.jpg"))
            self.assertEqual(len(files), 10)

            # No two frames may be identical: a duplicated frame gives the
            # tracker a stationary target that barely exercises the memory
            # bank. Hash the whole file -- the backdrop is deliberately shared
            # between frames, so only the region around the moving target
            # differs and any prefix check would compare identical bytes.
            digests = {hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
            self.assertEqual(len(digests), len(files))

            self.assertEqual(len(prompts.boxes), 1)
            box = prompts.boxes[0]
            self.assertEqual(box.frame_idx, 0)
            x1, y1, x2, y2 = (int(v) for v in box.xyxy)
            frame = cv2.imread(str(files[0]))
            patch = frame[y1:y2, x1:x2]
            self.assertGreater(patch.size, 0)
            # The target is drawn bright green-yellow against a muted backdrop.
            self.assertGreater(patch[:, :, 1].mean(), frame[:, :, 1].mean() + 40)

    def test_target_actually_moves(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, _ = build_synthetic_cache(
                Path(tmp) / "syn", frames=12, width=320, height=240, box=40)
            files = sorted(out.glob("*.jpg"))

            def centroid(path):
                img = cv2.imread(str(path))
                # The target is the brightest green region.
                mask = img[:, :, 1] > 180
                ys, xs = np.nonzero(mask)
                return (xs.mean(), ys.mean()) if xs.size else (0.0, 0.0)

            first, last = centroid(files[0]), centroid(files[-1])
            moved = abs(first[0] - last[0]) + abs(first[1] - last[1])
            self.assertGreater(moved, 20, "a stationary target barely exercises "
                                          "the memory bank")


class TestFrameCache(unittest.TestCase):
    def test_video_cache_is_capped_and_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            video = tmp / "clip.mp4"
            _write_video(video, frames=6)
            cache = build_frame_cache(str(video), None, "*.tif*", tmp / "cache", 4)
            names = sorted(p.name for p in cache.glob("*.jpg"))
            self.assertEqual(names, ["00000.jpg", "00001.jpg", "00002.jpg", "00003.jpg"])
            # A second call must reuse rather than re-decode.
            again = build_frame_cache(str(video), None, "*.tif*", cache, None)
            self.assertEqual(len(list(again.glob("*.jpg"))), 4)


class TestCalibration(unittest.TestCase):
    def test_preprocessing_matches_sam2_normalization(self):
        img = np.zeros((30, 40, 3), dtype=np.uint8)
        img[..., 0] = 255  # pure red
        out = preprocess_rgb(img, 64)
        self.assertEqual(out.shape, (3, 64, 64))
        self.assertEqual(out.dtype, np.float32)
        means = out.mean(axis=(1, 2))
        self.assertAlmostEqual(float(means[0]), (1.0 - IMG_MEAN[0]) / IMG_STD[0], places=4)
        self.assertAlmostEqual(float(means[1]), (0.0 - IMG_MEAN[1]) / IMG_STD[1], places=4)

    def test_batches_respect_limit_and_stride(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "clip.mp4"
            _write_video(video, frames=12)
            batches = list(calibration_batches(video=str(video), size=32, limit=5))
            self.assertEqual(len(batches), 5)
            self.assertEqual(batches[0].shape, (1, 3, 32, 32))
            strided = list(calibration_batches(video=str(video), size=32,
                                               limit=3, stride=4))
            self.assertEqual(len(strided), 3)

    def test_requires_exactly_one_source(self):
        with self.assertRaises(ValueError):
            list(calibration_batches(video="a.mp4", frames_dir="b/"))
        with self.assertRaises(ValueError):
            list(calibration_batches())


class TestMaskComparison(unittest.TestCase):
    def test_packbits_roundtrip_at_odd_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ref.npz"
            _write_dump(path, "ref", width=61)
            dump = load_dump(path)
            masks = unpack(dump["packed"][0], dump["width"])
            self.assertEqual(masks.shape, (2, 24, 61))
            self.assertEqual(int(masks.sum()), 2 * 10 * 20)

    def test_identical_dumps_score_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.npz", Path(tmp) / "b.npz"
            _write_dump(a, "pytorch_fp32")
            _write_dump(b, "trt_fp16")
            result = compare(load_dump(a), load_dump(b))
            self.assertAlmostEqual(result["mean_iou"], 1.0)
            self.assertEqual(result["frames_compared"], 6)

    def test_shifted_masks_lose_iou(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.npz", Path(tmp) / "b.npz"
            _write_dump(a, "pytorch_fp32", shift=0)
            _write_dump(b, "trt_int8", shift=4)
            result = compare(load_dump(a), load_dump(b))
            self.assertAlmostEqual(result["mean_iou"], 16 / 24, places=4)
            self.assertEqual(result["iou_below_0.95_pct"], 100.0)

    def test_two_empty_masks_agree(self):
        empty = np.zeros((4, 4), dtype=bool)
        self.assertEqual(iou(empty, empty), 1.0)

    def test_rejects_a_foreign_npz(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "junk.npz"
            np.savez(path, something=np.zeros(3))
            with self.assertRaises(SystemExit):
                load_dump(path)


class TestToolEntryPoints(unittest.TestCase):
    """Every tool must produce a useful message rather than a traceback when
    TensorRT is absent — that is the normal state on a dev machine."""

    def test_build_trt_engine_help(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "tools/build_trt_engine.py"), "--help"],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0)
        self.assertIn("--precision", proc.stdout)

    def test_build_trt_engine_without_tensorrt_explains_itself(self):
        try:
            import tensorrt  # noqa: F401
            self.skipTest("tensorrt IS installed; cannot test the missing path.")
        except ImportError:
            pass
        proc = subprocess.run(
            [sys.executable, str(ROOT / "tools/build_trt_engine.py"),
             "--onnx", "/tmp/nope.onnx", "--engine", "/tmp/nope.engine"],
            capture_output=True, text=True, timeout=120)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("system-site-packages", proc.stdout + proc.stderr)

    def test_jetson_doctor_json_mode(self):
        import json

        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts/jetson_doctor.py"), "--json"],
            capture_output=True, text=True, timeout=600)
        payload = json.loads(proc.stdout)
        self.assertIn("items", payload)
        self.assertTrue(all({"section", "level", "name"} <= set(i) for i in payload["items"]))
        # Exit code must signal blockers so CI / a shell script can gate on it.
        self.assertEqual(proc.returncode, 1 if payload["blockers"] else 0)


if __name__ == "__main__":
    unittest.main()
