"""Tests for the Orin benchmark tooling.

None of these need CUDA, TensorRT or the EdgeTAM weights — they cover the
plumbing that would otherwise only break once you are already sitting in
front of the Jetson.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.benchmark import (  # noqa: E402
    build_frame_cache,
    csv_table,
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
         "rest_ms": 15.0, "peak_torch_mb": 1800.0},
        {"variant": "pytorch_bf16", "fps_median": 22.0, "encoder_ms": None,
         "rest_ms": None, "peak_torch_mb": 2100.0},
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
