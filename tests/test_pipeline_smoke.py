"""Smoke tests that don't require EdgeTAM weights.

Run with:  python -m pytest tests/  (or python -m unittest tests/test_pipeline_smoke.py)
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.pipeline import PipelineConfig, _resolve_video_mode
from src.prompts import BoxPrompt, PointPrompt, PromptSet
from src.prompts.file_source import load_prompts
from src.trackers import available_trackers, build_tracker
from src.visualize import overlay_masks


class TestPrompts(unittest.TestCase):
    def test_promptset_object_ids(self):
        ps = PromptSet(
            boxes=[BoxPrompt(1, 0, (0, 0, 10, 10))],
            points=[PointPrompt(2, 0, (5, 5), 1)],
        )
        self.assertEqual(ps.object_ids(), [1, 2])
        self.assertFalse(ps.is_empty())

    def test_load_prompts_roundtrip(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"boxes": [{"obj_id": 1, "frame_idx": 0, "xyxy": [1, 2, 3, 4]}]}, f)
            path = f.name
        ps = load_prompts(path)
        self.assertEqual(len(ps.boxes), 1)
        self.assertEqual(ps.boxes[0].xyxy, (1.0, 2.0, 3.0, 4.0))
        Path(path).unlink()


class TestRegistry(unittest.TestCase):
    def test_known_trackers_registered(self):
        names = available_trackers()
        self.assertIn("edgetam", names)
        self.assertIn("edgetam_trt", names)
        self.assertIn("efficientsam3", names)

    def test_efficientsam3_stub_raises(self):
        t = build_tracker("efficientsam3")
        with self.assertRaises(NotImplementedError):
            t.prepare("/tmp/whatever")

    def _trt_tracker(self, **kwargs):
        base = dict(
            model_cfg="configs/edgetam.yaml",
            checkpoint="/tmp/does_not_exist.pt",
            device="cpu",
            precision="float32",
        )
        base.update(kwargs)
        return build_tracker("edgetam_trt", **base)

    def test_edgetam_trt_constructs_without_engines(self):
        # Non-strict mode must not raise at construction time even with no
        # engines at all; each module just stays on PyTorch.
        t = self._trt_tracker(
            memory_attention_engine="/tmp/does_not_exist.engine", strict=False
        )
        self.assertFalse(t.strict)
        t._load_engines()
        self.assertEqual(t._engines, {})

    def test_edgetam_trt_accepts_legacy_engine_path(self):
        # `engine_path` was the single-engine config key; it still points at
        # the image encoder so old configs keep working.
        t = self._trt_tracker(engine_path="/tmp/enc.engine")
        self.assertEqual(t.engine_paths["image_encoder"], "/tmp/enc.engine")
        self.assertIsNone(t.engine_paths["memory_attention"])

    def test_edgetam_trt_strict_rejects_missing_engine(self):
        t = self._trt_tracker(
            image_encoder_engine="/tmp/does_not_exist.engine",
            memory_attention_engine="/tmp/does_not_exist.engine",
            memory_encoder_engine="/tmp/does_not_exist.engine",
            sam_head_engine="/tmp/does_not_exist.engine",
            strict=True,
        )
        with self.assertRaises((FileNotFoundError, ImportError)):
            t._load_engines()

    def test_edgetam_trt_defaults_to_fp16(self):
        self.assertEqual(self._trt_tracker(precision="fp16").precision, "float16")
        self.assertEqual(
            build_tracker(
                "edgetam_trt",
                model_cfg="configs/edgetam.yaml",
                checkpoint="/tmp/x.pt",
                device="cpu",
            ).precision,
            "float16",
        )

    def test_precision_aliases(self):
        t = build_tracker(
            "edgetam",
            model_cfg="configs/edgetam.yaml",
            checkpoint="/tmp/x.pt",
            device="cpu",
            precision="bf16",
        )
        self.assertEqual(t.precision, "bfloat16")
        with self.assertRaises(ValueError):
            build_tracker(
                "edgetam",
                model_cfg="configs/edgetam.yaml",
                checkpoint="/tmp/x.pt",
                device="cpu",
                precision="int8",
            )


class TestVisualize(unittest.TestCase):
    def test_overlay_no_mask_returns_same_shape(self):
        frame = np.zeros((10, 20, 3), dtype=np.uint8)
        out = overlay_masks(frame, masks={})
        self.assertEqual(out.shape, frame.shape)

    def test_overlay_with_mask_changes_pixels(self):
        frame = np.zeros((10, 20, 3), dtype=np.uint8)
        mask = np.zeros((10, 20), dtype=bool)
        mask[2:8, 4:16] = True
        out = overlay_masks(frame, {1: mask}, draw_bbox=False)
        self.assertTrue((out[mask] != 0).any())


class TestVideoModeResolution(unittest.TestCase):
    def _cfg(self, path, mode):
        return PipelineConfig(video_path=Path(path), output_path=Path("/tmp/out.mp4"), video_mode=mode)

    def test_jpg_mode_is_forced(self):
        self.assertEqual(_resolve_video_mode(self._cfg("foo.mp4", "jpg")), "jpg")

    def test_mp4_mode_rejects_non_mp4(self):
        with self.assertRaises(ValueError):
            _resolve_video_mode(self._cfg("foo.avi", "mp4"))

    def test_auto_mode_falls_back_when_decord_missing(self):
        # In this environment decord is not installed, so auto -> jpg.
        try:
            import decord  # noqa: F401
            self.skipTest("decord IS installed; can't test fallback path here.")
        except ImportError:
            pass
        self.assertEqual(_resolve_video_mode(self._cfg("foo.mp4", "auto")), "jpg")


class TestFrameSequence(unittest.TestCase):
    def _write_tiff_seq(self, d: Path, n: int = 3) -> None:
        import cv2
        for i in range(n):
            # 16-bit single-channel TIFF, like raw thermal/scientific frames.
            img = np.full((8, 12), i * 1000 + 100, dtype=np.uint16)
            cv2.imwrite(str(d / f"frame_{i:06d}.tiff"), img)

    def test_list_and_load_16bit_grayscale_tiff(self):
        from src.io_utils import list_frame_files, load_frame_rgb8

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write_tiff_seq(d, n=3)
            files = list_frame_files(d, "*.tif*")
            self.assertEqual([f.name for f in files],
                             ["frame_000000.tiff", "frame_000001.tiff", "frame_000002.tiff"])
            rgb = load_frame_rgb8(files[1])
            self.assertEqual(rgb.shape, (8, 12, 3))
            self.assertEqual(rgb.dtype, np.uint8)

    def test_frames_metadata(self):
        from src.io_utils import frames_metadata

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write_tiff_seq(d, n=4)
            meta = frames_metadata(d, "*.tif*", fps=24.0)
            self.assertEqual((meta.width, meta.height, meta.frame_count), (12, 8, 4))
            self.assertEqual(meta.fps, 24.0)

    def test_missing_frames_raises(self):
        from src.io_utils import list_frame_files

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                list_frame_files(tmp, "*.tif*")


class TestFpsMetrics(unittest.TestCase):
    def test_warmup_excluded_from_average(self):
        from src.metrics import fps_summary

        # First frame slow (warm-up), rest fast.
        dt = [1.0, 0.1, 0.1, 0.1, 0.1]
        s = fps_summary(dt, warmup=1)
        self.assertEqual(s["frames"], 5)
        self.assertEqual(s["warmup"], 1)
        self.assertEqual(s["kept_frames"], 4)
        self.assertAlmostEqual(s["avg_fps_post_warmup"], 10.0, places=6)  # 4 / 0.4
        self.assertLess(s["avg_fps_all"], s["avg_fps_post_warmup"])

    def test_warmup_clamped_and_zero(self):
        from src.metrics import fps_summary

        dt = [0.1, 0.1, 0.1]
        # warmup of 0 -> post-warmup avg equals all-frames avg
        s0 = fps_summary(dt, warmup=0)
        self.assertEqual(s0["warmup"], 0)
        self.assertAlmostEqual(s0["avg_fps_post_warmup"], s0["avg_fps_all"])
        # warmup larger than frames is clamped to n-1 (keeps at least 1 frame)
        s = fps_summary(dt, warmup=99)
        self.assertEqual(s["warmup"], 2)
        self.assertEqual(s["kept_frames"], 1)


class TestEncodeStaysOutOfTheBudget(unittest.TestCase):
    """Writing the mp4 must never enter the reported frame time.

    The overlay and the encode exist to make something watchable; a consumer of
    the masks pays neither. They are timed, but on their own counter. This
    pins that down causally instead of by inspection: make writing a frame take
    250 ms and check the budget does not move.
    """

    class _Result:
        def __init__(self, idx, mask):
            self.frame_idx, self.masks = idx, {1: mask}

    class _Tracker:
        """Enough of the VideoTracker surface for _run_frames, no weights."""
        name, precision = "fake", "float32"

        def __init__(self, frames, shape):
            self.frames, self.shape = frames, shape

        def prepare(self, _root): pass

        def set_prompts(self, _p): pass

        def reset(self): pass

        def propagate(self):
            for i in range(self.frames):
                mask = np.zeros(self.shape, dtype=bool)
                mask[1:3, 1:3] = True
                yield TestEncodeStaysOutOfTheBudget._Result(i, mask)

    def _run(self, write_delay_s):
        import time as _time
        from contextlib import contextmanager
        import cv2
        import src.pipeline as P

        captured = {}

        def capture(cfg, tracker, meta, prompts, per_frame_dt, pre, infer, post,
                    encode=None, read=None):
            captured.update(per_frame_dt=per_frame_dt, encode=encode)

        real_writer, real_report = P.open_video_writer, P._report_timing

        @contextmanager
        def slow_writer(*a, **k):
            with real_writer(*a, **k) as emit:
                yield lambda frame: (_time.sleep(write_delay_s), emit(frame))[1]

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for i in range(4):
                cv2.imwrite(str(d / f"frame_{i:06d}.tiff"),
                            np.full((8, 12, 3), 40 * i, dtype=np.uint8))
            P.open_video_writer, P._report_timing = slow_writer, capture
            try:
                P.run(self._Tracker(4, (8, 12)),
                      PromptSet(boxes=[BoxPrompt(1, 0, (1, 1, 3, 3))]),
                      PipelineConfig(output_path=d / "out.mp4", frames_dir=d,
                                     frame_pattern="*.tif*"))
            finally:
                P.open_video_writer, P._report_timing = real_writer, real_report
        return captured

    def test_slow_writer_moves_only_the_encode_counter(self):
        delay_ms = 250.0
        fast, slow = self._run(0.0), self._run(delay_ms / 1000.0)

        budget_fast = max(fast["per_frame_dt"]) * 1000.0
        budget_slow = max(slow["per_frame_dt"]) * 1000.0
        self.assertLess(
            budget_slow, budget_fast + delay_ms / 2,
            f"a {delay_ms:.0f} ms write leaked into the frame budget "
            f"({budget_fast:.1f} -> {budget_slow:.1f} ms)",
        )
        # ...and landed where it belongs, so the test cannot pass by the
        # writer never having been called at all.
        self.assertGreater(min(slow["encode"]), delay_ms)
        self.assertLess(max(fast["encode"]), delay_ms)


if __name__ == "__main__":
    unittest.main()
