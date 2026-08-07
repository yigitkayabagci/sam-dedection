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

    def test_tail_is_reported_and_excludes_warmup(self):
        from src.metrics import fps_summary

        # A slow warm-up frame, then a steady band with one 60 ms spike.
        dt = [1.0] + [0.030] * 97 + [0.060, 0.032]
        s = fps_summary(dt, warmup=1)
        self.assertAlmostEqual(s["max_ms"], 60.0, places=6)
        self.assertAlmostEqual(s["p50_ms"], 30.0, places=6)
        # The 1000 ms warm-up frame must not be the max; that is the whole
        # reason the tail is computed on the post-warmup slice.
        self.assertLess(s["max_ms"], 100.0)

    def test_deadline_counts_only_the_frames_that_missed_it(self):
        from src.metrics import fps_summary

        dt = [0.030] * 8 + [0.040, 0.050]  # 33.3 ms slot at 30 FPS
        s = fps_summary(dt, warmup=0, deadline_ms=1000.0 / 30.0)
        self.assertEqual(s["missed"], 2)
        self.assertAlmostEqual(s["deadline_ms"], 33.333333, places=4)
        # No deadline given -> the tail is still measured, just not judged.
        self.assertEqual(fps_summary(dt)["missed"], 0)
        self.assertAlmostEqual(fps_summary(dt)["max_ms"], 50.0, places=6)

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
                    encode=None, read=None, source=None):
            captured.update(per_frame_dt=per_frame_dt, encode=encode, source=source)

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


class TestFrameSkipping(unittest.TestCase):
    """`realtime_fps` must cost frames, not correctness.

    A tracker slower than the source has to leave frames untracked -- that is
    the point -- but the run still has to end on the last frame of the clip and
    the mp4 still has to come out the length of the clip. Those two are what
    separate dropping stale frames from simply falling behind.
    """

    class _SlowTracker:
        """A tracker that takes `work_s` per frame and records what it saw."""

        name, precision = "fake", "float32"

        def __init__(self, shape, work_s):
            self.shape, self.work_s = shape, work_s
            self.tracked: list[int] = []

        def prepare(self, _root): pass

        def set_prompts(self, _p): pass

        def reset(self): pass

        def propagate(self):
            raise AssertionError("realtime_fps must route through propagate_frames")

        def propagate_frames(self, frame_indices):
            import time as _time
            from src.trackers import TrackingResult

            for idx in frame_indices:
                _time.sleep(self.work_s)
                self.tracked.append(idx)
                mask = np.zeros(self.shape, dtype=bool)
                mask[1:3, 1:3] = True
                yield TrackingResult(frame_idx=idx, masks={1: mask})

    class _ColdStartTracker:
        """Slow for its first `cold_frames` calls, fast after -- a stand-in for
        CUDA graph capture / kernel autotuning on a real backend's first calls."""

        name, precision = "fake", "float32"

        def __init__(self, shape, cold_frames, cold_work_s, hot_work_s):
            self.shape = shape
            self.cold_frames, self.cold_work_s, self.hot_work_s = (
                cold_frames, cold_work_s, hot_work_s)
            self.tracked: list[int] = []
            self.calls = 0

        def prepare(self, _root): pass

        def set_prompts(self, _p): pass

        def reset(self): pass

        def propagate(self):
            raise AssertionError("realtime_fps must route through propagate_frames")

        def propagate_frames(self, frame_indices):
            import time as _time
            from src.trackers import TrackingResult

            for idx in frame_indices:
                _time.sleep(self.cold_work_s if self.calls < self.cold_frames
                            else self.hot_work_s)
                self.calls += 1
                self.tracked.append(idx)
                mask = np.zeros(self.shape, dtype=bool)
                mask[1:3, 1:3] = True
                yield TrackingResult(frame_idx=idx, masks={1: mask})

    def _run(self, frames=24, fps=300.0, work_s=0.01, fps_warmup=0, tracker=None):
        from contextlib import contextmanager
        import cv2
        import src.pipeline as P

        written: list[int] = []
        captured = {}
        real_writer, real_report = P.open_video_writer, P._report_timing

        @contextmanager
        def counting_writer(*_a, **_k):
            yield lambda frame: written.append(1)

        def capture(cfg, tracker, meta, prompts, per_frame_dt, pre, infer, post,
                    encode=None, read=None, source=None):
            captured.update(per_frame_dt=per_frame_dt, source=source)

        tracker = tracker if tracker is not None else self._SlowTracker((8, 12), work_s)
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for i in range(frames):
                cv2.imwrite(str(d / f"frame_{i:06d}.tiff"),
                            np.full((8, 12, 3), (10 * i) % 256, dtype=np.uint8))
            P.open_video_writer, P._report_timing = counting_writer, capture
            try:
                P.run(tracker,
                      PromptSet(boxes=[BoxPrompt(1, 0, (1, 1, 3, 3))]),
                      PipelineConfig(output_path=d / "out.mp4", frames_dir=d,
                                     frame_pattern="*.tif*", realtime_fps=fps,
                                     fps_warmup=fps_warmup))
            finally:
                P.open_video_writer, P._report_timing = real_writer, real_report
        return tracker, written, captured

    def test_slow_tracker_drops_frames_but_still_finishes_on_the_last_one(self):
        frames = 24
        tracker, _written, captured = self._run(frames=frames)

        self.assertLess(len(tracker.tracked), frames, "nothing was dropped")
        self.assertEqual(tracker.tracked, sorted(set(tracker.tracked)))
        # Falling behind would end somewhere in the middle of the clip; keeping
        # only the freshest frame ends on the frame the source stopped at.
        self.assertEqual(tracker.tracked[-1], frames - 1)

        source = captured["source"]
        self.assertEqual(source.delivered, len(tracker.tracked))
        self.assertEqual(source.delivered + source.dropped, frames)

    def test_dropped_frames_still_reach_the_video_holding_the_last_mask(self):
        frames = 24
        tracker, written, _ = self._run(frames=frames)
        self.assertLess(len(tracker.tracked), frames)
        # One written frame per source frame, not per tracked frame -- otherwise
        # the mp4 would be shorter than the clip and play back fast.
        self.assertEqual(len(written), frames)

    def test_a_fast_tracker_keeps_every_frame_and_is_not_charged_for_waiting(self):
        frames = 6
        # A rate the whole loop clears with room to spare -- the tracker costs
        # nothing here, but reading and overlaying each frame still does.
        fps = 20.0
        tracker, written, captured = self._run(frames=frames, fps=fps, work_s=0.0)

        self.assertEqual(tracker.tracked, list(range(frames)))
        self.assertEqual(len(written), frames)
        self.assertEqual(captured["source"].dropped, 0)
        # Each frame spent ~1/fps idle waiting for the source. Charging that to
        # the model would report it as exactly as fast as the source; the budget
        # has to stay near the work actually done.
        self.assertLess(max(captured["per_frame_dt"]), 0.5 / fps)

    def test_warmup_runs_before_the_clock_so_cold_start_is_not_charged_as_drops(self):
        # A model that is 20x too slow for its first 5 calls (cold: CUDA graph
        # capture, kernel autotuning) but comfortably fast after must not lose
        # most of a short clip to that cold start. Racing the clock from frame
        # 0 would: 5 * 200ms = 1s of real time at 100 FPS is ~100 frame
        # opportunities gone before the model is even warm.
        frames, fps, warmup = 60, 100.0, 5
        tracker = self._ColdStartTracker((8, 12), cold_frames=warmup,
                                         cold_work_s=0.02, hot_work_s=0.001)
        tracker, _written, captured = self._run(frames=frames, fps=fps,
                                                fps_warmup=warmup, tracker=tracker)

        # The warm-up frames ran: frames 0..warmup-1, in order, before anything
        # paced by the source.
        self.assertEqual(tracker.tracked[:warmup], list(range(warmup)))
        source = captured["source"]
        # The source only starts once the model is hot, so a fast hot model
        # keeps essentially everything from there on -- nothing close to the
        # ~100-frame wipeout racing the clock from frame 0 would cause.
        self.assertGreater(source.delivered, frames - warmup - 3)
        self.assertEqual(source.delivered + source.dropped, frames - warmup)

        # And the report reflects only the hot steady state: the ~20ms cold
        # calls must not appear in what gets measured at all.
        self.assertLess(max(captured["per_frame_dt"]), 0.01)

    def test_warmup_is_not_subtracted_twice_from_a_realtime_report(self):
        # Regression: fps_warmup used to be re-applied by _report_timing on top
        # of frames _frame_stream had already run untimed and excluded before
        # the source ever started, silently stripping that many *more* off an
        # already-warmup-free list. On a short, heavily-skipped run that could
        # collapse the reported sample down to one point.
        frames, fps, warmup = 30, 50.0, 5
        tracker = self._ColdStartTracker((8, 12), cold_frames=warmup,
                                         cold_work_s=0.01, hot_work_s=0.001)
        _tracker, _written, captured = self._run(frames=frames, fps=fps,
                                                 fps_warmup=warmup, tracker=tracker)
        source = captured["source"]
        # Every delivered (post-warmup) frame must still be in the report --
        # none of them silently dropped a second time by warm-up accounting.
        self.assertEqual(len(captured["per_frame_dt"]), source.delivered)


class TestFixedFrameStride(unittest.TestCase):
    """`frame_stride`: a fixed pattern, no clock, no queue -- the same frames
    skipped on every run regardless of how long any of them took."""

    class _Tracker:
        name, precision = "fake", "float32"

        def __init__(self, shape):
            self.shape = shape
            self.tracked: list[int] = []

        def prepare(self, _root): pass

        def set_prompts(self, _p): pass

        def reset(self): pass

        def propagate(self):
            raise AssertionError("frame_stride must route through propagate_frames")

        def propagate_frames(self, frame_indices):
            from src.trackers import TrackingResult

            for idx in frame_indices:
                self.tracked.append(idx)
                mask = np.zeros(self.shape, dtype=bool)
                mask[1:3, 1:3] = True
                yield TrackingResult(frame_idx=idx, masks={1: mask})

    def _run(self, frames, stride, **cfg_kwargs):
        from contextlib import contextmanager
        import cv2
        import src.pipeline as P

        written: list[int] = []
        real_writer = P.open_video_writer
        tracker = self._Tracker((8, 12))

        @contextmanager
        def counting_writer(*_a, **_k):
            yield lambda frame: written.append(1)

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for i in range(frames):
                cv2.imwrite(str(d / f"frame_{i:06d}.tiff"),
                            np.full((8, 12, 3), (10 * i) % 256, dtype=np.uint8))
            P.open_video_writer = counting_writer
            try:
                P.run(tracker,
                      PromptSet(boxes=[BoxPrompt(1, 0, (1, 1, 3, 3))]),
                      PipelineConfig(output_path=d / "out.mp4", frames_dir=d,
                                     frame_pattern="*.tif*", frame_stride=stride,
                                     **cfg_kwargs))
            finally:
                P.open_video_writer = real_writer
        return tracker, written

    def test_only_every_nth_frame_reaches_the_model(self):
        tracker, _written = self._run(frames=10, stride=2)
        self.assertEqual(tracker.tracked, [0, 2, 4, 6, 8])

    def test_skipped_frames_still_reach_the_video_holding_the_last_mask(self):
        frames = 10
        _tracker, written = self._run(frames=frames, stride=2)
        # Same contract as realtime_fps: one written frame per source frame,
        # not per tracked one, or the mp4 would play back faster than the clip.
        self.assertEqual(len(written), frames)

    def test_stride_one_is_equivalent_to_off(self):
        tracker, _written = self._run(frames=6, stride=1)
        self.assertEqual(tracker.tracked, list(range(6)))

    def test_combining_with_realtime_fps_is_rejected(self):
        with self.assertRaises(ValueError):
            self._run(frames=6, stride=2, realtime_fps=30.0)


if __name__ == "__main__":
    unittest.main()
