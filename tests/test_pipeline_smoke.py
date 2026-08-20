"""Smoke tests that don't require EdgeTAM weights.

Run with:  python -m pytest tests/  (or python -m unittest tests/test_pipeline_smoke.py)
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

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


class TestFrameSkip(unittest.TestCase):
    """Half the inferences, all of the frames.

    Two claims to pin, and only one of them is visible in the output. That the
    output still covers every source frame can be read off the rendered frames;
    that inference did not run on half of them cannot -- a version that inferred
    everything and threw half the masks away would look identical while costing
    exactly as much. So these tests also ask the tracker what `prepare()`
    actually handed it.
    """

    class _Result:
        def __init__(self, idx, mask):
            self.frame_idx, self.masks = idx, {1: mask}

    class _Tracker:
        """Tracks whatever `prepare()` found, the way init_state does."""
        name, precision = "fake", "float32"

        def __init__(self, shape):
            self.shape, self.seen, self.prompt_frames = shape, [], []

        def prepare(self, root):
            self.seen = sorted(Path(root).glob("*.jpg"))

        def set_prompts(self, prompts):
            self.prompt_frames = [b.frame_idx for b in prompts.boxes]

        def reset(self): pass

        def propagate(self):
            for i in range(len(self.seen)):
                # A mask whose area names the inference it came from, so a held
                # frame can be told apart from a freshly inferred one.
                mask = np.zeros(self.shape, dtype=bool)
                mask[1:2 + i, 1:2] = True
                yield TestFrameSkip._Result(i, mask)

    def _run(self, stride, frames=6, prompt_frame=0):
        """Track `frames` constant-valued frames (frame i is all i*40).

        Returns the tracker and, per written output frame, which source frame it
        was drawn on and which inference's mask it carries.
        """
        import cv2
        import src.pipeline as P

        rendered = []
        tracker = self._Tracker((8, 12))
        real_overlay, real_report = P.overlay_masks, P._report_timing
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for i in range(frames):
                cv2.imwrite(str(d / f"frame_{i:06d}.tiff"),
                            np.full((8, 12, 3), i * 40, dtype=np.uint8))
            # The overlay is handed the source frame and the masks drawn on it:
            # the pixel value names the frame, the mask area names the inference.
            P.overlay_masks = lambda rgb, masks, *a, **k: (
                rendered.append((int(rgb[0, 0, 0]), int(masks[1].sum()))),
                real_overlay(rgb, masks, *a, **k))[1]
            P._report_timing = lambda *a, **k: None
            try:
                P.run(tracker,
                      PromptSet(boxes=[BoxPrompt(1, prompt_frame, (1, 1, 3, 3))]),
                      PipelineConfig(output_path=d / "out.mp4", frames_dir=d,
                                     frame_pattern="*.tif*", frame_stride=stride))
            finally:
                P.overlay_masks, P._report_timing = real_overlay, real_report
        return tracker, rendered

    def test_half_the_inferences_cover_every_frame(self):
        tracker, rendered = self._run(stride=2, frames=6)
        self.assertEqual(len(tracker.seen), 3, "model was given more than half the clip")
        # Every source frame is still written, in order (0, 40, ... 200)...
        self.assertEqual([value for value, _ in rendered], [0, 40, 80, 120, 160, 200])
        # ...and the odd ones carry the mask of the inference before them, which
        # is the whole point: 3 inferences, 6 decided frames, no mask invented
        # for a frame the model never saw.
        self.assertEqual([area for _, area in rendered], [1, 1, 2, 2, 3, 3])

    def test_stride_one_leaves_the_clip_alone(self):
        tracker, rendered = self._run(stride=1, frames=6)
        self.assertEqual(len(tracker.seen), 6)
        self.assertEqual([value for value, _ in rendered], [0, 40, 80, 120, 160, 200])
        # One inference per frame: no mask is ever reused.
        self.assertEqual([area for _, area in rendered], [1, 2, 3, 4, 5, 6])

    def test_last_group_is_not_dropped_when_it_is_partial(self):
        # 6 frames at stride 4 -> inferences on 0 and 4; frames 5 has no
        # inference of its own and must still come out, holding frame 4's.
        tracker, rendered = self._run(stride=4, frames=6)
        self.assertEqual(len(tracker.seen), 2)
        self.assertEqual([value for value, _ in rendered], [0, 40, 80, 120, 160, 200])
        self.assertEqual([area for _, area in rendered], [1, 1, 1, 1, 2, 2])

    def test_prompt_lands_on_the_frame_it_was_drawn_on(self):
        # A prompt on source frame 4 must reach the model as inferred frame 2,
        # which is that same image -- not frame 4 of the decimated clip.
        tracker, _ = self._run(stride=2, frames=6, prompt_frame=4)
        self.assertEqual(tracker.prompt_frames, [2])

    def test_uninferred_prompt_frame_snaps_backwards(self):
        from src.pipeline import _decimate_prompts

        ps = PromptSet(boxes=[BoxPrompt(1, 5, (0, 0, 1, 1))],
                       points=[PointPrompt(2, 4, (1, 1), 1)])
        out = _decimate_prompts(ps, 2)
        self.assertEqual(out.boxes[0].frame_idx, 2)   # frame 5 -> 4 -> inferred 2
        self.assertEqual(out.points[0].frame_idx, 2)
        # Everything else about the prompt is untouched.
        self.assertEqual(out.boxes[0].xyxy, (0, 0, 1, 1))

    def test_tracked_cache_holds_only_the_inferred_frames(self):
        import cv2
        from src.pipeline import _tracked_cache

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for i in range(6):
                cv2.imwrite(str(d / f"{i:05d}.jpg"),
                            np.full((8, 12, 3), i * 40, dtype=np.uint8))
            files = sorted(d.glob("*.jpg"))
            out = _tracked_cache(d, files[::2])
            self.assertEqual([p.name for p in sorted(out.glob("*.jpg"))],
                             ["00000.jpg", "00001.jpg", "00002.jpg"])
            # The full cache is untouched -- the overlay still needs every frame.
            self.assertEqual(len(sorted(d.glob("*.jpg"))), 6)
            # Renumbering must not reshuffle the content: 0, 2, 4 in order.
            values = [int(cv2.imread(str(p))[0, 0, 0]) for p in sorted(out.glob("*.jpg"))]
            for got, want in zip(values, [0, 80, 160]):
                self.assertLess(abs(got - want), 6, f"{values} is not [0, 80, 160]")

    def test_jpg_mode_gives_the_model_only_the_inferred_frames(self):
        # The other input path: extract_frames dumps every frame, and init_state
        # reads a directory, so the skipped frames have to be out of the one it
        # is given -- while staying available to the overlay.
        import src.pipeline as P
        from src.io_utils import open_video_writer

        tracker = self._Tracker((32, 32))
        real_report = P._report_timing
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            with open_video_writer(d / "in.mp4", 30.0, (32, 32)) as emit:
                for i in range(6):
                    emit(np.full((32, 32, 3), i * 40, dtype=np.uint8))
            P._report_timing = lambda *a, **k: None
            try:
                P.run(tracker, PromptSet(boxes=[BoxPrompt(1, 0, (1, 1, 3, 3))]),
                      PipelineConfig(output_path=d / "out.mp4", video_path=d / "in.mp4",
                                     video_mode="jpg", frame_stride=2))
            finally:
                P._report_timing = real_report
        self.assertEqual(len(tracker.seen), 3, "model was given more than half the clip")
        self.assertEqual([p.name for p in tracker.seen],
                         ["00000.jpg", "00001.jpg", "00002.jpg"])

    def test_output_keeps_the_source_length_and_rate(self):
        import cv2
        import src.pipeline as P

        tracker = self._Tracker((32, 32))
        real_report = P._report_timing
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for i in range(6):
                cv2.imwrite(str(d / f"frame_{i:06d}.tiff"),
                            np.full((32, 32, 3), i * 40, dtype=np.uint8))
            P._report_timing = lambda *a, **k: None
            try:
                P.run(tracker, PromptSet(boxes=[BoxPrompt(1, 0, (1, 1, 3, 3))]),
                      PipelineConfig(output_path=d / "out.mp4", frames_dir=d,
                                     frame_pattern="*.tif*", fps=30.0, frame_stride=2))
            finally:
                P._report_timing = real_report
            cap = cv2.VideoCapture(str(d / "out.mp4"))
            try:
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(cap.get(cv2.CAP_PROP_FPS))
            finally:
                cap.release()
        # 3 inferences, but the clip is the source's: 6 frames at 30 fps.
        self.assertEqual(count, 6)
        self.assertAlmostEqual(fps, 30.0, places=3)

    def test_spread_shares_a_cost_without_inventing_or_losing_any(self):
        from src.pipeline import _spread_over_source

        # 3 inferences carrying 2 source frames each.
        out = _spread_over_source([60.0, 40.0, 50.0], 2, 6)
        self.assertEqual(out, [30.0, 30.0, 20.0, 20.0, 25.0, 25.0])
        self.assertAlmostEqual(sum(out), 150.0)  # no time created or lost
        # A partial last group is shared over what it actually carries: 40 ms
        # over 4 frames, then 40 ms over the 2 that are left.
        self.assertEqual(_spread_over_source([40.0, 40.0], 4, 6),
                         [10.0, 10.0, 10.0, 10.0, 20.0, 20.0])
        # Per-written-frame work (overlay, encode) repeats instead of dividing.
        self.assertEqual(_spread_over_source([2.0, 2.0], 2, 4, share=False),
                         [2.0, 2.0, 2.0, 2.0])
        # Stride 1 is the identity, so an unskipped run charts exactly as before.
        self.assertEqual(_spread_over_source([10.0, 20.0], 1, 2), [10.0, 20.0])

    def test_charts_are_drawn_on_the_source_timeline(self):
        # The chart has to stay comparable with an unskipped run of the same
        # clip: same number of points, same unit, same x axis.
        import cv2
        import src.pipeline as P

        seen = {}
        tracker = self._Tracker((8, 12))
        real_latency, real_stage = P.write_latency_chart, P.write_stage_chart
        P.write_latency_chart = lambda ms, path, **k: (
            seen.update(latency=list(ms), warmup=k.get("warmup")), None)[1]
        P.write_stage_chart = lambda pre, infer, post, path, **k: (
            seen.update(infer=list(infer), encode=list(k.get("encode_ms") or [])), None)[1]
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for i in range(6):
                cv2.imwrite(str(d / f"frame_{i:06d}.tiff"),
                            np.full((8, 12, 3), i * 40, dtype=np.uint8))
            try:
                P.run(tracker, PromptSet(boxes=[BoxPrompt(1, 0, (1, 1, 3, 3))]),
                      PipelineConfig(output_path=d / "out.mp4", frames_dir=d,
                                     frame_pattern="*.tif*", frame_stride=2,
                                     fps_warmup=1, fps_chart=d / "lat.png",
                                     stage_chart=d / "stg.png"))
            finally:
                P.write_latency_chart, P.write_stage_chart = real_latency, real_stage

        # 3 inferences, 6 plotted points -- one per source frame, in pairs.
        self.assertEqual(len(seen["latency"]), 6)
        self.assertEqual(len(seen["infer"]), 6)
        self.assertEqual(len(seen["encode"]), 6)
        for i in (0, 2, 4):
            self.assertEqual(seen["latency"][i], seen["latency"][i + 1],
                             "an inference's cost is not shared evenly over its frames")
        # The warm-up shading moves onto that timeline too, or it would mark the
        # wrong frames.
        self.assertEqual(seen["warmup"], 2)

    def test_direct_mp4_mode_rejects_frame_skip(self):
        cfg = PipelineConfig(video_path=Path("foo.mp4"), output_path=Path("/tmp/out.mp4"),
                             video_mode="mp4", frame_stride=2)
        with self.assertRaises((ValueError, RuntimeError)):
            _resolve_video_mode(cfg)


class TestThermalConfigs(unittest.TestCase):
    """The 512 thermal configs, checked without EdgeTAM or a GPU.

    A config key the tracker does not accept is a `TypeError` at model-build
    time -- after the operator has waited for a checkpoint load. These build the
    tracker object itself, which is where every key is validated.
    """

    def _config(self, name: str) -> dict:
        import yaml

        return yaml.safe_load((ROOT / "configs" / name).read_text())

    def test_the_adapter_config_builds_a_tracker(self):
        cfg = self._config("edgetam_512_lora_adapter.yaml")
        tracker = build_tracker("edgetam", **cfg)
        self.assertTrue(tracker.lora_adapter.endswith("edgetam_lora_512.adapter.pt"))
        self.assertEqual(tracker.lora_scale, 1.0)
        # The base weights it loads are upstream's, untouched.
        self.assertIn("third_party/EdgeTAM", tracker.checkpoint)

    def test_the_merged_config_asks_for_no_adapter(self):
        tracker = build_tracker("edgetam", **self._config("edgetam_512_lora.yaml"))
        self.assertIsNone(tracker.lora_adapter)
        self.assertTrue(tracker.checkpoint.endswith("edgetam_lora_512.pt"))

    def test_the_deployment_configs_agree_on_which_checkpoint_ships(self):
        # Whatever the default is, SAMURAI and the INT8 build must track it --
        # engines calibrated against one checkpoint and scored against another
        # measure the difference between the checkpoints.
        default = self._config("edgetam_512_lora.yaml")["checkpoint"]
        for name in ("edgetam_samurai_512.yaml", "edgetam_sam2long_512.yaml",
                     "edgetam_trt_samurai_512.yaml", "edgetam_trt_int8_512.yaml"):
            self.assertEqual(self._config(name)["checkpoint"], default, name)


class TestEngineDirectories(unittest.TestCase):
    def test_an_engine_directory_never_serves_two_checkpoints(self):
        """Engines hold the weights they were exported from.

        A config that points at someone else's engine directory runs those
        engines with its own checkpoint as the scaffold -- and a module that
        falls back to PyTorch (a missing engine, `strict: false`) then swaps
        the weights out mid-frame with nothing said. It is invisible in the
        output and it is a one-character mistake to make in a YAML file.
        """
        import yaml

        seen: dict[str, dict[str, str]] = {}
        for path in sorted((ROOT / "configs").glob("*.yaml")):
            cfg = yaml.safe_load(path.read_text())
            engine = cfg.get("image_encoder_engine")
            if not engine:
                continue
            seen.setdefault(str(Path(engine).parent), {})[path.name] = cfg["checkpoint"]

        self.assertTrue(seen, "no engine-bearing config found")
        for directory, configs in seen.items():
            self.assertEqual(len(set(configs.values())), 1, f"{directory}: {configs}")


class TestCheckpointNote(unittest.TestCase):
    """Which weights a run actually loaded, said out loud before frame 0."""

    def _write(self, tmp: Path, meta: dict) -> Path:
        import torch

        path = tmp / "ckpt.pt"
        torch.save({"model": {}, "meta": meta}, path)
        return path

    def test_it_names_the_run_that_produced_the_checkpoint(self):
        from src.trackers.edgetam_tracker import _checkpoint_note

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), {"method": "lora", "lora_r": 16,
                                           "dataset": "Anti-UAV410",
                                           "train_sequences": ["a", "b"],
                                           "image_size": 512})
            note = _checkpoint_note(str(path), 512)
        self.assertIn("lora", note)
        self.assertIn("r=16", note)
        self.assertIn("Anti-UAV410", note)
        self.assertIn("2 sequences", note)
        self.assertNotIn("!!", note)      # nothing to warn about

    def test_it_warns_when_the_config_runs_it_at_another_resolution(self):
        from src.trackers.edgetam_tracker import _checkpoint_note

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(Path(tmp), {"method": "lora", "image_size": 512})
            self.assertIn("!!", _checkpoint_note(str(path), 768))

    def test_a_checkpoint_with_no_provenance_is_silent_not_loud(self):
        # Upstream's own checkpoint has no meta, and a missing file must not
        # turn a diagnostic into the thing that fails the run.
        from src.trackers.edgetam_tracker import _checkpoint_note

        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(_checkpoint_note(str(self._write(Path(tmp), {})), 512))
            self.assertIsNone(_checkpoint_note(str(Path(tmp) / "absent.pt"), 512))


class TestRecordModes(unittest.TestCase):
    """`tools/run_records.py`'s mode table, which nothing else validates.

    A mode names a config and a backend, and a wrong pairing does not fail: a
    TensorRT config with no engines falls back to PyTorch on whatever
    checkpoint it names, and the run looks like it worked. So the table is
    checked here rather than on the device.
    """

    def setUp(self):
        from tools import run_records

        self.rr = run_records

    def test_every_mode_names_a_config_that_exists(self):
        for name, spec in self.rr.MODES.items():
            self.assertTrue((ROOT / spec.config).is_file(), f"{name}: {spec.config}")

    def test_every_mode_names_a_registered_tracker(self):
        registered = set(available_trackers())
        for name, spec in self.rr.MODES.items():
            self.assertIn(spec.tracker, registered, name)

    def test_the_default_set_is_still_the_tensorrt_one(self):
        # Adding PyTorch modes must not make an Orin run silently grow.
        for name in self.rr.DEFAULT_MODES:
            self.assertEqual(self.rr.MODES[name].tracker, "edgetam_trt", name)

    def test_every_group_holds_real_modes(self):
        for group, names in self.rr.GROUPS.items():
            for name in names:
                self.assertIn(name, self.rr.MODES, f"{group} -> {name}")

    def test_the_samurai_group_changes_exactly_one_thing(self):
        # The point of a group is a comparison with one variable in it. Same
        # backend, same window, same weights -- only the memory gate differs.
        import yaml

        a, b = (self.rr.MODES[m] for m in self.rr.GROUPS["samurai"])
        self.assertEqual(a.tracker, b.tracker)
        self.assertEqual(a.crop, b.crop)
        configs = [yaml.safe_load((ROOT / m.config).read_text()) for m in (a, b)]
        self.assertEqual(configs[0]["checkpoint"], configs[1]["checkpoint"])
        self.assertNotIn("samurai", configs[0])
        self.assertTrue(configs[1]["samurai"]["enabled"])

    def test_the_weights_group_changes_exactly_one_thing(self):
        import yaml

        specs = [self.rr.MODES[m] for m in self.rr.GROUPS["weights"]]
        self.assertEqual({s.tracker for s in specs}, {"edgetam"})
        self.assertEqual({s.crop for s in specs}, {None})
        checkpoints = [yaml.safe_load((ROOT / s.config).read_text())["checkpoint"]
                       for s in specs]
        self.assertEqual(len(set(checkpoints)), len(checkpoints), checkpoints)

    def test_the_trt512_ladder_stays_on_one_set_of_weights(self):
        # A latency ladder that changed the weights halfway up would not be a
        # ladder. Window and memory gate vary; the checkpoint does not.
        import yaml

        specs = [self.rr.MODES[m] for m in self.rr.GROUPS["trt512"]]
        self.assertEqual({s.tracker for s in specs}, {"edgetam_trt"})
        checkpoints = {yaml.safe_load((ROOT / s.config).read_text())["checkpoint"]
                       for s in specs}
        self.assertEqual(len(checkpoints), 1, checkpoints)

    def test_groups_and_modes_expand_in_order_without_repeats(self):
        self.assertEqual(self.rr.expand_modes("samurai,lora512,lora512_crop"),
                         ["lora512", "lora512_samurai", "lora512_crop"])

    def test_an_unknown_mode_is_refused_by_name(self):
        with self.assertRaises(SystemExit):
            self.rr.expand_modes("lora512,nonsense")
        with self.assertRaises(SystemExit):
            self.rr.expand_modes(" , ")


if __name__ == "__main__":
    unittest.main()
