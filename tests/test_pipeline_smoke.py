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

    def test_edgetam_trt_constructs_without_engine(self):
        # Non-strict mode must not raise at construction time even if the engine
        # file is missing — only the runtime fallback message should fire later.
        t = build_tracker(
            "edgetam_trt",
            engine_path="/tmp/does_not_exist.engine",
            model_cfg="configs/edgetam.yaml",
            checkpoint="/tmp/does_not_exist.pt",
            device="cpu",
            precision="float32",
            strict=False,
        )
        self.assertEqual(t.engine_path, "/tmp/does_not_exist.engine")
        self.assertFalse(t.strict)
        # _try_load_engine returns False when the file is absent.
        self.assertFalse(t._try_load_engine())

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


if __name__ == "__main__":
    unittest.main()
