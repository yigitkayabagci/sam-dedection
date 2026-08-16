"""Calibration capture and the per-module quantisation plan.

Neither TensorRT nor Model Optimizer is needed here. What is tested is the part
that actually decides whether INT8 works: that real engine inputs are recorded
in the layout the engine sees, that a capture taken at the wrong resolution is
refused rather than silently used, and that the per-module plan says what the
measurements in docs/tensorrt_fp16.md say.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trackers.calibration import (  # noqa: E402
    CalibrationStore,
    RecordingEngine,
    wrap_engines,
)
from src.training.qat import (  # noqa: E402
    MODULE_ROOTS,
    RECURRENT,
    distillation_loss,
    insert_quantizers,
)
from tools.quantize_edgetam import (  # noqa: E402
    DEFAULT_PLAN,
    MODULES,
    _calibration_arrays,
    apply_overrides,
    plan_table,
    write_spec,
)


class FakeBinding:
    """The buffer-and-replay path: inputs exist only as buffer contents."""

    def __init__(self, shapes):
        self.inputs = {name: torch.zeros(shape) for name, shape in shapes.items()}
        self.outputs = {"out": torch.zeros(1)}
        self.runs = 0

    def run(self):
        self.runs += 1

    def enqueue(self):
        self.run()


class FakeEngine:
    def __init__(self, shapes=None):
        self.input_names = list(shapes or {"image": (1, 3, 8, 8)})
        self.output_names = ["out"]
        self.calls = []
        self._binding = FakeBinding(shapes or {"image": (1, 3, 8, 8)})

    def run(self, inputs, clone="__all__"):
        self.calls.append(inputs)
        return {"out": torch.zeros(1)}

    def binding(self, shapes):
        return self._binding

    def accepts(self, shapes):
        return True


class TestCalibrationStore(unittest.TestCase):
    def test_records_and_concatenates_along_the_batch_axis(self):
        store = CalibrationStore(limit=10)
        for _ in range(3):
            store.record({"image": torch.zeros(2, 3, 4, 4)})
        self.assertEqual(store.arrays()["image"].shape, (6, 3, 4, 4))

    def test_stops_at_the_limit(self):
        store = CalibrationStore(limit=5)
        for _ in range(20):
            store.record({"image": torch.zeros(1, 3, 4, 4)})
        self.assertEqual(len(store.arrays()["image"]), 5)
        self.assertTrue(store.full)

    def test_stride_spreads_samples_across_the_clip(self):
        # The memory bank fills over the first ~16 frames; sampling only the
        # opening would calibrate for a transient the engine barely runs in.
        store = CalibrationStore(limit=100, stride=3)
        for i in range(30):
            store.record({"x": torch.full((1, 2), float(i))})
        values = store.arrays()["x"][:, 0]
        np.testing.assert_array_equal(values, [2, 5, 8, 11, 14, 17, 20, 23, 26, 29])

    def test_saving_nothing_is_an_error_not_an_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError) as ctx:
                CalibrationStore().save(Path(tmp) / "empty.npz")
        self.assertIn("never called", str(ctx.exception))

    def test_roundtrip_through_npz(self):
        store = CalibrationStore()
        store.record({"a": torch.ones(2, 3), "b": torch.zeros(2, 4)})
        with tempfile.TemporaryDirectory() as tmp:
            path = store.save(Path(tmp) / "c.npz")
            with np.load(path) as data:
                self.assertEqual(sorted(data.files), ["a", "b"])
                self.assertEqual(data["a"].shape, (2, 3))


class TestRecordingEngine(unittest.TestCase):
    def test_records_the_run_path_and_still_returns_the_output(self):
        engine = FakeEngine()
        recorder = RecordingEngine(engine)
        out = recorder.run({"image": torch.ones(1, 3, 8, 8)})

        self.assertIn("out", out)
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(recorder.store.arrays()["image"].shape, (1, 3, 8, 8))

    def test_records_the_buffer_path_at_the_moment_it_runs(self):
        # Memory attention writes into persistent buffers and replays a
        # captured graph; there is no argument to intercept, so the buffers
        # have to be copied before run() hands them over.
        engine = FakeEngine({"memory": (1, 16, 4)})
        recorder = RecordingEngine(engine)
        binding = recorder.binding({"memory": (1, 16, 4)})
        binding.inputs["memory"].fill_(7.0)
        binding.run()

        recorded = recorder.store.arrays()["memory"]
        self.assertEqual(recorded.shape, (1, 16, 4))
        self.assertTrue((recorded == 7.0).all())
        self.assertEqual(engine._binding.runs, 1)

    def test_a_later_write_does_not_change_what_was_recorded(self):
        engine = FakeEngine({"memory": (1, 4, 2)})
        recorder = RecordingEngine(engine)
        binding = recorder.binding({"memory": (1, 4, 2)})
        binding.inputs["memory"].fill_(1.0)
        binding.run()
        binding.inputs["memory"].fill_(9.0)
        binding.run()

        recorded = recorder.store.arrays()["memory"]
        np.testing.assert_array_equal(recorded[:, 0, 0], [1.0, 9.0])

    def test_everything_else_passes_straight_through(self):
        recorder = RecordingEngine(FakeEngine())
        self.assertEqual(recorder.input_names, ["image"])
        self.assertEqual(recorder.output_names, ["out"])
        self.assertTrue(recorder.accepts({"image": (1, 3, 8, 8)}))

    def test_wrap_engines_replaces_them_in_place(self):
        engines = {"image_encoder": FakeEngine(), "sam_head": FakeEngine()}
        stores = wrap_engines(engines, limit=4, stride=1)

        self.assertEqual(sorted(stores), ["image_encoder", "sam_head"])
        for engine in engines.values():
            self.assertIsInstance(engine, RecordingEngine)


def spec_for(shape, batch_axis=0, name="image"):
    return {"inputs": [{"name": name, "shape": list(shape), "batch_axis": batch_axis}]}


class TestCalibrationArrays(unittest.TestCase):
    def _write(self, tmp, arrays):
        path = Path(tmp) / "c.npz"
        np.savez(path, **arrays)
        return path

    def test_accepts_a_capture_whose_batch_axis_differs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, {"image": np.zeros((37, 3, 512, 512), np.float32)})
            arrays = _calibration_arrays(path, spec_for((1, 3, 512, 512)))
        self.assertEqual(arrays["image"].shape[0], 37)

    def test_rejects_a_capture_taken_at_a_different_resolution(self):
        # The failure this catches is quiet and expensive: calibrating a 512
        # graph on 1024 activations produces scales for a network that does
        # not exist, and the engine builds fine.
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, {"image": np.zeros((8, 3, 1024, 1024), np.float32)})
            with self.assertRaises(ValueError) as ctx:
                _calibration_arrays(path, spec_for((1, 3, 512, 512)))
        self.assertIn("different image size", str(ctx.exception))

    def test_rejects_a_capture_missing_an_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, {"curr": np.zeros((4, 8), np.float32)})
            spec = {"inputs": [
                {"name": "curr", "shape": [1, 8], "batch_axis": 0},
                {"name": "memory", "shape": [1, 16], "batch_axis": 0},
            ]}
            with self.assertRaises(ValueError) as ctx:
                _calibration_arrays(path, spec)
        self.assertIn("memory", str(ctx.exception))

    def test_extra_arrays_in_the_capture_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, {"image": np.zeros((4, 3, 8, 8), np.float32),
                                     "leftover": np.zeros((4, 2), np.float32)})
            arrays = _calibration_arrays(path, spec_for((1, 3, 8, 8)))
        self.assertEqual(list(arrays), ["image"])


class TestPlan(unittest.TestCase):
    def test_the_plan_follows_the_measurements(self):
        # docs/tensorrt_fp16.md: the encoder collapses under min/max and is
        # recovered by clipping; the memory encoder is the exact opposite,
        # because it writes the memory bank and a clipping bias accumulates.
        self.assertEqual(DEFAULT_PLAN["image_encoder"].calibration, "entropy")
        self.assertEqual(DEFAULT_PLAN["memory_encoder"].calibration, "max")

    def test_every_module_has_a_plan_with_a_stated_reason(self):
        for name in MODULES:
            self.assertIn(name, DEFAULT_PLAN)
            self.assertTrue(DEFAULT_PLAN[name].reason.strip())

    def test_flops_shares_account_for_the_whole_frame(self):
        self.assertAlmostEqual(sum(p.flops_share for p in DEFAULT_PLAN.values()), 1.0,
                               places=2)

    def test_holding_a_module_back_leaves_the_others_alone(self):
        plan = apply_overrides(DEFAULT_PLAN, ["image_encoder"], {})

        self.assertFalse(plan["image_encoder"].quantize)
        self.assertTrue(plan["memory_attention"].quantize)
        self.assertTrue(DEFAULT_PLAN["image_encoder"].quantize, "mutated the default")

    def test_the_table_reports_the_flop_coverage_that_remains(self):
        self.assertIn("100.0%", plan_table(DEFAULT_PLAN))
        # Dropping the encoder leaves the memory path: 61.9 + 8.8 + 2.5.
        self.assertIn("73.2%", plan_table(apply_overrides(DEFAULT_PLAN,
                                                          ["image_encoder"], {})))

    def test_an_unknown_module_is_rejected(self):
        with self.assertRaises(KeyError):
            apply_overrides(DEFAULT_PLAN, ["memory_bank"], {})

    def test_an_unknown_calibrator_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_overrides(DEFAULT_PLAN, [], {"sam_head": "percentile"})

    def test_calibrator_can_be_overridden(self):
        plan = apply_overrides(DEFAULT_PLAN, [], {"image_encoder": "max"})
        self.assertEqual(plan["image_encoder"].calibration, "max")


class TestWriteSpec(unittest.TestCase):
    def test_precision_is_recorded_so_a_mixed_set_builds_in_one_command(self):
        # Without this, a module held at fp16 would land in a strongly-typed
        # build, where "no Q/DQ nodes" means fp32 rather than fp16.
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "a.spec.json"
            source.write_text(json.dumps({"module": "sam_head", "onnx": "a.onnx"}))
            destination = Path(tmp) / "b.spec.json"
            write_spec(source, destination, "fp16")
            spec = json.loads(destination.read_text())

        self.assertEqual(spec["precision"], "fp16")
        self.assertEqual(spec["module"], "sam_head")


class TestQatPolicy(unittest.TestCase):
    def test_module_roots_cover_the_same_four_engines(self):
        self.assertEqual(sorted(MODULE_ROOTS), sorted(MODULES))

    def test_qat_refuses_the_recurrent_memory_path_by_default(self):
        # Retraining what writes the memory bank is the one change whose error
        # accumulates instead of averaging out. It has to be asked for twice.
        for name in RECURRENT:
            with self.assertRaises(ValueError) as ctx:
                insert_quantizers(object(), [name], forward_loop=lambda m: None)
            self.assertIn("recurrent", str(ctx.exception))

    def test_an_unknown_module_is_rejected_before_modelopt_is_imported(self):
        with self.assertRaises(KeyError):
            insert_quantizers(object(), ["backbone"], forward_loop=lambda m: None)


class TestDistillationLoss(unittest.TestCase):
    def _outputs(self, mask_logit, score_logit, size=8):
        return {"pred_masks_high_res": torch.full((1, 1, size, size), mask_logit),
                "object_score_logits": torch.full((1, 1), score_logit)}

    def test_a_student_identical_to_its_teacher_scores_exactly_zero(self):
        # A KL, not a soft-target cross-entropy: the two differ by the
        # teacher's own entropy, which would put "perfect agreement" at an
        # arbitrary positive number that moves with the teacher.
        for logit in (0.5, 4.0, 12.0):
            teacher = self._outputs(logit, 2.0)
            loss = distillation_loss(self._outputs(logit, 2.0), teacher)
            self.assertAlmostEqual(float(loss), 0.0, places=4)

    def test_disagreement_costs(self):
        teacher = self._outputs(6.0, 2.0)
        close = distillation_loss(self._outputs(5.0, 2.0), teacher)
        far = distillation_loss(self._outputs(-6.0, 2.0), teacher)
        self.assertGreater(float(far), float(close))

    def test_the_object_score_is_matched_on_the_logit(self):
        # It decides whether no_obj_ptr enters the memory bank, so a shift here
        # changes the tracker's state for the next seven frames.
        teacher = self._outputs(4.0, 3.0)
        loss = distillation_loss(self._outputs(4.0, 0.0), teacher, mask_weight=0.0)
        self.assertAlmostEqual(float(loss), 9.0, places=4)

    def test_no_gradient_flows_back_into_the_teacher(self):
        teacher = self._outputs(4.0, 2.0)
        teacher["pred_masks_high_res"].requires_grad_(True)
        student = self._outputs(1.0, 1.0)
        student["pred_masks_high_res"].requires_grad_(True)
        student["object_score_logits"].requires_grad_(True)
        distillation_loss(student, teacher).backward()

        self.assertIsNone(teacher["pred_masks_high_res"].grad)
        self.assertIsNotNone(student["pred_masks_high_res"].grad)


if __name__ == "__main__":
    unittest.main()
