"""LoRA: the factorisation, the freeze, and the merge that has to be exact.

The merge is the load-bearing property. A LoRA checkpoint that is not
key-for-key what `build_sam2` expects cannot be evaluated through the same
tracker as the fine-tuned one, and then the comparison the whole module exists
for is between two deployments rather than two training methods.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training import lora  # noqa: E402
from src.training.finetune import FROZEN_MODULES  # noqa: E402
from tests.test_clip_loop import FakeSam2  # noqa: E402


class TestAdapters(unittest.TestCase):
    def test_a_fresh_adapter_is_the_identity(self):
        # B is zero at init, so the model starts exactly where the checkpoint
        # left it. An adapter that perturbed the output would make the first
        # validation number meaningless.
        for base, x in ((nn.Linear(16, 12), torch.randn(4, 16)),
                        (nn.Conv2d(6, 8, 3, padding=1), torch.randn(2, 6, 10, 10))):
            adapter = lora._adapt(base, r=2, alpha=8, dropout=0.0)
            torch.testing.assert_close(adapter(x), base(x))

    def test_the_merged_weight_reproduces_the_adapted_forward(self):
        torch.manual_seed(0)
        for base, x in ((nn.Linear(16, 12), torch.randn(4, 16)),
                        (nn.Conv2d(6, 8, 3, padding=1), torch.randn(2, 6, 10, 10)),
                        (nn.Conv2d(6, 8, 1), torch.randn(2, 6, 10, 10)),
                        (nn.Conv2d(6, 8, 3, stride=2, padding=1),
                         torch.randn(2, 6, 12, 12))):
            adapter = lora._adapt(base, r=2, alpha=8, dropout=0.0)
            with torch.no_grad():      # move B off zero, or this proves nothing
                adapter.lora_B.normal_(0, 0.2)
            adapted = adapter(x)

            merged = base
            with torch.no_grad():
                merged.weight.data += adapter.delta()
            torch.testing.assert_close(adapted, merged(x), rtol=1e-4, atol=1e-5)

    def test_a_grouped_convolution_is_left_alone(self):
        # Its weight is block diagonal; a dense B @ A cannot represent an
        # update to it, and pretending otherwise would silently corrupt it.
        self.assertIsNone(lora._adapt(nn.Conv2d(8, 8, 3, groups=8), r=2, alpha=8))
        with self.assertRaises(ValueError):
            lora.LoRAConv2d(nn.Conv2d(8, 8, 3, groups=8), r=2, alpha=8)

    def test_a_layer_narrower_than_the_rank_is_left_alone(self):
        self.assertIsNone(lora._adapt(nn.Linear(4, 32), r=4, alpha=8))
        self.assertIsNone(lora._adapt(nn.Conv2d(3, 64, 3), r=4, alpha=8))

    def test_scaling_follows_alpha_over_r(self):
        base = nn.Linear(16, 16)
        a = lora.LoRALinear(base, r=4, alpha=8)
        b = lora.LoRALinear(base, r=4, alpha=16)
        self.assertAlmostEqual(a.scaling, 2.0)
        self.assertAlmostEqual(b.scaling, 4.0)

    def test_alpha_defaults_to_twice_the_rank(self):
        model = FakeSam2()
        report = lora.inject(model, "encoder", r=2)
        self.assertEqual(report["alpha"], 4.0)


class TestInject(unittest.TestCase):
    def setUp(self):
        self.model = FakeSam2()
        self.before = list(self.model.state_dict())

    def test_it_adapts_the_same_modules_the_fine_tune_would_train(self):
        report = lora.inject(self.model, "head", r=2)
        adapted = set(report["adapted"])
        self.assertIn("sam_prompt_encoder", adapted)
        self.assertIn("obj_ptr_proj", adapted)
        self.assertNotIn("image_encoder", adapted)   # head stage stops short

        report = lora.inject(FakeSam2(), "encoder", r=2)
        self.assertIn("image_encoder", set(report["adapted"]))

    def test_the_memory_path_is_never_adapted(self):
        lora.inject(self.model, "encoder", r=2)
        for name, _ in self.model.named_modules():
            if any(name.startswith(m) for m in FROZEN_MODULES):
                self.assertNotIn("lora", name)

    def test_it_reports_what_it_skipped_rather_than_hiding_it(self):
        model = FakeSam2()
        model.image_encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.Conv2d(8, 8, 3, padding=1, groups=8),   # depthwise
        )
        report = lora.inject(model, "encoder", r=2)
        self.assertEqual(report["skipped"]["grouped"], 1)
        self.assertIn("grouped/depthwise", lora.summarise(report))

    def test_a_stage_with_nothing_to_adapt_is_an_error(self):
        model = FakeSam2()
        model.sam_mask_decoder = nn.Identity()
        model.sam_prompt_encoder = nn.Identity()
        model.obj_ptr_proj = nn.Identity()
        with self.assertRaises(RuntimeError):
            lora.inject(model, "head", r=2)

    def test_injection_does_not_change_what_the_model_computes(self):
        images = torch.randn(2, 3, FakeSam2().size, FakeSam2().size)
        reference = self.model.forward_image(images)["features"]
        lora.inject(self.model, "encoder", r=2)
        torch.testing.assert_close(self.model.forward_image(images)["features"],
                                   reference)


class TestFreeze(unittest.TestCase):
    def test_only_adapters_receive_gradients(self):
        model = FakeSam2()
        lora.inject(model, "encoder", r=2)
        counts = lora.freeze(model, "encoder")

        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(lora.is_lora_parameter(n) for n in trainable), trainable)
        self.assertEqual(sum(counts.values()),
                         sum(p.numel() for n, p in model.named_parameters()
                             if p.requires_grad))

    def test_it_trains_far_fewer_parameters_than_the_fine_tune(self):
        from src.training.finetune import apply_freeze

        full = sum(apply_freeze(FakeSam2(), "encoder").values())
        model = FakeSam2()
        lora.inject(model, "encoder", r=2)
        self.assertLess(sum(lora.freeze(model, "encoder").values()), full)

    def test_the_memory_path_stays_frozen_under_lora_too(self):
        model = FakeSam2()
        lora.inject(model, "encoder", r=2)
        lora.freeze(model, "encoder")
        for name, param in model.named_parameters():
            if name.startswith(FROZEN_MODULES):
                self.assertFalse(param.requires_grad, name)

    def test_freezing_without_injecting_is_an_error_not_a_silent_no_op(self):
        with self.assertRaises(RuntimeError):
            lora.freeze(FakeSam2(), "encoder")


class TestMerge(unittest.TestCase):
    def test_the_merged_model_is_key_for_key_the_original(self):
        # This is what lets a LoRA run be scored through the same tracker,
        # config, exporter and engine build as the fine-tuned one.
        model = FakeSam2()
        expected = list(model.state_dict())

        lora.inject(model, "encoder", r=2)
        self.assertNotEqual(list(model.state_dict()), expected)
        lora.merge(model)
        self.assertEqual(list(model.state_dict()), expected)

    def test_merging_preserves_the_function(self):
        torch.manual_seed(0)
        model = FakeSam2()
        lora.inject(model, "encoder", r=2)
        for name, param in model.named_parameters():
            if lora.is_lora_parameter(name) and name.endswith("lora_B"):
                with torch.no_grad():
                    param.normal_(0, 0.1)

        images = torch.randn(2, 3, model.size, model.size)
        adapted = model.forward_image(images)["features"]
        lora.merge(model)
        torch.testing.assert_close(model.forward_image(images)["features"],
                                   adapted, rtol=1e-4, atol=1e-5)

    def test_merge_reports_how_many_layers_it_folded(self):
        model = FakeSam2()
        report = lora.inject(model, "encoder", r=2)
        self.assertEqual(lora.merge(model), len(report["adapted"]))
        self.assertEqual(lora.merge(model), 0)     # idempotent

    def test_saving_merged_leaves_the_training_model_adapted(self):
        import tempfile

        model = FakeSam2()
        lora.inject(model, "encoder", r=2)
        lora.freeze(model, "encoder")

        with tempfile.TemporaryDirectory() as tmp:
            path = lora.save_merged_checkpoint(model, Path(tmp) / "m.pt",
                                               {"method": "lora"})
            saved = torch.load(path, weights_only=False)

        self.assertEqual(list(saved["model"]), list(FakeSam2().state_dict()))
        self.assertEqual(saved["meta"]["method"], "lora")
        # The run must be able to continue: the adapters are still there.
        self.assertTrue(any(p.requires_grad for p in model.parameters()))
        self.assertTrue(any(isinstance(m, lora.ADAPTED) for m in model.modules()))


if __name__ == "__main__":
    unittest.main()
