"""The training loop's contract with EdgeTAM, and the freeze policy.

EdgeTAM is not installed here, so a stand-in implements the four methods
`propagate` actually calls and records how it was called. That is the point:
these tests pin the *interface* the loop assumes -- which frames get a prompt,
where each result is filed so the next frame's memory bank finds it, that the
head wrapper is removed afterwards -- and they fail loudly if that assumption
drifts. Numerical fidelity to the real model is not, and cannot be, the claim;
`tools/compare_backends.py` on the device is.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.clip_loop import (  # noqa: E402
    Batch,
    box_prompt,
    check_model,
    clip_losses,
    propagate,
)
from src.training.finetune import (  # noqa: E402
    EMA,
    FROZEN_MODULES,
    Rates,
    apply_freeze,
    param_groups,
    save_checkpoint,
)

SIZE = 32


class FakeSam2(nn.Module):
    """A SAM 2-shaped model: same module names, same method contract.

    `track_step` reads `output_dict` the way the real one does, so the loop's
    memory bookkeeping is exercised rather than merely tolerated -- `seen`
    records how many past frames were visible on each call.
    """

    def __init__(self, size: int = SIZE) -> None:
        super().__init__()
        self.size = size
        self.image_encoder = nn.Conv2d(3, 8, 3, padding=1)
        self.sam_mask_decoder = nn.Conv2d(8, 1, 3, padding=1)
        self.sam_prompt_encoder = nn.Linear(4, 8)
        self.obj_ptr_proj = nn.Linear(8, 8)
        # The recurrent core the freeze policy must never train.
        self.memory_attention = nn.Conv2d(8, 8, 1)
        self.memory_encoder = nn.Conv2d(8, 8, 1)
        self.spatial_perceiver = nn.Linear(8, 8)
        self.maskmem_tpos_enc = nn.Parameter(torch.zeros(7, 1, 1, 8))
        self.no_obj_ptr = nn.Parameter(torch.zeros(1, 8))
        self.prompted: list[bool] = []
        self.seen: list[int] = []

    def forward_image(self, images):
        return {"features": self.image_encoder(images)}

    def _prepare_backbone_features(self, backbone_out):
        feats = backbone_out["features"]
        return None, [feats], [torch.zeros_like(feats)], [(self.size, self.size)]

    def _forward_sam_heads(self, backbone_features, point_inputs=None, **kwargs):
        low = self.sam_mask_decoder(backbone_features)
        batch = low.shape[0]
        pooled = backbone_features.mean(dim=(2, 3))
        ious = torch.stack([pooled.mean(1)] * 3, dim=-1)
        obj_ptr = self.obj_ptr_proj(pooled)
        score = pooled.mean(1, keepdim=True)
        return (low.repeat(1, 3, 1, 1), None, ious, low, low, obj_ptr, score)

    def track_step(self, frame_idx, is_init_cond_frame, current_vision_feats,
                   current_vision_pos_embeds, feat_sizes, point_inputs, mask_inputs,
                   output_dict, num_frames, **kwargs):
        self.prompted.append(point_inputs is not None)
        self.seen.append(len(output_dict["cond_frame_outputs"])
                         + len(output_dict["non_cond_frame_outputs"]))

        features = current_vision_feats[0]
        # Condition on the memory the loop filed, so a bookkeeping mistake
        # changes the numbers instead of passing silently.
        for stored in output_dict["non_cond_frame_outputs"].values():
            features = features + 0.01 * self.memory_attention(stored["maskmem"])
        _, _, _, low, high, obj_ptr, score = self._forward_sam_heads(
            features, point_inputs=point_inputs
        )
        return {
            "pred_masks": low,
            "pred_masks_high_res": high,
            "obj_ptr": obj_ptr,
            "object_score_logits": score,
            "maskmem": self.memory_encoder(features),
        }


def fake_batch(batch: int = 2, frames: int = 4, labelled=(1, 2)) -> Batch:
    torch.manual_seed(0)
    boxes = torch.tensor([[8.0, 8.0, 20.0, 20.0]]).repeat(batch, frames, 1)
    masks = []
    for t in range(frames):
        if t not in labelled:
            masks.append(None)
            continue
        mask = torch.zeros(batch, SIZE, SIZE, dtype=torch.bool)
        mask[:, 8:20, 8:20] = True
        masks.append(mask)
    return Batch(
        images=torch.randn(batch, frames, 3, SIZE, SIZE),
        boxes=boxes,
        exist=torch.ones(batch, frames),
        masks=masks,
    )


class TestCheckModel(unittest.TestCase):
    def test_rejects_something_that_is_not_a_sam2_model(self):
        with self.assertRaises(AttributeError) as ctx:
            check_model(nn.Linear(2, 2).eval())
        self.assertIn("forward_image", str(ctx.exception))

    def test_rejects_train_mode(self):
        # Batch-norm statistics must stay frozen so the checkpoint keeps
        # matching its exported engines.
        with self.assertRaises(RuntimeError) as ctx:
            check_model(FakeSam2().train())
        self.assertIn("eval()", str(ctx.exception))

    def test_accepts_an_eval_mode_model(self):
        check_model(FakeSam2().eval())


class TestBoxPrompt(unittest.TestCase):
    def test_a_box_becomes_two_labelled_corner_points(self):
        boxes = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        prompt = box_prompt(boxes)

        self.assertEqual(prompt["point_coords"].shape, (2, 2, 2))
        np.testing.assert_allclose(prompt["point_coords"][0].numpy(), [[1, 2], [3, 4]])
        np.testing.assert_array_equal(prompt["point_labels"].numpy(),
                                      [[2, 3], [2, 3]])


class TestPropagate(unittest.TestCase):
    def test_one_output_per_frame_with_the_keys_the_loss_needs(self):
        model = FakeSam2().eval()
        outputs = propagate(model, torch.randn(2, 5, 3, SIZE, SIZE),
                            torch.tensor([[8.0, 8.0, 20.0, 20.0]]).repeat(2, 1))

        self.assertEqual(len(outputs), 5)
        for out in outputs:
            self.assertEqual(set(out) >= {"pred_masks_high_res", "object_score_logits",
                                          "ious"}, True)
            self.assertEqual(out["object_score_logits"].shape, (2, 1))
            self.assertEqual(out["ious"].shape, (2, 1))

    def test_only_the_first_frame_carries_the_prompt(self):
        model = FakeSam2().eval()
        propagate(model, torch.randn(1, 4, 3, SIZE, SIZE), torch.zeros(1, 4))
        self.assertEqual(model.prompted, [True, False, False, False])

    def test_each_frame_sees_every_earlier_one(self):
        # If the loop filed results in the wrong place, the memory bank would
        # stay empty and this would be all zeros.
        model = FakeSam2().eval()
        propagate(model, torch.randn(1, 5, 3, SIZE, SIZE), torch.zeros(1, 4))
        self.assertEqual(model.seen, [0, 1, 2, 3, 4])

    def test_the_head_wrapper_leaves_nothing_behind(self):
        # Not just "the method works again": assigning the original back would
        # leave a bound method in the instance dict for good.
        model = FakeSam2().eval()
        propagate(model, torch.randn(1, 2, 3, SIZE, SIZE), torch.zeros(1, 4))
        self.assertNotIn("_forward_sam_heads", vars(model))

    def test_the_head_wrapper_is_removed_even_when_the_loop_raises(self):
        model = FakeSam2().eval()
        model.track_step = lambda **kw: (_ for _ in ()).throw(ValueError("boom"))
        with self.assertRaises(ValueError):
            propagate(model, torch.randn(1, 2, 3, SIZE, SIZE), torch.zeros(1, 4))
        self.assertNotIn("_forward_sam_heads", vars(model))

    def test_gradients_reach_the_encoder_through_the_whole_clip(self):
        # The claim being tested is that this is real BPTT: a loss on the last
        # frame has to move weights used on the first.
        model = FakeSam2().eval()
        outputs = propagate(model, torch.randn(1, 4, 3, SIZE, SIZE), torch.zeros(1, 4))
        outputs[-1]["pred_masks_high_res"].sum().backward()

        grad = model.image_encoder.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_a_model_that_never_calls_the_head_is_reported_clearly(self):
        model = FakeSam2().eval()
        model.track_step = lambda **kw: {"pred_masks": torch.zeros(1, 1, 4, 4)}
        with self.assertRaises(RuntimeError) as ctx:
            propagate(model, torch.randn(1, 2, 3, SIZE, SIZE), torch.zeros(1, 4))
        self.assertIn("SAM head was never called", str(ctx.exception))


class TestClipLosses(unittest.TestCase):
    def test_the_prompted_frame_is_excluded_by_default(self):
        model = FakeSam2().eval()
        _, terms = clip_losses(model, fake_batch(frames=4, labelled=(0,)))
        # Frame 0 is the only labelled one, and it is skipped, so no mask term.
        self.assertNotIn("mask", terms)

    def test_skip_first_false_scores_the_prompted_frame(self):
        model = FakeSam2().eval()
        _, terms = clip_losses(model, fake_batch(frames=4, labelled=(0,)),
                               skip_first=False)
        self.assertIn("mask", terms)

    def test_loss_is_finite_and_differentiable(self):
        model = FakeSam2().eval()
        loss, terms = clip_losses(model, fake_batch())

        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(model.image_encoder.weight.grad.abs().sum()), 0.0)
        self.assertIn("object_score", terms)

    def test_frames_with_no_visible_target_still_train_the_object_head(self):
        batch = fake_batch(frames=3, labelled=())
        batch.exist[:] = 0.0
        batch.boxes[:] = float("nan")
        _, terms = clip_losses(FakeSam2().eval(), batch)

        self.assertEqual(set(terms), {"object_score"})


class TestFreezePolicy(unittest.TestCase):
    def test_the_memory_path_is_never_trainable_in_any_stage(self):
        for stage in ("head", "encoder"):
            model = FakeSam2()
            apply_freeze(model, stage)
            for name, param in model.named_parameters():
                if any(name.startswith(m) for m in FROZEN_MODULES):
                    self.assertFalse(param.requires_grad, f"{stage}: {name} is trainable")

    def test_learned_memory_tokens_stay_frozen(self):
        model = FakeSam2()
        apply_freeze(model, "encoder")
        self.assertFalse(model.maskmem_tpos_enc.requires_grad)
        self.assertFalse(model.no_obj_ptr.requires_grad)

    def test_head_stage_leaves_the_encoder_alone(self):
        model = FakeSam2()
        counts = apply_freeze(model, "head")

        self.assertNotIn("image_encoder", counts)
        self.assertFalse(model.image_encoder.weight.requires_grad)
        self.assertTrue(model.sam_mask_decoder.weight.requires_grad)

    def test_encoder_stage_adds_the_encoder_and_keeps_the_head(self):
        model = FakeSam2()
        counts = apply_freeze(model, "encoder")

        self.assertIn("image_encoder", counts)
        self.assertIn("sam_mask_decoder", counts)

    def test_an_unknown_stage_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_freeze(FakeSam2(), "everything")

    def test_counts_are_parameter_numbers_not_tensor_counts(self):
        model = FakeSam2()
        counts = apply_freeze(model, "head")
        self.assertEqual(counts["sam_mask_decoder"],
                         sum(p.numel() for p in model.sam_mask_decoder.parameters()))


class TestParamGroups(unittest.TestCase):
    def test_trunk_and_head_get_different_rates(self):
        model = FakeSam2()
        model.image_encoder = nn.Module()
        model.image_encoder.trunk = nn.Conv2d(3, 8, 3)
        model.image_encoder.neck = nn.Conv2d(8, 8, 1)
        apply_freeze(model, "encoder")
        groups = {g["name"]: g for g in param_groups(model, Rates(head=1e-4, trunk=1e-5))}

        self.assertAlmostEqual(groups["trunk"]["lr"], 1e-5)
        self.assertAlmostEqual(groups["head"]["lr"], 1e-4)

    def test_frozen_parameters_never_enter_the_optimiser(self):
        model = FakeSam2()
        apply_freeze(model, "head")
        listed = {id(p) for g in param_groups(model) for p in g["params"]}

        for param in model.memory_attention.parameters():
            self.assertNotIn(id(param), listed)

    def test_biases_and_norms_get_no_weight_decay(self):
        model = FakeSam2()
        apply_freeze(model, "head")
        for group in param_groups(model, Rates(weight_decay=0.05)):
            expected = 0.0 if "no-decay" in group["name"] else 0.05
            self.assertAlmostEqual(group["weight_decay"], expected)
            for param in group["params"]:
                self.assertEqual(param.ndim > 1, "no-decay" not in group["name"])


class TestEma(unittest.TestCase):
    def test_shadow_tracks_only_trainable_parameters(self):
        model = FakeSam2()
        apply_freeze(model, "head")
        ema = EMA(model)

        self.assertIn("sam_mask_decoder.weight", ema.shadow)
        self.assertNotIn("memory_attention.weight", ema.shadow)

    def test_update_moves_the_shadow_towards_the_weights(self):
        model = FakeSam2()
        apply_freeze(model, "head")
        ema = EMA(model, decay=0.5)
        before = ema.shadow["sam_mask_decoder.weight"].clone()

        with torch.no_grad():
            model.sam_mask_decoder.weight.add_(1.0)
        ema.update(model)

        moved = ema.shadow["sam_mask_decoder.weight"] - before
        torch.testing.assert_close(moved, torch.full_like(moved, 0.5))

    def test_applied_swaps_in_and_restores_the_raw_weights(self):
        model = FakeSam2()
        apply_freeze(model, "head")
        ema = EMA(model, decay=0.0)  # shadow follows the weights exactly
        raw = model.sam_mask_decoder.weight.detach().clone()

        with torch.no_grad():
            model.sam_mask_decoder.weight.add_(3.0)
        with ema.applied(model):
            torch.testing.assert_close(model.sam_mask_decoder.weight, raw)
        torch.testing.assert_close(model.sam_mask_decoder.weight, raw + 3.0)

    def test_weights_are_restored_even_if_evaluation_raises(self):
        model = FakeSam2()
        apply_freeze(model, "head")
        ema = EMA(model)
        raw = model.sam_mask_decoder.weight.detach().clone()
        with self.assertRaises(ValueError):
            with ema.applied(model):
                raise ValueError("evaluation blew up")
        torch.testing.assert_close(model.sam_mask_decoder.weight, raw)


class TestSaveCheckpoint(unittest.TestCase):
    def test_saved_in_the_layout_edgetams_loader_expects(self):
        # build_sam2 reads torch.load(path)["model"] and load_state_dict's it
        # strictly, so the fine-tuned weights have to drop into the existing
        # configs and exporter with nothing else changed.
        import tempfile

        model = FakeSam2()
        with tempfile.TemporaryDirectory() as tmp:
            path = save_checkpoint(model, Path(tmp) / "ft.pt", {"stage": "head"})
            loaded = torch.load(path, map_location="cpu", weights_only=False)

        self.assertEqual(set(loaded) >= {"model", "meta"}, True)
        self.assertEqual(loaded["meta"]["stage"], "head")
        FakeSam2().load_state_dict(loaded["model"])  # strict by default


if __name__ == "__main__":
    unittest.main()
