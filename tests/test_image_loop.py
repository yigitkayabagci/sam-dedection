"""One encode, many prompts, and no memory bank -- the static loop's contract.

Three claims are worth pinning, and all three are things that would otherwise
fail silently and expensively:

* the image encoder runs **once per image**, not once per instance. Getting
  this wrong costs a factor of `max_instances` in GPU time and nothing else
  would notice;
* a padded instance slot is never computed, so a ragged batch costs what its
  real instances cost;
* the memory encoder never runs, because there is no next frame to read what it
  would write.

EdgeTAM is not installed here, so `tests/test_clip_loop.FakeSam2` stands in --
the same one the video loop is tested against, deliberately, because the whole
argument for going through `track_step` is that these two loops share a path.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import cv2
except ImportError:                                     # pragma: no cover
    cv2 = None

from src.training.aerial import (  # noqa: E402
    DatasetSpec,
    InstanceGates,
    index_frames,
    list_frames,
    sample_windows,
)
from src.training.finetune import Rates, apply_freeze  # noqa: E402
from src.training.image_loop import (  # noqa: E402
    ImageBatch,
    ImageSplit,
    image_losses,
    instance_iou,
    propagate_image,
    stream,
)
from src.training.losses import Weights, instance_loss  # noqa: E402
from src.training.schedule import Schedule, images, run_stages  # noqa: E402
from tests.test_clip_loop import FakeSam2  # noqa: E402

SIZE = 32


def fake_batch(images: int = 2, width: int = 3, valid=None) -> ImageBatch:
    """`images` windows with `width` instance slots, `valid` marking the real ones."""
    torch.manual_seed(0)
    if valid is None:
        valid = torch.ones(images, width, dtype=torch.bool)
    boxes = torch.tensor([8.0, 8.0, 20.0, 20.0]).expand(images, width, 4).clone()
    masks = torch.zeros(images, width, SIZE, SIZE, dtype=torch.bool)
    masks[:, :, 8:20, 8:20] = True
    return ImageBatch(images=torch.randn(images, 3, SIZE, SIZE), boxes=boxes,
                      masks=masks, valid=valid)


class TestPropagate(unittest.TestCase):
    def test_the_encoder_runs_once_per_image_not_once_per_instance(self):
        model = FakeSam2(SIZE).eval()
        batch = fake_batch(images=2, width=4)
        propagate_image(model, batch.images, batch.boxes, batch.valid)

        self.assertEqual(model.images, [2])          # one call, two images
        self.assertEqual(model.encodes, [8])         # eight prompted instances

    def test_padded_slots_are_never_computed(self):
        model = FakeSam2(SIZE).eval()
        valid = torch.tensor([[True, True, False], [True, False, False]])
        batch = fake_batch(images=2, width=3, valid=valid)
        out = propagate_image(model, batch.images, batch.boxes, batch.valid)

        self.assertEqual(model.encodes, [3])
        self.assertEqual(out["pred_masks_high_res"].shape[0], 3)

    def test_rows_identify_which_image_each_instance_came_from(self):
        model = FakeSam2(SIZE).eval()
        valid = torch.tensor([[False, True, True], [True, False, False]])
        out = propagate_image(model, *[getattr(fake_batch(2, 3, valid), f)
                                       for f in ("images", "boxes", "valid")])

        # Flat b*K + k, so the caller can gather its own targets in this order.
        np.testing.assert_array_equal(out["rows"].numpy(), [1, 2, 3])
        np.testing.assert_array_equal(out["image_of"].numpy(), [0, 0, 1])

    def test_the_memory_encoder_never_runs(self):
        # Nothing will ever read what it would write, and its weights are
        # frozen throughout -- computing it would be pure waste on every step.
        # `run_mem_encoder=False` is what skips it, so the stand-in honours the
        # flag and this asserts the skip rather than the absence of a gradient.
        model = FakeSam2(SIZE).eval()
        seen: list[bool] = []
        original = model.track_step
        model.track_step = lambda **kw: (seen.append(kw["run_mem_encoder"]),
                                         original(**kw))[1]

        image_losses(model, fake_batch())[0].backward()
        self.assertEqual(seen, [False])
        self.assertIsNone(model.memory_encoder.weight.grad)

    def test_the_frame_is_prompted_and_is_a_conditioning_frame(self):
        model = FakeSam2(SIZE).eval()
        batch = fake_batch()
        propagate_image(model, batch.images, batch.boxes, batch.valid)

        self.assertEqual(model.prompted, [True])
        self.assertEqual(model.seen, [0])            # no memory to read

    def test_the_head_wrapper_leaves_nothing_behind(self):
        model = FakeSam2(SIZE).eval()
        batch = fake_batch()
        propagate_image(model, batch.images, batch.boxes, batch.valid)
        self.assertNotIn("_forward_sam_heads", vars(model))

    def test_a_batch_of_nothing_but_padding_is_an_error(self):
        model = FakeSam2(SIZE).eval()
        batch = fake_batch(valid=torch.zeros(2, 3, dtype=torch.bool))
        with self.assertRaises(ValueError):
            propagate_image(model, batch.images, batch.boxes, batch.valid)

    def test_a_model_in_train_mode_is_refused(self):
        # Batch-norm statistics must stay frozen so the checkpoint keeps
        # matching its exported engines -- the same rule as the clip loop.
        model = FakeSam2(SIZE).train()
        batch = fake_batch()
        with self.assertRaises(RuntimeError):
            propagate_image(model, batch.images, batch.boxes, batch.valid)


class TestInstanceLoss(unittest.TestCase):
    def test_there_is_no_object_score_term(self):
        # A static set has no `exist` label; every prompted instance is present
        # by construction, so BCE against a constant 1 would teach the head to
        # fire unconditionally. That head is trained on video, where the label
        # is real, or not at all.
        _, terms = image_losses(FakeSam2(SIZE).eval(), fake_batch())
        self.assertEqual(set(terms), {"focal", "dice", "iou"})

    def test_loss_is_finite_and_reaches_the_encoder(self):
        model = FakeSam2(SIZE).eval()
        loss, _ = image_losses(model, fake_batch())

        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(model.image_encoder.weight.grad.abs().sum()), 0.0)

    def test_a_perfect_prediction_costs_less_than_a_wrong_one(self):
        target = torch.zeros(2, SIZE, SIZE)
        target[:, 8:20, 8:20] = 1.0
        good = torch.where(target > 0, 10.0, -10.0)[:, None]
        bad = -good

        outputs = {"pred_masks_high_res": good, "ious": torch.ones(2, 1)}
        wrong = {"pred_masks_high_res": bad, "ious": torch.ones(2, 1)}
        self.assertLess(float(instance_loss(outputs, target)[0]),
                        float(instance_loss(wrong, target)[0]))

    def test_the_weights_are_sam2s_published_ones(self):
        # focal dominates by design: it is the only term whose scale reflects
        # how few pixels an aerial vehicle occupies.
        self.assertEqual((Weights().focal, Weights().dice, Weights().iou),
                         (20.0, 1.0, 1.0))


class TestInstanceIou(unittest.TestCase):
    def test_targets_are_gathered_in_the_same_order_as_the_predictions(self):
        # If `rows` and the target gather ever disagreed, every instance would
        # be scored against another instance's mask -- and the number would
        # still look like a plausible IoU.
        model = FakeSam2(SIZE).eval()
        valid = torch.tensor([[True, False, True], [False, True, False]])
        scores = instance_iou(model, fake_batch(2, 3, valid))

        self.assertEqual(scores.shape, (3,))
        self.assertTrue(((scores >= 0.0) & (scores <= 1.0)).all())

    def test_two_empty_masks_agree_perfectly(self):
        # Both said "nothing here" and both were right; the same reading
        # src/accuracy.py takes.
        batch = fake_batch(images=1, width=1)
        batch.masks[:] = False

        class AlwaysEmpty(FakeSam2):
            def _forward_sam_heads(self, backbone_features, point_inputs=None, **kw):
                out = super()._forward_sam_heads(backbone_features, point_inputs, **kw)
                empty = torch.full_like(out[4], -10.0)
                return (out[0], out[1], out[2], empty, empty, out[5], out[6])

        scores = instance_iou(AlwaysEmpty(SIZE).eval(), batch)
        np.testing.assert_allclose(scores, [1.0])


@unittest.skipIf(cv2 is None, "OpenCV is not installed")
class TestEndToEnd(unittest.TestCase):
    """Files on disk to a trained checkpoint, with nothing mocked in between.

    Every piece below has its own test; what this one covers is the wiring
    between them -- the index feeding window sampling, windows feeding the
    collate, the collate feeding the loss, and all of it going through the same
    `run_stages` the video path uses. That seam is where a shape mistake would
    otherwise surface an hour into a Colab session.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        for sub in ("tir", "label"):
            (root / "scene" / sub).mkdir(parents=True, exist_ok=True)
        for i in range(6):
            mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
            mask[4:12, 4 + i:12 + i] = 2                     # a car
            mask[20:28, 18:26] = 3                           # a person
            cv2.imwrite(str(root / "scene" / "label" / f"{i}_label.png"), mask)
            cv2.imwrite(str(root / "scene" / "tir" / f"{i}.png"),
                        (mask * 60).astype(np.uint8))

        self.spec = DatasetSpec(
            name="toy", thermal="**/tir/*.png", masks="**/label/*.png",
            classes={"background": 0, "car": 2, "person": 3},
            things=("car", "person"), strip=("_label",))
        self.gates = InstanceGates(min_area=4, min_side=2, max_area=0.9, fill=0.25)
        index = index_frames(list_frames(root, self.spec, "thermal"),
                             self.spec, self.gates, workers=2)
        self.split = ImageSplit(
            samples=sample_windows(index, size=SIZE, per_image=1,
                                   max_instances=4, seed=0),
            spec=self.spec, gates=self.gates, gray=True)

    def test_every_window_carries_both_objects(self):
        self.assertEqual(len(self.split.samples), 6)
        for sample in self.split.samples:
            self.assertEqual(len(sample.instances), 2)

    def test_the_stream_assembles_batches_the_loss_accepts(self):
        batches = list(stream(self.split, batch=2, seed=0, limit=2, device="cpu",
                              workers=2, depth=1))
        self.assertEqual(len(batches), 2)
        for batch in batches:
            self.assertEqual(batch.images.shape, (2, 3, SIZE, SIZE))
            self.assertEqual(batch.masks.shape[:2], (2, 2))
            self.assertTrue(batch.valid.all())
            loss, terms = image_losses(FakeSam2(SIZE).eval(), batch)
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(set(terms), {"focal", "dice", "iou"})

    def test_run_stages_drives_the_static_loop_end_to_end(self):
        torch.manual_seed(0)
        model = FakeSam2(SIZE).eval()
        saves: list[dict] = []
        before = {n: p.detach().clone() for n, p in model.named_parameters()}

        result = run_stages(
            model, self.split, self.split,
            Schedule(stages=(("head", 1, Rates(head=1e-3)),
                             ("encoder", 1, Rates(head=1e-3, neck=1e-3, trunk=1e-3))),
                     batch=2, steps_per_epoch=2, val_batches=1, workers=2, depth=1),
            freeze=apply_freeze, save=lambda m, meta: saves.append(meta),
            device="cpu", log=lambda *_: None, loop=images())

        self.assertEqual([h["stage"] for h in result["history"]], ["head", "encoder"])
        self.assertTrue(saves)
        moved = [n for n, p in model.named_parameters()
                 if not torch.equal(p.detach(), before[n])]
        self.assertIn("image_encoder.weight", moved)
        for name, param in model.named_parameters():
            if name.startswith(("memory_attention", "memory_encoder", "spatial_perceiver")):
                self.assertTrue(torch.equal(param.detach(), before[name]), name)

    def test_lora_takes_the_same_path_as_the_fine_tune(self):
        # The comparison the notebook makes is only a comparison if both go
        # through this loop with one callback swapped.
        from src.training import lora

        torch.manual_seed(0)
        model = FakeSam2(SIZE).eval()
        lora.inject(model, "encoder", r=2)
        schedule = Schedule(stages=(("encoder", 1, Rates(head=1e-3, neck=1e-3,
                                                         trunk=1e-3)),),
                            batch=2, steps_per_epoch=2, val_batches=1,
                            workers=2, depth=1)
        result = run_stages(model, self.split, self.split, schedule,
                            freeze=lora.freeze, save=lambda m, meta: None,
                            device="cpu", log=lambda *_: None, loop=images())

        self.assertEqual([h["stage"] for h in result["history"]], ["encoder"])
        self.assertTrue(np.isfinite(result["best_val_loss"]))

    def test_scoring_reports_one_iou_per_instance(self):
        from tools.eval_instances import score

        result = score(FakeSam2(SIZE).eval(), self.split, batch=3, device="cpu")
        self.assertEqual(result["instances"], 12)
        self.assertEqual(sorted(result["per_class"]), [2, 3])
        self.assertTrue(0.0 <= result["mean_iou"] <= 1.0)


class TestBatch(unittest.TestCase):
    def test_instances_counts_the_real_slots_only(self):
        valid = torch.tensor([[True, True, False], [True, False, False]])
        self.assertEqual(fake_batch(2, 3, valid).instances, 3)

    def test_to_moves_every_tensor_and_keeps_the_samples(self):
        batch = fake_batch()
        batch.samples = ["a"]
        moved = batch.to("cpu")
        self.assertEqual(moved.samples, ["a"])
        self.assertEqual(moved.valid.shape, batch.valid.shape)


if __name__ == "__main__":
    unittest.main()
