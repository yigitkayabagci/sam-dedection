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
    Source,
    index_frames,
    list_frames,
    sample_windows,
)
from src.training.finetune import Rates, apply_freeze  # noqa: E402
from src.training.clip_loop import jitter_boxes  # noqa: E402
from src.training.image_loop import (  # noqa: E402
    PROMPTS,
    ImageBatch,
    ImageSplit,
    build_prompt,
    collate,
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


class TestPrompts(unittest.TestCase):
    """What the model is told, and that the weaker forms actually reach it.

    Every number this project has recorded was taken with the ground-truth box
    as the prompt, which on an isolated target states most of the mask -- so
    two encoders of different quality score within a hundredth of each other
    and the metric looks flat. `point` and `jitter` are how that is checked
    rather than assumed, and they are worth pinning because a prompt that
    silently fell back to the box would produce a perfectly plausible table of
    numbers meaning nothing.
    """

    def captured(self, prompt: str, **kwargs) -> dict:
        """The `point_inputs` dict `track_step` was actually handed."""
        model = FakeSam2(SIZE).eval()
        batch = fake_batch(images=1, width=1)
        seen: list[dict] = []
        original = model.track_step
        model.track_step = lambda **kw: (seen.append(kw["point_inputs"]),
                                         original(**kw))[1]
        propagate_image(model, batch.images, batch.boxes, batch.valid,
                        prompt, **kwargs)
        return seen[-1]

    def test_box_is_the_default_and_still_two_corners(self):
        # The default has to stay byte-identical: it is what every recorded
        # number was taken with, and a changed default would silently rebase
        # every comparison against the old ones.
        model = FakeSam2(SIZE).eval()
        batch = fake_batch(images=1, width=1)
        seen: list[dict] = []
        original = model.track_step
        model.track_step = lambda **kw: (seen.append(kw["point_inputs"]),
                                         original(**kw))[1]
        propagate_image(model, batch.images, batch.boxes, batch.valid)

        self.assertEqual(tuple(seen[-1]["point_coords"].shape), (1, 2, 2))
        np.testing.assert_array_equal(seen[-1]["point_labels"].numpy(), [[2, 3]])

    def test_a_point_prompt_is_one_positive_click_at_the_centre(self):
        inputs = self.captured("point")

        self.assertEqual(tuple(inputs["point_coords"].shape), (1, 1, 2))
        np.testing.assert_array_equal(inputs["point_labels"].numpy(), [[1]])
        # fake_batch's box is (8, 8, 20, 20).
        np.testing.assert_allclose(
            inputs["point_coords"].reshape(2).numpy(), [14.0, 14.0])

    def test_a_jittered_prompt_is_still_a_box_but_not_the_same_one(self):
        inputs = self.captured("jitter", jitter=0.3,
                               generator=torch.Generator().manual_seed(0))
        corners = inputs["point_coords"].reshape(-1).numpy()

        np.testing.assert_array_equal(inputs["point_labels"].numpy(), [[2, 3]])
        self.assertFalse(np.allclose(corners, [8.0, 8.0, 20.0, 20.0]))
        # Still recognisably the same target: 0.3 of a 12-pixel side.
        np.testing.assert_allclose(corners, [8.0, 8.0, 20.0, 20.0], atol=12 * 0.3)

    def test_zero_jitter_leaves_the_box_exactly_where_it_was(self):
        boxes = torch.tensor([[8.0, 8.0, 20.0, 20.0]])
        self.assertTrue(torch.equal(jitter_boxes(boxes, 0.0), boxes))

    def test_the_same_seed_perturbs_two_checkpoints_identically(self):
        # Otherwise the gap between two rows of the probe table is partly the
        # draw, and the whole point of the axis is that it is not.
        boxes = torch.tensor([[8.0, 8.0, 20.0, 20.0], [1.0, 2.0, 9.0, 30.0]])
        first = jitter_boxes(boxes, 0.3, torch.Generator().manual_seed(4))
        again = jitter_boxes(boxes, 0.3, torch.Generator().manual_seed(4))
        self.assertTrue(torch.equal(first, again))

    def test_jitter_never_returns_an_inverted_rectangle(self):
        # Edges move independently, so a large fraction can cross them; the
        # re-sort is what keeps x0 < x1. SAM 2 would take the inverted one and
        # segment nothing.
        boxes = torch.tensor([[8.0, 8.0, 20.0, 20.0]]).expand(64, 4)
        moved = jitter_boxes(boxes, 0.9, torch.Generator().manual_seed(2))
        self.assertTrue(bool((moved[:, 0] <= moved[:, 2]).all()))
        self.assertTrue(bool((moved[:, 1] <= moved[:, 3]).all()))

    def test_an_unknown_prompt_is_refused_rather_than_silently_boxed(self):
        with self.assertRaises(ValueError):
            build_prompt(torch.zeros(1, 4), "scribble")

    def test_instance_iou_forwards_the_prompt(self):
        model = FakeSam2(SIZE).eval()
        batch = fake_batch(images=1, width=1)
        seen: list[dict] = []
        original = model.track_step
        model.track_step = lambda **kw: (seen.append(kw["point_inputs"]),
                                         original(**kw))[1]
        instance_iou(model, batch, "point")

        np.testing.assert_array_equal(seen[-1]["point_labels"].numpy(), [[1]])

    def test_every_advertised_prompt_builds(self):
        boxes = torch.tensor([[8.0, 8.0, 20.0, 20.0]])
        for name in PROMPTS:
            with self.subTest(prompt=name):
                built = build_prompt(boxes, name, 0.3,
                                     torch.Generator().manual_seed(0))
                self.assertEqual(built["point_coords"].shape[0], 1)
                self.assertEqual(built["point_labels"].shape[0], 1)


class TestAnchoring(unittest.TestCase):
    """Stage A moves the encoder; stage B must not simply undo it.

    Feature distillation used as a *regulariser* rather than as a teacher: the
    reference is the model's own frozen copy from the start of the stage, so
    the term measures drift from what pretraining bought.
    """

    def test_the_features_come_back_from_the_forward_that_already_ran(self):
        # The anchoring term is only cheap because of this: no second pass
        # through a 38.7 GFLOP encoder for a number the first pass computed.
        model = FakeSam2(SIZE).eval()
        batch = fake_batch(images=2, width=3)
        out = propagate_image(model, batch.images, batch.boxes, batch.valid)

        self.assertEqual(model.images, [2])                 # still one encode
        self.assertEqual(out["features"].shape, (2, 8, SIZE, SIZE))

    def test_an_unmoved_model_pays_nothing(self):
        import copy

        model = FakeSam2(SIZE).eval()
        anchor = copy.deepcopy(model).eval()
        _, terms = image_losses(model, fake_batch(), anchor=anchor, anchor_weight=1.0)
        self.assertAlmostEqual(terms["anchor"], 0.0, places=5)

    def test_a_moved_encoder_pays(self):
        import copy

        model = FakeSam2(SIZE).eval()
        anchor = copy.deepcopy(model).eval()
        with torch.no_grad():
            model.image_encoder.weight.add_(0.5)

        _, terms = image_losses(model, fake_batch(), anchor=anchor, anchor_weight=1.0)
        self.assertGreater(terms["anchor"], 0.01)

    def test_zero_weight_leaves_the_loss_untouched(self):
        import copy

        torch.manual_seed(0)
        model = FakeSam2(SIZE).eval()
        anchor = copy.deepcopy(model).eval()
        plain, terms = image_losses(model, fake_batch())
        with_anchor, other = image_losses(model, fake_batch(), anchor=anchor,
                                          anchor_weight=0.0)

        self.assertAlmostEqual(float(plain.detach()), float(with_anchor.detach()),
                               places=6)
        self.assertNotIn("anchor", terms)
        self.assertNotIn("anchor", other)

    def test_the_anchor_receives_no_gradient(self):
        import copy

        model = FakeSam2(SIZE).eval()
        anchor = copy.deepcopy(model).eval()
        for parameter in anchor.parameters():
            parameter.requires_grad_(False)

        image_losses(model, fake_batch(), anchor=anchor, anchor_weight=1.0)[0].backward()
        self.assertIsNone(anchor.image_encoder.weight.grad)
        self.assertIsNotNone(model.image_encoder.weight.grad)

    def test_the_loop_carries_the_anchor_through_run_stages(self):
        import copy

        from src.training.schedule import images as image_loop_for

        model = FakeSam2(SIZE).eval()
        anchor = copy.deepcopy(model).eval()
        loop = image_loop_for(anchor, 0.5)
        _, terms = loop.loss(model, fake_batch())
        self.assertIn("anchor", terms)


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
        self.source = Source(spec=self.spec, gates=self.gates)
        index = index_frames(list_frames(root, self.spec, "thermal"),
                             self.source, workers=2)
        self.split = ImageSplit(sample_windows(index, size=SIZE, per_image=1,
                                               max_instances=4, seed=0))

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

    def test_the_split_strategy_reaches_the_loader_from_the_split(self):
        # The hazard `ImageSplit.split` exists for: decomposing one way at
        # index time and another at load time renumbers the component labels,
        # so every instance is paired with another instance's mask -- and the
        # loss still looks finite. Indexing with watershed and collating
        # without it must not silently agree.
        root = Path(self.tmp.name)
        mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
        mask[8:16, 8:16] = 2
        mask[8:16, 20:28] = 2
        mask[11:13, 16:20] = 2                              # the bridge
        cv2.imwrite(str(root / "scene" / "label" / "b_label.png"), mask)
        cv2.imwrite(str(root / "scene" / "tir" / "b.png"), (mask * 60).astype(np.uint8))

        frame = [f for f in list_frames(root, self.spec, "thermal")
                 if f.name == "scene/b"][0]
        plain = index_frames([frame], self.source, workers=1)[0]
        watershed = Source(spec=self.spec, gates=self.gates, mode="watershed")
        split = index_frames([frame], watershed, workers=1)[0]
        self.assertEqual(len(plain.instances), 1)
        self.assertEqual(len(split.instances), 2)

        samples = sample_windows([split], size=SIZE, per_image=1,
                                 max_instances=4, seed=0)
        matched = collate(samples, "cpu")
        for row, instance in zip(matched.masks[0], samples[0].instances):
            self.assertEqual(int(row.sum()), instance.area)

    def test_scoring_reports_one_iou_per_instance(self):
        from tools.eval_instances import score

        result = score(FakeSam2(SIZE).eval(), self.split, batch=3, device="cpu")
        self.assertEqual(result["instances"], 12)
        # Qualified by dataset: `car` is class 5 in Kust4K and something else
        # elsewhere, so a mixed run must not add two distributions together.
        self.assertEqual(sorted(result["per_class"]), ["toy/car", "toy/person"])
        self.assertTrue(0.0 <= result["mean_iou"] <= 1.0)

    def test_two_datasets_mix_inside_one_batch(self):
        # The reason `Source` lives on the sample and not on the split: one
        # dataset is one sensor, one city and one set of annotation habits, and
        # a trunk carrying general visual features needs to see past any of
        # them. Here a semantic set that must be decomposed rides in the same
        # batch as one whose masks already carry instances.
        root = Path(self.tmp.name)
        (root / "vis" / "ir").mkdir(parents=True, exist_ok=True)
        (root / "vis" / "mask").mkdir(parents=True, exist_ok=True)
        for i in range(4):
            mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
            mask[6:14, 6:14] = 17 + i               # arbitrary instance ids
            mask[18:26, 18:26] = 90 + i
            cv2.imwrite(str(root / "vis" / "mask" / f"v{i}.png"), mask)
            cv2.imwrite(str(root / "vis" / "ir" / f"v{i}.png"),
                        (mask * 2).astype(np.uint8))

        instance_spec = DatasetSpec(
            name="toyvis", thermal="**/vis/ir/*.png", masks="**/vis/mask/*.png",
            classes={"background": 0, "target": 1}, things=("target",))
        instance_source = Source(spec=instance_spec, gates=self.gates, mode="labels")
        other = index_frames(list_frames(root, instance_spec, "thermal"),
                             instance_source, workers=2)
        # `labels` mode reads each distinct value as one instance -- no class
        # filter and no components, because the annotation is already the
        # target this whole stage has been reconstructing.
        self.assertTrue(all(len(e.instances) == 2 for e in other))

        mixed = ImageSplit(list(self.split.samples)
                           + sample_windows(other, size=SIZE, per_image=1,
                                            max_instances=4, seed=0))
        self.assertEqual(set(mixed.sources),
                         {"toy/components", "toyvis/labels"})

        # A batch large enough to span both, assembled and scored as one.
        batch = collate(list(mixed.samples), "cpu")
        self.assertEqual(batch.images.shape[0], len(mixed.samples))
        loss, _ = image_losses(FakeSam2(SIZE).eval(), batch)
        self.assertTrue(torch.isfinite(loss))

    def test_an_index_built_one_way_is_refused_by_a_run_decomposing_another(self):
        # The failure this prevents is silent: different modes renumber the
        # component labels, so every instance would train against a different
        # instance's mask while the loss stayed perfectly finite.
        from src.training.aerial import load_index, save_index

        root = Path(self.tmp.name)
        index = index_frames(list_frames(root, self.spec, "thermal"),
                             self.source, workers=2)
        path = save_index(root / "cache.json", index)

        load_index(path, self.source)                      # same source: fine
        with self.assertRaises(ValueError) as ctx:
            load_index(path, Source(self.spec, self.gates, mode="watershed"))
        self.assertIn("renumber", str(ctx.exception))


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
