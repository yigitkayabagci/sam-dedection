"""The loop both methods run through.

The point of this suite is the property the comparison depends on: swapping
`freeze` and `save` changes *what moves*, and nothing else. Same stages, same
batch order, same schedule, same validation clips, same rule for keeping a
checkpoint. If those drifted apart, the fine-tune-versus-LoRA number would be
measuring two notebooks.
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

from src.training.antiuav import Clip, Sequence, SequenceLabels  # noqa: E402
from src.training.finetune import Rates, apply_freeze  # noqa: E402
from src.training.schedule import Schedule, Split, run_stages, validate  # noqa: E402
from tests.test_clip_loop import FakeSam2  # noqa: E402
from tools.train_encoder import scaled_rates  # noqa: E402

SIZE = 32
FRAMES = 4


def make_split(sequences: int = 3, clips_each: int = 8) -> Split:
    clips, stores = [], {}
    for s in range(sequences):
        name = f"s{s}"
        exist = np.ones(FRAMES * clips_each + FRAMES, dtype=bool)
        boxes = np.tile(np.array([8.0, 8.0, 20.0, 20.0], np.float32), (exist.size, 1))
        sequence = Sequence(name=name, split="train",
                            frames=tuple(Path(f"/nonexistent/{i}.jpg")
                                         for i in range(exist.size)),
                            labels=SequenceLabels(exist=exist, boxes=boxes))
        for c in range(clips_each):
            clips.append(Clip(sequence, tuple(range(c, c + FRAMES)), (0, 0),
                              (SIZE, SIZE), SIZE))
        mask = np.zeros((SIZE, SIZE), dtype=bool)
        mask[8:20, 8:20] = True
        stores[name] = {i: mask for i in range(exist.size)}
    return Split(clips=clips, stores=stores)


def patch_pixels(test):
    """Make `clip_tensor` synthesise frames instead of decoding them."""
    import src.training.clip_loop as clip_loop

    original = clip_loop.clip_tensor
    clip_loop.clip_tensor = lambda clip, device="cpu": torch.randn(
        len(clip.indices), 3, clip.size, clip.size, device=device)
    test.addCleanup(lambda: setattr(clip_loop, "clip_tensor", original))


class TestScaledRates(unittest.TestCase):
    """Two knobs, in order: the batch's scaling rule keeps the table's shape,
    and an absolute override changes it."""

    BASE = Rates(head=5e-5, neck=5e-5, trunk=1e-5, weight_decay=1e-4)

    def test_scale_one_and_no_override_is_the_table_itself(self):
        self.assertEqual(scaled_rates(self.BASE), self.BASE)
        self.assertEqual(scaled_rates(self.BASE, 1.0, {}), self.BASE)

    def test_the_scale_multiplies_every_part_and_leaves_decay_alone(self):
        out = scaled_rates(self.BASE, 4.0)
        self.assertAlmostEqual(out.head, 2e-4)
        self.assertAlmostEqual(out.neck, 2e-4)
        self.assertAlmostEqual(out.trunk, 4e-5)
        self.assertEqual(out.weight_decay, self.BASE.weight_decay)

    def test_an_override_is_absolute_and_wins_over_the_scale(self):
        out = scaled_rates(self.BASE, 4.0, {"head": 1e-5, "trunk": 1e-4})
        self.assertAlmostEqual(out.head, 1e-5, msg="not 1e-5 * 4")
        self.assertAlmostEqual(out.trunk, 1e-4)
        self.assertAlmostEqual(out.neck, 2e-4, msg="unnamed parts still scale")

    def test_a_zero_override_leaves_that_part_alone(self):
        out = scaled_rates(self.BASE, 2.0, {"head": 0.0, "trunk": 1e-4})
        self.assertAlmostEqual(out.head, 1e-4)
        self.assertAlmostEqual(out.trunk, 1e-4)

    def test_the_shape_can_be_inverted_so_the_trunk_leads(self):
        out = scaled_rates(self.BASE, 1.0, {"head": 1e-5, "trunk": 1e-4})
        self.assertGreater(out.trunk, out.head,
                           "a modality shift lives in the trunk; a run that "
                           "wants the encoder to learn it must be able to say so")


class TestValidate(unittest.TestCase):
    def setUp(self):
        patch_pixels(self)
        self.model = FakeSam2(SIZE).eval()
        self.split = make_split()

    def test_the_slice_is_bounded_while_generating_not_afterwards(self):
        # The bug this replaces: `[... for b in batches(...)][:limit]` builds
        # every batch first, so the limit costs the entire val set per epoch.
        seen = []
        import src.training.schedule as schedule_mod
        real = schedule_mod.clip_losses
        schedule_mod.clip_losses = lambda m, b, **k: (seen.append(1), real(m, b, **k))[1]
        self.addCleanup(lambda: setattr(schedule_mod, "clip_losses", real))

        validate(self.model, self.split, Schedule(batch=2, val_batches=3), "cpu")
        self.assertEqual(len(seen), 3)

    def test_the_same_clips_are_scored_every_time(self):
        # Fixed seed, fixed slice: an epoch-to-epoch change in the number is
        # the model, never the sample it was measured on.
        seen = []
        import src.training.schedule as schedule_mod
        real = schedule_mod.clip_losses

        def record(model, batch, **kwargs):
            seen.append([(c.sequence.name, c.indices) for c in batch.clips])
            return real(model, batch, **kwargs)

        schedule_mod.clip_losses = record
        self.addCleanup(lambda: setattr(schedule_mod, "clip_losses", real))

        schedule = Schedule(batch=2, val_batches=2)
        validate(self.model, self.split, schedule, "cpu")
        first = list(seen)
        seen.clear()
        validate(self.model, self.split, schedule, "cpu")
        self.assertEqual(first, seen)


class TestRunStages(unittest.TestCase):
    def setUp(self):
        patch_pixels(self)
        torch.manual_seed(0)
        self.model = FakeSam2(SIZE).eval()
        self.train, self.val = make_split(), make_split(sequences=2)
        self.schedule = Schedule(
            stages=(("head", 1, Rates(head=1e-3)),
                    ("encoder", 1, Rates(head=1e-3, neck=1e-3, trunk=1e-4))),
            batch=2, steps_per_epoch=3, val_batches=2, workers=2, depth=1)

    def run_it(self, **kwargs):
        saves = []
        result = run_stages(
            self.model, self.train, self.val, self.schedule,
            freeze=kwargs.get("freeze", apply_freeze),
            save=kwargs.get("save", lambda m, meta: saves.append(meta)),
            device="cpu", log=lambda *_: None)
        return result, saves

    def test_it_runs_every_stage_and_epoch(self):
        result, _ = self.run_it()
        self.assertEqual([h["stage"] for h in result["history"]], ["head", "encoder"])
        self.assertEqual(result["stages"], ["head", "encoder"])

    def test_the_first_epoch_always_saves_and_later_ones_only_improve(self):
        result, saves = self.run_it()
        self.assertTrue(result["history"][0]["saved"])
        self.assertEqual(len(saves), sum(h["saved"] for h in result["history"]))
        self.assertEqual(result["best_val_loss"],
                         min(h["val_loss"] for h in result["history"]))

    def test_the_freeze_callback_decides_what_moves(self):
        stages_seen = []

        def freeze(model, stage):
            stages_seen.append(stage)
            return apply_freeze(model, stage)

        self.run_it(freeze=freeze)
        self.assertEqual(stages_seen, ["head", "encoder"])

    def test_lora_and_the_fine_tune_take_the_same_path_through_the_loop(self):
        # Same stages, same number of saves, same history shape -- the only
        # difference between the two runs is which parameters received a
        # gradient. That is what makes the comparison a comparison.
        from src.training import lora

        self.schedule = Schedule(
            stages=(("encoder", 1, Rates(head=1e-3, neck=1e-3, trunk=1e-3)),),
            batch=2, steps_per_epoch=3, val_batches=2, workers=2, depth=1)
        torch.manual_seed(0)
        full, _ = self.run_it()

        torch.manual_seed(0)
        self.model = FakeSam2(SIZE).eval()
        lora.inject(self.model, "encoder", r=2)
        adapted, _ = self.run_it(freeze=lora.freeze)

        self.assertEqual([h["stage"] for h in full["history"]],
                         [h["stage"] for h in adapted["history"]])
        self.assertEqual(len(full["history"]), len(adapted["history"]))
        self.assertEqual([h["epoch"] for h in full["history"]],
                         [h["epoch"] for h in adapted["history"]])

    def test_the_weights_actually_move(self):
        before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        self.run_it()
        moved = [n for n, p in self.model.named_parameters()
                 if not torch.equal(p.detach(), before[n])]
        self.assertTrue(moved)

    def test_the_memory_path_never_moves(self):
        before = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        self.run_it()
        for name, param in self.model.named_parameters():
            if name.startswith(("memory_attention", "memory_encoder", "spatial_perceiver")):
                self.assertTrue(torch.equal(param.detach(), before[name]), name)

    def test_meta_reaches_the_checkpoint(self):
        self.schedule = Schedule(
            stages=(("head", 1, Rates(head=1e-3)),), batch=2, steps_per_epoch=2,
            val_batches=1, workers=2, depth=1, meta={"method": "lora", "r": 8})
        _, saves = self.run_it()
        self.assertEqual(saves[0]["method"], "lora")
        self.assertEqual(saves[0]["r"], 8)

    def test_a_lora_run_saves_a_checkpoint_the_stock_loader_accepts(self):
        from src.training import lora

        self.model = FakeSam2(SIZE).eval()
        lora.inject(self.model, "encoder", r=2)
        self.schedule = Schedule(
            stages=(("encoder", 1, Rates(head=1e-3, neck=1e-3, trunk=1e-3)),),
            batch=2, steps_per_epoch=2, val_batches=1, workers=2, depth=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lora.pt"
            self.run_it(freeze=lora.freeze,
                        save=lambda m, meta: lora.save_merged_checkpoint(m, path, meta))
            saved = torch.load(path, weights_only=False)
        self.assertEqual(list(saved["model"]), list(FakeSam2(SIZE).state_dict()))


if __name__ == "__main__":
    unittest.main()
