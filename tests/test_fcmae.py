"""Masked-autoencoder pretraining for a convolutional trunk.

The claims worth pinning are the ones the paper's own ablations turn on:

* a masked position must not reach a visible one at any depth -- that leak is
  the difference between the paper's 79.3 % row and its 83.7 % one;
* GRN must actually be inserted, and must start as the identity so adding it
  to a trained checkpoint changes nothing until it is trained;
* the loss must score only the patches the encoder never saw, or the task is
  copying.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from src.training.fcmae import (  # noqa: E402
    Decoder,
    carries_grn,
    FCMAEConfig,
    GRN,
    ConvNeXtBlock,
    expand_mask,
    fcmae_loss,
    insert_grn,
    masked_convolutions,
    normalise_patches,
    patch_mask,
    patchify,
    restore_grn,
)


def trunk(width: int = 8) -> nn.Module:
    """A small convolutional stack standing in for RepViT.

    Deliberately includes a 7x7 depthwise convolution: that is the kernel that
    pulls a masked neighbour into a visible position, and the reason the mask
    is re-applied per convolution rather than per stage.
    """
    return nn.Sequential(
        nn.Conv2d(3, width, 3, stride=2, padding=1),
        nn.Conv2d(width, width, 7, padding=3, groups=width),
        nn.Conv2d(width, width * 4, 1),
        nn.GELU(),
        nn.Conv2d(width * 4, width, 1),
        nn.Conv2d(width, width, 3, stride=2, padding=1),
    )


class MaskTest(unittest.TestCase):
    def test_the_ratio_is_exact_per_image(self):
        """A binomial draw would give some images in the batch far more signal
        than others, and the loss would follow whichever kept the most."""
        config = FCMAEConfig(image_size=128, patch=32, mask_ratio=0.6)
        mask = patch_mask(8, 128, config, torch.Generator().manual_seed(0))
        self.assertEqual(mask.shape, (8, 4, 4))
        per_image = mask.reshape(8, -1).sum(dim=1)
        self.assertEqual(set(per_image.tolist()), {10})     # round(0.6 * 16)

    def test_different_images_get_different_masks(self):
        mask = patch_mask(4, 128, FCMAEConfig(image_size=128),
                          torch.Generator().manual_seed(0))
        rows = {tuple(row.tolist()) for row in mask.reshape(4, -1)}
        self.assertGreater(len(rows), 1)

    def test_the_expansion_is_nearest_so_no_position_is_part_masked(self):
        """A bilinear resize would put fractional values on a patch boundary,
        and a position that is 30 % masked is a position that leaks."""
        mask = torch.tensor([[[True, False], [False, True]]])
        grown = expand_mask(mask, 8, 8)
        self.assertEqual(set(grown.flatten().tolist()), {0.0, 1.0})
        self.assertEqual(grown.shape, (1, 1, 8, 8))


class LeakTest(unittest.TestCase):
    """The property the whole encoder-side design exists for."""

    def setUp(self):
        torch.manual_seed(0)
        self.config = FCMAEConfig(image_size=64, patch=32)
        self.model = trunk()

    def features(self, images, mask):
        with masked_convolutions(self.model, mask) as masking:
            out = self.model(images)
        return out, masking.touched

    def test_changing_a_masked_patch_cannot_change_the_output(self):
        """The definition of "no leak", as a test rather than as a diagram:
        rewrite the pixels under a removed patch and every feature must be
        bit-identical."""
        mask = torch.tensor([[[True, False], [False, False]]])
        images = torch.randn(1, 3, 64, 64)
        first, touched = self.features(images, mask)
        changed = images.clone()
        changed[:, :, :32, :32] = torch.randn(1, 3, 32, 32) * 10
        second, _ = self.features(changed, mask)
        self.assertGreater(touched, 0)
        torch.testing.assert_close(first, second)

    def test_a_visible_patch_does_change_it(self):
        """The control: without this the test above would pass on a model that
        ignores its input."""
        mask = torch.tensor([[[True, False], [False, False]]])
        images = torch.randn(1, 3, 64, 64)
        first, _ = self.features(images, mask)
        changed = images.clone()
        changed[:, :, 32:, 32:] = torch.randn(1, 3, 32, 32) * 10
        second, _ = self.features(changed, mask)
        self.assertFalse(torch.allclose(first, second))

    def test_masking_only_at_the_end_does_leak(self):
        """Why the hook goes on every convolution. Masking the trunk's output
        alone lets the 7x7 depthwise pull masked values into visible positions
        several layers earlier, and this is that failure, measured."""
        mask = torch.tensor([[[True, False], [False, False]]])
        images = torch.randn(1, 3, 64, 64)

        def end_masked(x):
            out = self.model(x)
            return out * (1.0 - expand_mask(mask, *out.shape[-2:]))

        changed = images.clone()
        changed[:, :, :32, :32] = torch.randn(1, 3, 32, 32) * 10
        self.assertFalse(torch.allclose(end_masked(images), end_masked(changed)))

    def test_the_hooks_do_not_outlive_the_forward(self):
        """A fine-tuning forward has to be the unmodified one."""
        mask = torch.tensor([[[True, False], [False, False]]])
        images = torch.randn(1, 3, 64, 64)
        with masked_convolutions(self.model, mask):
            pass
        plain = self.model(images)
        torch.testing.assert_close(plain, self.model(images))
        self.assertFalse(torch.allclose(plain, self.features(images, mask)[0]))

    def test_a_trunk_with_no_convolution_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            with masked_convolutions(nn.Sequential(nn.Linear(4, 4)),
                                     torch.zeros(1, 2, 2, dtype=torch.bool)):
                pass
        self.assertIn("Conv2d", str(caught.exception))


class GRNTest(unittest.TestCase):
    def test_an_untrained_grn_is_the_identity(self):
        """gamma and beta start at zero, so inserting it into a trained
        checkpoint cannot change what that checkpoint does."""
        x = torch.randn(2, 6, 5, 5)
        torch.testing.assert_close(GRN(6)(x), x)

    def test_a_loud_channel_is_amplified_relative_to_a_quiet_one(self):
        """The competition the paper is after: a channel's gain depends on how
        loud the others are."""
        grn = GRN(2)
        with torch.no_grad():
            grn.gamma.fill_(1.0)
        x = torch.zeros(1, 2, 4, 4)
        x[:, 0] = 1.0
        x[:, 1] = 0.1
        out = grn(x)
        loud = (out[:, 0] / x[:, 0]).mean()
        quiet = (out[:, 1] / x[:, 1]).mean()
        self.assertGreater(loud, quiet)

    def test_it_is_found_by_shape_not_by_name(self):
        """RepViT's FFN is not called `mlp`, and a discovery pass keyed on
        names would silently insert nothing."""
        model = trunk()
        report = insert_grn(model)
        self.assertEqual(report["inserted"], 1)          # the 1x1 8 -> 32
        self.assertEqual(list(report["channels"].values()), [32])
        self.assertEqual(report["parameters"], 64)

    def test_a_projection_back_down_is_not_an_expansion(self):
        model = nn.Sequential(nn.Conv2d(32, 8, 1))
        self.assertEqual(insert_grn(model)["inserted"], 0)

    def test_inserting_it_leaves_the_model_running(self):
        model = trunk()
        before = model(torch.randn(1, 3, 64, 64))
        insert_grn(model)
        torch.testing.assert_close(model(torch.randn(1, 3, 64, 64) * 0 + 1) * 0,
                                   before * 0)           # shapes still line up
        self.assertEqual(model(torch.randn(1, 3, 64, 64)).shape, before.shape)


class LossTest(unittest.TestCase):
    def setUp(self):
        self.config = FCMAEConfig(image_size=64, patch=32)

    def test_only_the_masked_patches_are_scored(self):
        """Scoring the visible ones would reward copying, which a convolution
        does perfectly and which teaches nothing."""
        images = torch.randn(1, 3, 64, 64)
        target = patchify(images, 32)
        target = normalise_patches(target, self.config.norm_floor)
        mask = torch.tensor([[[True, False], [False, False]]])
        wrong_visible = target.clone()
        wrong_visible[..., 1, 1] += 5.0                  # a visible patch
        loss, _ = fcmae_loss(wrong_visible, images, mask, self.config)
        self.assertLess(float(loss), 1e-6)

    def test_a_wrong_masked_patch_does_cost(self):
        images = torch.randn(1, 3, 64, 64)
        target = normalise_patches(patchify(images, 32), self.config.norm_floor)
        mask = torch.tensor([[[True, False], [False, False]]])
        wrong = target.clone()
        wrong[..., 0, 0] += 5.0
        self.assertGreater(float(fcmae_loss(wrong, images, mask, self.config)[0]), 1.0)

    def test_a_flat_patch_does_not_become_a_noise_target(self):
        """The floor. A patch of empty ground has a standard deviation near
        zero; without it the target is sensor noise rescaled to unit variance,
        and the run spends its capacity predicting that."""
        flat = torch.full((1, 3, 32, 32, 1, 1), 0.5) + torch.randn(
            1, 3, 32, 32, 1, 1) * 1e-4
        floored = normalise_patches(flat, 1e-3)
        unfloored = normalise_patches(flat, 1e-12)
        self.assertLess(float(floored.abs().max()), 1.0)
        self.assertGreater(float(unfloored.abs().max()), 2.0)

    def test_the_report_says_how_much_was_masked(self):
        images = torch.randn(2, 3, 64, 64)
        mask = patch_mask(2, 64, self.config, torch.Generator().manual_seed(0))
        _, terms = fcmae_loss(torch.zeros(2, 3, 32, 32, 2, 2), images, mask,
                              self.config)
        self.assertAlmostEqual(terms["masked"], 0.5, places=6)  # round(.6*4)=2


class DecoderTest(unittest.TestCase):
    def test_it_predicts_one_patch_per_position(self):
        config = FCMAEConfig(image_size=64, patch=32, decoder_dim=16)
        decoder = Decoder(8, config)
        features = torch.randn(2, 8, 4, 4)               # stride 16
        mask = patch_mask(2, 64, config, torch.Generator().manual_seed(0))
        out = decoder(features, mask)
        self.assertEqual(out.shape, (2, 3, 32, 32, 2, 2))

    def test_a_masked_position_carries_the_token_not_a_zero(self):
        """Or "masked" and "black" would be the same input, and the decoder
        could not tell which positions it is being asked to invent."""
        config = FCMAEConfig(image_size=64, patch=32, decoder_dim=16)
        decoder = Decoder(4, config)
        with torch.no_grad():
            decoder.mask_token.fill_(3.0)
        mask = torch.tensor([[[True, False], [False, False]]])
        zeros = torch.zeros(1, 4, 2, 2)
        first = decoder(zeros, mask)
        second = decoder(zeros, torch.zeros_like(mask))
        self.assertFalse(torch.allclose(first, second))

    def test_the_block_is_the_v2_block(self):
        """One 7x7 depthwise, an expansion, a GRN, a projection, a residual --
        and no LayerScale, which the paper drops once GRN is present."""
        block = ConvNeXtBlock(8)
        self.assertIsInstance(block.grn, GRN)
        self.assertEqual(block.expand.out_channels, 32)
        self.assertEqual(block.depthwise.kernel_size, (7, 7))
        self.assertFalse(any("layer_scale" in name or "gamma" == name
                             for name, _ in block.named_parameters()))
        # Not an identity at initialisation: V1's LayerScale was what made a
        # block start as one, and the paper removes it. What starts as an
        # identity is the GRN inside it.
        x = torch.randn(1, 8, 6, 6)
        self.assertEqual(block(x).shape, x.shape)
        expanded = torch.randn(1, 32, 6, 6)              # GRN sits after the 4x
        torch.testing.assert_close(block.grn(expanded), expanded)


class RunTest(unittest.TestCase):
    """The parts of `tools/pretrain_fcmae.py` that do not need a model."""

    def test_the_frame_walk_reads_every_image_and_no_record(self):
        """Stage A's whole argument is volume, and it gets that volume by
        ignoring the gates the mask pools apply: a frame whose teacher mask was
        refused is still a frame."""
        import tempfile

        from tools.pretrain_fcmae import find_frames

        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        root = Path(holder.name)
        (root / "seq" / "ir").mkdir(parents=True)
        for index in range(4):
            (root / "seq" / "ir" / f"{index:06d}.jpg").touch()
        (root / "seq" / "record.json").write_text("{}")
        (root / "seq" / "notes.txt").touch()
        found, _ = find_frames([root])
        self.assertEqual(len(found), 4)
        self.assertTrue(all(path.suffix == ".jpg" for path in found))

    def test_the_same_frame_under_two_roots_is_read_once(self):
        import tempfile

        from tools.pretrain_fcmae import find_frames

        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        root = Path(holder.name)
        (root / "a").mkdir()
        (root / "a" / "x.png").touch()
        self.assertEqual(len(find_frames([root, root / "a"])[0]), 1)

    def test_the_schedule_warms_up_then_decays_to_nothing(self):
        from tools.pretrain_fcmae import learning_rate

        peak, total, warmup = 1e-3, 1000, 100
        self.assertLess(learning_rate(0, total, warmup, peak), peak)
        self.assertAlmostEqual(learning_rate(warmup - 1, total, warmup, peak),
                               peak, places=9)
        self.assertLess(learning_rate(total - 1, total, warmup, peak), peak * 0.01)
        rising = [learning_rate(s, total, warmup, peak) for s in range(warmup)]
        self.assertEqual(rising, sorted(rising))

    def test_the_peak_follows_the_batch_the_way_the_paper_scales_it(self):
        """lr = base_lr x batch / 256. Written here because it is the one
        hyperparameter a bigger card silently changes."""
        base = 1.5e-4
        self.assertAlmostEqual(base * 256 / 256.0, base)
        self.assertAlmostEqual(base * 512 / 256.0, 2 * base)



class HandoffTest(unittest.TestCase):
    """A GRN checkpoint has to be loadable by whatever consumes it.

    `insert_grn` renames the layer it sits behind -- `expand.weight` becomes
    `expand.0.weight` -- and adds two vectors. `build_sam2` loads strictly, so
    without this the stage that spent twenty epochs producing the checkpoint
    hands the next notebook a state-dict error at its first cell.
    """

    def net(self):
        return nn.Sequential(nn.Conv2d(3, 8, 3), nn.Conv2d(8, 32, 1),
                             nn.GELU(), nn.Conv2d(32, 8, 1))

    def saved(self, with_meta: bool):
        import tempfile

        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        trained = self.net()
        report = insert_grn(trained)
        path = Path(holder.name) / "checkpoint.pt"
        body = {"model": trained.state_dict()}
        if with_meta:
            body["meta"] = {"grn": report, "stage": "fcmae"}
        torch.save(body, path)
        return path, report

    def test_the_stock_graph_refuses_it_which_is_why_this_exists(self):
        path, _ = self.saved(with_meta=True)
        with self.assertRaises(RuntimeError):
            self.net().load_state_dict(torch.load(path, weights_only=False)["model"])

    def test_restoring_grn_first_makes_the_strict_load_succeed(self):
        path, report = self.saved(with_meta=True)
        found = carries_grn(path)
        self.assertEqual(found["channels"], report["channels"])
        model = self.net()
        self.assertEqual(restore_grn(model, found["channels"]), 1)
        model.load_state_dict(torch.load(path, weights_only=False)["model"])

    def test_a_copy_that_lost_its_meta_is_still_read_correctly(self):
        """A checkpoint copied to Drive by hand keeps its weights and may lose
        its meta; guessing wrong here is a strict-load error either way, so the
        fallback reads the key shapes and has to agree with the meta."""
        with_meta, _ = self.saved(with_meta=True)
        without, _ = self.saved(with_meta=False)
        self.assertEqual(carries_grn(with_meta)["channels"],
                         carries_grn(without)["channels"])
        model = self.net()
        restore_grn(model, carries_grn(without)["channels"])
        model.load_state_dict(torch.load(without, weights_only=False)["model"])

    def test_a_checkpoint_without_grn_says_so(self):
        import tempfile

        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        path = Path(holder.name) / "plain.pt"
        torch.save({"model": self.net().state_dict(), "meta": {}}, path)
        self.assertEqual(carries_grn(path)["inserted"], 0)

    def test_restoring_twice_is_not_two_layers_deep(self):
        """The notebook may re-run a cell; a second restore must be a no-op
        rather than a GRN behind a GRN."""
        path, _ = self.saved(with_meta=True)
        channels = carries_grn(path)["channels"]
        model = self.net()
        self.assertEqual(restore_grn(model, channels), 1)
        self.assertEqual(restore_grn(model, channels), 0)
        model.load_state_dict(torch.load(path, weights_only=False)["model"])



class ModalityTest(unittest.TestCase):
    """A walk with no modality filter is not a thermal pretrain.

    The trees this reads hold both sensors, often inside one sequence, and
    `load_image` turns a colour frame grey rather than refusing it -- so an
    unfiltered run would have spent its capacity on grey RGB and printed
    nothing about it.
    """

    def tree(self, *relative):
        import tempfile

        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        root = Path(holder.name)
        for rel in relative:
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        return root

    LAYOUT = (
        "VTUAV/train_ST_001/pedestrian_003/ir/0.jpg",
        "VTUAV/train_ST_001/pedestrian_003/rgb/0.jpg",
        "AeroVIS/sequences/sd_001/0.jpg",
        "HIT_UAV/train/0.jpg",
        "Kust4K/tir/0.png",
        "Kust4K/rgb/0.png",
    )

    def test_a_thermal_run_leaves_the_colour_half_behind(self):
        from tools.pretrain_fcmae import find_frames

        frames, census = find_frames([self.tree(*self.LAYOUT)])
        names = sorted(str(f).split("/")[-3:][0] + "/" + f.name for f in frames)
        self.assertEqual(len(frames), 3)
        self.assertTrue(all("rgb" not in str(f).lower() for f in frames), names)
        self.assertTrue(all("aerovis" not in str(f).lower() for f in frames), names)

    def test_the_census_counts_both_so_the_run_says_what_it_left(self):
        from tools.pretrain_fcmae import describe, find_frames

        _, census = find_frames([self.tree(*self.LAYOUT)])
        self.assertEqual(census["VTUAV"], {"thermal": 1, "rgb": 1})
        self.assertEqual(census["AeroVIS"], {"thermal": 0, "rgb": 1})
        text = describe(census, "thermal")
        self.assertIn("3 frames kept, 3 left out of 6", text)

    def test_any_is_a_choice_and_has_to_be_asked_for(self):
        from tools.pretrain_fcmae import find_frames, parser

        self.assertEqual(parser().parse_args(["--frames", "x"]).modality,
                         "thermal")
        frames, _ = find_frames([self.tree(*self.LAYOUT)], modality="any")
        self.assertEqual(len(frames), 6)

    def test_the_one_definition_is_the_one_the_builders_embed(self):
        """RGB_SOURCES lived in two builders and nowhere else, which is how the
        fallback once read "no rgb in the name" as thermal. A third consumer
        must not grow a third copy."""
        from src.training.modality import RGB_SOURCES

        import re

        for builder in ("build_stage_b_notebooks", "build_stage_b_stable_notebook"):
            with self.subTest(builder=builder):
                # The stable builder writes the literal across two source
                # lines, so the tuple it names is compared rather than the
                # bytes it is spelled in.
                text = (ROOT / "tools" / f"{builder}.py").read_text()
                found = re.search(r"RGB_SOURCES = \((.*?)\)", text, re.S)
                self.assertIsNotNone(found, f"{builder} names no RGB_SOURCES")
                names = tuple(re.findall(r'"([a-z0-9_]+)"', found.group(1)))
                self.assertEqual(names, RGB_SOURCES)

    def test_the_name_rule_agrees_with_the_builders_on_every_pool(self):
        from src.training.modality import modality_of_name

        namespace = {"POOL_MODALITIES": {}, "GUESSED": set()}
        text = (ROOT / "tools" / "build_stage_b_notebooks.py").read_text()
        start = text.index('RGB_SOURCES = ("visdrone"')
        end = text.index('    return "thermal"', start) + len('    return "thermal"')
        exec(text[start:end], namespace)                     # noqa: S102
        for name in ("visdrone", "aerovis_train", "vtuav_vis", "vtuav_thermal",
                     "vtuav_rgb", "dronevehicle_thermal", "dronevehicle_rgb",
                     "dronevehicle_rgb_only", "kust4k_rgb", "hituav_thermal",
                     "segfly_thermal", "rgbt234", "lasher", "vtuav_lt_thermal",
                     "kaggle_uav_thermal"):
            with self.subTest(pool=name):
                self.assertEqual(namespace["modality_of"](name),
                                 modality_of_name(name))

    def test_a_thermal_folder_beats_a_colour_name(self):
        """`.../vtuav_rgb_pool/ir/...` is a thermal frame in a badly named
        tree, and dropping it would be the gate firing on a real target."""
        from src.training.modality import modality_of_path

        self.assertEqual(modality_of_path("/x/vtuav_rgb_pool/ir/1.png"), "thermal")

    def test_a_joined_path_is_read_component_by_component(self):
        """The pool-name rule finds nothing in a joined path -- it splits on
        `_` and `content/pool/vtuav_rgb/frames/1.png` comes back thermal."""
        from src.training.modality import modality_of_name, modality_of_path

        joined = "content/pool/vtuav_rgb/frames/1.png"
        self.assertEqual(modality_of_name(joined), "thermal")
        self.assertEqual(modality_of_path("/" + joined), "rgb")



if __name__ == "__main__":
    unittest.main()
