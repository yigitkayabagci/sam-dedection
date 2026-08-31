"""The three axes run_records.py varies, and the ways they must not be confused.

A mode is an input configuration -- a resolution and whether the frame is
cropped or resized into it. `--weights` is a different question entirely: which
trained checkpoint runs inside that configuration. The comparison is only worth
anything while each row of the summary holds one of them fixed, so what is
pinned here is that no combination can silently produce a row labelled one
thing and measured on another:

  * every (weights, size) pair resolves to a config that exists, declares that
    size, and takes its engines from a directory of its own;
  * a pair with no config raises instead of falling back to stock;
  * the output folder separates weights, policy and skip runs from the baseline.

`--policy` is the third, and it is the one that has to be held to a stricter
rule than the other two, because it is the only one that edits a config on the
way past. An overlay that set a `checkpoint:` would make `--policy` a second
way to change the weights and every `--weights` label in the summary would stop
being true, so what is pinned is that an overlay touches runtime blocks and
nothing else -- and that `plain` really is plain, which is only true while no
backend config in the table carries a policy block of its own.

None of it needs torch, EdgeTAM or an engine -- it is table arithmetic, which
is why it can guard every combination rather than the one being worked on.
"""
from __future__ import annotations

import sys
import unittest
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from tools.run_records import (MODES, POLICIES, POLICY_KEYS, POLICY_NOTE,  # noqa: E402
                               TRAINED_AT, WEIGHTS, config_for, digest,
                               cache_for, engines_missing, folder,
                               overlay_for, pointers_missing, provenance,
                               staged_config)


def body(config: str) -> dict:
    return yaml.safe_load((ROOT / config).read_text()) or {}


class Tables(unittest.TestCase):
    def test_every_mode_names_a_size_a_weights_set_could_serve(self):
        for mode, (size, crop) in MODES.items():
            self.assertIn(size, WEIGHTS["stock"], mode)
            self.assertIn(crop, (None, size),
                          f"{mode}: a crop is either off or the model input")

    def test_every_config_in_the_table_exists(self):
        for weights, table in WEIGHTS.items():
            for size, config in table.items():
                self.assertTrue((ROOT / config).is_file(),
                                f"{weights}/{size} -> {config} is not there")

    def test_every_config_declares_the_size_it_is_filed_under(self):
        """The table's key and the YAML's own `image_size` cannot disagree.

        They are what makes `--modes full768` mean 768: the mode picks the key,
        the tracker obeys the YAML, and a mismatch would run one and report the
        other. An absent `image_size` is not a mismatch -- it means the Hydra
        config's own 1024 stands, which is what configs/edgetam_trt.yaml relies
        on and why it is filed under 1024 while declaring nothing.
        """
        for weights, table in WEIGHTS.items():
            for size, config in table.items():
                declared = body(config).get("image_size", 1024)
                self.assertEqual(int(declared), size,
                                 f"{weights}/{size} -> {config}")

    def test_every_config_is_a_tensorrt_one(self):
        """run_records always runs `--tracker edgetam_trt`."""
        for weights, table in WEIGHTS.items():
            for size, config in table.items():
                self.assertIn("image_encoder_engine", body(config),
                              f"{weights}/{size} -> {config} has no engines")

    def test_each_weights_set_takes_its_engines_from_its_own_directory(self):
        """Engines are weight-specific: sharing a directory overwrites one set."""
        seen: dict[str, tuple[str, int]] = {}
        for weights, table in WEIGHTS.items():
            for size, config in table.items():
                directory = Path(body(config)["image_encoder_engine"]).parent.name
                self.assertNotIn(directory, seen,
                                 f"{weights}/{size} shares {directory}/ with "
                                 f"{seen.get(directory)}")
                seen[directory] = (weights, size)

    def test_each_weights_set_names_one_checkpoint(self):
        """A set that mixed checkpoints across sizes would not be one model."""
        for weights, table in WEIGHTS.items():
            checkpoints = {body(c)["checkpoint"] for c in table.values()}
            self.assertEqual(len(checkpoints), 1,
                             f"{weights} spans {checkpoints}")

    def test_pool_deep_is_the_thermal_stage_b_checkpoint_at_its_own_size(self):
        self.assertEqual(TRAINED_AT["pool_deep"], 512)
        self.assertIn("thermal_deep", body(WEIGHTS["pool_deep"][512])["checkpoint"])
        self.assertNotIn(1024, WEIGHTS["pool_deep"],
                         "a 512-trained checkpoint at 1024 measures the "
                         "resolution, not the training")


class ConfigFor(unittest.TestCase):
    def test_resolves_a_pair_that_exists(self):
        self.assertEqual(config_for("pool_deep", 512, "full512"),
                         WEIGHTS["pool_deep"][512])

    def test_refuses_a_pair_that_does_not_rather_than_falling_back(self):
        """Falling back to stock would label the row with the wrong weights."""
        with self.assertRaises(SystemExit) as raised:
            config_for("pool_deep", 1024, "full1024")
        message = str(raised.exception)
        self.assertIn("full1024", message)
        self.assertIn("stock", message)


class Folder(unittest.TestCase):
    @staticmethod
    def args(weights="stock", frame_skip=1, policy="plain"):
        return Namespace(weights=weights, frame_skip=frame_skip, policy=policy)

    def test_a_plain_stock_run_keeps_the_bare_name(self):
        """Runs written before there was a second axis stay where they are."""
        self.assertEqual(folder("full512", self.args()), "full512")

    def test_other_weights_land_beside_the_baseline(self):
        self.assertEqual(folder("full512", self.args(weights="pool_deep")),
                         "full512_pool_deep")

    def test_a_skip_run_still_gets_its_own_folder(self):
        self.assertEqual(folder("full512", self.args(frame_skip=2)),
                         "full512_skip2")

    def test_both_axes_at_once_stay_distinct(self):
        names = {folder("full512", self.args(w, s))
                 for w in WEIGHTS for s in (1, 2)}
        self.assertEqual(len(names), 2 * len(WEIGHTS))

    def test_a_policy_run_lands_beside_the_baseline_not_on_it(self):
        """The ladder is only readable while its rungs are separate folders."""
        self.assertEqual(folder("full512", self.args(policy="guard")),
                         "full512_guard")
        self.assertEqual(
            folder("full512", self.args(weights="pool_deep", policy="ego")),
            "full512_pool_deep_ego")

    def test_all_three_axes_at_once_stay_distinct(self):
        names = {folder("full512", self.args(w, s, p))
                 for w in WEIGHTS for s in (1, 2) for p in POLICIES}
        self.assertEqual(len(names), 2 * len(WEIGHTS) * len(POLICIES))


class Policies(unittest.TestCase):
    """The third axis, and the rule that keeps it from becoming the second one.

    An overlay states runtime behaviour. The moment one of them could set a
    `checkpoint:` or an engine path, `--policy` would be a way to change the
    weights while the summary still printed the `--weights` it was given, and
    every row comparing the two would be wrong in a way nothing would catch.
    """

    def test_every_named_overlay_exists(self):
        for policy, path in POLICIES.items():
            if path is not None:
                self.assertTrue((ROOT / path).is_file(), policy)

    def test_the_baseline_names_no_overlay(self):
        self.assertIsNone(POLICIES["plain"])
        self.assertEqual(overlay_for("plain"), {})

    def test_an_overlay_sets_runtime_blocks_and_nothing_else(self):
        for policy in POLICIES:
            with self.subTest(policy=policy):
                self.assertLessEqual(set(overlay_for(policy)), set(POLICY_KEYS))

    def test_no_overlay_can_reach_the_weights(self):
        """The rule the docstring above exists for, stated as a test."""
        for policy in POLICIES:
            blocks = overlay_for(policy)
            for key in ("checkpoint", "model_cfg", "image_size",
                        "image_encoder_engine", "sam_head_engine"):
                self.assertNotIn(key, blocks, f"{policy} sets {key}")

    def test_a_stray_key_is_refused_rather_than_merged(self):
        import tempfile

        from tools import run_records

        with tempfile.TemporaryDirectory() as tmp:
            stray = Path(tmp) / "stray.yaml"
            stray.write_text("guard: {enabled: true}\ncheckpoint: other.pt\n")
            POLICIES["_stray"] = str(stray.relative_to(ROOT)) \
                if stray.is_relative_to(ROOT) else str(stray)
            # overlay_for reads ROOT / path, and an absolute path joins to
            # itself, so this works from a temp dir either way.
            try:
                with self.assertRaises(SystemExit) as caught:
                    run_records.overlay_for("_stray")
                self.assertIn("checkpoint", str(caught.exception))
            finally:
                POLICIES.pop("_stray")

    def test_ego_motion_never_ships_without_the_filter_that_consumes_it(self):
        """A shift with no `samurai:` is a decode and a flow bought for nothing.

        `EdgeTAMTracker._shift_for` is handed to `samurai.install` and has no
        other consumer, so an ego-only overlay would cost a millisecond a frame
        and change no output.
        """
        for policy in POLICIES:
            blocks = overlay_for(policy)
            if "ego_motion" in blocks:
                self.assertIn("samurai", blocks, policy)

    def test_the_ladder_adds_one_thing_per_rung(self):
        rungs = ["plain", "samurai", "ego", "guard"]
        seen = [set(overlay_for(p)) for p in rungs]
        for lower, upper in zip(seen, seen[1:]):
            self.assertTrue(lower < upper,
                            "each rung is the one below it plus exactly one block")
        self.assertEqual([len(s) for s in seen], [0, 1, 2, 3])

    def test_the_guard_overlay_configures_every_field_the_guard_has(self):
        """A half-configured guard is a setting nobody wrote.

        `staged_config` merges per block, so the overlay's `guard:` is the whole
        of it -- a field left out here silently takes the dataclass default
        rather than the backend config's value.
        """
        import ast

        source = (ROOT / "src/trackers/stabiliser.py").read_text()
        fields = {
            node.target.id
            for cls in ast.walk(ast.parse(source))
            if isinstance(cls, ast.ClassDef) and cls.name == "GuardConfig"
            for node in cls.body if isinstance(node, ast.AnnAssign)
        }
        self.assertTrue(fields, "GuardConfig has fields to check")
        configured = set(overlay_for("guard")["guard"]) - {"enabled"}
        self.assertEqual(configured, fields)

    def test_the_guard_reads_frames_at_the_size_it_can_match_a_template_on(self):
        """The guard shares ego_motion's FrameMotion, so `reduce` is its resolution.

        At 4 a 512-pixel input is judged on a 128-pixel frame, where a 20-pixel
        target is 5 across -- enough for a flow field, thin for the template
        match that re-acquires a lost track.
        """
        self.assertEqual(overlay_for("guard")["ego_motion"]["reduce"], 2)
        self.assertEqual(overlay_for("ego")["ego_motion"]["reduce"], 4)

    def test_every_policy_that_changes_a_run_explains_itself_in_the_summary(self):
        for policy in POLICIES:
            if policy != "plain":
                self.assertIn(policy, POLICY_NOTE)
                self.assertIn("Training-free", POLICY_NOTE[policy])

    def test_the_backend_configs_carry_no_policy_of_their_own(self):
        """Which is what makes `plain` a baseline rather than a mixture.

        If one of these ever gains a `samurai:` block, a `--policy plain` row
        stops being the no-policy row and the ladder loses its bottom rung.
        """
        for weights, table in WEIGHTS.items():
            for size, config in table.items():
                for key in POLICY_KEYS:
                    self.assertNotIn(key, body(config),
                                     f"{config} ({weights}/{size}) sets {key}")


class StagedConfig(unittest.TestCase):
    """What cli.py is actually pointed at, per run."""

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.outdir = Path(self.tmp.name)

    def test_the_baseline_reads_the_backend_yaml_untouched(self):
        """A plain run must read the same file it read before this axis existed."""
        config = WEIGHTS["pool_deep"][512]
        self.assertEqual(staged_config(config, "plain", self.outdir), config)
        self.assertFalse((self.outdir / "config.yaml").exists())

    def test_a_policy_run_writes_the_merge_beside_its_own_output(self):
        config = WEIGHTS["pool_deep"][512]
        staged = Path(staged_config(config, "guard", self.outdir))
        self.assertEqual(staged, self.outdir / "config.yaml")
        merged = yaml.safe_load(staged.read_text())
        base = body(config)
        # everything the backend said
        for key in ("checkpoint", "image_size", "image_encoder_engine"):
            self.assertEqual(merged[key], base[key], key)
        # plus everything the policy said
        for key in overlay_for("guard"):
            self.assertEqual(merged[key], overlay_for("guard")[key], key)

    def test_the_merge_is_per_block_not_per_key(self):
        """Half a `guard:` from a file and half from a config is nobody's setting."""
        config = WEIGHTS["pool_deep"][512]
        staged = Path(staged_config(config, "guard", self.outdir))
        self.assertEqual(yaml.safe_load(staged.read_text())["guard"],
                         overlay_for("guard")["guard"])

    def test_paths_stay_relative_so_the_copy_resolves_from_anywhere(self):
        """cli.py resolves against the repository root, not the config's directory."""
        staged = Path(staged_config(WEIGHTS["pool_deep"][512], "ego", self.outdir))
        merged = yaml.safe_load(staged.read_text())
        self.assertFalse(Path(merged["checkpoint"]).is_absolute())
        self.assertFalse(Path(merged["image_encoder_engine"]).is_absolute())

    def test_it_says_where_it_came_from(self):
        staged = Path(staged_config(WEIGHTS["pool_deep"][512], "guard", self.outdir))
        head = staged.read_text().splitlines()[0:3]
        self.assertIn("run_records.py", head[0])
        self.assertIn(WEIGHTS["pool_deep"][512], head[0])
        self.assertIn("guard", head[1])


class FrameCache(unittest.TestCase):
    """Where the decoded frames are staged, and why the key is not the mode.

    The pipeline crops in the pass that writes the cache, so the cache is the
    cropped view. Two modes share it exactly when they stage the same bytes.
    """

    @staticmethod
    def args(cache_dir="/mnt/ssd/cache", frame_skip=1):
        return Namespace(cache_dir=cache_dir, frame_skip=frame_skip)

    def test_the_default_is_left_to_the_pipeline(self):
        self.assertIsNone(cache_for(Path("rec"), "full768", self.args(None)))

    def test_every_full_mode_stages_the_same_frames(self):
        """The resize to image_size happens in the model, not on disk."""
        names = {cache_for(Path("rec"), m, self.args())
                 for m in ("full512", "full768", "full1024")}
        self.assertEqual(len(names), 1)

    def test_a_crop_is_different_pixels_and_gets_its_own(self):
        full = cache_for(Path("rec"), "full768", self.args())
        crop = cache_for(Path("rec"), "crop768", self.args())
        self.assertNotEqual(full, crop)
        self.assertNotEqual(crop, cache_for(Path("rec"), "crop512", self.args()))

    def test_two_records_never_share_one(self):
        self.assertNotEqual(cache_for(Path("rec_a"), "full768", self.args()),
                            cache_for(Path("rec_b"), "full768", self.args()))

    def test_a_skip_gets_its_own_so_a_longer_run_leaves_no_tail(self):
        """A skipped run writes fewer files; the tracker indexes the directory.

        Sharing would leave the tail of a full-length run in place to be read
        as extra frames -- a correctness matter, not a saving.
        """
        self.assertNotEqual(cache_for(Path("rec"), "full768", self.args()),
                            cache_for(Path("rec"), "full768",
                                      self.args(frame_skip=2)))

    def test_it_lands_under_the_directory_it_was_given(self):
        path = cache_for(Path("/media/ssd/frames/rec_028"), "crop768",
                         self.args("/media/ssd/cache"))
        self.assertEqual(path, Path("/media/ssd/cache/rec_028/crop768"))


class AllPointers(unittest.TestCase):
    """SAMURAI does two things; an ordinary export leaves one of them out.

    Mask re-selection needs the sam_head engine's `obj_ptr_all`, emitted only
    under `--all-pointers`. Without it the tracker warns once and carries on
    with the memory gate -- fine for a run that never asked for SAMURAI, and a
    silent halving of the thing being measured for one that did.
    """

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def config(self, outputs, spec=True, engine=True):
        import json

        head = self.dir / "edgetam_sam_head.engine"
        if engine:
            head.touch()
        if spec:
            head.with_suffix(".spec.json").write_text(json.dumps(
                {"outputs": [{"name": n} for n in outputs]}))
        path = self.dir / "config.yaml"
        path.write_text(yaml.safe_dump({"sam_head_engine": str(head)}))
        return str(path)

    ORDINARY = ["pred_masks", "ious", "obj_ptr", "object_score_logits"]

    def test_a_policy_without_samurai_does_not_care(self):
        self.assertIsNone(pointers_missing(self.config(self.ORDINARY), "plain"))

    def test_every_samurai_rung_is_refused_on_an_ordinary_export(self):
        config = self.config(self.ORDINARY)
        for policy in ("samurai", "ego", "guard"):
            self.assertIsNotNone(pointers_missing(config, policy), policy)

    def test_an_all_pointers_export_passes(self):
        config = self.config(self.ORDINARY + ["obj_ptr_all"])
        for policy in POLICIES:
            self.assertIsNone(pointers_missing(config, policy), policy)

    def test_a_missing_engine_is_the_other_preflight_s_question(self):
        """`engines_missing` already answers it; two errors for one cause is noise."""
        self.assertIsNone(
            pointers_missing(self.config(self.ORDINARY, engine=False), "guard"))

    def test_a_deleted_spec_is_not_evidence_either_way(self):
        self.assertIsNone(
            pointers_missing(self.config(self.ORDINARY, spec=False), "guard"))


class Preflight(unittest.TestCase):
    def test_reports_the_directory_an_unbuilt_config_wants(self):
        """Engines are gitignored, so nothing in a clean checkout has them."""
        directory = engines_missing(WEIGHTS["pool_deep"][512])
        self.assertEqual(directory, "models512_pool_deep")

    def test_the_build_command_it_prints_serves_every_policy(self):
        """A set built from the printed command has to work under --policy too.

        `--all-pointers` costs one pass of a 256-wide MLP over three tokens;
        leaving it out costs SAMURAI's mask re-selection, discovered minutes
        into a run as one warning line.
        """
        source = (ROOT / "tools/run_records.py").read_text()
        marker = "python tools/export_edgetam_onnx.py --outdir"
        self.assertIn(marker, source)
        printed = source[source.index(marker):source.index(marker) + 600]
        self.assertIn("--all-pointers", printed)

    def test_says_nothing_about_a_config_with_no_engines_at_all(self):
        self.assertIsNone(engines_missing("configs/edgetam_512.yaml"))


class Provenance(unittest.TestCase):
    """The record that answers "which weights produced this folder", later.

    A config path proves which file was read. Only a hash proves which engine
    ran: two sets built from different checkpoints carry the same module names
    and the same shapes, so nothing about a run distinguishes them afterwards.
    """

    @staticmethod
    def args(weights, policy="plain"):
        return Namespace(weights=weights, frame_skip=1, policy=policy)

    def record(self, weights, size=768, mode="full768", crop=None,
               policy="plain"):
        config = config_for(weights, size, mode)
        return provenance(mode, config, crop, self.args(weights, policy),
                          ["python", "cli.py", "--config", config])

    def test_names_the_weights_the_size_and_the_config(self):
        row = self.record("pool_deep")
        self.assertEqual(row["weights"], "pool_deep")
        self.assertEqual(row["image_size"], 768)
        self.assertEqual(row["trained_at"], 512)
        self.assertEqual(row["config"], WEIGHTS["pool_deep"][768])

    def test_records_the_size_the_1024_config_only_implies(self):
        """configs/edgetam_trt.yaml declares no image_size; the record still does."""
        self.assertEqual(self.record("stock", 1024, "full1024")["image_size"], 1024)

    def test_carries_the_crop_so_a_mode_is_reconstructable(self):
        self.assertEqual(self.record("stock", 768, "crop768", 768)["center_crop"], 768)
        self.assertIsNone(self.record("stock", 768, "full768")["center_crop"])

    def test_lists_every_engine_the_config_names(self):
        row = self.record("pool_deep")
        self.assertEqual(sorted(row["engines"]),
                         ["image_encoder", "memory_attention",
                          "memory_encoder", "sam_head"])

    def test_it_records_the_policy_by_name_and_by_value(self):
        """The name alone would not survive an edit to configs/policies/."""
        row = self.record("pool_deep", policy="guard")
        self.assertEqual(row["policy"], "guard")
        self.assertIn("guard", row["policy_blocks"])
        self.assertEqual(row["policy_blocks"]["guard"]["caution_below"], 1.0)

    def test_a_baseline_record_says_so_rather_than_saying_nothing(self):
        row = self.record("pool_deep")
        self.assertEqual(row["policy"], "plain")
        self.assertEqual(row["policy_blocks"], {})

    def test_the_policy_never_moves_the_weights_the_row_is_labelled_with(self):
        """The whole reason overlays are restricted -- checked end to end."""
        plain = self.record("pool_deep")
        guarded = self.record("pool_deep", policy="guard")
        self.assertEqual(plain["checkpoint"], guarded["checkpoint"])
        self.assertEqual(plain["engines"], guarded["engines"])
        self.assertEqual(plain["image_size"], guarded["image_size"])

    def test_the_command_is_kept_verbatim(self):
        row = self.record("stock")
        self.assertIn("--config", row["command"])
        self.assertTrue(all(isinstance(c, str) for c in row["command"]))

    def test_two_weights_sets_never_record_the_same_checkpoint(self):
        """What separates the folders is the weights, and the record says so.

        Engine *content* is compared by hash, but only once they are built --
        in a clean checkout they are gitignored and absent, which the record
        reports as null rather than papering over. The path and config fields
        hold either way, and that is what makes an absent engine visible as
        absent instead of looking like a match.
        """
        stock, deep = self.record("stock"), self.record("pool_deep")
        self.assertNotEqual(stock["config"], deep["config"])
        for row in (stock, deep):
            self.assertEqual(sorted(row["engines"]), sorted(deep["engines"]))
        paths = {w: yaml.safe_load((ROOT / self.record(w)["config"]).read_text())
                 ["image_encoder_engine"] for w in ("stock", "pool_deep")}
        self.assertNotEqual(paths["stock"], paths["pool_deep"])

    def test_an_engine_that_is_not_built_is_recorded_as_absent(self):
        """Engines are gitignored; the record says so rather than inventing one."""
        self.assertIsNone(self.record("pool_deep")["engines"]["image_encoder"])

    def test_digest_hashes_content_not_the_name(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            same_a = Path(tmp) / "a.engine"
            same_b = Path(tmp) / "b.engine"
            other = Path(tmp) / "c.engine"
            same_a.write_bytes(b"weights")
            same_b.write_bytes(b"weights")
            other.write_bytes(b"other weights")
            self.assertEqual(digest(same_a)["sha256"], digest(same_b)["sha256"])
            self.assertNotEqual(digest(same_a)["sha256"], digest(other)["sha256"])
            self.assertEqual(digest(same_a)["bytes"], 7)
            self.assertIsNone(digest(Path(tmp) / "missing.engine"))


if __name__ == "__main__":
    unittest.main()
