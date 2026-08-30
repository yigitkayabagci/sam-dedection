"""`--dataset spec:path[:modality[:mode]]`, and the stratified split.

Two things worth pinning. The flag is what makes a run multi-dataset, and it is
positional with defaults falling off the end, so a typo in the third field has
to be an error rather than a silently different mode. And the split has to be
stratified: a 4 024-frame set beside a 100-sequence one can land almost
entirely in `test` under a single permutation, and names collide across
datasets so a name-keyed lookup would train on one dataset's frame while
scoring another's.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.aerial import (  # noqa: E402
    SPECS,
    Frame,
    FrameIndex,
    InstanceGates,
    Source,
    gates_tag,
    split_index,
)
from src.training.datasets import describe, parse  # noqa: E402


class TestParse(unittest.TestCase):
    def test_the_short_form_defaults_to_thermal_components(self):
        request = parse("kust4k:/data/Kust4K")

        self.assertEqual(request.root, Path("/data/Kust4K"))
        self.assertEqual(request.modality, "thermal")
        self.assertEqual(request.source.mode, "components")
        self.assertTrue(request.source.gray)

    def test_the_long_form_sets_modality_and_mode(self):
        request = parse("vtuav_vis:/data/VTUAV:thermal:labels")

        self.assertEqual(request.source.spec.name, "vtuav_vis")
        self.assertEqual(request.source.mode, "labels")
        # The gates ride on the label because they decide which components
        # exist and `decompose` numbers only the ones it keeps -- an index
        # cached under other gates pairs instances with their neighbour's mask.
        self.assertTrue(request.label.startswith("vtuav_vis:thermal:labels:"))
        self.assertEqual(request.label,
                         f"vtuav_vis:thermal:labels:{gates_tag()}")

    def test_rgb_turns_the_grey_replication_off(self):
        # A thermal frame is one channel replicated to three; an RGB one is
        # already three, and replicating its blue channel would train on a
        # different image than the file holds.
        self.assertFalse(parse("segfly:/data/SegFly:rgb").source.gray)

    def test_gates_are_shared_across_datasets(self):
        gates = InstanceGates(min_area=999)
        request = parse("kust4k:/data/K", gates)
        self.assertEqual(request.source.gates.min_area, 999)

    def test_the_role_field_decides_which_splits_a_dataset_feeds(self):
        self.assertEqual(parse("kust4k:/data/K").source.role, "all")
        self.assertEqual(
            parse("kust4k:/data/K:thermal:components:train").source.role, "train")
        self.assertEqual(
            parse("vtuav_vis:/data/V:thermal:labels:eval").source.role, "eval")

    def test_role_stays_out_of_the_cache_key(self):
        # Changing which split a dataset feeds must not throw away an index
        # that took a full pass over it to build.
        self.assertEqual(parse("kust4k:/data/K:thermal:components:train").label,
                         parse("kust4k:/data/K:thermal:components").label)

    def test_a_run_with_nothing_to_score_on_says_so(self):
        text = describe([parse("kust4k:/data/K:thermal:components:train")])
        self.assertIn("nothing to", text)

    def test_each_field_is_validated_rather_than_ignored(self):
        for bad in ("kust4k", "kust4k:", ":/data/K", "nosuchset:/data/K",
                    "kust4k:/data/K:sideways", "kust4k:/data/K:thermal:magic",
                    "kust4k:/data/K:thermal:components:sometimes"):
            with self.assertRaises(ValueError, msg=bad):
                parse(bad)

    def test_the_error_shows_the_syntax(self):
        with self.assertRaises(ValueError) as ctx:
            parse("kust4k")
        self.assertIn("spec:path", str(ctx.exception))

    def test_describe_names_every_dataset_in_a_run(self):
        text = describe([parse("kust4k:/data/K"),
                         parse("vtuav_vis:/data/V:thermal:labels")])
        self.assertIn("kust4k", text)
        self.assertIn("vtuav_vis", text)
        self.assertIn("labels", text)


class TestStratifiedSplit(unittest.TestCase):
    def index(self, source: Source, count: int, prefix: str = "f"):
        return [FrameIndex(frame=Frame(name=f"{prefix}{i:04d}",
                                       image=Path(f"{i}.png"),
                                       mask=Path(f"{i}_m.png")),
                           size=(64, 64), instances=(), rejects={}, source=source)
                for i in range(count)]

    def setUp(self):
        self.big = Source(SPECS["kust4k"])
        self.small = Source(SPECS["vtuav_vis"], mode="labels")

    def test_every_dataset_is_split_in_the_same_proportions(self):
        # The failure without stratification: 20 frames beside 2 000 can land
        # entirely on one side of the line by chance.
        index = self.index(self.big, 2000) + self.index(self.small, 20, "v")
        parts = split_index(index, seed=0)

        for name, share in (("train", 0.8), ("val", 0.1), ("test", 0.1)):
            for source in (self.big, self.small):
                total = sum(1 for e in index if e.source is source)
                got = sum(1 for e in parts[name] if e.source is source)
                self.assertAlmostEqual(got / total, share, delta=0.06,
                                       msg=f"{name} {source.name}")

    def test_the_three_parts_partition_the_index(self):
        index = self.index(self.big, 100) + self.index(self.small, 40, "v")
        parts = split_index(index, seed=3)
        seen = [id(e) for part in parts.values() for e in part]

        self.assertEqual(len(seen), 140)
        self.assertEqual(len(set(seen)), 140)

    def test_colliding_frame_names_are_not_confused(self):
        # `000123.png` exists in most of these datasets. Splitting on entries
        # rather than on names is what keeps two of them apart.
        index = self.index(self.big, 30) + self.index(self.small, 30)
        parts = split_index(index, seed=1)
        for part in parts.values():
            for entry in part:
                self.assertIn(entry.source, (self.big, self.small))
        self.assertEqual(sum(len(p) for p in parts.values()), 60)

    def test_it_is_reproducible_from_its_seed(self):
        index = self.index(self.big, 50) + self.index(self.small, 50, "v")
        first = [e.frame.name for e in split_index(index, seed=5)["test"]]
        again = [e.frame.name for e in split_index(index, seed=5)["test"]]
        self.assertEqual(first, again)

    def test_a_train_role_dataset_never_reaches_val_or_test(self):
        # The point: a set whose instances were reconstructed from a semantic
        # map is a fine thing to learn from and a bad thing to be scored on --
        # where the decomposition fused two cars, a model that separates them
        # correctly would be marked wrong.
        reconstructed = Source(SPECS["kust4k"], role="train")
        real = Source(SPECS["vtuav_vis"], mode="labels")
        index = self.index(reconstructed, 100) + self.index(real, 40, "v")
        parts = split_index(index, seed=0)

        for name in ("val", "test"):
            for entry in parts[name]:
                self.assertIs(entry.source, real, name)
        self.assertEqual(sum(1 for e in parts["train"] if e.source is reconstructed),
                         100)

    def test_an_eval_role_dataset_never_reaches_train(self):
        held_out = Source(SPECS["vtuav_vis"], mode="labels", role="eval")
        index = self.index(self.big, 100) + self.index(held_out, 40, "v")
        parts = split_index(index, seed=0)

        self.assertFalse(any(e.source is held_out for e in parts["train"]))
        self.assertEqual(sum(1 for name in ("val", "test")
                             for e in parts[name] if e.source is held_out), 40)

    def test_an_eval_role_dataset_splits_between_val_and_test(self):
        held_out = Source(SPECS["vtuav_vis"], mode="labels", role="eval")
        parts = split_index(self.index(held_out, 200), seed=0)

        self.assertGreater(len(parts["val"]), 0)
        self.assertGreater(len(parts["test"]), 0)
        # Default fractions give val and test equal shares of what is left.
        self.assertAlmostEqual(len(parts["val"]) / 200, 0.5, delta=0.05)

    def test_two_sources_of_the_same_length_do_not_get_the_same_permutation(self):
        # Same length, same seed: without the per-source offset both datasets
        # would hold out the frames at identical positions, which is not a
        # correctness bug but is a needless correlation between them.
        index = self.index(self.big, 60) + self.index(self.small, 60, "v")
        parts = split_index(index, seed=0)
        first = [e.frame.name[1:] for e in parts["test"] if e.source is self.big]
        other = [e.frame.name[1:] for e in parts["test"] if e.source is self.small]
        self.assertNotEqual(first, other)


if __name__ == "__main__":
    unittest.main()
