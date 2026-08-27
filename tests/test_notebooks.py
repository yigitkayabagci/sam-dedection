"""Every notebook, checked the way a *Run all* would fail.

Syntax alone catches almost nothing in a notebook. What actually breaks a run
is a name used before any cell binds it -- a rename that missed a cell, a
setting added to the prose but not to the settings cell -- and on notebook 07
that surfaces forty minutes and a dataset download into the session.

This is not hypothetical. `tools/check_notebook.py` was written for this suite
and immediately found three settings that a generator edit had silently failed
to apply: the notebook rebuilt cleanly, parsed cleanly, and referred to five
names that did not exist.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.check_notebook import check, strip_magics  # noqa: E402

NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))

# Which generator owns which notebook, by its number prefix. Anything not
# listed comes from `build_notebooks.py`; a wrong answer here sends someone to
# a generator that rebuilds a different file and leaves theirs as stale as it
# was, which is a confusing few minutes.
BUILDERS = {
    "12_": "build_probe_notebook",
    "13_": "build_pool_notebooks",
    "14_": "build_pool_notebooks",
    "15_": "build_shared_pool_notebook",
    "16_": "build_vtuav_pool_notebooks",
    "17_": "build_vtuav_pool_notebooks",
    "18_": "build_kust4k_pool_notebook",
    "19_": "build_stage_b_notebooks",
    "20_": "build_stage_b_notebooks",
    "21_": "build_data_readiness_notebook",
}

# 15-20 were all asked for the same way -- cells only, no prose, no comments --
# but they are not the same length of job. The cap is per notebook because a
# single number would either let 15 grow or refuse 19 the four cells it needs
# to train, score twice and draw the result.
COMMENT_FREE = {
    "15_dronevehicle_shared_pool.ipynb": 6,
    "16_vtuav_rgb_pool.ipynb": 6,
    "17_vtuav_thermal_pool.ipynb": 6,
    "19_thermal_stage_b_pool.ipynb": 8,
    "21_pool_data_readiness.ipynb": 6,
    "20_thermal_stage_b_pool_rgb.ipynb": 8,
}


def reload_calls(path: Path) -> list[str]:
    """Every real call to `reload(...)` in a notebook's code cells.

    Parsed rather than grepped, so the comment in 07 that *explains* why
    `importlib.reload(torch)` is wrong does not count as an instance of it.
    """
    import ast
    import json

    found = []
    cells = json.loads(path.read_text())["cells"]
    for number, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        python, _ = strip_magics("".join(cell["source"]))
        try:
            tree = ast.parse(python)
        except SyntaxError:                              # reported elsewhere
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (func.attr if isinstance(func, ast.Attribute)
                    else func.id if isinstance(func, ast.Name) else "")
            if name == "reload":
                found.append(f"cell {number}: {ast.unparse(node)}")
    return found


def stamp_of(path: Path) -> str | None:
    """The build id a generated notebook embeds, or None if it has none."""
    import json

    found = [line.split('"')[1]
             for cell in json.loads(path.read_text())["cells"]
             if cell["cell_type"] == "code"
             for line in cell["source"] if line.startswith("STAMP    = ")]
    assert len(found) <= 1, f"{path.name} embeds {len(found)} stamps"
    return found[0] if found else None


class TestEveryNotebook(unittest.TestCase):
    def test_there_are_notebooks_to_check(self):
        # A glob that quietly matched nothing would make every case below pass.
        self.assertGreaterEqual(len(NOTEBOOKS), 7)

    def test_every_name_is_defined_before_it_is_used(self):
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                self.assertEqual(check(path), [])

    def test_no_notebook_reloads_an_imported_module(self):
        """`importlib.reload` on a C-extension module always raises.

        Shipped once, in notebook 07's dependency cell, as a check that torch
        had not been replaced by a pip install. Re-running `torch/__init__.py`
        re-registers its `triton` TORCH_LIBRARY namespace and C++ refuses the
        second registration, so the guard raised **unconditionally** -- the
        worst way for a safety check to fail, because it fires when nothing is
        wrong and tells you nothing when something is.

        The same is true of numpy and cv2, and the answer is the same in every
        case: read the installed version from `importlib.metadata`, which comes
        off disk and needs no reimport. Nothing a notebook does needs a reload.
        """
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                self.assertEqual(reload_calls(path), [])

    def test_each_notebook_carries_the_stamp_the_repo_records(self):
        """The embedded build id must equal the one in `.stamps.json`.

        That pair is what lets cell 1 say "you are running an old file" -- a
        confusion that cost two full runs once, because the traceback pointed
        at a line the repo no longer contained. The check is only as good as
        the two staying in step, so a hand-edited notebook, or a rebuild whose
        `.stamps.json` was not committed, has to fail here.
        """
        import json

        stamps_file = ROOT / "notebooks" / ".stamps.json"
        self.assertTrue(stamps_file.is_file(), "notebooks/.stamps.json is missing")
        stamps = json.loads(stamps_file.read_text())

        # Only the generated pair is stamped; 01-06 are written by hand. The
        # two sets have to agree in both directions, so a generated notebook
        # missing from the file fails just as loudly as a stale stamp.
        embedded = {path.name: stamp_of(path) for path in NOTEBOOKS}
        self.assertEqual(sorted(k for k, v in embedded.items() if v),
                         sorted(stamps))

        for name, stamp in sorted(stamps.items()):
            with self.subTest(notebook=name):
                # Four generators now, and sending someone to the wrong one is
                # a confusing few minutes: rebuilding with `build_notebooks.py`
                # leaves 12 exactly as stale as it was.
                builder = BUILDERS.get(name[:3], "build_notebooks")
                self.assertEqual(
                    embedded.get(name), stamp,
                    f"{name} is out of step with .stamps.json -- run: "
                    f"python tools/{builder}.py")

    def test_the_two_notebooks_do_not_share_a_stamp(self):
        # They are built from one file and most cells are identical, so the
        # hash has to fold in the variant or a swapped pair looks correct.
        import json

        stamps = json.loads((ROOT / "notebooks" / ".stamps.json").read_text())
        generated = [stamps[p.name] for p in NOTEBOOKS if p.name in stamps]
        self.assertEqual(len(generated), len(set(generated)))

    def test_every_generated_notebook_names_a_generator_that_exists(self):
        # The mapping is only useful if the file it points at is real, and a
        # renamed generator would otherwise fail silently on the day someone
        # needs it.
        import json

        stamps = json.loads((ROOT / "notebooks" / ".stamps.json").read_text())
        for name in stamps:
            with self.subTest(notebook=name):
                builder = BUILDERS.get(name[:3], "build_notebooks")
                self.assertTrue((ROOT / "tools" / f"{builder}.py").is_file(),
                                f"{name} points at tools/{builder}.py")

    def test_the_comment_free_notebook_stays_comment_free(self):
        """15 was asked for as cells only -- no markdown, no comments.

        It is generated, so the way that erodes is a helpful line added to the
        generator's cell text months later, not an edit to the notebook.
        """
        import json

        for name, cap in COMMENT_FREE.items():
            with self.subTest(notebook=name):
                path = ROOT / "notebooks" / name
                self.assertTrue(path.is_file())
                cells = json.loads(path.read_text())["cells"]
                self.assertTrue(all(c["cell_type"] == "code" for c in cells),
                                f"no markdown cells in {name}")
                self.assertLessEqual(len(cells), cap, f"{name} stays short")
                commented = [line for cell in cells for line in cell["source"]
                             if line.lstrip().startswith("#")]
                self.assertEqual(commented, [])

    def test_the_two_vtuav_arms_differ_only_in_their_modality(self):
        """16 and 17 are one job with one variable, run on two runtimes.

        They must write to different pools and different Drive folders or a
        parallel run has each arm's zip land on the other's -- and to
        different DATA_ROOTs, because a shared extraction tree makes the
        second arm skip staging and then index nothing.
        """
        rgb = settings_of(ROOT / "notebooks" / "16_vtuav_rgb_pool.ipynb")
        ir = settings_of(ROOT / "notebooks" / "17_vtuav_thermal_pool.ipynb")
        self.assertEqual(sorted(rgb), sorted(ir))
        differ = {k for k in rgb if rgb[k] != ir[k]}
        self.assertEqual(
            differ, {"MODALITY", "POOL", "EXTRACT_MODE", "MIRROR_DIR",
                     "DATA_ROOT", "NOTEBOOK", "STAMP"},
            f"unexpected differences: {sorted(differ)}")
        self.assertNotEqual(rgb["MIRROR_DIR"], ir["MIRROR_DIR"])
        self.assertNotEqual(rgb["POOL"], ir["POOL"])
        self.assertNotEqual(rgb["DATA_ROOT"], ir["DATA_ROOT"])

    def test_the_two_stage_b_arms_differ_only_in_the_data_they_mix(self):
        """19 and 20 answer "does feeding RGB alongside thermal help?".

        They answer it only if the RGB windows are the single difference: the
        schedule, the seed, the split, the gates, the prompt and the two grades
        have to be the same source, or the gap between their numbers is not
        attributable to anything. The three names allowed to move are the
        variable itself and the two places a parallel run would otherwise
        collide -- the Drive folder and the checkpoint's `RUN` tag.
        """
        thermal = settings_of(ROOT / "notebooks" / "19_thermal_stage_b_pool.ipynb")
        mixed = settings_of(ROOT / "notebooks" / "20_thermal_stage_b_pool_rgb.ipynb")
        self.assertEqual(sorted(thermal), sorted(mixed))
        differ = {k for k in thermal if thermal[k] != mixed[k]}
        self.assertEqual(differ, {"MODALITIES", "RUN", "MIRROR_DIR",
                                  "NOTEBOOK", "STAMP"},
                         f"unexpected differences: {sorted(differ)}")
        self.assertEqual(thermal["MODALITIES"], '["thermal"]')
        self.assertEqual(mixed["MODALITIES"], '["thermal", "rgb"]')
        self.assertNotEqual(thermal["MIRROR_DIR"], mixed["MIRROR_DIR"])

    def test_the_stage_b_arms_start_from_stock_and_run_no_stage_a(self):
        """The point of these two is stage B alone: `--base` is the stock
        checkpoint and nothing distils into it first, so what the pools bought
        is the only thing in the number."""
        import json

        for name in ("19_thermal_stage_b_pool.ipynb",
                     "20_thermal_stage_b_pool_rgb.ipynb"):
            with self.subTest(notebook=name):
                source = "".join(
                    line for cell in json.loads(
                        (ROOT / "notebooks" / name).read_text())["cells"]
                    for line in cell["source"])
                self.assertIn('"--base", BASE_CKPT', source)
                self.assertIn('"--anchor-weight", "0.0"', source)
                self.assertNotIn("pretrain_encoder", source)
                self.assertNotIn("DISTILL", source)

    def test_the_reload_check_would_catch_the_line_that_shipped(self):
        # Otherwise the case above passes because the notebooks are clean *and*
        # because the check does nothing, and those look identical.
        import json
        import tempfile

        cell = ["# importlib.reload(torch) is wrong -- this comment is not it\n",
                "import importlib, torch as _t\n",
                "importlib.reload(_t)\n"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.ipynb"
            path.write_text(json.dumps({
                "cells": [{"cell_type": "code", "metadata": {}, "outputs": [],
                           "execution_count": None, "source": cell}],
                "metadata": {}, "nbformat": 4, "nbformat_minor": 0}))
            self.assertEqual(reload_calls(path), ["cell 0: importlib.reload(_t)"])


def settings_of(path: Path) -> dict[str, str]:
    """The `NAME = "value"` lines a generated notebook's code cells bind."""
    import json
    import re

    found = {}
    for cell in json.loads(path.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        for line in cell["source"]:
            match = re.match(r'([A-Z][A-Z0-9_]*)\s*=\s*(.+?)(?:\s+#.*)?$',
                             line.rstrip("\n"))
            if match:
                found.setdefault(match.group(1), match.group(2))
    return found


class TestTheTwoTeacherArms(unittest.TestCase):
    """07 and 10 are one experiment with one variable, and have to stay so.

    The pair only measures the teacher if nothing else moved between them.
    They are generated from one file, so today they agree by construction --
    but a variant-specific edit is one keyword argument away, and a run of
    both that differed in the schedule *and* the teacher would produce a
    number nobody could attribute. That is a whole GPU day to discover.
    """

    SAM = ROOT / "notebooks" / "07_encoder_aerial_rgbt.ipynb"
    DINO = ROOT / "notebooks" / "10_encoder_teacher_dinov3.ipynb"

    def test_both_arms_exist(self):
        # Otherwise every case below passes on a missing file.
        self.assertTrue(self.SAM.is_file())
        self.assertTrue(self.DINO.is_file())

    def test_the_arms_name_the_two_teachers(self):
        self.assertEqual(settings_of(self.SAM)["TEACHER"],
                         '"facebook/sam2.1-hiera-base-plus"')
        self.assertIn("dinov3", settings_of(self.DINO)["TEACHER"])

    def test_nothing_but_the_teacher_and_its_bookkeeping_differs(self):
        """Every code cell identical but for the teacher, the mirror, the name.

        MIRROR and the NOTEBOOK/STAMP pair *have* to differ -- that is what
        keeps two simultaneous runs from overwriting each other. Anything
        else differing means the comparison has a second variable in it.
        """
        import json

        allowed = {"TEACHER", "MIRROR", "NOTEBOOK", "STAMP"}
        a, b = settings_of(self.SAM), settings_of(self.DINO)
        self.assertEqual(sorted(a), sorted(b), "the arms bind different names")
        differ = {k for k in a if a[k] != b[k]}
        self.assertEqual(differ, allowed,
                         f"unexpected differences: {sorted(differ - allowed)}; "
                         f"expected but identical: {sorted(allowed - differ)}")

    def test_the_arms_write_to_different_places(self):
        """They are meant to run at the same time on two runtimes.

        Same Drive, so a shared MIRROR would have each run's checkpoints and
        its instance-index cache land on the other's. INDEX is derived from
        MIRROR, so this one assertion covers both.
        """
        self.assertNotEqual(settings_of(self.SAM)["MIRROR"],
                            settings_of(self.DINO)["MIRROR"])

    def test_the_settings_reader_actually_reads_the_settings(self):
        # A regex that matched nothing would make every case above vacuous.
        found = settings_of(self.SAM)
        for name in ("TEACHER", "MIRROR", "SEED", "STEPS_PER_EPOCH", "SIZE"):
            self.assertIn(name, found)
        self.assertEqual(found["SEED"], "0")

    def test_every_variant_field_moves_the_stamp(self):
        """Change anything a variant carries and the build id has to change.

        It once folded in six of the eight fields, hand-listed. The two it
        missed were `teacher` and `distill` -- so editing 10's teacher, the
        single setting that notebook exists to change, rebuilt it to an
        identical stamp. Cell 1 compares that stamp against `.stamps.json` to
        tell you whether the file in the runtime is the file in the repo; one
        that cannot move always answers "current", which is worse than having
        no check at all. Reading the fields off the dataclass is what fixed
        it, and this case is what keeps a ninth field from reopening it.
        """
        import dataclasses
        import importlib
        import unittest.mock

        # The generator is a script: it reads `sys.argv[1]` at import time to
        # pick the variant. Importing it needs a script's argv.
        with unittest.mock.patch.object(sys, "argv", ["build_notebooks.py"]):
            builder = importlib.import_module("tools.build_notebooks")
        variant = builder.VARIANTS["dino"]
        names = [f.name for f in dataclasses.fields(variant)]
        self.assertGreaterEqual(len(names), 8)

        # The generator's own function, not a copy of it -- a reimplementation
        # here would keep passing after a regression in the real one.
        base = builder.stamp_for(variant)
        for name in names:
            with self.subTest(field=name):
                moved = dataclasses.replace(
                    variant, **{name: str(getattr(variant, name)) + " x"})
                self.assertNotEqual(builder.stamp_for(moved), base,
                                    f"changing {name!r} left the stamp alone")


class TestMagicHandling(unittest.TestCase):
    """The IPython lines, which are where this check earns its keep."""

    def test_a_shell_line_inside_an_if_keeps_the_block_valid(self):
        # Blanking the line instead would leave `if x:` with no body and
        # report a syntax error the notebook does not have.
        python, _ = strip_magics("if flag:\n    !python tool.py\nelse:\n    pass\n")
        compile(python, "<cell>", "exec")

    def test_interpolated_names_in_a_shell_line_count_as_uses(self):
        _, used = strip_magics("!python tool.py --out {CHECKPOINT} --n {COUNT}")
        self.assertEqual([name for name, _ in used], ["CHECKPOINT", "COUNT"])

    def test_a_backslash_continued_command_is_followed_to_its_end(self):
        source = "!python tool.py --a {ALPHA} \\\n    --b {BETA} \\\n    --c {GAMMA}\n"
        python, used = strip_magics(source)
        compile(python, "<cell>", "exec")
        self.assertEqual([name for name, _ in used], ["ALPHA", "BETA", "GAMMA"])

    def test_a_cell_magic_is_not_mistaken_for_python(self):
        python, _ = strip_magics("%%time\nx = 1\n")
        compile(python, "<cell>", "exec")


class TestTheCheckerCatchesThings(unittest.TestCase):
    """A checker that passes everything is not a check."""

    def notebook(self, *cells: str) -> Path:
        import json
        import tempfile

        payload = {"cells": [{"cell_type": "code", "metadata": {}, "outputs": [],
                              "execution_count": None,
                              "source": (c + "\n").splitlines(keepends=True)}
                             for c in cells],
                   "metadata": {}, "nbformat": 4, "nbformat_minor": 0}
        handle = tempfile.NamedTemporaryFile("w", suffix=".ipynb", delete=False)
        json.dump(payload, handle)
        handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return Path(handle.name)

    def test_a_name_from_a_later_cell_is_reported(self):
        problems = check(self.notebook("print(LATER)", "LATER = 1"))
        self.assertEqual(len(problems), 1)
        self.assertIn("LATER", problems[0])

    def test_a_name_a_shell_line_interpolates_is_reported(self):
        problems = check(self.notebook("!python tool.py --out {MISSING}"))
        self.assertEqual(len(problems), 1)
        self.assertIn("MISSING", problems[0])

    def test_an_earlier_cell_satisfies_a_later_one(self):
        self.assertEqual(check(self.notebook("EARLIER = 1", "print(EARLIER)")), [])

    def test_a_name_assigned_further_down_the_same_cell_is_reported(self):
        """The shape that shipped: a loop over a path the next line defines.

        The cross-cell pass cannot see this -- it treats everything a cell
        binds as available throughout the cell, which is the namespace *after*
        the cell has run, not during its first pass. The notebook reported
        clean and died on `NameError` at the exact line.
        """
        problems = check(self.notebook(
            "MIRROR = 1\n"
            "for d in (MIRROR, INDEX):\n"
            "    pass\n"
            "INDEX = 2\n"))
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("INDEX", problems[0])
        self.assertIn("used before", problems[0])

    def test_a_function_may_use_a_global_defined_below_it(self):
        # Legal Python: the body resolves LATER when it is called, not where
        # it is written. Reporting this would make the check unusable.
        self.assertEqual(check(self.notebook(
            "def f():\n    return LATER\nLATER = 1\nprint(f())\n")), [])

    def test_a_loop_target_is_not_reported_inside_its_own_loop(self):
        self.assertEqual(check(self.notebook(
            "for d in (1, 2):\n    print(d)\n")), [])

    def test_a_loop_carried_name_is_not_a_false_positive(self):
        # `prev` is read on a line above the one that binds it, but the
        # binding lives in the same statement and runs on the first iteration
        # -- Python is fine with it, so the check has to be too.
        self.assertEqual(check(self.notebook(
            "total = 0\nfor i in range(3):\n"
            "    if i:\n        total += prev\n    prev = i\n")), [])

    def test_a_comprehension_variable_is_not_reported(self):
        self.assertEqual(check(self.notebook(
            "xs = [1]\nys = [x * 2 for x in xs]\nprint(ys)\n")), [])

    def test_a_name_used_after_its_assignment_is_fine(self):
        self.assertEqual(check(self.notebook(
            "INDEX = 2\nfor d in (INDEX,):\n    pass\n")), [])

    def test_imports_and_builtins_are_not_reported(self):
        self.assertEqual(check(self.notebook("import os\nprint(len(os.sep))")), [])

    def test_a_comprehension_variable_does_not_leak_a_false_positive(self):
        self.assertEqual(check(self.notebook("xs = [i * 2 for i in range(3)]")), [])

    def test_a_function_parameter_is_not_a_missing_global(self):
        self.assertEqual(check(self.notebook("def f(a, b=1):\n    return a + b")), [])

    def test_a_name_bound_only_inside_a_branch_still_counts(self):
        # A notebook namespace keeps whatever ran, and this checker is not an
        # execution -- treating it otherwise would flood the report.
        self.assertEqual(check(self.notebook("if x_flag := True:\n    Y = 1",
                                             "print(Y)")), [])


if __name__ == "__main__":
    unittest.main()
