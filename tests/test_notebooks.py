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
                self.assertEqual(
                    embedded.get(name), stamp,
                    f"{name} is out of step with .stamps.json -- run: "
                    f"python tools/build_notebooks.py")

    def test_the_two_notebooks_do_not_share_a_stamp(self):
        # They are built from one file and most cells are identical, so the
        # hash has to fold in the variant or a swapped pair looks correct.
        import json

        stamps = json.loads((ROOT / "notebooks" / ".stamps.json").read_text())
        generated = [stamps[p.name] for p in NOTEBOOKS if p.name in stamps]
        self.assertEqual(len(generated), len(set(generated)))

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
