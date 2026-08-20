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


class TestEveryNotebook(unittest.TestCase):
    def test_there_are_notebooks_to_check(self):
        # A glob that quietly matched nothing would make every case below pass.
        self.assertGreaterEqual(len(NOTEBOOKS), 7)

    def test_every_name_is_defined_before_it_is_used(self):
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                self.assertEqual(check(path), [])


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
