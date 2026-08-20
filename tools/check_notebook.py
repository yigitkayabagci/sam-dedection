#!/usr/bin/env python3
"""Walk a notebook's code cells in order and find what a *Run all* would break on.

Syntax checking a notebook catches almost nothing, because the way notebooks
actually fail is different: a cell uses a name a later cell defines, or a name
that was renamed three edits ago, and nothing notices until the run reaches it
-- which on this notebook is forty minutes and a dataset download later.

So this reads the cells in order and tracks what is defined by the time each
one runs. A name that is used before anything binds it is reported with the
cell and line, which is the check a notebook can actually be held to without a
GPU.

Two things it deliberately does *not* try to be. It is not a type checker, and
it is not an execution: a name bound inside an `if` that never runs still
counts as bound, exactly as Python would leave it. And it understands IPython's
`!command` and `%magic` lines, because in this repo those carry `{variable}`
interpolation and a stale name in one of them is a real failure.

    python tools/check_notebook.py notebooks/07_encoder_aerial_rgbt.ipynb
"""
from __future__ import annotations

import argparse
import ast
import builtins
import json
import re
from pathlib import Path

# `!cmd --flag {NAME}` and `%%time` are not Python, but the braces in them are
# evaluated by IPython, so the names inside are real uses.
SHELL = re.compile(r"^\s*[!%]")
INTERPOLATED = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)")

# Names Colab provides, plus the ones IPython puts in a notebook's namespace.
AMBIENT = {"get_ipython", "In", "Out", "exit", "quit", "display", "_", "__"}


class Scope(ast.NodeVisitor):
    """Names one statement binds, and names it uses -- split by when they run.

    Bindings are collected from everywhere -- including inside functions and
    branches -- because a notebook's namespace keeps whatever ran. Uses are
    collected the same way, and the ones a comprehension or a function
    parameter introduced are subtracted, since those never escape.

    Uses are split in two, because they fail at two different times. A name
    inside a `def` or a `lambda` body runs when the function is *called*, so
    anything the cell binds later is fair game (`deferred`). Everything else
    runs the moment its statement does (`immediate`), and for those "bound
    somewhere in this cell" is not enough -- the settings cell of notebook 07
    once used `INDEX` in a mkdir loop three lines above the line that defined
    it, and this checker, then cell-scoped, waved it through.
    """

    def __init__(self) -> None:
        self.bound: set[str] = set()
        self.used: list[tuple[str, int]] = []          # runs later, if at all
        self.immediate: list[tuple[str, int]] = []     # runs with the statement
        self.local: set[str] = set()
        self._deferred = 0

    # -- bindings ---------------------------------------------------------
    def visit_FunctionDef(self, node):
        self.bound.add(node.name)
        for argument in ast.walk(node.args):
            if isinstance(argument, ast.arg):
                self.local.add(argument.arg)
        self._deferred += 1
        try:
            self.generic_visit(node)
        finally:
            self._deferred -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        for argument in ast.walk(node.args):
            if isinstance(argument, ast.arg):
                self.local.add(argument.arg)
        self._deferred += 1
        try:
            self.generic_visit(node)
        finally:
            self._deferred -= 1

    def visit_ClassDef(self, node):
        self.bound.add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.bound.add((alias.asname or alias.name).split(".")[0])

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.bound.add(alias.asname or alias.name)

    def visit_Global(self, node):
        self.bound.update(node.names)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.bound.add(node.name)
        self.generic_visit(node)

    def visit_comprehension(self, node):
        for name in ast.walk(node.target):
            if isinstance(name, ast.Name):
                self.local.add(name.id)
        self.generic_visit(node)

    # -- uses -------------------------------------------------------------
    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bound.add(node.id)
        elif self._deferred:
            self.used.append((node.id, node.lineno))
        else:
            self.immediate.append((node.id, node.lineno))


def strip_magics(source: str) -> tuple[str, list[tuple[str, int]]]:
    """Python-only source, plus the names an IPython line interpolated.

    A magic becomes `pass` **at its own indentation**, not a blank line: these
    cells put `!python tools/...` inside `if PRETRAIN:`, and blanking it would
    leave an `if` with no body and report a syntax error the notebook does not
    have. Continuation lines of a `\\`-continued command blank out, since the
    `pass` has already satisfied the block.
    """
    lines, interpolated = [], []
    continued = False
    for number, line in enumerate(source.split("\n"), start=1):
        if continued or SHELL.match(line):
            interpolated += [(name, number) for name in INTERPOLATED.findall(line)]
            was_continuation, continued = continued, line.rstrip().endswith("\\")
            indent = line[: len(line) - len(line.lstrip())]
            lines.append("" if was_continuation else f"{indent}pass")
            continue
        lines.append(line)
    return "\n".join(lines), interpolated


def check(path: Path) -> list[str]:
    """Every undefined-at-that-point name, as a readable line each."""
    notebook = json.loads(path.read_text())
    known = set(dir(builtins)) | AMBIENT
    problems: list[str] = []

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        python, interpolated = strip_magics(source)
        try:
            tree = ast.parse(python)
        except SyntaxError as exc:
            problems.append(f"cell {index}: syntax error at line {exc.lineno}: {exc.msg}")
            continue

        # Statement by statement, because the cell runs that way. An immediate
        # use may lean on earlier cells, on earlier statements of this one, or
        # on its own statement's bindings -- a `for` target, a loop-carried
        # name -- but not on a binding that has not happened yet. A deferred
        # use (inside a `def` or `lambda`) runs after the whole cell has, so
        # for those anything the cell binds anywhere counts, which is also the
        # rule for the names a shell line interpolates.
        cell_bound: set[str] = set()
        deferred: list[tuple[str, int]] = []
        cell_local: set[str] = set()
        for statement in tree.body:
            scope = Scope()
            scope.visit(statement)
            for name, line in scope.immediate:
                if (name not in known and name not in cell_bound
                        and name not in scope.bound and name not in scope.local):
                    problems.append(f"cell {index}, line {line}: {name!r} is "
                                    f"used before anything defines it")
            deferred += [(n, l) for n, l in scope.used if n not in scope.local]
            cell_bound |= scope.bound
            cell_local |= scope.local
        for name, line in deferred + interpolated:
            if name not in known and name not in cell_bound and name not in cell_local:
                problems.append(f"cell {index}, line {line}: {name!r} is not "
                                f"defined by any earlier cell")
        known |= cell_bound
    return problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("notebook", nargs="+", type=Path)
    args = p.parse_args(argv)

    failed = 0
    for path in args.notebook:
        problems = check(path)
        cells = sum(1 for c in json.loads(path.read_text())["cells"]
                    if c["cell_type"] == "code")
        if problems:
            failed = 1
            print(f"{path}: {len(problems)} problem(s) across {cells} code cells")
            for line in problems:
                print(f"  {line}")
        else:
            print(f"{path}: {cells} code cells, every name defined before use")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
