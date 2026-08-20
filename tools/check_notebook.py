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
    """Names a cell binds, and names it uses at module level.

    Bindings are collected from everywhere -- including inside functions and
    branches -- because a notebook's namespace keeps whatever ran. Uses are
    collected the same way, and the ones a comprehension or a function
    parameter introduced are subtracted, since those never escape.
    """

    def __init__(self) -> None:
        self.bound: set[str] = set()
        self.used: list[tuple[str, int]] = []
        self.local: set[str] = set()

    # -- bindings ---------------------------------------------------------
    def visit_FunctionDef(self, node):
        self.bound.add(node.name)
        for argument in ast.walk(node.args):
            if isinstance(argument, ast.arg):
                self.local.add(argument.arg)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        for argument in ast.walk(node.args):
            if isinstance(argument, ast.arg):
                self.local.add(argument.arg)
        self.generic_visit(node)

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
        else:
            self.used.append((node.id, node.lineno))


class ModuleScope(Scope):
    """`Scope`, but blind to what happens inside a function body.

    For the ordering check only. A function may refer to a global defined
    further down the cell -- the name is looked up when it is *called*, not
    where it is written -- so descending into the body would report uses that
    Python is perfectly happy with. A class body is different: it runs where it
    is written, so `Scope`'s handling of it is kept.
    """

    def visit_FunctionDef(self, node):
        self.bound.add(node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        pass


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

        scope = Scope()
        scope.visit(tree)
        # Every name the cell binds is available to the whole cell, the way a
        # notebook namespace behaves once the cell has run.
        for name, line in scope.used + interpolated:
            if name not in known and name not in scope.bound and name not in scope.local:
                problems.append(f"cell {index}, line {line}: {name!r} is not "
                                f"defined by any earlier cell")

        # ...but the cell still has to survive its own first run, top to
        # bottom, and the loop above cannot see that: it treats a name assigned
        # on the last line as available on the first. That is the namespace
        # *after* the cell, not during it. Shipped exactly that way once -- a
        # `mkdir` loop over a path the next statement went on to define -- and
        # this file reported the notebook clean.
        ready = set(known)
        for statement in tree.body:
            here = ModuleScope()
            here.visit(statement)
            for name, line in here.used:
                # `name in scope.bound` keeps this to genuine ordering faults:
                # a name nothing in the cell binds was already reported above.
                if (name in scope.bound and name not in ready
                        and name not in here.bound and name not in here.local):
                    problems.append(
                        f"cell {index}, line {line}: {name!r} is used before "
                        f"the line that assigns it")
            ready |= here.bound
        known |= scope.bound
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
