"""`--dataset spec:path[:modality[:mode[:role]]]` -- one flag, several datasets.

The encoder is the part of EdgeTAM that carries *general* visual features, and
a run on one dataset shows it one sensor, one city and one set of annotation
habits. Kust4K is 4 024 frames at 640x512; that is a reasonable thing to
fine-tune a head on and a thin thing to move a trunk with. So every entry point
takes the flag more than once and the windows concatenate:

    --dataset vtuav_vis:/data/VTUAV:thermal:labels
    --dataset kust4k:/data/Kust4K:thermal:components:train
    --dataset segfly:/data/SegFly:thermal:watershed:train

Each `Source` reaches the batch on its own samples, so a step mixes them rather
than alternating between them -- and mixing is the point, because it is what
stops the trunk fitting any one set's statistics.

The fifth field, `role`, is the one worth reading twice. `train` keeps a
dataset out of val and test entirely, and the reason is what a held-out number
means: a set whose instances were *reconstructed* from a semantic map carries a
target that is itself uncertain, so a model that correctly separates two cars
the decomposition fused is scored **wrong**. Put the reconstructed sets in
training and measure on annotation somebody actually drew.

The syntax is positional with defaults falling off the end, deliberately: a
`--dataset` that needed five flags beside it would not be a repeatable flag,
and every field after the path has an obvious default (`thermal`,
`components` because that is what a semantic set needs, and `all`).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .aerial import MODES, ROLES, SPECS, InstanceGates, Source

MODALITIES = ("thermal", "rgb")


@dataclass(frozen=True)
class Request:
    """One `--dataset` argument, parsed. `root` is where its files are."""

    source: Source
    root: Path
    modality: str

    @property
    def label(self) -> str:
        """Identifies the dataset *and how it is decoded*, for cache filenames.

        `role` is deliberately absent: it decides which split a frame lands in,
        not how its mask is read, so changing it must not invalidate an index
        that took a full pass over the dataset to build.
        """
        return f"{self.source.spec.name}:{self.modality}:{self.source.mode}"


def parse(argument: str, gates: InstanceGates | None = None) -> Request:
    """`"kust4k:/data/Kust4K"` up to `"kust4k:/data/K:thermal:components:train"`.

    Split on `:` from the left with the path second, so a Windows-style drive
    letter is the one thing this cannot take -- which is fine, since none of
    these datasets fits on a machine that has one.
    """
    parts = argument.split(":")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"--dataset {argument!r} should be spec:path[:modality[:mode[:role]]], "
            f"e.g. kust4k:/content/data/Kust4K or "
            f"vtuav_vis:/content/data/VTUAV:thermal:labels")

    name, root = parts[0], parts[1]
    modality = parts[2] if len(parts) > 2 and parts[2] else "thermal"
    mode = parts[3] if len(parts) > 3 and parts[3] else "components"
    role = parts[4] if len(parts) > 4 and parts[4] else "all"

    if name not in SPECS:
        raise ValueError(f"unknown spec {name!r}; known: {sorted(SPECS)}")
    if modality not in MODALITIES:
        raise ValueError(f"modality must be one of {MODALITIES}, got {modality!r}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")

    source = Source(spec=SPECS[name], gates=gates or InstanceGates(), mode=mode,
                    gray=modality == "thermal", role=role)
    return Request(source=source, root=Path(root), modality=modality)


def describe(requests: list[Request]) -> str:
    """One line per dataset, so a mixed run says what it is made of."""
    lines = ["| dataset | modality | mode | role | root |", "|---|---|---|---|---|"]
    for request in requests:
        lines.append(f"| {request.source.spec.name} | {request.modality} | "
                     f"`{request.source.mode}` | `{request.source.role}` | "
                     f"`{request.root}` |")
    if not any(r.source.role in ("all", "eval") for r in requests):
        lines += ["", "!! every dataset is role=train, so there is nothing to "
                      "score on. At least one needs `all` or `eval`."]
    return "\n".join(lines)
