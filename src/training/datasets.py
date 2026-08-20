"""`--dataset spec:path[:modality[:mode]]` -- one flag, several datasets.

The encoder is the part of EdgeTAM that carries *general* visual features, and
a run on one dataset shows it one sensor, one city and one set of annotation
habits. Kust4K is 4 024 frames at 640x512; that is a reasonable thing to
fine-tune a head on and a thin thing to move a trunk with. So every entry point
takes the flag more than once and the windows concatenate:

    --dataset kust4k:/data/Kust4K
    --dataset vtuav_vis:/data/VTUAV:thermal:labels
    --dataset segfly:/data/SegFly:thermal:watershed

Each `Source` reaches the batch on its own samples, so a step mixes them rather
than alternating between them -- and mixing is the point, because it is what
stops the trunk fitting any one set's statistics.

The syntax is positional with defaults falling off the end, deliberately: a
`--dataset` that needed four flags beside it would not be a repeatable flag,
and every field after the path has an obvious default (`thermal`, and
`components` because that is what a semantic set needs).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .aerial import MODES, SPECS, InstanceGates, Source

MODALITIES = ("thermal", "rgb")


@dataclass(frozen=True)
class Request:
    """One `--dataset` argument, parsed. `root` is where its files are."""

    source: Source
    root: Path
    modality: str

    @property
    def label(self) -> str:
        return f"{self.source.spec.name}:{self.modality}:{self.source.mode}"


def parse(argument: str, gates: InstanceGates | None = None) -> Request:
    """`"kust4k:/data/Kust4K"` or `"vtuav_vis:/data/VTUAV:thermal:labels"`.

    Split on `:` from the left with the path second, so a Windows-style drive
    letter is the one thing this cannot take -- which is fine, since none of
    these datasets fits on a machine that has one.
    """
    parts = argument.split(":")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"--dataset {argument!r} should be spec:path[:modality[:mode]], "
            f"e.g. kust4k:/content/data/Kust4K or "
            f"vtuav_vis:/content/data/VTUAV:thermal:labels")

    name, root = parts[0], parts[1]
    modality = parts[2] if len(parts) > 2 and parts[2] else "thermal"
    mode = parts[3] if len(parts) > 3 and parts[3] else "components"

    if name not in SPECS:
        raise ValueError(f"unknown spec {name!r}; known: {sorted(SPECS)}")
    if modality not in MODALITIES:
        raise ValueError(f"modality must be one of {MODALITIES}, got {modality!r}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    source = Source(spec=SPECS[name], gates=gates or InstanceGates(), mode=mode,
                    gray=modality == "thermal")
    return Request(source=source, root=Path(root), modality=modality)


def describe(requests: list[Request]) -> str:
    """One line per dataset, so a mixed run says what it is made of."""
    lines = ["| dataset | modality | mode | root |", "|---|---|---|---|"]
    for request in requests:
        lines.append(f"| {request.source.spec.name} | {request.modality} | "
                     f"`{request.source.mode}` | `{request.root}` |")
    return "\n".join(lines)
