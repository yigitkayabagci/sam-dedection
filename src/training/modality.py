"""Which sensor a path or a pool name came from, decided in one place.

`RGB_SOURCES` existed in two notebook builders and nowhere else, which is how
`modality_of` once read "no rgb in the name" as thermal and would have trained
`visdrone` as a thermal pool. It lives here now so a third consumer --
`tools/pretrain_fcmae.py`, which walks directories rather than pool names --
cannot grow a fourth copy that disagrees; `tests/test_fcmae.py` holds the
builders' embedded copies to this list.

The two questions are different and both are needed. A **pool** is named
(`vtuav_thermal`, `aerovis_train`), so its modality is a property of the name.
A **frame on disk** is a path, and the archives this project extracts put both
sensors side by side -- `.../pedestrian_003/ir/000000.jpg` beside
`.../pedestrian_003/rgb/000000.jpg` -- so the directory decides where the name
cannot.
"""
from __future__ import annotations

from pathlib import Path

# Sets whose names say nothing about their sensor and which are RGB anyway.
# Without this the fallback below reads silence as thermal.
RGB_SOURCES = ("visdrone", "aerovis", "vtuav_vis", "rgbt234", "lasher")

# Directory names the extracted archives use for the colour half. `vis` is
# VTUAV's spelling; `ir` and `tir` are the thermal ones, and a path carrying
# either of those is thermal whatever else it says.
RGB_DIRS = ("rgb", "vis", "visible", "color", "colour")
THERMAL_DIRS = ("ir", "tir", "thermal", "infrared")


def modality_of_name(name: str) -> str:
    """`"rgb"` or `"thermal"` for a pool or dataset name.

    A faithful port of the `modality_of` the stage-B builders embed, minus the
    per-run `POOL_MODALITIES` override that only a notebook has: an explicit
    `rgb` token settles it, then the named RGB sets, and the fallback is
    thermal because this project's pools are. That fallback is exactly why
    `RGB_SOURCES` has to be complete rather than indicative -- it once read
    "no rgb in the name" as thermal and would have trained VisDrone as one.
    """
    lowered = name.lower()
    if (lowered.endswith("_rgb") or "_rgb_" in lowered
            or "rgb" in lowered.split("_")):
        return "rgb"
    if any(source in lowered for source in RGB_SOURCES):
        return "rgb"
    return "thermal"


def modality_of_path(path: str | Path) -> str:
    """`"rgb"` or `"thermal"` for one frame on disk.

    A *directory component* beats everything else, because the archives put
    both sensors under one sequence and only the folder tells them apart. A
    thermal component wins over a colour one -- `.../vtuav_rgb_pool/ir/...` is
    a thermal frame in a badly named tree, and dropping it would be the gate
    firing on a real target.

    Then the name test, applied **per component**. `modality_of_name` reads
    pool names, and its token rule (`"rgb" in name.split("_")`) finds nothing
    in a joined path -- `content/pool/vtuav_rgb/frames/000123.png` splits into
    two useless halves and comes back thermal, which is the whole bug this
    function exists to avoid.
    """
    parts = [part.lower() for part in Path(path).parts]
    if any(part in THERMAL_DIRS for part in parts):
        return "thermal"
    if any(part in RGB_DIRS for part in parts):
        return "rgb"
    if any(modality_of_name(part) == "rgb" for part in parts):
        return "rgb"
    return "thermal"
