from __future__ import annotations

import json
from pathlib import Path

from .types import BoxPrompt, PointPrompt, PromptSet


def load_prompts(path: str | Path) -> PromptSet:
    """Load prompts from a JSON file.

    Schema:
      {
        "boxes":  [{"obj_id": 1, "frame_idx": 0, "xyxy": [x1, y1, x2, y2]}, ...],
        "points": [{"obj_id": 1, "frame_idx": 0, "xy": [x, y], "label": 1}, ...]
      }
    """
    data = json.loads(Path(path).read_text())
    boxes = [
        BoxPrompt(
            obj_id=int(b["obj_id"]),
            frame_idx=int(b.get("frame_idx", 0)),
            xyxy=tuple(float(v) for v in b["xyxy"]),
        )
        for b in data.get("boxes", [])
    ]
    points = [
        PointPrompt(
            obj_id=int(p["obj_id"]),
            frame_idx=int(p.get("frame_idx", 0)),
            xy=tuple(float(v) for v in p["xy"]),
            label=int(p.get("label", 1)),
        )
        for p in data.get("points", [])
    ]
    ps = PromptSet(boxes=boxes, points=points)
    if ps.is_empty():
        raise ValueError(f"Prompt file {path} contains no prompts.")
    return ps


def save_prompts(prompts: PromptSet, path: str | Path) -> Path:
    """Write a PromptSet in the schema `load_prompts` reads.

    Lets a selection made once -- interactively, on the full frame -- be reused
    across runs and configurations instead of being picked again each time.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "boxes": [{"obj_id": b.obj_id, "frame_idx": b.frame_idx, "xyxy": list(b.xyxy)}
                  for b in prompts.boxes],
        "points": [{"obj_id": p.obj_id, "frame_idx": p.frame_idx, "xy": list(p.xy),
                    "label": p.label} for p in prompts.points],
    }, indent=2) + "\n")
    return path
