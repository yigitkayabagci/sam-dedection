from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class BoxPrompt:
    obj_id: int
    frame_idx: int
    xyxy: Tuple[float, float, float, float]


@dataclass(frozen=True)
class PointPrompt:
    obj_id: int
    frame_idx: int
    xy: Tuple[float, float]
    label: int  # 1 = foreground, 0 = background


@dataclass
class PromptSet:
    boxes: List[BoxPrompt] = field(default_factory=list)
    points: List[PointPrompt] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.boxes and not self.points

    def object_ids(self) -> List[int]:
        ids = {p.obj_id for p in self.boxes} | {p.obj_id for p in self.points}
        return sorted(ids)
