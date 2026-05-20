from __future__ import annotations

import cv2
import numpy as np

# Fixed palette so the same obj_id keeps the same colour across frames.
_PALETTE_BGR = [
    (255, 56, 56),   # red-ish
    (56, 255, 56),   # green
    (56, 56, 255),   # blue
    (255, 255, 56),  # yellow
    (255, 56, 255),  # magenta
    (56, 255, 255),  # cyan
]


def _color(obj_id: int) -> tuple[int, int, int]:
    return _PALETTE_BGR[(obj_id - 1) % len(_PALETTE_BGR)]


def overlay_masks(
    frame_rgb: np.ndarray,
    masks: dict[int, np.ndarray],
    alpha: float = 0.5,
    draw_bbox: bool = True,
) -> np.ndarray:
    """Blend per-object binary masks onto an RGB frame and optionally draw bboxes."""
    out = frame_rgb.copy()
    for obj_id, mask in masks.items():
        if mask.dtype != bool:
            mask = mask.astype(bool)
        if not mask.any():
            continue
        color = np.array(_color(obj_id)[::-1], dtype=np.uint8)  # BGR -> RGB
        out[mask] = (alpha * color + (1 - alpha) * out[mask]).astype(np.uint8)

        if draw_bbox:
            ys, xs = np.where(mask)
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), tuple(int(c) for c in color), 2)
            cv2.putText(
                out,
                f"id={obj_id}",
                (int(x1), max(0, int(y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                tuple(int(c) for c in color),
                2,
                cv2.LINE_AA,
            )
    return out
