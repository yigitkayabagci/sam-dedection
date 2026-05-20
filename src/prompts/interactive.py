from __future__ import annotations

import cv2
import numpy as np

from .types import BoxPrompt, PointPrompt, PromptSet

_WINDOW = "select target (ENTER=accept, ESC=cancel)"


def pick_box(first_frame: np.ndarray, obj_id: int = 1) -> PromptSet:
    """Open a window and let the user drag a bounding box around the target."""
    bgr = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR) if first_frame.shape[-1] == 3 else first_frame
    x, y, w, h = cv2.selectROI(_WINDOW, bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(_WINDOW)
    if w == 0 or h == 0:
        raise RuntimeError("No bounding box selected.")
    box = BoxPrompt(obj_id=obj_id, frame_idx=0, xyxy=(float(x), float(y), float(x + w), float(y + h)))
    return PromptSet(boxes=[box])


def pick_points(first_frame: np.ndarray, obj_id: int = 1) -> PromptSet:
    """Left-click = foreground, right-click = background, ENTER to finish."""
    bgr = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR) if first_frame.shape[-1] == 3 else first_frame
    display = bgr.copy()
    points: list[PointPrompt] = []

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append(PointPrompt(obj_id, 0, (float(x), float(y)), label=1))
            cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append(PointPrompt(obj_id, 0, (float(x), float(y)), label=0))
            cv2.circle(display, (x, y), 5, (0, 0, 255), -1)

    cv2.namedWindow(_WINDOW)
    cv2.setMouseCallback(_WINDOW, on_mouse)
    while True:
        cv2.imshow(_WINDOW, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):  # ENTER
            break
        if key == 27:  # ESC
            points.clear()
            break
    cv2.destroyWindow(_WINDOW)
    if not points:
        raise RuntimeError("No points selected.")
    return PromptSet(points=points)
