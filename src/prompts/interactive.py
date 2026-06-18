from __future__ import annotations

import cv2
import numpy as np

from ..visualize import _color
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


_MULTI_BOX_WINDOW = "select MULTIPLE boxes (draw+ENTER each, ESC=done)"
_MULTI_PT_WINDOW = "multi-target points (L=fg R=bg, n=next obj, 1-9=pick obj, ENTER=done)"


def pick_boxes(first_frame: np.ndarray, start_obj_id: int = 1) -> PromptSet:
    """Select MULTIPLE boxes in one shot; each box becomes a new object id.

    Draw a box, press SPACE/ENTER to confirm it, draw the next; press ESC when
    done. obj_id is assigned start_obj_id, start_obj_id+1, ... in draw order.
    """
    bgr = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR) if first_frame.shape[-1] == 3 else first_frame
    rois = cv2.selectROIs(_MULTI_BOX_WINDOW, bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(_MULTI_BOX_WINDOW)
    boxes: list[BoxPrompt] = []
    for i, roi in enumerate(rois):
        x, y, w, h = (int(v) for v in roi)
        if w == 0 or h == 0:
            continue
        boxes.append(BoxPrompt(obj_id=start_obj_id + i, frame_idx=0,
                               xyxy=(float(x), float(y), float(x + w), float(y + h))))
    if not boxes:
        raise RuntimeError("No boxes selected.")
    return PromptSet(boxes=boxes)


def pick_points_multi(first_frame: np.ndarray, start_obj_id: int = 1) -> PromptSet:
    """Select MULTIPLE objects with points in one window.

    Controls:
      left-click   = foreground point for the active object
      right-click  = background point for the active object
      n            = start a new object (active id += 1)
      1..9         = switch active object to that id
      ENTER        = finish, ESC = cancel
    Each object id gets its own colour, matching the output overlay palette.
    """
    bgr = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR) if first_frame.shape[-1] == 3 else first_frame
    base = bgr.copy()
    points: list[PointPrompt] = []
    state = {"obj": start_obj_id}

    def redraw() -> np.ndarray:
        disp = base.copy()
        for p in points:
            color = _color(p.obj_id)  # BGR
            cv2.circle(disp, (int(p.xy[0]), int(p.xy[1])), 5, color, -1)
            if p.label == 0:  # background: draw an outline ring to distinguish
                cv2.circle(disp, (int(p.xy[0]), int(p.xy[1])), 7, (255, 255, 255), 1)
        cv2.putText(disp, f"active obj={state['obj']}  (L=fg R=bg n=next 1-9=pick ENTER=done)",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _color(state["obj"]), 2, cv2.LINE_AA)
        return disp

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append(PointPrompt(state["obj"], 0, (float(x), float(y)), label=1))
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append(PointPrompt(state["obj"], 0, (float(x), float(y)), label=0))

    cv2.namedWindow(_MULTI_PT_WINDOW)
    cv2.setMouseCallback(_MULTI_PT_WINDOW, on_mouse)
    while True:
        cv2.imshow(_MULTI_PT_WINDOW, redraw())
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):  # ENTER
            break
        if key == 27:  # ESC
            points.clear()
            break
        if key in (ord("n"), ord("N")):
            state["obj"] += 1
        elif ord("1") <= key <= ord("9"):
            state["obj"] = key - ord("0")
    cv2.destroyWindow(_MULTI_PT_WINDOW)
    if not points:
        raise RuntimeError("No points selected.")
    return PromptSet(points=points)
