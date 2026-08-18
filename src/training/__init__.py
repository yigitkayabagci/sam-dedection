"""Training-side code: the dataset, the labels, the loop, the losses.

Nothing here runs at inference time. It exists so the checkpoint can be
specialised to 512x512 single-channel thermal drone footage, and so the same
loop can be reused for quantisation-aware training.

Kept import-light on purpose: `antiuav` parses annotations and computes crop
geometry with nothing but the standard library and numpy, so it is testable
without OpenCV, without EdgeTAM and without a GPU. cv2, torch and transformers
are imported inside the functions that actually need them.
"""
from .antiuav import (
    Clip,
    Sequence,
    SequenceLabels,
    clip_tensor,
    crop_window,
    frame_shape,
    iter_clips,
    list_sequences,
    load_labels,
    load_window,
    map_boxes,
    sample_clips,
)
from .labels import (
    Gates,
    MaskStore,
    frames_to_label,
    label_sequence,
    load_masks,
    open_masks,
    save_masks,
    summarise,
    zoom_window,
)

__all__ = [
    "Clip",
    "Gates",
    "MaskStore",
    "Sequence",
    "SequenceLabels",
    "clip_tensor",
    "crop_window",
    "frame_shape",
    "frames_to_label",
    "iter_clips",
    "label_sequence",
    "list_sequences",
    "load_labels",
    "load_masks",
    "load_window",
    "map_boxes",
    "open_masks",
    "sample_clips",
    "save_masks",
    "summarise",
    "zoom_window",
]
