"""Training-side code: the dataset, the labels, the loop, the losses.

Nothing here runs at inference time. It exists so the checkpoint can be
specialised to 512x512 single-channel thermal drone footage, and so the same
loop can be reused for quantisation-aware training.

Kept import-light on purpose: `antiuav` parses annotations and computes crop
geometry with nothing but the standard library and numpy, so it is testable
without OpenCV, without EdgeTAM and without a GPU. cv2, torch and transformers
are imported inside the functions that actually need them. `image_loop`,
`distill`, `pretrain` and `automask` need torch at module level and are
therefore *not* re-exported here -- import them by path.

Two data paths, deliberately parallel:

    antiuav   -> clip_loop  -> losses.frame_loss       video, one box per frame
    aerial    -> image_loop -> losses.instance_loss    stills, many prompts each

with `schedule.run_stages` shared between them, because the stages, the
optimiser, the EMA and the checkpoint rule are the same problem either way.
"""
from .aerial import (
    MODES,
    DatasetSpec,
    FrameIndex,
    Instance,
    InstanceGates,
    SPECS,
    Sample,
    Source,
    decompose,
    index_frames,
    list_frames,
    list_pairs,
    probe_classes,
    sample_windows,
    split_frames,
)
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
    "DatasetSpec",
    "MODES",
    "FrameIndex",
    "Gates",
    "Instance",
    "InstanceGates",
    "MaskStore",
    "SPECS",
    "Sample",
    "Sequence",
    "Source",
    "SequenceLabels",
    "clip_tensor",
    "crop_window",
    "decompose",
    "frame_shape",
    "frames_to_label",
    "index_frames",
    "iter_clips",
    "label_sequence",
    "list_frames",
    "list_pairs",
    "list_sequences",
    "load_labels",
    "load_masks",
    "load_window",
    "map_boxes",
    "open_masks",
    "probe_classes",
    "sample_clips",
    "sample_windows",
    "save_masks",
    "split_frames",
    "summarise",
    "zoom_window",
]
