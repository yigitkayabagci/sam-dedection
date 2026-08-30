"""Training-side code: the dataset, the labels, the loop, the losses.

Nothing here runs at inference time. It exists so the checkpoint can be
specialised to 512x512 single-channel thermal drone footage, and so the same
loop can be reused for quantisation-aware training.

Kept import-light on purpose: `antiuav` parses annotations and computes crop
geometry with nothing but the standard library and numpy, so it is testable
without OpenCV, without EdgeTAM and without a GPU. cv2, torch and transformers
are imported inside the functions that actually need them. `image_loop` and
`distill` need torch at module level and are therefore *not* re-exported here
-- import them by path.

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
from .aerial_video import (
    ImageMaskStore,
    PoolMaskStore,
    birdsai_sequences,
    empty_stores,
    flight_name,
    pool_sequence_stores,
    source_name,
    split_flights,
    video_clips,
    vtuav_sequences,
    vtuav_vis_sequences,
    weighted_clip_sample,
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
    "ImageMaskStore",
    "MaskStore",
    "PoolMaskStore",
    "SPECS",
    "Sample",
    "Sequence",
    "Source",
    "SequenceLabels",
    "birdsai_sequences",
    "clip_tensor",
    "crop_window",
    "decompose",
    "empty_stores",
    "flight_name",
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
    "pool_sequence_stores",
    "sample_clips",
    "sample_windows",
    "save_masks",
    "source_name",
    "split_frames",
    "split_flights",
    "summarise",
    "video_clips",
    "vtuav_sequences",
    "vtuav_vis_sequences",
    "weighted_clip_sample",
    "zoom_window",
]
