"""What a checkpoint says about itself, and what to do when it disagrees.

`src/training/finetune.save_checkpoint` writes `{"model": state_dict, "meta":
{...}}`, and `meta["image_size"]` records the resolution the run trained at.
Reading it back matters more than it sounds, because the failure it catches is
invisible everywhere else:

EdgeTAM holds no resolution in any parameter. The RepViT trunk and the neck are
fully convolutional, and the positional encodings are computed per frame from
the actual feature-map size rather than read out of a learned table. So a
512-trained checkpoint loads into a 768 build under `strict=True` with the same
982 keys, the same shapes, and not one warning. Nothing raises, nothing logs,
and the only symptom is a model that quietly scores worse than it should.

`tools/train_encoder.py` has warned about this since the mismatch was first
found, but only on the training side -- the tracker, the ONNX exporter and
everything downstream of them loaded the checkpoint and said nothing. This
module is that check in one place, so the inference path can make the same
noise the training path already makes.

The note is a warning and never an error. Running a checkpoint off its training
resolution is a legitimate measurement (that is exactly what
`configs/edgetam_768_pool_deep.yaml` exists to take); it just has to be a
deliberate one.
"""
from __future__ import annotations

from pathlib import Path


def trained_size(checkpoint: str | Path | None) -> int | None:
    """The input size `checkpoint` records having been trained at, or None.

    None covers every case where the question cannot be answered: no path, a
    file that is not there, a blob that is not a dict, and the stock EdgeTAM
    checkpoint -- which carries no `meta` at all, having been trained at 1024
    long before this repository existed.
    """
    if not checkpoint:
        return None

    import torch

    try:
        blob = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    except Exception:
        return None
    meta = blob.get("meta") if isinstance(blob, dict) else None
    size = (meta or {}).get("image_size") if isinstance(meta, dict) else None
    if isinstance(size, bool) or not isinstance(size, (int, float)):
        return None
    return int(size)


def recorded_meta(checkpoint: str | Path | None) -> dict:
    """Everything `save_checkpoint` recorded about the run that wrote this file.

    `{}` for the cases where there is nothing to read: no path, a missing file,
    a blob that is not a dict, and stock EdgeTAM -- which carries no `meta` at
    all. An empty dict is the honest answer to "what produced this?", and it is
    a different answer from "stock produced it".

    The field worth reading first is `base`, the checkpoint the run started
    from. A stage that continues another one has no other way to tell whether
    it is continuing the arm it meant to: the weights load strictly and score
    normally whichever base they came from, so the wrong one costs a full run
    before anything looks odd.
    """
    if not checkpoint:
        return {}

    import torch

    try:
        blob = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    except Exception:
        return {}
    meta = blob.get("meta") if isinstance(blob, dict) else None
    return dict(meta) if isinstance(meta, dict) else {}


def size_mismatch_note(checkpoint: str | Path | None,
                       image_size: int | None) -> str | None:
    """One line to print when a checkpoint is being run off its trained size.

    Returns None when there is nothing to say -- the sizes agree, or the
    checkpoint never recorded one, or the caller did not override the size at
    all (in which case the Hydra config's own 1024 applies and the stock
    checkpoint is right about it).
    """
    if image_size is None:
        return None
    was = trained_size(checkpoint)
    if was is None or was == int(image_size):
        return None
    return (
        f"!! {checkpoint} records image_size={was}, running at {image_size}. "
        f"The weights load either way -- EdgeTAM keeps no resolution in any "
        f"parameter -- so this runs and may simply be worse. Compare it "
        f"against a stock run at {image_size}, never against numbers taken at "
        f"{was}."
    )


def warn_size_mismatch(checkpoint: str | Path | None,
                       image_size: int | None) -> str | None:
    """`size_mismatch_note`, printed to stderr. Returns what it printed."""
    import sys

    note = size_mismatch_note(checkpoint, image_size)
    if note:
        print(note, file=sys.stderr)
    return note
