"""Stage A, as four arms that all produce the same artifact.

`docs/encoder_training_todo.md` has one pretraining route in it -- distil an
RGB foundation model into the thermal encoder over registered pairs -- and no
way to say what it should be compared against. This module is the missing
frame. It makes stage A a **choice between four arms**, holds the data,
schedule, seed and output format fixed across all of them, and hands stage B a
single kind of file no matter which arm ran:

| arm | what supervises the encoder | what data it can eat |
|---|---|---|
| `none` | nothing -- stock EdgeTAM weights | none |
| `mask` | the model's own features on the unmasked frame (`automask.py`) | **anything**, one modality is enough |
| `distil` | a frozen teacher's features on a paired input | needs a teacher-side image |
| `both` | `distil` warmed up, then `mask` | whatever `distil` can eat |

The arms are not equally likely to win and the table is not a menu to pick
from by taste: `none` is the baseline every other arm has to beat, and an arm
that does not beat it cost GPU hours and bought nothing. That comparison is the
only output of this stage that means anything, and it is measured in stage B,
not here.

**One artifact.** Every arm writes an ordinary EdgeTAM state dict --
`{"model": state_dict, "meta": {...}}`, the format `finetune.save_checkpoint`
has always written and `build_sam2` loads unchanged -- plus a `stage_a.json`
manifest beside it naming the arm, the corpora, the seed and the step count. So
`tools/train_encoder.py --base <that file>` is the whole handoff, and the
manifest is what lets a stage-B run log say what it started from six weeks
later. `none` writes one too: a copy of the stock checkpoint and a manifest
saying `arm=none`, because a baseline that is handled by a different code path
is a baseline that drifts.

## The modality question, which is the real content here

The teacher is an **RGB** model. The deployment is **thermal**. What each of
the two sees is not a detail -- it decides which datasets can be used at all,
and the answer is different for the three kinds of corpus that exist:

    paired    registered RGB-T          teacher <- RGB,        student <- thermal
    rgb       aerial RGB, no thermal    teacher <- RGB colour, student <- its own luminance
    thermal   thermal, no RGB twin      teacher <- thermal,    student <- thermal

`paired` is the published route (Gupta, Hoffman & Malik, CVPR 2016, and
everything since): the teacher works on the modality it was trained on, the
student on the modality it will be deployed on, and the registration is what
makes one supervise the other. Nothing else here is as well founded.

`rgb` is a **proxy, and is labelled as one**. Feeding the student the same
colour image the teacher gets would teach it to use a channel the sensor does
not have, and the deployed encoder has never seen colour in its life. Feeding
it the *luminance* of that image keeps the structure of the real task -- one
channel in, three-channel semantics out -- while being honest that the sensor
is wrong. It is what lets an aerial RGB corpus with no thermal twin (VisDrone
and friends) contribute at all, and it is the arm to verify the plumbing on
first, because a pipeline that cannot learn luminance-to-colour is broken for
reasons that have nothing to do with thermal.

`thermal` is the one that has to be **measured before it is used**, and
`modality_gap` is the measurement. Pushing a thermal frame through an RGB
teacher produces *something*; whether that something carries information about
the scene or is closer to noise is a number, not an opinion, and the number is
cheap: on registered pairs, compare the teacher's features on the RGB half
against its features on the thermal half, and read that against two references
-- how far apart two *unrelated* frames land (chance), and how far the same
frame lands from itself under a mild photometric change (this teacher's own
floor). If the thermal gap sits near chance, thermal-only corpora are worthless
to the `distil` arm and belong to the `mask` arm, which needs no teacher at
all. If it sits near the floor, every thermal frame on disk is usable.

There is a published answer to expect, and it is split in a way that decides
the design. The Caltech aerial RGB-T set (arXiv 2403.08997, ECCV 2024) is this
project's domain -- a drone under 120 m with co-registered halves -- and it
measured both halves of the question on the same frames:

* **A foundation model's *outputs* collapse on thermal.** SAM's instance masks
  on the thermal half, scored against its own masks on the registered RGB half,
  reach AP **0.018**, against 0.538 on a *grayscale* version of the RGB. Open-
  vocabulary segmenters lose 0.13-0.27 mIoU the same way. So distilling a
  teacher's *predictions* from thermal is worthless, and the one paper that
  tried response-level distillation RGB-to-thermal scored 2.9 mAP50 **below not
  distilling at all** (arXiv 2606.11572).
* **Its *features* mostly survive.** On the same set, a **frozen** DINOv2 read
  with a nonlinear head scores 0.706 mIoU on thermal segmentation against 0.725
  for the best end-to-end network trained on thermal, and beats the
  thermal-specific FTNet's 0.613. The patch features transfer; the head does
  not.

Which is exactly what this stage does -- it copies a dense feature map and
never touches a prediction -- and it is why the `thermal->thermal` route is
worth measuring rather than dismissing.

The same literature carries one more finding that ought to change the order of
work here rather than only the settings. On this project's own model scale --
a 4.8 M EfficientViT, aerial thermal segmentation -- the same paper reports
scratch **0.687**, generic **ImageNet 0.725**, and self-supervised pretraining
on 40 k *thermal* images **0.714**. Ordinary RGB pretraining beat thermal
self-supervision at this size. It is one table and it is not this task, but it
says plainly which arm is the favourite and which is the long shot, and it is
the reason `none` is not a formality.

That the two arms want opposite data is the reason both exist. `distil` is
starved by the scarcity of registered pairs; `mask` does not care, and the
thermal-only sets -- which are the large ones, and the ones in the deployment's
own modality -- are its natural food.

## Adding a dataset

A corpus is one row. Either name a spec that `aerial.SPECS` already knows:

    Corpus("kust4k", "/content/data/Kust4K", "paired", spec="kust4k")

or spell the globs out, which is what a set nobody has written a spec for
needs:

    Corpus("visdrone", "/content/data/VisDrone", "rgb", rgb="**/images/*.jpg")

Nothing else has to be touched: `index` derives the crop from the pictures that
are actually on disk, `summarise` prints what it found per corpus, and the
loops read a flat list of `Item`.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from .aerial import IMAGE_SUFFIXES, SPECS, DatasetSpec, load_image, normalise
from .distill import (Projector, distill_loss, encoder_features,
                      shared_window, teacher_maps)
from .finetune import EMA, Rates, apply_freeze, param_groups

MODALITIES = ("paired", "rgb", "thermal")
ARMS = ("none", "mask", "distil", "both")

# What each modality feeds to which side. `drop` is how a corpus is kept for
# the mask arm and kept out of the distillation arm without deleting its row.
ROUTES = {
    # route            student half   student gray   teacher half   teacher gray
    "rgb->thermal":   ("thermal",     True,          "rgb",         False),
    "rgb->gray":      ("rgb",         True,          "rgb",         False),
    "thermal->thermal": ("thermal",   True,          "thermal",     True),
    "rgb->rgb":       ("rgb",         False,         "rgb",         False),
    "drop":           ("thermal",     True,          None,          False),
}

DEFAULT_ROUTE = {
    "paired": "rgb->thermal",
    "rgb": "rgb->gray",
    "thermal": "thermal->thermal",
}


@dataclass(frozen=True)
class Corpus:
    """One unlabelled image source, and how its two halves are to be read.

    Deliberately not `aerial.Source`, which is stage B's and carries a palette,
    gates and a role. Stage A reads no labels, so a corpus is a root, a pair of
    globs and a statement of which modalities are on disk -- and that is the
    whole of it, which is what makes a new dataset one line rather than a spec.

    `spec` borrows an existing `DatasetSpec`'s globs, border and strip rules
    when the set already has one, so the two stages cannot end up reading
    different halves of the same archive. Anything given explicitly wins over
    the spec, which is how a download whose layout differs from its paper is
    corrected without editing `aerial.py`.
    """

    name: str
    root: str | Path
    modality: str = "paired"
    thermal: str = ""
    rgb: str = ""
    spec: str = ""
    route: str = ""
    crop: float | str | None = "auto"
    border: int | None = None
    limit: int | None = None
    strip: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.modality not in MODALITIES:
            raise ValueError(
                f"{self.name}: modality must be one of {MODALITIES}, "
                f"got {self.modality!r}")
        if self.route and self.route not in ROUTES:
            raise ValueError(
                f"{self.name}: route must be one of {sorted(ROUTES)}, "
                f"got {self.route!r}")
        if self.spec and self.spec not in SPECS:
            raise ValueError(
                f"{self.name}: no spec named {self.spec!r} -- "
                f"have {sorted(SPECS)}")

    # -- resolution -------------------------------------------------------
    @property
    def routing(self) -> str:
        return self.route or DEFAULT_ROUTE[self.modality]

    def globs(self) -> tuple[str, str]:
        """`(thermal, rgb)` globs, spec-filled and then overridden."""
        base = SPECS[self.spec] if self.spec else None
        thermal = self.thermal or (base.thermal if base else "") or ""
        rgb = self.rgb or (base.rgb if base else "") or ""
        return thermal, rgb

    def padding(self) -> int:
        if self.border is not None:
            return int(self.border)
        return int(SPECS[self.spec].border) if self.spec else 0

    def strips(self) -> tuple[str, ...]:
        return self.strip or (tuple(SPECS[self.spec].strip) if self.spec else ())

    def wants(self) -> tuple[str, str | None]:
        """`(student half, teacher half)` as modality names."""
        student, _, teacher, _ = ROUTES[self.routing]
        return student, teacher

    def check(self) -> None:
        """Refuse a row whose route asks for a half the modality does not have.

        The failure this prevents is silent rather than loud: a `thermal`
        corpus routed `rgb->thermal` simply finds no RGB glob, pairs nothing,
        and contributes zero items to a run that reports a total and looks
        fine.
        """
        student, teacher = self.wants()
        thermal, rgb = self.globs()
        have = {"thermal": bool(thermal), "rgb": bool(rgb)}
        if self.modality == "rgb":
            have["thermal"] = False
        if self.modality == "thermal":
            have["rgb"] = False
        for side, half in (("student", student), ("teacher", teacher)):
            if half and not have[half]:
                raise ValueError(
                    f"{self.name}: route {self.routing!r} wants the {half} half "
                    f"for the {side}, but a {self.modality!r} corpus with "
                    f"thermal={thermal!r} rgb={rgb!r} has none. Give the glob, "
                    f"name a spec, or change the route.")


class Item(NamedTuple):
    """One training sample: which file each side reads, and how.

    A flat tuple and not a reference back to its `Corpus` because the batcher
    shuffles items from several corpora together, and a per-item crop and
    border is exactly what lets a 1920x1080 set and a bordered 640x512 one
    share a batch -- the same reason `distill.Pair` carries its own.

    `teacher` is None for a corpus routed `drop`, and for every item when the
    arm has no teacher. `collate` reads the student half either way, so the
    mask arm consumes the same index the distillation arm does.
    """

    student: Path
    teacher: Path | None
    crop: float | None
    border: int
    corpus: str
    student_gray: bool = True
    teacher_gray: bool = False


@dataclass
class Batch:
    """A student input, and the teacher's input for the same window."""

    student: torch.Tensor                  # [B, 3, S, S]
    teacher: torch.Tensor | None = None    # [B, 3, S, S], or None

    def __len__(self) -> int:
        return int(self.student.shape[0])

    def to(self, device: str) -> "Batch":
        return Batch(
            student=self.student.to(device, non_blocking=True),
            teacher=(None if self.teacher is None
                     else self.teacher.to(device, non_blocking=True)))


# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------


def _spec_for(corpus: Corpus) -> DatasetSpec:
    """A throwaway spec carrying this corpus's globs, for `aerial.list_pairs`.

    Reusing that function rather than re-deriving the pairing key is not
    tidiness: the key has to strip the glob's own fixed directories or two
    sequences that both number frames from zero collide, and `aerial._key`
    already gets that right and is tested. A corpus with hand-written globs
    gets the same treatment as one with a spec.
    """
    thermal, rgb = corpus.globs()
    base = SPECS[corpus.spec] if corpus.spec else None
    from dataclasses import replace as _replace

    if base is not None:
        return _replace(base, thermal=thermal, rgb=rgb,
                        border=corpus.padding(), strip=corpus.strips())
    return DatasetSpec(
        name=corpus.name, masks="", classes={}, things=(),
        thermal=thermal, rgb=rgb, strip=corpus.strips(),
        border=corpus.padding(), palette_source="stage A reads no labels")


def _listing(root: Path, pattern: str) -> list[Path]:
    return sorted(p for p in root.glob(pattern)
                  if p.suffix.lower() in IMAGE_SUFFIXES)


def _auto_crop(sample: Path, corpus: Corpus, size: int) -> float | None:
    """The crop fraction that keeps native pixels, read off a real image.

    Set by hand it is the most error-prone knob in the stage: too high on a
    1920x1080 source and every small target is resampled away before training
    sees it. Derived, it is `size` over the shorter side once the border is
    off, or no crop at all when the source is already at or below `size`.
    """
    import cv2

    probe = cv2.imread(str(sample), cv2.IMREAD_UNCHANGED)
    if probe is None:
        raise FileNotFoundError(f"Could not read {sample}")
    spec = _spec_for(corpus)
    _, (width, height) = spec.inset(int(probe.shape[0]), int(probe.shape[1]))
    shorter = min(height, width)
    return size / shorter if shorter > size else None


def index(corpora: Sequence[Corpus], size: int = 512,
          seed: int = 0, log=print) -> list[Item]:
    """Every usable sample across every corpus, tagged with how to read it.

    Corpora are indexed independently and concatenated, and each carries its
    own crop and border, so mixing a bordered 640x512 set with an uncropped
    1920x1080 one needs no coordination between the rows.

    `limit` subsamples **across** a corpus rather than truncating it, for the
    reason `distill.subsample` gives: most of these sets come from video, so
    the first N files are two flights.
    """
    from .aerial import list_pairs

    out: list[Item] = []
    for corpus in corpora:
        corpus.check()
        root = Path(corpus.root)
        if not root.exists():
            log(f"  !! {corpus.name}: {root} does not exist -- skipped")
            continue
        student_half, teacher_half = corpus.wants()
        thermal, rgb = corpus.globs()
        found: list[tuple[Path, Path | None]] = []

        if teacher_half is not None and teacher_half != student_half:
            # Two different halves of the same frame: pair them by stem.
            pairs = list_pairs(root, _spec_for(corpus))
            picked = {"thermal": 0, "rgb": 1}
            found = [(p[picked[student_half]], p[picked[teacher_half]])
                     for p in pairs]
        else:
            pattern = thermal if student_half == "thermal" else rgb
            files = _listing(root, pattern)
            if not files:
                log(f"  !! {corpus.name}: nothing matched {pattern!r} under "
                    f"{root} -- skipped")
                continue
            found = [(f, f if teacher_half is not None else None) for f in files]

        if not found:
            log(f"  !! {corpus.name}: no samples -- skipped")
            continue
        found = subsample(found, corpus.limit, seed=seed)

        crop = corpus.crop
        if crop == "auto":
            crop = _auto_crop(found[0][0], corpus, size)
        _, student_gray, _, teacher_gray = ROUTES[corpus.routing]
        border = corpus.padding()
        out += [Item(s, t, crop, border, corpus.name, student_gray, teacher_gray)
                for s, t in found]
        shown = "none" if crop is None else f"{crop:.3f}"
        log(f"  {corpus.name:14s} {len(found):>8,} samples  {corpus.routing:<17s}"
            f" crop={shown:>5s} border={border}")
    return out


def subsample(items: Sequence, count: int | None, seed: int = 0) -> list:
    """`count` items spread across the whole list, not the first `count` of it.

    The same argument as `distill.subsample`, kept here so a corpus can be
    capped without importing that module's pair-shaped version.
    """
    items = list(items)
    if count is None or count >= len(items):
        return items
    order = np.random.default_rng(seed).permutation(len(items))[:count]
    return [items[int(i)] for i in sorted(order)]


def summarise(items: Sequence[Item]) -> str:
    """Per-corpus counts and routes, for the cell that prints what will run."""
    if not items:
        return "no samples indexed"
    rows: dict[str, dict] = {}
    for item in items:
        row = rows.setdefault(item.corpus, {"n": 0, "teacher": 0})
        row["n"] += 1
        row["teacher"] += item.teacher is not None
    width = max(len(name) for name in rows)
    lines = [f"{'corpus':<{width}}  {'samples':>9}  {'with teacher':>12}"]
    for name, row in sorted(rows.items()):
        lines.append(f"{name:<{width}}  {row['n']:>9,}  {row['teacher']:>12,}")
    total = sum(r["n"] for r in rows.values())
    paired = sum(r["teacher"] for r in rows.values())
    lines.append(f"{'total':<{width}}  {total:>9,}  {paired:>12,}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Pixels
# --------------------------------------------------------------------------


def collate(items: Sequence[Item], size: int = 512, device: str = "cpu",
            executor=None, rng: np.random.Generator | None = None,
            want_teacher: bool = True, jitter: int = 0) -> Batch:
    """Read a batch, each item under its own crop, border and gray flags.

    The crop placement is drawn once per **item** and used for both halves, so
    a registered pair stays registered: a teacher looking at a different corner
    of the frame from the student is not a teacher, it is noise with a
    confident gradient. `distill.shared_window` does the placement in
    normalised coordinates, which is what makes it survive two halves that were
    published at different resolutions.

    **`jitter` deliberately breaks that, by a few pixels, on purpose.** These
    archives are not registered to the pixel and it is worth being specific
    about how far off they are: DroneVehicle's own authors report over 20 % of
    its boxes deviating by more than 3 px or 3 degrees, and on KAIST -- the
    canonical *aligned* multispectral set -- over half the boxes are shifted,
    mostly by 0-10 px. The measured shape of the damage (AR-CNN, ICCV 2019,
    which shifted one modality and re-scored) is that 1-2 px is free, 4 px
    costs a few points, and 6 px costs about 65 % relative. That paper's remedy
    was to *train under* the shift rather than try to remove it, which cut the
    variance across shifts from 11.55 to 1.24. This is the same remedy: offset
    the teacher's window by up to `jitter` pixels, so the student cannot learn
    to depend on a correspondence the data does not actually have.

    It applies only where the two halves are different files. A corpus routed
    `rgb->gray` reads one image twice and is perfectly registered by
    construction; jittering it would be inventing a problem.
    """
    mapper = map if executor is None else executor.map
    rng = rng or np.random.default_rng()
    places = rng.random((len(items), 2))
    shifts = (rng.integers(-jitter, jitter + 1, size=(len(items), 2))
              if jitter else np.zeros((len(items), 2), dtype=int))

    def read(args):
        path, gray, place, crop, border, offset = args
        import cv2

        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        height = int(raw.shape[0]) - 2 * border
        width = int(raw.shape[1]) - 2 * border
        if height <= 0 or width <= 0:
            raise ValueError(
                f"border={border} leaves nothing of {path} "
                f"({raw.shape[1]}x{raw.shape[0]})")
        if crop:
            origin, window = shared_window((height, width), place, crop)
        else:
            origin, window = (0, 0), (width, height)
        x0 = min(max(origin[0] + int(offset[0]), 0), max(width - window[0], 0))
        y0 = min(max(origin[1] + int(offset[1]), 0), max(height - window[1], 0))
        return load_image(path, (x0 + border, y0 + border), window, size, gray)

    still = np.zeros(2, dtype=int)
    student = list(mapper(read, [(i.student, i.student_gray, p, i.crop, i.border, still)
                                 for i, p in zip(items, places)]))
    teacher = None
    if want_teacher:
        missing = [i.corpus for i in items if i.teacher is None]
        if missing:
            raise ValueError(
                f"want_teacher=True but {len(missing)} item(s) have no teacher "
                f"half (corpora: {sorted(set(missing))}). Filter the index with "
                f"`with_teacher` before streaming, or route those corpora.")
        teacher = list(mapper(read, [
            (i.teacher, i.teacher_gray, p, i.crop, i.border,
             s if i.teacher != i.student else still)
            for i, p, s in zip(items, places, shifts)]))
    return Batch(
        student=normalise(np.stack(student), device),
        teacher=None if teacher is None else normalise(np.stack(teacher), device))


def with_teacher(items: Sequence[Item], log=print) -> list[Item]:
    """The items a teacher-side arm can actually use, and a word about the rest."""
    usable = [i for i in items if i.teacher is not None]
    dropped = len(items) - len(usable)
    if dropped:
        names = sorted({i.corpus for i in items if i.teacher is None})
        log(f"{dropped:,} of {len(items):,} samples have no teacher half and are "
            f"skipped by this arm ({', '.join(names)}). The mask arm reads them.")
    if not usable:
        raise SystemExit(
            "no sample in the index has a teacher half -- this arm has nothing "
            "to train on. Either add a `paired` or `rgb` corpus, or run the "
            "mask arm, which needs no teacher.")
    return usable


def stream(items: Sequence[Item], batch: int, size: int = 512,
           seed: int | None = None, limit: int | None = None,
           device: str = "cuda", workers: int = 8, depth: int = 2,
           want_teacher: bool = True, jitter: int = 0):
    """Shuffled, prefetched batches -- the same threading as every other mode."""
    from .loader import batch_clips, prefetch_with

    rng = np.random.default_rng(seed)
    chunks = batch_clips(items, batch, seed=seed, limit=limit)
    return prefetch_with(
        chunks,
        lambda chunk, pool: collate(chunk, size, "cpu", pool, rng,
                                    want_teacher, jitter),
        device=device, workers=workers, depth=depth,
    )


# --------------------------------------------------------------------------
# The measurement that decides which corpora the distillation arm can eat
# --------------------------------------------------------------------------


def to_pixels(images: torch.Tensor) -> torch.Tensor:
    """Undo `normalise`, back to [0, 1] RGB."""
    from .aerial import MEAN, STD

    mean = torch.as_tensor(MEAN, device=images.device)[:, None, None]
    std = torch.as_tensor(STD, device=images.device)[:, None, None]
    return images * std + mean


def from_pixels(pixels: torch.Tensor) -> torch.Tensor:
    from .aerial import MEAN, STD

    mean = torch.as_tensor(MEAN, device=pixels.device)[:, None, None]
    std = torch.as_tensor(STD, device=pixels.device)[:, None, None]
    return (pixels - mean) / std


def to_luminance(images: torch.Tensor) -> torch.Tensor:
    """The same image with its colour thrown away, replicated back to 3 channels.

    Rec. 601 weights, which is what `cv2.COLOR_BGR2GRAY` uses, so a tensor
    grayed here and a file read with `load_image(gray=True)` agree.
    """
    pixels = to_pixels(images)
    weights = torch.as_tensor([0.299, 0.587, 0.114],
                              device=images.device)[None, :, None, None]
    grey = (pixels * weights).sum(dim=1, keepdim=True).expand(-1, 3, -1, -1)
    return from_pixels(grey)


def photometric(images: torch.Tensor, rng: torch.Generator | None = None,
                gain: float = 0.2, bias: float = 0.1) -> torch.Tensor:
    """A mild per-image brightness/contrast change -- the teacher's noise floor.

    Not an augmentation policy and not meant to be one. It exists so that
    `modality_gap`'s cross-modal number has something to be read *against*: a
    cosine of 0.6 means nothing until you know what the same teacher scores
    against the same picture under a change that everyone agrees is
    inconsequential.
    """
    pixels = to_pixels(images)
    shape = (pixels.shape[0], 1, 1, 1)
    scale = 1.0 + (torch.rand(shape, device=pixels.device, generator=rng)
                   * 2 - 1) * gain
    shift = (torch.rand(shape, device=pixels.device, generator=rng)
             * 2 - 1) * bias
    return from_pixels((pixels * scale + shift).clamp(0.0, 1.0))


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-position cosine similarity between two dense maps."""
    import torch.nn.functional as F

    if a.shape[-2:] != b.shape[-2:]:
        b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
    return float((F.normalize(a.float(), dim=1)
                  * F.normalize(b.float(), dim=1)).sum(dim=1).mean())


@torch.no_grad()
def modality_gap(teacher, items: Sequence[Item], *, size: int = 512,
                 samples: int = 256, batch: int = 8, seed: int = 0,
                 device: str = "cuda", log=print) -> dict:
    """How much of the teacher survives when it is shown thermal instead of RGB.

    Four numbers, all mean per-position cosine similarity between two of the
    teacher's own dense feature maps, and all on the **same** registered
    windows so nothing but the input differs:

    | name | compares | reads as |
    |---|---|---|
    | `floor` | RGB against the same RGB, brightness and contrast nudged | this teacher's own noise -- the ceiling any other row could reach |
    | `luminance` | RGB against its own grayscale | what losing *colour alone* costs |
    | `thermal` | RGB against the registered thermal frame | **the number this exists for** |
    | `chance` | RGB against an unrelated frame's RGB | the bottom of the scale |

    The decision it drives: if `thermal` sits near `chance`, an RGB teacher
    shown a thermal frame is describing nothing, so thermal-only corpora cannot
    feed the distillation arm and belong to the mask arm. If it sits near
    `luminance`, the teacher is mostly reading shape and layout -- which
    thermal has -- and every thermal frame on disk is usable. In between is the
    interesting case and the one where the ordering against `luminance`
    matters more than the absolute value.

    It needs **registered pairs** and says so rather than quietly scoring an
    item against itself: a corpus routed `rgb->gray` has the same file on both
    sides, and its `thermal` row would come out equal to `luminance` and mean
    nothing.
    """
    usable = [i for i in items
              if i.teacher is not None and i.teacher != i.student]
    if not usable:
        raise SystemExit(
            "modality_gap needs registered RGB-T pairs -- no indexed item has a "
            "teacher file different from its student file. Add a `paired` "
            "corpus, or skip the probe.")
    usable = subsample(usable, samples, seed=seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    totals = {"floor": [], "luminance": [], "thermal": [], "chance": []}

    for chunk in _chunks(usable, batch):
        pair = collate(chunk, size, device, want_teacher=True,
                       rng=np.random.default_rng(seed))
        rgb = teacher.features(pair.teacher)
        totals["floor"].append(_cosine(rgb, teacher.features(
            photometric(pair.teacher, generator))))
        totals["luminance"].append(_cosine(rgb, teacher.features(
            to_luminance(pair.teacher))))
        totals["thermal"].append(_cosine(rgb, teacher.features(pair.student)))
        if len(chunk) > 1:
            totals["chance"].append(_cosine(rgb, rgb.roll(1, dims=0)))

    out = {name: float(np.mean(values)) for name, values in totals.items() if values}
    out["samples"] = len(usable)
    out["teacher"] = getattr(teacher, "model_id", "?")
    log(summarise_gap(out))
    return out


def _chunks(items: Sequence, size: int):
    """Whole batches, and the whole list when there are not enough for one.

    Dropping the tail is right for training and wrong here: a probe asked for
    64 samples with a batch of 8 would return nothing at all if the index
    happened to hold 7, and report a mean over an empty list.
    """
    if len(items) < size:
        if items:
            yield list(items)
        return
    for start in range(0, len(items) - size + 1, size):
        yield list(items[start:start + size])


def summarise_gap(gap: dict) -> str:
    """The four numbers, in order, with the reading spelled out.

    The verdict is a rule and not a judgement so that two runs cannot disagree
    about what the same numbers meant: the thermal row is placed on the scale
    between chance and the teacher's own floor, and the fraction of that scale
    it reaches is the only thing that decides.
    """
    order = ("floor", "luminance", "thermal", "chance")
    lines = [f"teacher {gap.get('teacher', '?')} on {gap.get('samples', 0)} "
             f"registered windows", ""]
    lines += [f"  {name:<10} {gap[name]:.4f}" for name in order if name in gap]
    if not {"thermal", "chance", "floor"} <= gap.keys():
        return "\n".join(lines)
    span = gap["floor"] - gap["chance"]
    reached = (gap["thermal"] - gap["chance"]) / span if span > 1e-6 else 0.0
    lines += ["", f"  thermal reaches {reached:.0%} of the way from chance to "
                  f"this teacher's own floor"]
    if reached < 0.25:
        verdict = ("near chance: an RGB teacher shown a thermal frame is not "
                   "describing the scene. Route thermal-only corpora to the "
                   "mask arm (`route=\"drop\"`) rather than distilling on them.")
    elif reached < 0.6:
        verdict = ("partial: usable, but worth less per sample than a "
                   "registered pair. Keep thermal-only corpora if pairs are "
                   "scarce; drop them if they would outnumber the pairs.")
    else:
        verdict = ("close to the floor: the teacher is reading shape and "
                   "layout, which thermal has. Thermal-only corpora can feed "
                   "the distillation arm directly.")
    return "\n".join(lines + ["", f"  -> {verdict}"])


# --------------------------------------------------------------------------
# The published feature-distillation loss, beside this repo's cosine one
# --------------------------------------------------------------------------


def centre(features: torch.Tensor) -> torch.Tensor:
    """Parameter-free LayerNorm across channels, position by position.

    The *whitening* step, and it is worth naming because two separate lines of
    work arrived at it independently: feature distillation (arXiv 2205.14141)
    applies an un-affine LayerNorm to the teacher's map before the loss, and
    data2vec applies the same normalisation to its EMA target. The reason is
    the same in both -- a frozen model's channels arrive on wildly different
    scales, and without it a handful of large channels decide the whole loss.

    Cosine already removes the per-position magnitude; what this removes is the
    per-position *offset*, which cosine does not.
    """
    import torch.nn.functional as F

    moved = features.float().permute(0, 2, 3, 1)
    return F.layer_norm(moved, (moved.shape[-1],)).permute(0, 3, 1, 2)


def feature_loss(student: torch.Tensor, teacher: torch.Tensor,
                 beta: float = 2.0, whiten: bool = True
                 ) -> tuple[torch.Tensor, dict[str, float]]:
    """Whitened smooth-L1 between two dense maps -- arXiv 2205.14141's recipe.

    The canonical published feature-distillation objective, transcribed rather
    than adapted: whiten the teacher's map with a scale-and-bias-free
    LayerNorm, bridge the channel counts with a 1x1 convolution on the student,
    feed both sides the *same* augmented view, and take a smooth L1 with
    **beta = 2.0**. That paper's ablation is the reason for each part -- against
    an un-normalised target it reports up to +1.0 top-1, against an l2-normalised
    one up to +1.1, and it notes the property that matters more than either:
    with whitening the hyperparameters stop depending on which teacher is used.

    **Why it is offered beside `distill.distill_loss` rather than replacing
    it.** The cosine loss this repo already runs is scale-free by construction,
    which is a real argument in its favour when the student's raw magnitude is
    what EdgeTAM's own mask decoder was trained to read. The two disagree about
    exactly one thing -- whether the *magnitude* of the teacher's features is
    worth copying -- and that disagreement is a run, not an argument. Holding
    everything else fixed and switching this one setting is what makes it one.
    """
    import torch.nn.functional as F

    if student.shape[-2:] != teacher.shape[-2:]:
        teacher = F.interpolate(teacher, size=student.shape[-2:],
                                mode="bilinear", align_corners=False)
    target = centre(teacher) if whiten else teacher.float()
    loss = F.smooth_l1_loss(student.float(), target, beta=beta)
    with torch.no_grad():
        cosine = 1.0 - (F.normalize(student.float(), dim=1)
                        * F.normalize(target, dim=1)).sum(dim=1).mean()
    return loss, {"smooth_l1": float(loss.detach()), "cosine": float(cosine)}


def channel_normalise(features: torch.Tensor) -> torch.Tensor:
    """Mean-centre and L2-normalise each channel over its own map.

    FreqKD's preprocessing step (arXiv 2606.11572), and it reports it as worth
    **+1.4 mAP50 on its own** -- larger than most of the loss choices it is
    applied under. It removes the per-channel offset and scale that differ
    between two models trained by different people on different modalities,
    leaving the spatial pattern, which is the only part that could transfer.
    """
    flat = features.float().flatten(2)
    flat = flat - flat.mean(dim=2, keepdim=True)
    flat = flat / flat.norm(dim=2, keepdim=True).clamp(min=1e-6)
    return flat.reshape(features.shape)


def frequency_loss(student: torch.Tensor, teacher: torch.Tensor,
                   cutoff: float = 0.5, high_weight: float = 0.1
                   ) -> tuple[torch.Tensor, dict[str, float]]:
    """Strict on low spatial frequencies, forgiving on high ones.

    The one loss in this file that was measured on **this exact problem** --
    distilling a frozen RGB DINO-family teacher into a thermal student -- and
    the measurement is worth quoting, because it is mostly a list of things
    that do not work. On KAIST, against a student that was merely initialised
    from the RGB weights and not distilled at all (61.7 mAP50): uniform
    full-band MSE scores **61.1, worse than not distilling**; response-level
    distillation **58.8, much worse**; cosine 62.1; and this frequency-split
    loss **64.1**. (arXiv 2606.11572, single runs.)

    The reason is measurable and is the useful part. On paired frames, the two
    modalities' feature maps diverge about **2.4x more in the high band than in
    the low band** -- a thermal image's fine texture is genuinely different from
    a colour image's, while its layout is the same object in the same place. A
    loss that weights both equally spends most of its gradient forcing the
    student to reproduce texture that its sensor does not measure, which is why
    plain MSE goes backwards.

    So: FFT the two maps over the spatial axes, split at `cutoff` in normalised
    radial frequency, take a hard squared error below it and a saturating
    `log(1 + |.|^2)` above it at `high_weight`. The saturation matters twice
    over -- it also caps what any single badly-registered position can
    contribute, which is the failure mode of pairs that are aligned to a few
    pixels rather than exactly.
    """
    import torch.nn.functional as F

    if student.shape[-2:] != teacher.shape[-2:]:
        teacher = F.interpolate(teacher, size=student.shape[-2:],
                                mode="bilinear", align_corners=False)
    left = channel_normalise(student)
    right = channel_normalise(teacher)
    difference = torch.fft.fft2(left) - torch.fft.fft2(right)
    height, width = difference.shape[-2:]
    fy = torch.fft.fftfreq(height, device=difference.device)[:, None]
    fx = torch.fft.fftfreq(width, device=difference.device)[None, :]
    radius = (fy * fy + fx * fx).sqrt() / (0.5 * 2 ** 0.5)
    low = radius <= cutoff
    power = difference.abs().pow(2)
    low_term = (power * low).mean()
    high_term = torch.log1p(power * (~low)).mean()
    total = low_term + high_weight * high_term
    return total, {"low": float(low_term.detach()),
                   "high": float(high_term.detach())}


def gram_loss(student: torch.Tensor, teacher: torch.Tensor,
              positions: int = 256, seed: int = 0) -> torch.Tensor:
    """Match the *relations* between positions, not the positions themselves.

    `||X_s X_s^T - X_t X_t^T||^2` on L2-normalised features -- DINOv3's Gram
    anchoring (arXiv 2508.10104) and, in a different literature, the relational
    distillation that beat plain similarity distillation by **+2.1 to +2.7
    zero-shot mIoU** on image-to-LiDAR transfer (arXiv 2409.00845, averaged
    over three runs). Both arrive at the same object for the same reason: a
    per-position loss asks the student to land where the teacher landed, and a
    relational one asks only that the *pattern of agreements and disagreements*
    between positions survive.

    Two properties make it the right extra term for this project specifically.
    It is **projector-free** -- a Gram matrix is `N x N` whatever the channel
    count -- so it constrains the student's own representation rather than the
    scaffolding in front of it. And it is **tolerant of misregistration in a
    way a positional loss cannot be**: it never asks what is at position *p*,
    only whether *p* and *q* look alike, which is exactly the part of a pair
    that survives a few pixels of shift.

    Sampled at `positions` cells: the matrix is quadratic in the grid, and a
    32x32 map has 1 024 positions.
    """
    import torch.nn.functional as F

    left = F.normalize(student.float(), dim=1).flatten(2)
    right = F.normalize(teacher.float(), dim=1)
    if right.shape[-2:] != student.shape[-2:]:
        right = F.interpolate(right, size=student.shape[-2:],
                              mode="bilinear", align_corners=False)
    right = F.normalize(right, dim=1).flatten(2)

    count = left.shape[-1]
    if count > positions:
        pick = torch.randperm(
            count, device=left.device,
            generator=torch.Generator(device=left.device).manual_seed(seed)
        )[:positions]
        left, right = left[..., pick], right[..., pick]
    return (torch.einsum("bcn,bcm->bnm", left, left)
            - torch.einsum("bcn,bcm->bnm", right, right)).pow(2).mean()


LOSSES = ("cosine", "feature", "frequency")


# --------------------------------------------------------------------------
# The loop every arm shares
# --------------------------------------------------------------------------


def run(model, items: Sequence[Item], step, *, save, extra=(),
        extra_lr: float = 1e-3, size: int = 512, epochs: int = 1,
        batch: int = 8, steps_per_epoch: int = 400,
        rates: Rates = Rates(neck=1e-4, trunk=5e-5),
        freeze=apply_freeze, stage: str = "backbone",
        want_teacher: bool = True, jitter: int = 0, grad_clip: float = 1.0,
        ema_decay: float = 0.999, workers: int = 8, depth: int = 2,
        seed: int = 0, device: str = "cuda", progress=None, log=print,
        meta: dict | None = None) -> dict:
    """Optimise `model` against whatever `step(batch)` returns, and save the EMA.

    One loop for both training arms, because the only thing that differs
    between them is the loss: `distil` compares the student against a frozen
    teacher's map, `automask` against the model's own map on the unmasked
    frame. Everything the two share -- the freeze, the parameter groups, the
    one-cycle schedule, the EMA, the checkpoint format -- is here exactly once,
    which is the only reason two arms' results can be set beside each other.

    `extra` is for parameters the model does not own: `distil`'s projector, and
    nothing else so far. They get their own learning rate and no weight decay,
    and they are not saved -- the checkpoint is an ordinary EdgeTAM state dict.

    Deliberately not `schedule.run_stages`. There is one stage and not two,
    there is no validation set to select an epoch on, and the objective is a
    proxy for the thing that matters: what decides whether stage A paid for
    itself is stage B run from this checkpoint against stage B run without it.
    Reusing the two-stage runner would hide all three facts.
    """
    from .finetune import summarise_freeze

    counts = freeze(model, stage)
    log(summarise_freeze(counts, model))

    extra = list(extra)
    groups = param_groups(model, rates)
    if extra:
        groups = groups + [{"params": extra, "lr": extra_lr,
                            "weight_decay": 0.0, "name": "extra"}]
    opt = torch.optim.AdamW(groups)
    trainable = [p for g in opt.param_groups for p in g["params"]]
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in opt.param_groups],
        total_steps=max(epochs * steps_per_epoch, 1), pct_start=0.1)
    ema = EMA(model, decay=ema_decay)

    history = []
    for epoch in range(epochs):
        batches = stream(items, batch, size, seed=seed + 100 * epoch,
                         limit=steps_per_epoch, device=device, workers=workers,
                         depth=depth, want_teacher=want_teacher, jitter=jitter)
        if progress is not None:
            batches = progress(batches, total=steps_per_epoch,
                               desc=f"stage A e{epoch}")

        losses, tracked = [], {}
        for chunk in batches:
            loss, terms = step(chunk)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            opt.step()
            sched.step()
            ema.update(model)
            losses.append(float(loss.detach()))
            for name, value in terms.items():
                tracked.setdefault(name, []).append(float(value))
            if hasattr(batches, "set_postfix"):
                batches.set_postfix(**{k: f"{v:.4f}" for k, v in terms.items()})

        mean = float(np.mean(losses)) if losses else float("nan")
        row = {"epoch": epoch, "loss": mean,
               **{k: float(np.mean(v)) for k, v in tracked.items()}}
        history.append(row)
        log(f"  epoch {epoch}: " + "  ".join(
            f"{k} {v:.4f}" for k, v in row.items() if k != "epoch"))

    with ema.applied(model):
        save(model, {**(meta or {}), "epochs": epochs,
                     "steps": epochs * steps_per_epoch,
                     "samples": len(items),
                     "final_loss": history[-1]["loss"] if history else float("nan")})

    del opt, sched, ema
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {"history": history, "samples": len(items),
            "steps": epochs * steps_per_epoch,
            "final_loss": history[-1]["loss"] if history else float("nan")}


def distil(model, items: Sequence[Item], teacher, *, save, size: int = 512,
           loss: str = "cosine", beta: float = 2.0, whiten: bool = True,
           cutoff: float = 0.5, high_weight: float = 0.1,
           gram_weight: float = 0.0, gram_positions: int = 256,
           layers: int = 1, moment_weight: float = 0.0, tolerance: int = 0,
           projector_lr: float = 1e-3, device: str = "cuda", log=print,
           **kwargs) -> dict:
    """The teacher arm: reproduce a frozen model's dense map from the other half.

    Thin over `run`, and thin on purpose -- the loss, the projector and the
    teacher wrapper are all `distill.py`'s, unchanged, because 07-11 already
    depend on them and a second copy of a distillation loss is a second thing
    to keep correct. What this adds is the index: `distill.pretrain` reads
    registered pairs and only registered pairs, and this reads whatever `index`
    produced, including luminance-routed RGB corpora and thermal-only ones.
    """
    usable = with_teacher(items, log=log)
    student_dim = encoder_features(
        model, torch.zeros(1, 3, size, size, device=device)).shape[1]
    projector = Projector(student_dim, teacher.dim).to(device)
    log(f"projector {student_dim} -> {teacher.dim} "
        f"({sum(p.numel() for p in projector.parameters()) / 1e6:.2f} M, "
        f"discarded after)")

    if loss not in LOSSES:
        raise ValueError(f"loss must be one of {LOSSES}, got {loss!r}")
    if loss == "feature" and tolerance:
        # The neighbourhood max is defined on cosine similarity and has no
        # smooth-L1 counterpart; silently ignoring it would leave a run
        # believing it had absorbed a misalignment it had not.
        raise ValueError(
            "tolerance is a cosine-loss setting and does not apply to the "
            "smooth-L1 feature loss. Use loss='cosine' for unregistered pairs.")

    def score(left, right):
        if loss == "feature":
            return feature_loss(left, right, beta=beta, whiten=whiten)
        if loss == "frequency":
            return frequency_loss(left, right, cutoff=cutoff,
                                  high_weight=high_weight)
        return distill_loss(left, right, moment_weight, tolerance)

    def step(chunk: Batch):
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=device.startswith("cuda")):
            targets = teacher_maps(teacher, chunk.teacher, layers)
            raw = encoder_features(model, chunk.student)
            student = projector(raw)
        scored = [score(student, target) for target in targets]
        # The *same* projected student against each teacher block, averaged --
        # a constraint rather than extra capacity. Averaging and not summing so
        # that layers=2 does not silently double the learning rate.
        total = sum(value for value, _ in scored) / len(scored)
        terms = {k: float(np.mean([t[k] for _, t in scored]))
                 for k in scored[0][1]}
        if gram_weight:
            # On the *raw* student, deliberately: a relational term that went
            # through the projector would be constraining the scaffolding, and
            # a Gram matrix needs no channel alignment to begin with.
            relation = sum(gram_loss(raw, target, gram_positions)
                           for target in targets) / len(targets)
            total = total + gram_weight * relation
            terms["gram"] = float(relation.detach())
        return total, terms

    result = run(model, usable, step, save=save,
                 extra=projector.parameters(), extra_lr=projector_lr,
                 size=size, want_teacher=True, device=device, log=log,
                 meta={"arm": "distil", "teacher": teacher.model_id,
                       "teacher_dim": teacher.dim, "teacher_size": teacher.size,
                       "loss": loss, "beta": beta, "whiten": whiten,
                       "layers": layers, "gram_weight": gram_weight,
                       "cutoff": cutoff, "high_weight": high_weight,
                       "tolerance": tolerance, "moments": moment_weight},
                 **kwargs)
    del projector
    return result | {"arm": "distil", "teacher": teacher.model_id,
                     "loss": loss}


# --------------------------------------------------------------------------
# The handoff to stage B
# --------------------------------------------------------------------------


def baseline(base: str | Path, out: str | Path, meta: dict | None = None) -> Path:
    """The `none` arm: the stock weights, written where every other arm writes.

    A baseline reached by a different code path is a baseline that drifts --
    stage B loading a stock checkpoint from one place and a pretrained one from
    another will eventually differ in something nobody meant to change. So
    `none` produces a file of the same shape in the same folder with the same
    manifest beside it, and the only difference is that nothing was trained.
    """
    import shutil

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(base), out)
    write_manifest(out, {"arm": "none", "base": str(base), "steps": 0,
                         "samples": 0, **(meta or {})})
    return out


def write_manifest(checkpoint: str | Path, report: dict) -> Path:
    """`stage_a.json` beside the checkpoint: what produced it, in one file.

    Stage B takes a path and asks no questions -- `--base` is a filename -- so
    without this the record of which arm, which corpora and which seed produced
    a checkpoint lives in a Colab cell that scrolls away. The manifest travels
    with the file into Drive and is what a stage-B run log can name six weeks
    later.
    """
    path = Path(checkpoint).with_suffix(".stage_a.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"checkpoint": Path(checkpoint).name, **report}
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return path


def read_manifest(checkpoint: str | Path) -> dict:
    """The manifest beside a checkpoint, or `{}` if it has none."""
    path = Path(checkpoint).with_suffix(".stage_a.json")
    return json.loads(path.read_text()) if path.is_file() else {}


def stage_b_command(checkpoint: str | Path, out: str | Path = "checkpoints/edgetam_stage_b.pt") -> str:
    """The exact stage-B invocation for this checkpoint, ready to paste.

    Printed at the end of every arm. The handoff is one flag and it is still
    worth spelling out: the failure it prevents is a stage-B run that silently
    started from the stock weights and was written up as if it had not.
    """
    return (f"python tools/train_encoder.py --base {checkpoint} --out {out} \\\n"
            f"    --dataset <spec>:<root> ...        # stage B's own settings")
