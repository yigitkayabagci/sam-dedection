"""SAMURAI: choose the mask that moves right, and refuse to remember bad frames.

The failure this exists to fix is specific and was observed on a real recording.
EdgeTAM emits an `object_score_logits` per frame; when it drops below zero the
mask is zeroed and a constant "no object" vector (`no_obj_ptr`) is written into
the memory bank instead of a real appearance. The next seven frames read that
back. **One hard frame -- a contrast shift, motion blur, a bird crossing --
poisons the memory, and nothing in the architecture ever un-poisons it.**

SAMURAI (Yang et al., 2024) addresses exactly this, and it is *training-free*.
Two changes, both to decisions EdgeTAM already makes:

**1. Which of the three candidate masks to keep.** Stock SAM 2 takes the one
its IoU head rates highest -- an appearance judgement with no memory of where
the object was going. SAMURAI adds a Kalman filter over the box and scores each
candidate by how well it agrees with the predicted motion:

    M* = argmax  a * s_kf(M) + (1 - a) * s_mask(M)

**2. Which frames enter the memory bank.** Stock SAM 2 stores the last N
frames, unconditionally, whatever happened on them. SAMURAI stores a frame only
if the mask, the object score and the motion all agree it was a good one. This
is the half that matters here: **a poisoned frame is never written.**

Why this is nearly free on the accelerated path
-----------------------------------------------
The memory-bank bookkeeping is the part this project deliberately left in
PyTorch -- "which past frames to attend to, temporal position encodings, object
pointer collection" (`docs/tensorrt_fp16.md`). That is precisely what SAMURAI
modifies. No engine changes shape: filtering changes *which* slots are live,
and the padded-slot design already accepts anything from 0 to `num_maskmem`.
The Kalman filter is an 8-state update on numpy, microseconds per frame.

Integration is by patching methods on the built predictor, the same idiom
`edgetam_trt_tracker.py` uses and for the same reason: `track_step` is one long
method and two of its leaves need replacing, not a fork of the class.
"""
from __future__ import annotations

from dataclasses import dataclass

from collections import deque

import numpy as np


# --------------------------------------------------------------------------
# Motion model
# --------------------------------------------------------------------------


class KalmanBoxTracker:
    """Constant-velocity Kalman filter over `(x, y, aspect, height)`.

    Centre, aspect ratio and height rather than corners: a drone's apparent
    size changes smoothly with range while its aspect stays roughly constant,
    so the velocity terms stay small and the filter is not fighting the
    parameterisation. Process and measurement noise scale with height, which is
    the standard trick for making one set of constants work across a target
    that spans 5 to 200 pixels -- and Anti-UAV410 spans exactly that.
    """

    def __init__(self, position_weight: float = 1 / 20, velocity_weight: float = 1 / 160):
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
        self.mean: np.ndarray | None = None
        self.covariance: np.ndarray | None = None

        self._motion = np.eye(8)
        for i in range(4):
            self._motion[i, i + 4] = 1.0   # position += velocity
        self._observation = np.eye(4, 8)

    # -- noise ------------------------------------------------------------

    def _std(self, height: float, initial: bool = False) -> np.ndarray:
        p, v = self.position_weight, self.velocity_weight
        scale = 2.0 if initial else 1.0
        return np.array([
            scale * p * height, scale * p * height, 1e-2, scale * p * height,
            (10 if initial else 1) * v * height, (10 if initial else 1) * v * height,
            1e-5, (10 if initial else 1) * v * height,
        ])

    # -- interface --------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self.mean is not None

    def reset(self) -> None:
        self.mean = self.covariance = None

    def initiate(self, box: np.ndarray) -> None:
        measurement = xyxy_to_xyah(box)
        self.mean = np.r_[measurement, np.zeros(4)]
        self.covariance = np.diag(np.square(self._std(measurement[3], initial=True)))

    def predict(self, shift: tuple[float, float] | None = None) -> np.ndarray | None:
        """Advance one frame and return the predicted box, or None if unstarted.

        `shift` is the **camera's** own displacement between the two frames, in
        the same pixels the boxes are in, and it is the one thing published
        SAMURAI has no term for. Its filter lives in image coordinates, so on a
        drone -- where a degree of yaw translates the whole scene -- a target
        that never moved appears to accelerate, the constant-velocity model
        learns that acceleration, and the frame the camera stops it predicts a
        jump. Feeding the background's measured motion in as a control input
        puts the prediction back in the frame the ground sits still in, which
        is where a constant-velocity assumption is true.

        The estimate has its own error, so it also widens the position
        covariance a little rather than being trusted as exact -- a control
        input measured from pixels is a measurement, not an odometer.
        """
        if self.mean is None:
            return None
        noise = np.diag(np.square(self._std(self.mean[3])))
        self.mean = self._motion @ self.mean
        self.covariance = self._motion @ self.covariance @ self._motion.T + noise
        if shift is not None:
            dx, dy = float(shift[0]), float(shift[1])
            self.mean[0] += dx
            self.mean[1] += dy
            slack = (0.15 * (abs(dx) + abs(dy))) ** 2
            self.covariance[0, 0] += slack
            self.covariance[1, 1] += slack
        return self.predicted_box()

    def update(self, box: np.ndarray) -> None:
        if self.mean is None:
            self.initiate(box)
            return
        measurement = xyxy_to_xyah(box)
        std = self._std(self.mean[3])[:4]
        std[2] = 1e-1
        noise = np.diag(np.square(std))

        projected_mean = self._observation @ self.mean
        projected_cov = self._observation @ self.covariance @ self._observation.T + noise
        gain = np.linalg.solve(projected_cov.T,
                               (self.covariance @ self._observation.T).T).T
        innovation = measurement - projected_mean
        self.mean = self.mean + gain @ innovation
        self.covariance = self.covariance - gain @ projected_cov @ gain.T

    def predicted_box(self) -> np.ndarray | None:
        return None if self.mean is None else xyah_to_xyxy(self.mean[:4])


def xyxy_to_xyah(box: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = (float(v) for v in box)
    width, height = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    return np.array([x0 + width / 2, y0 + height / 2, width / height, height])


def xyah_to_xyxy(state: np.ndarray) -> np.ndarray:
    cx, cy, aspect, height = (float(v) for v in state)
    width = aspect * height
    return np.array([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2])


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SamuraiConfig:
    """SAMURAI's published defaults, with what each one guards.

    `kf_weight`         how much the motion model gets to overrule the IoU
                        head. Small on purpose: appearance is usually right,
                        and the motion score is there to break ties and to
                        veto a candidate that jumped across the frame.
    `stable_frames`     how long the filter has to agree with the tracker
                        before it is trusted. A Kalman filter initialised on
                        one box has no velocity estimate; using it immediately
                        would be worse than not using it.
    `stable_iou`        agreement required during that warm-up.
    `memory_iou`,       the memory gate. A frame enters the bank only if all
    `memory_obj_score`, three agree it was a good one. This is the half that
    `memory_kf_score`   stops one bad frame poisoning the next seven.
    """

    enabled: bool = True
    kf_weight: float = 0.15
    stable_frames: int = 15
    stable_iou: float = 0.3
    memory_iou: float = 0.5
    memory_obj_score: float = 0.0
    memory_kf_score: float = 0.0

    # `kf_gate` closes the hole the three thresholds above cannot: they decide
    # which frames enter the memory bank, and they work -- but only once. The
    # filter is updated with whatever box was chosen, so a mask that jumped to
    # a look-alike drags the filter onto it, and within two frames the filter
    # agrees with the distractor and the memory gate falls silent. Measured on
    # a 26-pixel target jumping 90 pixels: `kf_score` is 0.000 on the jump and
    # rejected, 0.045 on the next frame and *accepted*, 0.964 two frames later.
    # Fifteen of the sixteen frames after the jump entered the bank.
    #
    # This is the classical answer -- gating, as SORT and DeepSORT do it: a
    # filter is not updated with a measurement it does not believe. Below this
    # IoU against its own prediction the update is skipped and the filter
    # coasts. `kf_gate_patience` is the other half and it is not optional: a
    # filter that never updates would lock out a target that really did
    # manoeuvre, so after this many consecutive refusals the filter is
    # re-initiated on the tracker's box -- the tracker is then the one more
    # likely to be right.
    #
    # Swept on both cases at once. At 0.10 with patience 8, a jump puts 8 of 16
    # frames into the bank instead of 15, and a genuine acceleration still puts
    # in all 16 -- no cost at all on the honest case. Above 0.20 the manoeuvre
    # starts paying. The geometry is why: on a 26-pixel box a 90-pixel jump
    # scores 0 and a 13-pixel change of velocity scores about 0.33, so anything
    # between separates them and 0.10 sits inside with margin.
    #
    # **0.0 is off**, and off is the default: this changes what the filter
    # learns, so it is a thing to measure against `samurai` rather than to
    # assume.
    kf_gate: float = 0.0
    kf_gate_patience: int = 8

    # A mask that stays put and *swells* passes every check above: its IoU
    # against the prediction stays high because the prediction is inside it.
    # Measured on a box growing in place -- x2 scores 0.25 against the
    # prediction, x3 scores 0.111, and every one of the ten frames after the
    # growth entered the memory bank at all three sizes. That is the balloon,
    # written into memory as the target's appearance.
    #
    # This is the ratio to the running median of the areas already accepted, so
    # it follows a target that really is approaching. 0 is off. 3.0 is the same
    # number `stabiliser.GuardConfig.max_area_ratio` uses, and for the same
    # reason: a target's area between two frames at video rate cannot triple.
    memory_area_ratio: float = 0.0

    @property
    def permissive(self) -> bool:
        """True when nothing can be rejected, so behaviour is stock SAM 2.

        Worth being able to state exactly: it is what the no-op regression test
        asserts against, and it is the configuration to fall back to when
        something looks wrong.
        """
        return (self.kf_weight == 0.0 and self.memory_iou <= 0.0
                and self.memory_obj_score <= float("-inf")
                and self.memory_kf_score <= 0.0 and self.kf_gate <= 0.0
                and self.memory_area_ratio <= 0.0)


def boxes_from_masks(masks: np.ndarray) -> np.ndarray:
    """Tight boxes around each of a stack of mask logits (positive = object)."""
    from ..accuracy import box_from_mask

    return np.stack([box_from_mask(mask > 0) for mask in masks])


def mask_boxes(mask_logits):
    """`[B, M, H, W]` mask logits -> `[B, M, 4]` xyxy boxes, on the same device.

    Done in torch rather than by moving the masks to the CPU first. The motion
    model needs numbers on the host either way, so a synchronisation is
    unavoidable -- but the difference between copying `[B, 3, 128, 128]` and
    `[B, 3, 4]` across the bus every frame is about 98 KB against 48 bytes, and
    this project removed a per-frame stall once already
    (`docs/tensorrt_fp16.md`, "No per-frame cudaStreamSynchronize").

    Empty candidates come back as NaN, the same way an absent box is spelled
    everywhere else here.
    """
    import torch

    positive = mask_logits > 0
    rows, columns = positive.any(-1), positive.any(-2)

    def extent(flags):
        length = flags.shape[-1]
        index = torch.arange(length, device=flags.device)
        first = torch.where(flags, index, length).amin(-1).float()
        last = torch.where(flags, index, -1).amax(-1).float() + 1
        return first, last

    y0, y1 = extent(rows)
    x0, x1 = extent(columns)
    boxes = torch.stack((x0, y0, x1, y1), dim=-1)
    return boxes.masked_fill(~positive.flatten(2).any(-1)[..., None], float("nan"))


def box_iou_against(candidates: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
    """IoU of each candidate box against one reference. All zeros if there is none."""
    from ..accuracy import box_iou

    if reference is None or np.isnan(reference).any():
        return np.zeros(len(candidates))
    usable = ~np.isnan(candidates).any(axis=1)
    scores = np.zeros(len(candidates))
    if usable.any():
        scores[usable] = box_iou(candidates[usable],
                                 np.repeat(reference[None], usable.sum(), axis=0))
    return scores


@dataclass(frozen=True)
class Selection:
    """Which candidate mask won this frame, and by what.

    `weighted` is the full score vector, so the caller can hand it back to
    upstream in place of the raw IoU predictions -- which makes SAM 2's own
    `argmax` pick this candidate, and everything downstream of that choice
    (the mask, the object pointer, the memory write) stays consistent without
    any of it being recomputed here.
    """

    index: int
    kf_score: float
    mask_score: float
    weighted: np.ndarray
    area_ratio: float = 1.0


@dataclass
class FrameRecord:
    """What the memory gate needs to know about one already-tracked frame."""

    iou: float
    object_score: float
    kf_score: float
    # Area against the running median of the frames already accepted. 1.0 when
    # there is no history to compare against, which is the neutral value.
    area_ratio: float = 1.0

    def acceptable(self, cfg: SamuraiConfig) -> bool:
        return (self.iou > cfg.memory_iou
                and self.object_score > cfg.memory_obj_score
                and self.kf_score > cfg.memory_kf_score
                and (cfg.memory_area_ratio <= 0.0
                     or self.area_ratio <= cfg.memory_area_ratio))


class MotionAwareMemory:
    """SAMURAI's state: the filter, the warm-up counter, the per-frame record.

    Held separately from the model so it can be tested, inspected and reset
    without one, and so a tracker can hold one per object.
    """

    def __init__(self, config: SamuraiConfig | None = None) -> None:
        self.config = config or SamuraiConfig()
        self.kalman = KalmanBoxTracker()
        self.stable = 0
        # Consecutive frames the filter has refused to learn from. Only ever
        # non-zero when `kf_gate` is on.
        self.coasted = 0
        # Areas of the frames already accepted, for `memory_area_ratio`.
        self.areas: deque = deque(maxlen=12)
        self.records: dict[int, FrameRecord] = {}

    def reset(self) -> None:
        self.kalman.reset()
        self.stable = 0
        self.coasted = 0
        self.areas.clear()
        self.records.clear()

    # -- mask selection ---------------------------------------------------

    def select(self, boxes: np.ndarray, ious: np.ndarray,
               shift: tuple[float, float] | None = None) -> "Selection":
        """Pick one of the candidate boxes for this frame, and score the choice.

        Takes boxes rather than masks so the policy is independent of how the
        mask was represented -- and so the caller can extract them wherever it
        is cheapest, which on the accelerated path is on the GPU.

        During warm-up the appearance score alone decides and the filter merely
        watches, because a filter initialised on a single box has no velocity
        estimate yet -- trusting it early is worse than not using it.
        """
        cfg = self.config
        appearance = np.asarray(ious, dtype=np.float64).reshape(-1)
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)

        predicted = self.kalman.predict(shift) if self.kalman.ready else None
        kf_scores = box_iou_against(boxes, predicted)

        warming = self.stable < cfg.stable_frames
        if warming or cfg.kf_weight == 0.0:
            weighted = appearance
        else:
            weighted = cfg.kf_weight * kf_scores + (1 - cfg.kf_weight) * appearance
        chosen = int(np.argmax(weighted))

        box = boxes[chosen]
        if not np.isnan(box).any():
            if not self.kalman.ready:
                self.kalman.initiate(box)
                self.stable = 1
                self.coasted = 0
            elif warming:
                # Count a warm-up frame only when the filter and the tracker
                # already agree; a filter fed its own disagreements would
                # converge on the wrong track and then be trusted.
                if appearance[chosen] > cfg.stable_iou:
                    self.kalman.update(box)
                    self.stable += 1
                else:
                    self.stable = 0
            elif cfg.kf_gate <= 0.0 or kf_scores[chosen] >= cfg.kf_gate:
                self.kalman.update(box)
                self.coasted = 0
            else:
                # The filter does not believe this box. Coast on the prediction
                # rather than learning from it -- and give up coasting after
                # `kf_gate_patience`, because a filter that disagrees for that
                # long is the one that is wrong.
                self.coasted += 1
                if self.coasted >= cfg.kf_gate_patience:
                    self.kalman.reset()
                    self.kalman.initiate(box)
                    self.stable = 1
                    self.coasted = 0
        area = float(max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0))
        typical = float(np.median(self.areas)) if self.areas else 0.0
        ratio = area / typical if typical > 0 and area > 0 else 1.0
        if area > 0 and (cfg.memory_area_ratio <= 0.0
                         or ratio <= cfg.memory_area_ratio):
            # Only an accepted area joins the history, or one balloon frame
            # raises the median and the next one looks reasonable beside it.
            self.areas.append(area)
        return Selection(index=chosen, kf_score=float(kf_scores[chosen]),
                         mask_score=float(appearance[chosen]),
                         weighted=weighted.astype(np.float32),
                         area_ratio=float(ratio))

    # -- memory gate ------------------------------------------------------

    def record(self, frame_idx: int, iou: float, object_score: float,
               kf_score: float, area_ratio: float = 1.0) -> None:
        self.records[int(frame_idx)] = FrameRecord(float(iou), float(object_score),
                                                   float(kf_score),
                                                   float(area_ratio))

    def keep(self, frame_idx: int) -> bool:
        """Whether this frame is fit to enter the memory bank.

        Unrecorded frames are kept. That is the safe default and the honest
        one: a frame this policy never saw (a prompted frame, or one tracked
        before SAMURAI was installed) has not failed anything.
        """
        record = self.records.get(int(frame_idx))
        return True if record is None else record.acceptable(self.config)


# --------------------------------------------------------------------------
# Installation
# --------------------------------------------------------------------------


class Samurai:
    """The policy, per tracked object, plus the two patches that apply it.

    One `MotionAwareMemory` per batch row. SAM 2 batches objects, and each row
    has its own trajectory, so they cannot share a filter. The *memory gate* is
    necessarily shared, because the bank stores one entry per frame for every
    row at once -- so a frame is kept only if it was good for **every** object.
    That is the conservative direction (never store a frame where anything was
    lost) and it is the honest limitation of applying a single-object method to
    a multi-object bank.
    """

    def __init__(self, config: SamuraiConfig | None = None) -> None:
        self.config = config or SamuraiConfig()
        self.states: list[MotionAwareMemory] = []
        self.frame_idx = 0
        self.rejected: set[int] = set()
        # The camera's own displacement for the frame about to be scored, as a
        # fraction of the frame's width and height. Normalised on purpose: the
        # caller measures it on the pixels it has (a decoded frame, at whatever
        # resolution it was stored) and the boxes this scores live in the mask
        # logits' own grid, which is `image_size // 4` and a different number at
        # 512, 768 and 1024. A fraction is the one form that needs no agreement
        # between the two about resolution.
        self.shift: tuple[float, float] | None = None

    def state(self, row: int) -> MotionAwareMemory:
        while len(self.states) <= row:
            self.states.append(MotionAwareMemory(self.config))
        return self.states[row]

    def reset(self) -> None:
        for state in self.states:
            state.reset()
        self.states.clear()
        self.frame_idx = 0
        self.rejected.clear()
        self.shift = None

    def _shift_for(self, mask_logits) -> tuple[float, float] | None:
        """This frame's camera motion, in the mask grid's own pixels."""
        if self.shift is None:
            return None
        height, width = mask_logits.shape[-2:]
        return (float(self.shift[0]) * float(width),
                float(self.shift[1]) * float(height))

    def keep(self, frame_idx: int) -> bool:
        return all(state.keep(frame_idx) for state in self.states)

    def acceptable(self, candidates, limit: int) -> list[int]:
        """The most recent `limit` acceptable frames, oldest first.

        Searching backwards and stopping at `limit` is what keeps this cheap:
        at steady state the newest frames are almost always acceptable, so it
        looks at `limit` of them. It walks further only after a bad patch,
        which is exactly when it should.
        """
        kept: list[int] = []
        for frame_idx in sorted(candidates, reverse=True):
            if self.keep(frame_idx):
                kept.append(frame_idx)
                if len(kept) >= limit:
                    break
        return list(reversed(kept))

    # -- the mask choice --------------------------------------------------

    def rescore(self, mask_logits, iou_pred, object_score_logits):
        """Replace the IoU predictions with motion-aware scores, in place.

        Rewriting the scores rather than re-deriving the outputs is what keeps
        this small: SAM 2 selects its mask, its object pointer and what it
        writes to memory from a single `argmax` over these values, so changing
        them moves all three together and nothing has to be recomputed or kept
        consistent by hand.
        """
        import torch

        # **One** synchronisation per frame, and one is the number that matters:
        # every `.cpu()` on a CUDA tensor blocks until the stream drains, so
        # three of them is three stalls whatever the bytes moved. The three
        # small tensors are concatenated on the device and split on the host --
        # 3 boxes, 3 IoUs and an object score per row, tens of bytes, one stop.
        boxes_gpu = mask_boxes(mask_logits.detach()).float()
        ious_gpu = iou_pred.detach().float()
        scores_gpu = object_score_logits.detach().float().reshape(-1)
        flat = torch.cat((boxes_gpu.reshape(-1), ious_gpu.reshape(-1),
                          scores_gpu)).cpu().numpy()
        cut = boxes_gpu.numel()
        boxes = flat[:cut].reshape(boxes_gpu.shape)
        ious = flat[cut:cut + ious_gpu.numel()].reshape(ious_gpu.shape)
        scores = flat[cut + ious_gpu.numel():]

        out = iou_pred.clone()
        for row in range(boxes.shape[0]):
            state = self.state(row)
            selection = state.select(boxes[row], ious[row],
                                     self._shift_for(mask_logits))
            out[row] = torch.as_tensor(selection.weighted, dtype=out.dtype,
                                       device=out.device)
            state.record(self.frame_idx, selection.mask_score, float(scores[row]),
                         selection.kf_score, selection.area_ratio)
            if not state.keep(self.frame_idx):
                self.rejected.add(self.frame_idx)
        # A single-candidate frame -- the prompted one, in configurations that
        # do not use multimask there -- has nothing to choose between, but the
        # filter still had to see it. Returning the scores unchanged keeps this
        # a pure observation of that frame.
        return iou_pred if mask_logits.shape[1] < 2 else out


def _filtered_memory(output_dict: dict, samurai: Samurai, frame_idx: int,
                     num_maskmem: int, stride: int) -> dict:
    """`output_dict` with the unfit frames removed from the memory candidates.

    Two behaviours, and the difference is worth understanding:

    * **stride 1** (SAM 2's default, and EdgeTAM's): the chosen frames are
      *renumbered* onto the indices upstream is about to look up, so a rejected
      frame is replaced by an older good one and the bank stays full. Upstream's
      own selection logic runs untouched -- none of it is duplicated here.
    * **any other stride**, or reverse tracking: fall back to plain filtering.
      Bad frames still never enter the bank; they are simply not backfilled,
      which leaves the bank shorter rather than wrong.

    A backfilled frame takes the temporal position encoding of the slot it
    lands in rather than its true age. That is SAMURAI's behaviour too: the
    encoding marks recency of *use*, and a frame that was skipped was not used.
    """
    entries = output_dict.get("non_cond_frame_outputs", {})
    limit = max(num_maskmem - 1, 0)
    past = [f for f in entries if f < frame_idx]
    # The common case by far: the frames upstream is about to pick are all
    # acceptable, so there is nothing to change and nothing to copy. Checking
    # only those `limit` frames is also what keeps this cheap at steady state.
    recent = sorted(past, reverse=True)[:limit]
    if stride == 1 and all(samurai.keep(f) for f in recent):
        return output_dict

    chosen = samurai.acceptable(past, limit)
    if stride == 1:
        wanted = list(range(frame_idx - limit, frame_idx))[-len(chosen):] if chosen else []
        remapped = {w: entries[c] for w, c in zip(wanted, chosen)}
    else:
        remapped = {c: entries[c] for c in chosen}
    return {**output_dict, "non_cond_frame_outputs": remapped}


def install(predictor, config: SamuraiConfig | None = None,
            shift_for=None) -> Samurai:
    """Patch a built predictor. Returns the state, for inspection and reset.

    Patches two leaves rather than subclassing: `track_step` is one long method
    and only two of the decisions inside it change. The same idiom, and the
    same reasoning, as `edgetam_trt_tracker.py`.

    `shift_for(frame_idx)` is optional and returns the camera's own
    displacement between the previous frame and this one, as a fraction of the
    frame's width and height (or None where it could not be measured). It is
    asked for inside the memory hook, which is the one place that runs *before*
    the SAM head on every frame and knows which frame is about to be scored --
    so the control input reaches the filter for the frame it belongs to rather
    than the one after it.
    """
    samurai = Samurai(config)
    if not samurai.config.enabled:
        return samurai

    decoder = predictor.sam_mask_decoder
    decode = decoder.forward
    prepare = predictor._prepare_memory_conditioned_features
    num_maskmem = int(getattr(predictor, "num_maskmem", 7))
    stride = int(getattr(predictor, "memory_temporal_stride_for_eval", 1))

    def wrapped_decoder(*args, **kwargs):
        masks, iou_pred, tokens, object_score_logits = decode(*args, **kwargs)
        return (masks, samurai.rescore(masks, iou_pred, object_score_logits),
                tokens, object_score_logits)

    def wrapped_prepare(frame_idx, is_init_cond_frame, current_vision_feats,
                        current_vision_pos_embeds, feat_sizes, output_dict,
                        num_frames, track_in_reverse=False):
        # Runs before the SAM head on every frame, which makes it the place the
        # frame number is known; the head wrapper reads it back when recording.
        samurai.frame_idx = int(frame_idx)
        samurai.shift = (shift_for(int(frame_idx)) if shift_for is not None
                         and not track_in_reverse else None)
        if not is_init_cond_frame and not track_in_reverse:
            output_dict = _filtered_memory(output_dict, samurai, int(frame_idx),
                                           num_maskmem, stride)
        return prepare(frame_idx, is_init_cond_frame, current_vision_feats,
                       current_vision_pos_embeds, feat_sizes, output_dict,
                       num_frames, track_in_reverse)

    decoder.forward = wrapped_decoder
    predictor._prepare_memory_conditioned_features = wrapped_prepare
    cfg = samurai.config
    print(f"[samurai] motion-aware memory on: kf_weight={cfg.kf_weight}, "
          f"warm-up {cfg.stable_frames} frames, memory gate "
          f"iou>{cfg.memory_iou} obj>{cfg.memory_obj_score} kf>{cfg.memory_kf_score}"
          f"{', ego-motion compensated' if shift_for is not None else ''}")
    return samurai
