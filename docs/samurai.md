# SAMURAI: motion-aware memory, for about zero milliseconds

## The failure

EdgeTAM emits an `object_score_logits` on every tracked frame. When it drops
below zero (`third_party/EdgeTAM/sam2/modeling/sam2_base.py`) two things happen:
the frame's mask is zeroed, and a **constant "no object" vector** (`no_obj_ptr`)
is written into the memory bank in place of a real appearance.

The problem is not the zeroed mask — that frame was bad anyway. The problem is
the memory. The next `num_maskmem` frames read `no_obj_ptr` back and condition
on it. **One hard frame — a contrast shift, motion blur, a bird crossing the
sensor — poisons the memory, and nothing in the architecture ever un-poisons
it.** Observed on a real recording.

That is not a precision problem, so no amount of fp16, INT8 or fine-tuning
addresses it. It is a *policy* problem: SAM 2 stores the last N frames
unconditionally, whatever happened on them.

## What SAMURAI changes

Two decisions EdgeTAM already makes, both training-free
([Yang et al., 2024](https://arxiv.org/abs/2411.11922)):

**1. Which of the three candidate masks to keep.** Stock SAM 2 takes the one
its IoU head rates highest — an appearance judgement with no memory of where
the object was going. SAMURAI adds a Kalman filter over the box and scores each
candidate on agreement with the predicted motion:

```
M* = argmax  a * s_kf(M) + (1 - a) * s_mask(M)          a = 0.15
```

`a` is small on purpose. Appearance is usually right; the motion term is there
to break ties and to veto a candidate that has jumped across the frame.

**2. Which frames enter the memory bank.** A frame is stored only if the mask
quality, the object score and the motion all agree it was a good one:

```
iou > 0.5     and     object_score > 0.0     and     kf_score > 0.0
```

The middle threshold is exactly where EdgeTAM starts writing `no_obj_ptr`. This
is the half that matters here: **a poisoned frame is never written.**

## Why it is nearly free on the accelerated path

The memory-bank bookkeeping — "which past frames to attend to, temporal
position encodings, object pointer collection" — is the part this project
deliberately left in PyTorch (`docs/tensorrt_fp16.md`, *What stays in PyTorch*).
That is precisely what SAMURAI modifies.

* **No engine changes shape.** Filtering changes *which* slots are live, and the
  fixed-slot design already accepts anything from 0 to `num_maskmem` — the
  additive mask handles it.
* **The Kalman filter is 8 states on numpy.** Microseconds per frame, on the CPU,
  off the critical path.
* **Mask re-selection is a rescore.** SAM 2 derives the mask, the object pointer
  and the memory write from one `argmax` over the IoU predictions, so replacing
  those values moves all three together and nothing is recomputed.

The one measurable cost is on the TensorRT path, and only when SAMURAI's choice
differs from the engine's: one bilinear upsample of the chosen candidate,
128²→512², roughly 0.2 ms.

## How it is wired in

`src/trackers/samurai.py`, patching two leaves of the built predictor — the
same idiom and the same reasoning as `edgetam_trt_tracker.py`: `track_step` is
one long method and only two decisions inside it change.

| patch | what it does |
|---|---|
| `sam_mask_decoder.forward` | replaces the IoU predictions with motion-aware scores, so SAM 2's own `argmax` selects SAMURAI's candidate |
| `_prepare_memory_conditioned_features` | hands upstream an `output_dict` with the unfit frames removed |

The second one is worth a note. Rather than reimplementing SAM 2's memory
selection (40 lines of stride arithmetic that would then have to be kept in
step with upstream), the filtered frames are **renumbered onto the indices
upstream is about to look up**. Upstream's own logic runs untouched, and a
rejected frame is backfilled by an older good one instead of leaving a hole.
That renumbering assumes consecutive indices, which holds at
`memory_temporal_stride_for_eval == 1` (SAM 2's default); at any other stride
it falls back to plain filtering — bad frames still never enter the bank, they
are simply not backfilled, so the bank runs shorter rather than wrong.

### On the TensorRT path

The `sam_head` engine picks its candidate internally, so the decoder patch
never fires there. Instead `EdgeTAMTRTTracker._reselect` applies the same
policy at the engine's boundary — which needs one extra engine output:

```bash
python tools/export_edgetam_onnx.py --outdir models512/ --image-size 512 --all-pointers
```

`obj_ptr_all` is every candidate's object pointer, not just the winner's. The
pointer written to the memory bank has to describe the mask that was kept, and
the engine only returns the one it chose. Cost: one extra pass of a 256-wide
MLP over three tokens.

Without that output the tracker says so once and **switches off the mask
re-selection only** — the memory gate still applies, because that lives
entirely in Python. Existing engines therefore keep working and still get most
of the benefit.

## Turning it on, and off

```yaml
samurai:
  enabled: true
  kf_weight: 0.15        # how far motion may overrule appearance
  stable_frames: 15      # warm-up before the filter is trusted
  stable_iou: 0.3
  memory_iou: 0.5        # the gate
  memory_obj_score: 0.0
  memory_kf_score: 0.0
```

Omit the block, or set `enabled: false`, and the model is exactly upstream's.
`tests/test_edgetam_trt_integration.py::test_samurai_with_a_permissive_config_is_stock_edgetam`
pins that: thresholds that reject nothing and `kf_weight: 0` must reproduce
stock EdgeTAM pixel for pixel. Without that guarantee, "SAMURAI helped" and
"SAMURAI changed something unrelated" would be indistinguishable.

## Measuring it

Mean IoU is the wrong statistic. One 60-frame loss and sixty 1-frame losses
give the same mean and are completely different bugs — and memory poisoning is
firmly the first kind. `tools/eval_antiuav.py` reports **dropout episodes**:
how often a visible target was lost, and for how long each time.

```bash
python tools/eval_antiuav.py --data <anti-uav410> --split val --limit 20 \
    --tracker edgetam --config configs/edgetam_512.yaml         --json results/stock.json
python tools/eval_antiuav.py --data <anti-uav410> --split val --limit 20 \
    --tracker edgetam --config configs/edgetam_samurai_512.yaml --json results/samurai.json
```

If the memory gate is doing what it is supposed to, the episodes get
**shorter**, not just rarer — recovery is the thing being bought. A drop in
episode *count* with unchanged lengths would mean the mask re-selection helped
and the gate did not.

Anti-UAV410's `Occlusion` and `Out-of-View` attributes select the sequences
where this is decided; `notebooks/04_samurai_long_video.ipynb` sweeps the
thresholds over them.

## Status

Implemented and unit-tested (`tests/test_samurai.py`, 39 cases); **no on-device
measurement yet.** Nothing here is a result.

## SAM2Long, as a variant to measure against

SAM2Long ([Ding et al., 2024](https://arxiv.org/abs/2410.16268)) attacks the
same problem from the other side: instead of choosing better, it declines to
choose. It carries several candidate memory pathways and picks between them by
constrained tree search. Also training-free, better published long-video numbers.

**It is implemented** (`src/trackers/sam2long.py`,
`configs/edgetam_sam2long_512.yaml`) but **off by default**, because of what it
costs on this hardware. Memory attention is **61.9 % of the per-frame
arithmetic**, and N pathways means running it N times — three pathways puts a
measured ~10 ms frame near ~22 ms. Inside the 20–30 ms ceiling, but spending
most of it on one technique. SAMURAI costs approximately nothing.

So the recommendation is: **SAMURAI as the default, SAM2Long as the thing you
switch on to find out whether the extra 12 ms buys anything.** They compose —
SAMURAI chooses within a pathway and filters what it remembers, SAM2Long chooses
between pathways — and `configs/edgetam_sam2long_512.yaml` runs both.

The implementation needs no new inference loop. SAM 2 already gives every batch
row its own memory bank, object pointer and object score, so **N pathways are N
identically-prompted objects**: the tracker replicates the prompt, the rows
diverge (using the same IoU-rewrite lever SAMURAI uses), and the output
collapses to the best cumulative score. On TensorRT the only requirement is
`--max-batch >= pathways × objects`.

Divergence is deliberately rare — pathways split only when the top two
candidates are within `uncertainty_margin`, or when the object score has
dropped below `object_score_floor` (the case where `no_obj_ptr` is about to be
written and being wrong is most expensive). That is the paper's constrained
search, and it is what stops the hypotheses fanning out into noise.

**Not reproduced:** the paper also refines *which* memories a pathway keeps
using object-occurrence statistics. Each pathway here uses stock SAM 2 memory
selection, or SAMURAI's when that is also on.
