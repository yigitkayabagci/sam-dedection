# INT8 for EdgeTAM: which path, for which module, and why

`docs/tensorrt_fp16.md` ends with a measured INT8 *simulation* and an open
question: does a real, calibrated engine survive? This is the answer's
scaffolding — the tools, the plan, and the reasoning that produced it.

The short version: **INT8 damage is not uniform across EdgeTAM, and neither is
the right calibrator.** A single global setting is why the first attempt looked
impossible.

## The measurement everything follows from

From `tools/analyze_precision.py`, activations quantised per tensor, IoU
against the fp32 run over a 30-frame clip:

| module | share of FLOPs | absmax (TensorRT's default) | percentile-clipped |
|---|---:|---:|---:|
| memory attention | **61.9 %** | 0.9998 | 0.9998 |
| memory encoder | 8.8 % | **0.9999** | 0.9397 |
| SAM mask decoder | 2.5 % | 0.9993 | 0.9990 |
| image encoder | 26.8 % | **0.8170** | 0.9936 |

Two entries decide the whole strategy, and they point in opposite directions.

**The image encoder collapses under min/max and recovers under clipping.**
RepViT is a depthwise-separable stack whose per-channel activation ranges
differ by orders of magnitude, while TensorRT scales activations per tensor. A
handful of outliers stretch the scale until the bulk of the distribution has no
resolution left. The failure is not gradual: on 5 frames of 30 the tracker
loses the object outright. → **entropy calibration.**

**The memory encoder is the exact opposite: clipping hurts it, badly.** And the
*shape* of that failure names the mechanism. Rounding error scatters; clipping
error is systematic, always removing the same tail. The memory encoder's output
*is* the stored memory — it is the write port of the recurrent loop — so a
systematic bias goes into the bank and is read back for the next seven frames.
The measured curve is a sustained decline across the clip, not the isolated
dropouts every other module shows. → **max calibration, never clipped.**

**The memory attention is 62 % of the frame's arithmetic and barely notices INT8
either way.** That is the best ratio in the model, and it is why this is worth
doing at all. It only *reads* the memory bank, so it inherits none of the
accumulation problem the module that writes it has.

## What that gives

```bash
python tools/quantize_edgetam.py plan
```

| module | share of FLOPs | precision | why |
|---|---:|---|---|
| `image_encoder` | 26.8 % | INT8 / entropy | min/max collapses it to 0.8170; clipping recovers 0.9936 |
| `memory_attention` | 61.9 % | INT8 / max | most of the compute, least of the damage |
| `memory_encoder` | 8.8 % | INT8 / max | clipping bias accumulates through the loop |
| `sam_head` | 2.5 % | INT8 / max | 0.9993, and small enough that the choice barely matters |

Every engine is independent (`src/trackers/edgetam_trt_tracker.py`), so a
module that fails its accuracy gate drops back to fp16 **on its own** while the
others keep their speedup:

```bash
python tools/quantize_edgetam.py quantize --fp16 image_encoder
```

That still covers 73.2 % of the frame's arithmetic.

## Calibration data has to be real

The image encoder is easy — feed it thermal frames. The other three take
`pix_feat`, a padded `memory` bank and a sigmoid-scaled `mask_for_mem`. Nobody
can write those distributions down, and calibrating on noise sets scales for a
network that does not exist.

So calibration is **recorded from a real tracking run, at the engine boundary,
in the layout the engine actually receives** — padded memory slots, additive
mask and all (`src/trackers/calibration.py`). The recorder wraps anything
presenting the `TRTEngine` interface, which means it works against:

* `tests/reference_engines.py` — PyTorch, no TensorRT, no GPU. This is the
  default, and it has to be: calibration happens *before* an INT8 engine exists.
* real fp16 engines on the Orin, via `calibration_dir:` in the config, for a
  second opinion taken on the hardware that will run them.

Two details that are easy to get wrong and expensive to debug:

* **Samples are strided, not taken from the opening frames.** The memory bank
  fills over the first ~16 frames of a clip; calibrating on that transient
  tunes for a state the engine spends almost none of its time in.
* **A capture taken at the wrong resolution is refused, not used.** Calibrating
  a 512 graph on 1024 activations builds a perfectly valid engine that is
  simply wrong. `_calibration_arrays` checks every non-batch axis.

## PTQ, QAT, distillation: which is which here

| | what it is | used for |
|---|---|---|
| **PTQ** | calibrate scales on real data, no training | **the default and usually the end of it.** Three of four modules already clear 0.999 in simulation |
| **QAT** | fine-tune with fake quantisation in the graph, so the weights absorb rounding | **only for a module PTQ leaves short** — in practice the image encoder. ~10 % of the fine-tune schedule |
| **distillation** | match a teacher's outputs | **twice, for two different reasons** — see below |

Distillation appears at both ends of this pipeline and the two uses are
unrelated:

1. **SAM 2.1-Large → EdgeTAM, for labels** (`src/training/labels.py`). Anti-UAV410
   annotates boxes; EdgeTAM predicts masks. The teacher fills that gap offline.
2. **fp16 EdgeTAM → INT8 EdgeTAM, during QAT** (`src/training/qat.py`). The
   teacher here is *the same network before quantisation*, because the target
   is to reproduce it exactly. That is also why a short QAT schedule works: the
   weights only have to move far enough to absorb rounding.

The QAT loss is a **KL divergence, not a soft-target cross-entropy**. The two
differ by the teacher's own entropy — constant with respect to the student, but
it puts the "perfect agreement" floor at an arbitrary positive number that
moves with the teacher. As a KL, zero means identical, which is the only
reading that makes the curve interpretable.

**QAT will not reopen the memory path.** `insert_quantizers` refuses
`memory_attention` and `memory_encoder` unless asked twice, for the same reason
`finetune.py` freezes them: a change to what writes the memory bank accumulates
rather than averaging out. If a memory-path module fails its gate, leave that
engine at fp16.

## Building, and the cost that is not free

```bash
python tools/quantize_edgetam.py capture  --data <anti-uav410> --calib outputs/calib/
python tools/quantize_edgetam.py quantize --outdir models512/ --qdq-outdir models512_int8/
python tools/build_trt_engines.py --outdir models512_int8/ --precision auto --max-batch 4
```

`--precision auto` reads the per-module choice `quantize_edgetam.py` recorded
in each spec. This matters more than it looks: INT8 builds are **strongly
typed** so TensorRT obeys the calibrated Q/DQ nodes instead of re-deciding
precision layer by layer — but under strong typing a graph with *no* Q/DQ nodes
runs at the type it declares, which is fp32. A global `--precision int8` would
therefore make every module held back at fp16 quietly *slower*. `auto` builds
the mixed set correctly in one command.

One real cost, stated rather than buried: **a strongly-typed engine takes its
IO types from the graph, which the exporter writes as fp32.** The fp16 build
uses `--inputIOFormats/--outputIOFormats` to halve boundary traffic and avoid
reformat kernels; INT8 gives that back. Whether the INT8 compute wins by more
than the boundary loses is a measurement, and it is per module.

## The gate

Per module: accept INT8 if Anti-UAV410 validation **state accuracy** drops by
at most 1.0 point against the fp16 engines, and mask IoU against the fp16 run
stays above 0.99. Otherwise that module goes back to fp16 and the rest stand.

```bash
python tools/check_trt_parity.py --outdir models512_int8/ --image-size 512
python tools/eval_antiuav.py --data <anti-uav410> --split val \
    --tracker edgetam_trt --config configs/edgetam_trt_int8_512.yaml
```

Both are needed and they answer different questions. `check_trt_parity.py`
compares each engine against its module in isolation with a fresh input every
call — which cannot see accumulation, and EdgeTAM feeds its own masks back
through the memory bank. `eval_antiuav.py` runs the whole loop over real
sequences, so a bias that only shows up after two hundred frames of feedback
has somewhere to appear. Given what the memory encoder did under clipping, that
second check is not optional.

## Status

Tooling complete and unit-tested (`tests/test_quantization.py`); **no engine
built and no number measured yet.** Nothing in this document is a result — the
table at the top is from the fp16 work, and everything downstream of it is a
plan derived from that table. `notebooks/03_quantization_v2_int8.ipynb` runs the
calibration and PTQ; the gate has to be run on the Orin.

FP8 is not considered: Orin is sm_87 and has no FP8 tensor cores.
