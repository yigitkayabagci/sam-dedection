# EdgeTAM on Jetson AGX Orin: what was built, what was measured, and how

This is the record a report can be written from. It covers what the work
changed, every experiment run against it, the numbers those experiments
produced, and the commands that reproduce them.

**Companion document:** `docs/tensorrt_fp16.md` is the engineering detail — why
each module was hard to export, how the memory bank was made fixed-shape, what
the CUDA graph requires. This document is the experimental record. Where the
two overlap, that one is authoritative on mechanism and this one on results.

> **Numbers marked _(measured on device)_ come from real runs on the user's
> Orin AGX and are quotable. Numbers marked _(CPU, structural)_ were measured
> on a development machine without CUDA: they establish that a relationship
> holds, not what it costs on the target.** Anything not marked is a property
> of the code, not a measurement.

> **Not every tool in this suite measures real-time-representative frame
> time.** §4.2 audits each one directly — three tools bulk-decode the clip
> before timing starts and report a model-only number; one (`cli.py`)
> decodes per frame and reports the deployable budget. Read that section
> before quoting a ms/frame figure without knowing which kind it is.

---

## 1. What the work set out to do

EdgeTAM (a SAM 2 variant for video object tracking) ran on the Orin with only
its **image encoder** accelerated by TensorRT; the rest stayed in PyTorch. The
goal was to bring the whole per-frame graph onto TensorRT in fp16, reach a
**40 ms/frame** budget, and give up no accuracy.

The starting assumption — that the encoder is where the time goes — turned out
to be wrong, and measuring that was the first experiment.

---

## 2. The system under test

### 2.1 Per-frame computation

Four modules run on every tracked frame:

| module | what it is |
|---|---|
| `image_encoder` | RepViT-M1 trunk + FPN neck (+ the two high-res 1×1 convs, fused in) |
| `memory_attention` | 2-layer self/cross attention over the memory bank |
| `memory_encoder` | memory encoder + 2D Spatial Perceiver |
| `sam_head` | SAM mask decoder, specialised to the propagation path |

Everything else EdgeTAM does per frame is Python bookkeeping.

### 2.2 Why four engines and not one

The modules are separated where **Python sits between them**, not arbitrarily:

- Between `image_encoder` and `memory_attention`: the memory bank is assembled.
  It is a Python `dict` keyed by frame index; which past frames get read is
  decided per frame by `output_dict["non_cond_frame_outputs"].get(prev_frame_idx)`
  plus `select_closest_cond_frames`, which sorts candidates by distance to the
  current frame. A TensorRT engine has no dict, no `.get()`, and no
  data-dependent branch.
- Between `sam_head` and `memory_encoder`: this frame's masks are written back
  into that same dict for future frames to read.
- Between `memory_attention` and `sam_head`: nothing. That boundary is not
  forced.

The variable-length memory was solved by padding to a fixed 7 slots and
carrying occupancy in an additive attention mask — proven exactly equivalent to
the variable-length path by test (`test_padded_memory_matches_variable_length`).
What could **not** be moved into the engine is the decision of *which* frames
occupy those slots, because that decision is a dictionary lookup and a sort.

### 2.3 Could it be one engine?

Yes, in principle: a fused engine could take `(image, padded_bank, bank_pos,
bank_mask)` and return `(masks, obj_ptr, new_maskmem, new_maskmem_pos)`. Bank
assembly and write-back would still sit outside it, because they read and write
a Python dict.

What fusion could remove is only the **launch gap** — the time between one
engine returning and the next being launched. `tools/analyze_glue.py` measures
it directly. The decision was left to that measurement rather than taken on
faith; the threshold set was *under 0.5 ms of a ~26 ms frame → not worth it,
over 1.5 ms → worth it*, weighed against losing per-module fallback, per-module
parity debugging, and a much larger single export.

**Status: measured on device, §3.7 — launch gap is 0.217 ms (0.5% of frame).
Decision: not fusing.**

---

## 3. Experiments

### 3.1 Where the compute actually is

**Question.** Is the image encoder the bottleneck?

**Method.** `tools/analyze_cost.py` — FLOP counting via a `TorchDispatchMode`
over `torch.utils.flop_counter.flop_registry`, on the real checkpoint at
1024×1024.

**Result.** _(CPU, structural — FLOPs are hardware-independent)_

| module | share of per-frame FLOPs |
|---|---|
| memory_attention | **61.9%** |
| image_encoder | 26.8% |
| remainder | 11.3% |

**Reading.** Accelerating only the encoder addressed roughly a quarter of the
work. The memory attention — the part left in PyTorch — was the majority.

**Note on method.** The first attempt used `FlopCounterMode`, which fails under
`torch.inference_mode` (`'NoneType' object has no attribute 'next_functions'`)
and silently returned zero under `no_grad` because composite ops do not
decompose. The final version uses a custom dispatch mode, and adds a manual
SDPA rule because CPU attention lowers to
`aten._scaled_dot_product_flash_attention_for_cpu`, which has no registered
formula.

---

### 3.2 Is the rewrite lossless?

**Question.** The four modules were rewritten to be exportable — fixed-shape
memory, real-arithmetic RoPE instead of complex, additive masking. Does that
change what the model computes?

**Method.** `tools/analyze_precision.py --verify-graphs` — track a clip through
the rewritten modules at fp32 and compare masks against stock EdgeTAM, real
checkpoint, 1024×1024.

**Result.** _(measured, real checkpoint)_ **IoU 1.0000 on all 20 frames**, with
`strict=True` so any fallback to the original path would have raised.

Backed by unit tests that isolate each rewrite:

| claim | test |
|---|---|
| real-arithmetic RoPE == complex RoPE | `test_rope_real_matches_complex_self_attention`, `..._cross_attention` |
| padded + masked memory == variable-length memory | `test_padded_memory_matches_variable_length` |
| fused memory encoder + Perceiver == the two separately | `test_memory_encoder_graph_matches` |
| specialised SAM head == `_forward_sam_heads`, present *and* occluded | `test_sam_head_graph_matches`, `..._when_object_absent` |
| ONNX == PyTorch at batch ≠ trace batch | `test_onnx_matches_pytorch_at_any_batch` |
| whole patched tracker == stock EdgeTAM, 1 and 2 objects | `test_patched_tracker_matches_pytorch` |

48 tests, all passing.

---

### 3.3 Per-module engine parity and speed

**Question.** Does each fp16 engine reproduce its PyTorch module, and how much
faster is it?

**Method.** `tools/check_trt_parity.py` — fp32 PyTorch as reference, reporting
the engine's relative L2 error **next to PyTorch's own fp16 autocast error** on
the same output. Timed three ways: autocast, TensorRT enqueue, TensorRT CUDA
graph replay.

**Result.** _(measured on device, clocks **not** pinned)_

| module | torch fp16 | TensorRT + graph | speedup | worst rel L2 | vs torch fp16 |
|---|---|---|---|---|---|
| image_encoder | 49.58 ms | 14.30 ms | 3.47× | 2.48e-02 | 2.9× |
| memory_attention | 14.03 ms | 7.60 ms | 1.85× | 9.10e-04 | 7.9× |
| memory_encoder | 16.63 ms | 2.27 ms | 7.31× | 8.93e-03 | 3.0× |
| sam_head | 16.70 ms | 1.39 ms | 12.06× | 4.54e-03 | 2.3× |
| **total** | **96.94 ms** | **25.56 ms** | **3.79×** | | |

**On the pass/fail rule.** The first run failed `image_encoder` against a flat
2e-2 threshold. That threshold was a chosen number, not a property of the
model, and the failure was misleading. Where the encoder's error comes from was
measured on the real checkpoint at 1024²:

| source | rel L2 |
|---|---|
| fp16 output storage (engine IO boundary) | 2.1e-04 |
| fp16 input image | 0.9e-04 |
| **measured end to end** | **2.4e-02** |

The IO boundary is two orders of magnitude too small to matter — so `--io-fp32`
would not help. The error accumulates through **depth**:

| module | conv/linear layers | abs. rel L2 |
|---|---|---|
| image_encoder | **132** | 2.4e-02 |
| sam_head | 52 | 1.0e-03 |
| memory_attention | 20 | 9.0e-04 |
| memory_encoder | 13 | 7.9e-03 |

A ratio threshold alone is no better: `memory_attention` sits at 7.9× PyTorch's
fp16 because autocast keeps softmax and layer norm in fp32 there, giving it an
unusually clean baseline — while its absolute error is 9e-04 and irrelevant. So
a module now fails only when an output is **both** above `--rel-tol` **and**
above `--max-ratio` times PyTorch's own fp16. All four pass.

**A hypothesis that was tested and rejected.** The parity harness feeds the
encoder `randn` — white noise, not a natural image — and the suspicion was that
this inflated its error. Measured: a smooth, spatially correlated input is if
anything *more* sensitive to perturbation (ratio 0.3–0.8×). The reported figure
is not a test artefact.

---

### 3.4 Do the masks change over a whole clip?

**Question.** Per-module parity uses a fresh random input every call. EdgeTAM is
recurrent — every mask it emits is written into the memory bank and read back
later — so a small per-module error has hundreds of frames to compound. Does it?

**Method.** `tools/compare_backends.py` — track one 500-frame clip twice
(fp32 PyTorch reference, then TensorRT fp16), compare masks frame by frame with
IoU. IoU rather than logit error because a logit that shifts by 1e-3 changes
nothing while a logit that crosses zero changes a pixel. Reference masks are
kept bit-packed between runs so both models never share the GPU.

**Result.** _(measured on device, 500 frames, 1024×1024)_

| | |
|---|---|
| mean IoU | **0.9993** |
| min IoU | 0.9973 (frame 405) |
| p01 IoU | 0.9980 |
| below 0.99 | **0 of 500** |
| first quarter → last quarter | 0.9995 → 0.9991 |
| trend | **−0.0012 IoU per 1000 frames** |

**Reading.** The trend is the number that matters, and it is flat. A constant
offset and a slow drift look identical in a mean but mean nothing alike: the
first is fp16 rounding at the mask boundary, the second is the recurrent loop
amplifying it. At −0.0012/1000 frames the loop is not amplifying anything.

The residual is boundary jitter — fp16 moves logits near the mask edge across
zero, flipping a thin rim of pixels. At 0.9993 on a target of ~11,000 px that
is roughly 8 pixels, on a perimeter of ~380.

**A guard worth noting.** The tool refuses to report a score if the TensorRT
backend silently fell back to PyTorch — that configuration would score a
perfect 1.0000 against PyTorch and prove nothing. It is the one failure mode of
this test that looks like a pass.

---

### 3.5 End-to-end throughput

**Question.** Does the whole tracking loop hit 40 ms/frame?

**Method.** `tools/benchmark_tracking.py` — the real `propagate()` loop over a
500-frame synthetic clip, stock PyTorch and TensorRT back to back on the same
frames. Per-module breakdown comes from a second, separately timed pass, because
measuring it needs a `cudaStreamSynchronize` after every module.

**Result.** _(measured on device, 500 frames, 20 warm-up excluded)_

| | ms/frame | FPS | p50 / p90 / p99 |
|---|---|---|---|
| stock PyTorch (bf16) | 88.30 | 11.32 | 84.35 / 101.29 / 109.13 |
| **TensorRT fp16 + CUDA graphs** | **26.40** | **37.88** | 25.01 / 30.84 / **39.90** |
| | **3.34×** | | |

Per-module, inside that 26.40 ms:

| module | ms |
|---|---|
| image_encoder | 10.39 |
| memory_attention | 8.36 |
| memory_encoder | 3.14 |
| sam_head | 2.05 |
| accounted | 23.94 |
| other | 2.46 |

**Reading.** 40 ms met, and **p99 is also under 40 ms** — the tail matters more
than the mean for a real-time pipeline, since it is the tail that drops frames.

Note the baseline is bf16 (EdgeTAM's own default recipe) while the engines are
fp16. On Orin these use the same tensor cores at the same rate, so the
comparison is not distorted; `--precision float16` on the baseline makes it
exact if wanted.

---

### 3.6 What CUDA graphs are worth

**Question.** TensorRT is the goal; are CUDA graphs earning their place?

**Method.** `tools/benchmark_tracking.py --cuda-graph-ab` — the same backend
twice, graphs on and off, in one process.

**Result.** _(measured on device, 500 frames)_

| | ms/frame | p50 | p99 | p99 − p50 |
|---|---|---|---|---|
| graphs on | 33.40 | 33.26 | **37.70** | 4.44 |
| graphs off | 35.77 | 37.65 | **47.18** | 9.53 |

**Reading.** **The case for CUDA graphs is the tail, not the mean.** Without
them p99 is 47.18 ms — over the 40 ms budget; with them it is 37.70 ms, under
it. The jitter band roughly halves. That is what graph replay does: it removes
per-call kernel-launch dispatch, which is exactly the source of launch jitter.

**The mean difference in this table is not quotable.** The same configuration
measured 26.40 ms in §3.5 and 33.40 ms here — 26% apart, same settings — because
clocks were not pinned. The tail argument survives that, because both arms were
measured back to back in one run; the mean does not.

---

### 3.7 Where the non-engine time goes

**Question.** The benchmark reports ~2.5 ms of "other" per frame. Would fusing
the four engines into one remove it?

**Method.** `tools/analyze_glue.py` — splits that line by category, following
the real call tree (`image_encoder` runs inside `_get_image_feature`, a *sibling*
of `track_step`, not a child — an early version of this tool got that wrong and
charged the encoder's time to bookkeeping).

**Categories.** bank assembly (dict → engine input buffer), bank write-back
(engine output → dict), launch gap (between engine calls — **the only one
fusion could remove**), mask postprocess, other frame work.

**Result.** _(measured on device, 200 frames requested, 180 synchronized,
`--offload-video`)_

| | ms | % of frame |
|---|---|---|
| image_encoder | 14.419 | |
| memory_attention | 11.874 | |
| memory_encoder | 4.405 | |
| sam_head | 2.261 | |
| **engines total** | **32.958** | **83.3%** |
| bank assembly | 2.392 | 6.0% |
| bank write-back | 0.847 | 2.1% |
| launch gap | **0.217** | **0.5%** |
| mask postprocess | 0.455 | 1.2% |
| other frame work | 2.690 | 6.8% |
| **frame total** | **39.561** | |

**Reading.** Launch gap — the only category a fused engine could remove — is
0.217 ms, under the 0.5 ms *not-worth-it* threshold set in §2.3. **Decision:
not fusing.** The rest of the non-engine time (bank assembly + write-back +
other frame work, 5.93 ms) is Python dict read/write and survives fusion
regardless, per §2.2.

**Note on the frame total.** 39.56 ms here is higher than §3.5's 26.40 ms for
the same backend: this pass forces a `cudaStreamSynchronize` after every
module to attribute time correctly, which §3.5's throughput number does not
do. Use §3.5 for FPS; use this table only for the split between engines and
glue.

Verified on CPU (before this run) that the accounting sums to the measured
frame total exactly.

---

### 3.8 fp16 vs bf16 vs INT8

**Question.** Is fp16 the right precision?

**Method.** `tools/analyze_precision.py` — track a clip at fp32, re-track with
reduced-precision activations, compare masks. Pessimistic by construction: it
rounds normalisation and activation outputs too, which TensorRT often keeps in
fp32, so a clean fp16 result here is safe rather than optimistic.

**Result.** _(measured, real checkpoint)_

| precision | mask IoU vs fp32 |
|---|---|
| fp16 | **0.9999** |
| bf16 | 0.9997 |

fp16 wins, and the reason is structural: on Orin bf16 uses the same tensor cores
at the same rate while carrying 8 bits of significand to fp16's 11, and
TensorRT has fewer bf16 tactics. bf16's only advantage is dynamic range, which
matters only if activations overflow fp16 — measured, they do not.

**INT8.** Per-module, with symmetric per-tensor activation quantisation:

- The memory **encoder** is the module that compounds clipping bias (0.9397
  mean IoU, 7 of 30 frames below 0.99). It is the *write port of the recurrent
  loop* — its error goes into the bank and is read back.
- The image encoder is **calibration-sensitive, not structurally hostile.** An
  earlier conclusion here was wrong and was corrected: min/max calibration
  collapses it, while percentile clipping (0.9930) or fusion-modelled placement
  (0.9926) both rescue it. If an INT8 build disappoints, check which calibrator
  it used before concluding the architecture is at fault.
- Whole-model INT8 is not the sum of its per-module results.

**A tool flaw found and fixed, disclosed because it affected earlier numbers:**
`fake_quant_int8_percentile` used `torch.randint` to subsample, making it
non-deterministic — the same configuration scored 0.7157 and 0.5520 on
different runs. Replaced with a strided subsample.

#### 3.8.1 INT8 above is a PyTorch simulation, not a TensorRT engine

`analyze_precision.py`'s `fake_quant_int8*` hooks round each layer's *output*
to an 8-bit grid and dequantise it back to float inside an otherwise-fp32
forward pass (`PrecisionSimulation`, a `register_forward_hook`). Activations
only — weights are untouched, no engine is built. `build_trt_engines.py`'s
`PRECISION_FLAGS` currently holds only `fp16` / `bf16` / `fp32`
(`tools/build_trt_engines.py:98`); there is no `--int8` path or calibration
cache in this repo.

**Planned, not started: NVIDIA TensorRT Model Optimizer (`nvidia-modelopt`,
formerly AMMO).** NVIDIA's PyTorch-side PTQ/QAT toolkit for producing a
network TensorRT compiles straight to INT8 kernels. Mechanically:

1. Wraps `nn.Linear` / `nn.Conv2d` (etc.) with fake-quant Q/DQ nodes,
   including weights, per-channel.
2. Calibrates each Q node's scale from a short pass over representative
   input (max / entropy / percentile calibrators).
3. Exports the calibrated model to ONNX with the Q/DQ nodes baked in;
   TensorRT's parser builds native INT8 kernels around them directly
   ("explicit quantisation").
4. Optional QAT: fine-tune with Q/DQ active if PTQ alone is not accurate
   enough.

**Status: not started.** Needs `pip install nvidia-modelopt`, a calibration
dataset, export to a Q/DQ ONNX, new engine builds, and a
`check_trt_parity.py` / `compare_backends.py` run once those exist.

---

### 3.9 Resolution: 1024 vs 512

**Question.** What does halving the input resolution buy, and cost?

**Setup.** `configs/edgetam_512.yaml` (PyTorch) and `configs/edgetam_trt_512.yaml`
(TensorRT), with engines in a separate `models512/` — engines are shape-specific.

**Why it works at all.** The checkpoint was trained at 1024. The backbone and
neck are fully convolutional and positional encodings are computed per frame,
so a different size loads and runs — **except** for one table that does not
self-adjust: the memory attention's cross-attention (`RoPEAttentionv2`)
precomputes its rotary tables at construction from a hardcoded `q_sizes`
(`[64, 64]` for 1024). `src/trackers/_hydra_overrides.py` keeps `q_sizes` in
lockstep with `image_size // backbone_stride`. Without it the export fails with
*"Query RoPE table covers 4096 tokens but the feature map has 1024."*

**Token count.** 512 → 32×32 = 1024 spatial tokens per frame, against 64×64 =
4096 at 1024. Four times fewer.

**Comparison discipline.** A 512 run must be compared against a **512** PyTorch
baseline; against the 1024 baseline the measurement is of the resolution change,
not of TensorRT. Three tools had to be taught this — `check_trt_parity.py`
built its reference at 1024 regardless, `benchmark_tracking.py` pinned both
backends to their default YAMLs, and `run_experiment.py` declared a
`--baseline-config` it never passed on.

**Both resolutions emit masks at source frame resolution** — verified directly:
a 512 model and a 1024 model both return 240×320 masks for a 320×240 clip. So
512-TRT can be compared against 1024-TRT with IoU and no resampling in between,
which is what `--reference-config configs/edgetam_trt.yaml` does.

**Status: engines and configs ready; on-device results pending.**

---

### 3.10 Prompt sensitivity

**Question.** Does target size change the frame time? Does object count?

**Method.** `tools/sweep_prompt.py` — a separate tracking run per configuration
over a freshly generated clip.

**Expectation.** Target size should be flat: the model resizes every frame to a
fixed input, so a 12 px blob and a 120 px blob produce identically shaped
tensors. Object count should rise: objects are the batch dimension.

**Status: script ready, on-device results pending.**

---

## 4. Measurement methodology

### 4.1 What counts as a frame

Three timed stages, and one deliberately excluded:

| stage | what | in the budget |
|---|---|---|
| `pre` | decode + resize to model input + normalise, **per frame** | ✅ |
| `inference` | the four modules + EdgeTAM's bookkeeping | ✅ |
| `post` | masks resized back to source resolution | ✅ |
| overlay + mp4 encode | drawing the mask, writing a video | ❌ *demo only* |

This definition was arrived at by correcting two mistakes worth recording,
because both produced numbers that looked reasonable and were not:

1. **`pre` was not preprocessing.** It was re-reading the source frame off disk
   so an overlay could be drawn — the model had already read its own copy.
   Meanwhile the preprocessing a deployment actually pays happens in bulk inside
   `init_state`, lands in setup, and appeared in no frame time at all. Fixed by
   `_LazyFrames`, which replaces `inference_state["images"]` after `prepare()` so
   the per-frame decode happens inside the tracking loop.

2. **Video encoding was inside the frame budget.** Streaming frames to the writer
   (instead of collecting 1.4 GB of RGB and encoding at the end) put the encode
   call inside the timed region. It is now measured separately and excluded.

### 4.2 Per-tool coverage, audited directly

Every tool in the suite was instrumented and re-run to confirm — not assume —
what it includes. The audit method for each row was: monkeypatch the exact
call site the claim depends on, log every invocation with its index and
duration, run a real clip, and read the log. Numbers below are from those
runs, not from reading the source and reasoning about it.

| tool (step) | preprocess (`pre`) | model | postprocess (`post`) | overlay / encode |
|---|---|---|---|---|
| `benchmark_tracking.py` (3) | ❌ bulk, in setup | ✅ | ✅ *(lumped into one number)* | N/A, no video |
| `sweep_prompt.py` (5) | ❌ bulk, in setup | ✅ | ✅ *(lumped into one number)* | N/A, no video |
| `analyze_glue.py` | ❌ bulk, in setup | ✅ *(split per module)* | ✅ *(separate line)* | N/A, no video |
| `cli.py` frames-dir / jpg mode (4) | ✅ **per frame**, separated | ✅ | ✅ **per frame**, separated | ✅ measured, excluded from budget |
| `cli.py` direct-mp4 mode | ✅ **per frame**, separated | ✅ | ✅ **per frame**, separated | ✅ measured, excluded from budget |

**Only `cli.py` (step 4) measures a real-time-representative frame budget.**
Steps 3 and 5 decode the whole clip inside `prepare()`, before any per-frame
timer starts — deliberately, so they isolate the model — which means their
ms/frame numbers have **zero preprocessing in them**. A camera does not offer
that: every frame arrives raw and has to be decoded before anything else can
run. Quoting a step-3 number as "the real-time frame time" **overstates
achievable FPS** by however much decode+resize costs on the device, an amount
this repo has not measured for a camera source — only for JPEG, which was
6–18 ms/frame on this dev machine's CPU (§4.2.2) and will differ on the Orin
and differ again for whatever a real camera pipeline hands over (raw/NV12,
likely resizable on the VIC instead of the CPU — see §7).

**What "the model" itself includes**, for every tool: the four TensorRT
engines (or their PyTorch equivalents) plus EdgeTAM's per-frame bookkeeping —
memory-bank dict lookup/write, object-pointer accounting. Steps 3 and 5 do not
separate postprocessing (mask resize to source resolution) from this; it is
real work and it is included, just not broken out. Step 4 (`cli.py`) is the
only tool that reports it as its own `post` line.

#### 4.2.1 A universal blind spot: frame 0 is never timed for decode

Verified by logging every call to the frame-decode function, with real
indices:

```
JPG mode  (5-frame clip):  __getitem__ called for [1, 2, 3, 4]      -- 0 missing
mp4 mode (30-frame clip):  __getitem__ called for [1, 2, ..., 29]   -- 0 missing
```

Frame 0 is never decoded inside the timed loop, in **either** mode. The cause
is upstream, not a bug in this repo's instrumentation: `SAM2VideoPredictor.
init_state()` ends with

```python
# Warm up the visual backbone and cache the image feature on frame 0
self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
```

— an unconditional, hardcoded warm-up of frame 0's image encoder, inside
`init_state()`, i.e. inside `prepare()`, i.e. inside setup. It runs before
this repo's per-frame timer (or its lazy-decode wrapper) is even installed, so
there is no hook available to time it from outside without patching upstream
SAM2 code, which has not been done.

Effect, confirmed against the raw per-frame log: frame 0's wall time was
**115.6 ms** (JPG mode) and **95.2 ms** (mp4 mode) against **1.2–2.6 s** for
every other frame in the same *unpinned-clock CPU* run — both anomalously low,
consistent with frame 0 skipping decode *and* the image-encoder share of
inference (both already computed during setup).

**Practical impact: small and already covered by convention, not zero.**
`--fps-warmup` (default 20 in `run_experiment.py`) excludes frame 0 from every
reported statistic already, for the originally-stated reason ("model load /
CUDA warm-up") — this finding gives that convention a second, more specific
justification. The one place it is *not* covered is `--fps-warmup 0` or the
"FPS all" line (computed over every frame, warm-up included) — that figure is
inflated by however much one artificially-cheap frame out of the total pulls
it up. On a 500-frame run that is 1 in 500; on a short clip (`--frames 5`,
common in a smoke test) it is 1 in 5 and worth noticing.

**Rule going forward: never run with `--fps-warmup 0`, and never quote "FPS
all"; quote the post-warm-up figure.**

#### 4.2.2 A real gap, found and fixed: direct-mp4 mode measured zero preprocessing

Before this audit, `cli.py --video foo.mp4` (decord-decoded, no JPG cache) had
no lazy-decode wrapper at all — only the frames-dir/jpg paths had one. Verified
before the fix, on a real 30-frame clip:

```
pre (decode + resize to model input)   median 0.0 ms · mean 0.0 · min 0.0 · max 0.0
```

Flat zero for every frame, silently — no error, no warning, a chart that looks
complete. The true decode cost was not absent, it was misattributed: `_split_
frame`'s arithmetic is `infer = total − pre − post`, so with `pre` pinned at
zero by an empty log, all of it landed inside `inference` instead, understating
what the model itself costs and overstating what preprocessing costs (as
exactly zero, which is wrong in the other direction).

**Fixed** by adding `_LazyVideoFrames`, mirroring the existing JPG-mode
`_LazyFrames` but backed by decord's random-access indexing (`VideoReader[i]`)
instead of PIL, so decode is deferred to the moment each frame is actually
needed rather than done in one bulk loop inside `init_state`. If the wrapper
ever fails to attach, `cli.py` now prints an explicit warning naming the
consequence, rather than reporting a silent zero:

```
[pipeline] WARNING: could not install per-frame preprocessing timing for mp4
mode -- decode+resize cost will be folded into 'inference' rather than
reported as 'pre'. Total frame time is still correct.
```

Verified after the fix, same clip:

```
pre (decode + resize to model input)   median 7.2 ms · mean 8.0 · min 6.0 · max 18.3
```

**This gap did not affect any number already in this document.**
`run_experiment.py`'s video step always uses `--frames-dir` (via
`make_synthetic_video.py`), never `--video`, so it was always on the
already-correct JPG-mode path. The gap only mattered for direct `cli.py
--video some.mp4` runs, which nothing in the reproduction commands (§5) does.

### 4.3 Two frame times, both real

| | measures | use for |
|---|---|---|
| `benchmark_tracking.py` | the model alone; frames decoded up front | comparing engine sets or resolutions |
| `cli.py` | pre + model + post, per frame | **the deployable frame budget** |

The difference between them is per-frame preprocessing. Quoting one unlabelled
is what made 37.88 FPS and 23.0 FPS look contradictory when they were measuring
different things.

### 4.4 Reproducibility hazards found the hard way

- **Unpinned clocks cost ±30%.** `memory_attention` measured 11.33 ms and
  7.60 ms in two identical runs; the same benchmark configuration produced
  26.40 ms and 33.40 ms. `sudo nvpmodel -m 0 && sudo jetson_clocks` before any
  number that goes in a report. `tools/capture_environment.py` records whether
  they were pinned.
- **Two steps must use the same clip.** `run_experiment.py` originally passed
  `--radius` only to the video step, so the speed step benchmarked a different
  clip than the video it was compared against.
- **`--offload-video` must match across tools.** Without it EdgeTAM holds every
  frame on the GPU as fp32 — ~6.3 GB for 500 frames at 1024² — and on a Jetson's
  shared memory that slows the model itself.
- **1.4 GB of accumulated frames competes with the model.** CPU and GPU share
  memory *and bandwidth* on a Jetson; holding the rendered clip in a list was
  contending for the resource the model is bound by.

### 4.5 The synthetic clip

`write_synthetic_clip` in `tools/benchmark_tracking.py`: a blurred noise
background with N blobs on smooth sinusoidal trajectories, seeded (`rng =
default_rng(0)`) so every run gets identical frames. Smooth motion is
deliberate — a clip where the target teleports makes the memory bank useless and
is not what production looks like. `--radius` sets target size; the default is
proportional to the frame (~60 px at 720p).

---

## 5. Reproducing everything

```bash
# 0. Pin clocks first — without this nothing below is repeatable.
sudo nvpmodel -m 0 && sudo jetson_clocks

# 1. Record the environment (goes in the report).
python tools/capture_environment.py --out results/1024/environment --engines models/

# 2. Export + build, on the Orin. Engines are GPU- and TensorRT-version-specific.
python tools/export_edgetam_onnx.py --outdir models/ --verify
python tools/build_trt_engines.py   --outdir models/ --max-batch 4

# 3. The whole measurement suite, into one folder.
python tools/run_experiment.py --outdir results/1024

# 4. Prompt sensitivity.
python tools/sweep_prompt.py --radii 12,30,60,120 --objects 1,2,3 \
    --frames 200 --offload-video --out results/1024/05_sweep

# 5. Where the non-engine time goes (decides the fusion question).
python tools/analyze_glue.py --frames 200 --offload-video

# 6. The same at 512.
python tools/export_edgetam_onnx.py --outdir models512/ --image-size 512 --verify
python tools/build_trt_engines.py   --outdir models512/ --max-batch 4
python tools/run_experiment.py --outdir results/512 \
    --engines models512/ \
    --config configs/edgetam_trt_512.yaml \
    --baseline-config configs/edgetam_512.yaml \
    --reference-config configs/edgetam_trt.yaml \
    --image-size 512
```

`run_experiment.py` writes `SUMMARY.md` plus:

| file | contents |
|---|---|
| `01_parity.txt` | per-module engine-vs-PyTorch error and speed |
| `02_accuracy.txt` / `.json` | per-frame mask IoU, with the drift trend |
| `03_speed.txt` | FPS, p50/p90/p99, per-module ms, both backends |
| `03_speed_*.png` | per-frame latency, one chart per backend |
| `04_video.mp4` | the tracked clip |
| `04_video_latency.png` | per-frame latency: pre + model + post |
| `04_video_stages.png` | one histogram per stage + the demo-only cost |

---

## 6. Engineering obstacles, for the record

Two ONNX export failures worth documenting, because both surfaced as TensorRT
internal errors naming an ONNX node rather than a PyTorch line:

1. **`OneHot` → `Tile`.** `MaskDecoder.predict_masks` broadcasts the positional
   encoding with `torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)`.
   With a traced repeat count torch lowers that to OneHot → Tile, and TensorRT
   reports *"an IIOneHotLayer cannot be used to compute a shape tensor"*,
   blaming the Tile. `image_pe`'s batch axis is asserted to be 1 one line
   earlier, so repeating it **is** expanding it.

2. **`Tensor.squeeze(dim)` → `If`.** Torch cannot prove at trace time that the
   axis is 1, so it emits both outcomes behind a runtime branch whose arms
   differ in rank (`[B, C]` vs `[B, 1, C]`) — which TensorRT also rejects.
   Indexing with `[:, 0]` is unconditional.

The exporter now scans each graph for parser-hostile ops before writing the
spec, so a regression surfaces in seconds on a laptop rather than minutes into
a build on the device. Plain `Tile` is not flagged — the sine positional
encodings use four and they build; only a `Tile` whose repeats trace back to a
`OneHot` is.

**Two benign warnings every Orin build prints:**

- `DLA requests all profiles have same min, max, and opt value` — DLA is not
  being targeted; it has no attention kernels and EdgeTAM is mostly attention.
- `Detected layernorm nodes in FP16 ... Forcing Reduce or Pow Layers in FP32` —
  read the second sentence: TensorRT already applied the fix. The sum-of-squares
  in a layer norm is the one place fp16 can overflow, so it puts those layers in
  fp32 and leaves the rest in fp16. It warns because EdgeTAM's hand-written
  `LayerNorm2d` exports as separate Reduce/Pow/Div nodes rather than one
  `LayerNormalization`.

---

## 7. Open questions

| question | how to answer it | why it matters |
|---|---|---|
| What does 512 cost in accuracy? | `run_experiment.py --outdir results/512 --reference-config configs/edgetam_trt.yaml` | 4× fewer tokens; small objects are where it would show |
| Does target size affect frame time? | `sweep_prompt.py` | if not flat, something depends on content |
| What does camera preprocessing cost? | JPEG decode is now measured (`cli.py`'s `pre`, §4.2) — 6–18 ms/frame on a dev CPU, not yet on the Orin. A camera pipeline is not covered at all | a CSI camera gives NV12 and can resize on the VIC instead of the CPU, which is likely much cheaper than PIL-decoding a JPEG — but that is a guess until measured |
| Would `--opt-level 5` help memory_attention? | rebuild that one engine | it is 62% of FLOPs and has the *lowest* speedup (1.85×) |
| Does real (calibrated) INT8 survive, not just the simulation? | NVIDIA Model Optimizer PTQ, per §3.8.1 — not yet run | §3.8 only covers activations, no weight quantisation |

---

## 8. Honest limits

- **Every on-device number in §3.3, §3.5 and §3.6 was taken with clocks
  unpinned.** They are directionally sound and the ratios measured back to back
  within one run are reliable; the absolute means should be re-taken pinned
  before publication.
- **The synthetic clip is not real footage.** Smooth motion, high contrast,
  one or a few rigid blobs. It is a fair harness for *comparing backends* —
  both see identical frames — and a poor proxy for tracking difficulty.
- **Preprocessing in these measurements is JPEG decode**, not a camera pipeline.
  A real deployment's `pre` will differ, probably downward.
- **The 40 ms result is for one object at 1024×1024.** Multi-object numbers
  exist but are less exercised.
