# EdgeTAM on TensorRT: the whole per-frame graph, fp16, CUDA graphs

Before this change, only the image encoder ran on TensorRT. That is the part
everyone accelerates first — but on EdgeTAM it is not where the time goes.

## Where the time actually goes

Measured with `tools/analyze_cost.py` on the real checkpoint, one tracked frame
at 1024×1024, one object (`hidden_dim=256`, `mem_dim=64`, `num_maskmem=7`,
Perceiver emitting 256 + 256 latents per memory):

| module | GFLOP/frame | share | of which attention | params (M) | was on TensorRT? |
|---|---:|---:|---:|---:|---|
| image encoder (RepViT-M1 + FPN + conv_s0/s1) | 38.7 | 26.8 % | 0.0 | 4.92 | yes |
| **memory attention** (2 × self+cross+FFN) | **89.3** | **61.9 %** | **65.2** | 2.96 | no |
| memory encoder + spatial Perceiver | 12.6 | 8.8 % | 0.6 | 1.62 | no |
| SAM mask decoder | 3.6 | 2.5 % | 0.0 | 4.41 | no |
| **total** | **144.4** | | 65.8 | 13.9 | |

Accelerating only the image encoder covered **26.8 %** of the per-frame
compute. The other **73.2 %** stayed in PyTorch — and note the parameter
column: memory attention is 21 % of the weights but 62 % of the work, so model
size is a bad proxy for where the time goes.

The memory attention dominates because its self-attention runs over all 4096
image tokens with a *single* 256-dimensional head: the score matrix alone is
4096 × 4096, and that happens twice per layer, twice per frame. Attention is
45 % of the entire frame's arithmetic and essentially all of it lives in this
one module — which is exactly the shape fp16 tensor cores and TensorRT's fused
MHA kernels are built for.

All four now run on TensorRT in fp16, each replayed from a captured CUDA graph.

Reproduce with:

```bash
python tools/analyze_cost.py --trace-frames 24
```

## The three things that made this non-trivial

Worth saying plainly first: the image encoder was accelerated first because it
is the *easy* one. It is a feed-forward stack of convolutions with one fixed
input shape and one fixed output shape, no state, no complex arithmetic. It
exports to ONNX in a single call with no rewriting whatsoever. Every other
module fails at least one of those conditions, which is why they were left
behind — not because they were cheap.

### 1. The memory bank changes length every frame

`_prepare_memory_conditioned_features` concatenates however many memories exist
right now: one spatial memory after the first frame, growing to `num_maskmem`
(7) plus a pointer region that fills over ~16 frames. A TensorRT engine inside
a CUDA graph has exactly one shape, forever.

Traced on the real model (`tools/analyze_cost.py --trace-frames 24`), the
memory-attention key length over the first frames of a video:

| frame | spatial memories | pointer tokens | key length |
|---:|---:|---:|---:|
| 1 | 1 | 4 | 516 |
| 2 | 2 | 8 | 1032 |
| … | | | |
| 7 | 7 | 28 | 3612 |
| 8 | 7 | 32 | 3616 |
| … | | | |
| 16 | 7 | 64 | 3648 |
| 17+ | 7 | 64 | 3648 |

**16 distinct key lengths across 23 tracked frames.** Each would need its own
CUDA graph capture, its own TensorRT context and its own set of buffers.

The fix is a **fixed-slot buffer plus an additive attention mask**:

```
[ slot 0 | slot 1 | ... | slot 6 | object-pointer region ]
  <--------- 7 × 512 = 3584 ---------> <---- 96 ---->     = 3680 tokens
```

Unused slots are masked with an additive `-30000` logit. That is not an
approximation of the shorter sequence — `exp(-30000)` underflows to exactly 0
in fp16 and fp32 alike, so a masked position contributes exactly nothing and
the softmax is bit-equivalent to attending over the live prefix only. The
value chosen is finite on purpose: a true `-inf` would produce NaN if a row
were ever fully masked.

Two properties of EdgeTAM make the padding harmless:

* **RoPE is slot-position independent.** `apply_rotary_enc_v2` tiles the same
  16×16 frequency table across every spatial slot, so `tile(f, 7)` gives every
  slot the identical block. Which slots are occupied never changes the table,
  and it folds into the graph as a constant.
* **Temporal position encoding is already baked into `memory_pos`** by the
  caller before we see it, so we only copy, never recompute.

The alternative — a dynamic-shape engine with one CUDA graph captured per
sequence length — would need ~16 captures per video and a slower engine at
steady state, for no accuracy gain. Masking also gives constant per-frame
latency, which matters more than average latency on a real-time pipeline.

`tests/test_edgetam_graphs.py::test_padded_memory_matches_variable_length`
runs stock `MemoryAttention` against the padded version with deliberate
garbage in the unused slots, at three fill levels.

### 2. ONNX has no complex numbers

`apply_rotary_enc` / `apply_rotary_enc_v2` multiply `view_as_complex(x)` by a
complex table. Pointing `torch.onnx.export` at the stock module does not
produce a slow graph or a subtly wrong one — it refuses outright:

```
RuntimeError: ScalarType ComplexFloat is an unexpected tensor scalar type
```

The offending operations are `aten.view_as_complex` and `aten.view_as_real`.
Expanding `(a + ib)(cos + i·sin)` into two real multiply-adds is the same
arithmetic and exports to plain `Mul`/`Sub`/`Add`; the cos/sin tables become
graph constants. `test_rope_real_matches_complex_*` pins both forms against
upstream, and `test_stock_memory_attention_cannot_be_exported` pins the failure
so the rewrite can be revisited if a future torch gains complex support.

The cross-attention key rotation has one extra wrinkle worth knowing about:
within each slot, only the *trailing* 256 tokens (the Perceiver's 2D latents)
get rotated — the leading 256 (1D latents) are left alone. That split is
EdgeTAM's, and the wrapper reproduces it exactly.

### 3. CUDA graphs need static addresses

A captured graph replays the exact kernel launches it recorded, against the
exact device pointers it recorded. So `TRTEngine` owns persistent input and
output buffers, hands TensorRT their addresses once, warms up (TensorRT defers
some per-context setup to the first `enqueue`, which will otherwise get baked
in wrongly or fail the capture), captures, and from then on each frame is:
copy into the buffers → `graph.replay()`.

Buffers are cached per input-shape combination, with one execution context
each. Object count is constant for a whole video, so exactly one is built in
practice.

Because the output buffers are reused, anything that outlives a frame must be
copied out. The tracker is explicit about which:

| output | outlives the frame? | |
|---|---|---|
| `maskmem`, `maskmem_pos` | yes — read by later frames' memory bank | cloned |
| `obj_ptr`, `object_score_logits` | yes — go into `output_dict` | cloned |
| `low_res_masks` | yes — stored as `pred_masks` | cloned |
| `high_res_masks` | no — consumed by the memory encoder immediately | not cloned |
| FPN levels, `pix_feat` | no — consumed within the frame | not cloned |

`tests/test_edgetam_trt_integration.py` substitutes a fake engine whose output
buffers *are* overwritten on every call, so a missing clone shows up as a mask
mismatch rather than as a rare heisenbug on the Orin.

## Other wins found along the way

These are independent of TensorRT and mostly cost nothing:

* **`conv_s0`/`conv_s1` folded into the image encoder engine.**
  `SAM2Base.forward_image` applies them right after the encoder, so the engine
  can output 32- and 64-channel high-res levels instead of 256-channel ones:
  ~45 MB → ~6 MB of engine output per frame. The tracker detects the fusion
  from the engine's channel count and stops `forward_image` re-applying them.
* **fp16 engine I/O.** With TensorRT's default fp32 boundary, every engine gets
  reformat kernels wrapped around it. `--inputIOFormats/--outputIOFormats`
  halve that traffic. (Pass `--io-fp32` to the build driver to compare.)
* **Positional encodings cached.** `PositionEmbeddingSine` is a pure function
  of (H, W), but the old code re-materialised all three FPN levels every frame
  — ~44 MB of sine tables, of which EdgeTAM reads only the smallest.
* **No per-frame `cudaStreamSynchronize`.** The previous runtime synchronised
  after every inference, stalling the CPU on the GPU once per frame for no
  reason. Stream ordering already guarantees correctness.
* **`fill_hole_area` auto-disabled when `sam2._C` is missing.** Upstream calls
  `fill_holes_in_mask_scores` every frame; without the compiled extension it
  raises, warns, and skips — same output, one exception and one warning per
  frame. The tracker checks once and turns it off. Override with
  `fill_hole_area:` in the config.

## What fp16 costs, and why INT8 did not work

Measured with `tools/analyze_precision.py` on the real checkpoint: track a clip
at fp32, re-track it with reduced-precision activations, compare masks. 30
frames, one object, IoU against the fp32 run.

| configuration | mean IoU | worst frame | frames < 0.99 |
|---|---:|---:|---:|
| bf16 — what was already in production | 0.9997 | 0.9993 | 0 / 30 |
| **fp16 — what the engines run** | **0.9999** | **0.9998** | **0 / 30** |
| int8, all modules | 0.8169 | 0.0000 | 20 / 30 |

fp16 is not a compromise against the previous setup — it is more precise than
it. bfloat16 has 8 mantissa bits to fp16's 11, and EdgeTAM normalises before
every matmul, so activations sit comfortably inside fp16's range. fp16 scored
exactly 1.0000 on every frame of the clip.

Reduced precision is *simulated*, not run through autocast: a TensorRT fp16
layer on Orin reads and writes fp16 but accumulates in fp32, so rounding layer
outputs while accumulating in fp32 models the deployed engine more closely
than autocast (which lowers the accumulation too) — and is ~90× faster than
fp16 autocast on a CPU with no half-precision SIMD.

### INT8, one module at a time

| quantised module | mean IoU | worst frame | share of FLOPs |
|---|---:|---:|---:|
| image encoder | 0.8170 | 0.0000 | 26.8 % |
| memory attention | 0.9998 | 0.9993 | 61.9 % |
| memory encoder | 0.9999 | 0.9996 | 8.8 % |
| SAM mask decoder | 0.9993 | 0.9989 | 2.5 % |

Quantising the image encoder alone costs as much as quantising the entire
model (0.8170 vs 0.8169). The other three modules are individually harmless.
And the failure mode is not gentle: on 5 of 30 frames the tracker loses the
object outright (IoU 0.0000) and then re-acquires it.

This explains a failed INT8 experiment cleanly. The image encoder was the only
module with an engine, so it was the only place INT8 could be tried — and it
is the worst candidate in the model. It is RepViT: depthwise-separable
convolutions, whose per-channel activation ranges differ by orders of
magnitude. TensorRT quantises *activations* per tensor (per-channel is a
weights-only option), which is exactly the case that collapses on
depthwise-separable stacks — the same reason MobileNet-family networks need
per-channel treatment. The memory path is the opposite: dense matmuls with a
LayerNorm in front of every one, so activation ranges are bounded by
construction.

**So the next INT8 experiment should be the inverse of the failed one**:
quantise the memory path (73.2 % of the frame's arithmetic, all three modules
individually harmless) and keep the encoder in fp16. Before this work that was
not an option, because those three modules had no engines at all.

Caveats worth stating: activations are quantised, weights are not (TensorRT
does weights per-channel, which is more forgiving), so these are a lower bound
on INT8 damage. Run it yourself with:

```bash
python tools/analyze_precision.py --frames 30
```

## Workflow

Everything runs **on the Orin**. Engines are specific to the GPU and to the
TensorRT version; do not copy them from another machine.

```bash
# 1. Export the four ONNX graphs (+ .spec.json build metadata).
python tools/export_edgetam_onnx.py --outdir models/ --verify

# 2. Build fp16 engines. --max-batch is the largest object count they accept.
python tools/build_trt_engines.py --outdir models/ --max-batch 4

# 3. Check numerics and measure the speedup, per module.
python tools/check_trt_parity.py --outdir models/

# 4. Measure end-to-end tracking throughput, PyTorch vs TensorRT on the same clip.
python tools/benchmark_tracking.py --frames 500 --offload-video

# 5. Track.
python cli.py --tracker edgetam_trt --config configs/edgetam_trt.yaml \
    --video samples/road.mp4 --output outputs/road.mp4 \
    --prompt file --prompt-file examples/car_box_example.json \
    --fps-warmup 5 --fps-chart outputs/fps.png
```

`--verify` in step 1 runs each graph through onnxruntime on CPU and asserts
parity with PyTorch, so an export problem surfaces before you spend minutes in
`trtexec`.

Step 3 reports, per module, the fp16 engine's drift from fp32 PyTorch *next to*
the drift PyTorch's own fp16 autocast produces. That second column is the
baseline you were already running — an engine in the same range is not a
precision regression.

Step 4 runs the real `propagate()` loop on a generated clip and reports FPS,
p50/p90/p99 latency and where the milliseconds go, for the stock backend and
the TensorRT one back to back. `--offload-video` matters for long clips:
EdgeTAM otherwise preloads every frame to the GPU as fp32 at model resolution,
which is ~6.3 GB for 500 frames at 1024×1024.

## Knobs

| where | what |
|---|---|
| `export_edgetam_onnx.py --max-ptr-tokens` | Object-pointer capacity. Default `4 × (16 + 8) = 96`. Raise it if you prompt on many frames; frames that overflow fall back to PyTorch (and say so once). |
| `export_edgetam_onnx.py --static-batch` | Pin the batch dimension. Marginally faster; the tracker then falls back whenever the object count differs. |
| `build_trt_engines.py --max-batch / --opt-batch` | Optimisation profile. `opt` is what kernels are tuned for; `max` is the ceiling before PyTorch fallback. |
| `build_trt_engines.py --opt-level` | `trtexec --builderOptimizationLevel`, 0–5. Level 5 searches much harder and builds much slower — worth trying once on the memory attention. |
| `configs/edgetam_trt.yaml: use_cuda_graph` | Set `false` to isolate whether a problem is TensorRT or the graph capture. |
| `configs/edgetam_trt.yaml: strict` | `true` turns every silent PyTorch fallback into an error. Use it once all four engines are built — otherwise a broken engine looks like a mysterious slowdown. |

## Bringing it up incrementally

Every engine is optional. Delete or rename one and that module goes back to
PyTorch, everything else keeps running. `check_trt_parity.py --module
memory_attention` and `--module sam_head` narrow things down the same way.

If the engine and the model disagree structurally — a memory sequence length
that does not match `num_maskmem × tokens_per_slot`, a missing output, an FPN
level with unexpected channels — the tracker raises at patch time with the
mismatch spelled out, rather than producing quietly wrong masks.

## What is validated, and where

| claim | test |
|---|---|
| real-arithmetic RoPE == complex RoPE | `test_rope_real_matches_complex_self_attention`, `..._cross_attention` |
| padded + masked memory == variable-length memory | `test_padded_memory_matches_variable_length` |
| fused memory encoder + Perceiver == the two run separately | `test_memory_encoder_graph_matches` |
| specialised SAM head == `_forward_sam_heads`, present *and* occluded | `test_sam_head_graph_matches`, `..._when_object_absent` |
| ONNX == PyTorch, and stays correct at batch ≠ trace batch | `test_onnx_matches_pytorch_at_any_batch` |
| the whole patched tracker == stock EdgeTAM, 1 and 2 objects | `test_patched_tracker_matches_pytorch` |
| pointer overflow falls back instead of corrupting | `test_memory_attention_falls_back_when_pointers_overflow` |

These run on CPU with random weights at 256×256 — enough to prove the
rewrites, since none of them depend on tensor size. TensorRT itself, fp16
accuracy and CUDA graph capture can only be checked on the device:
`tools/check_trt_parity.py`.

## Known limits and further work

* **Memory features are stored as bf16.** `_run_single_frame_inference` casts
  `maskmem_features` to bf16 before storing it, so an fp16 memory round-trips
  through bf16's shorter mantissa. That is EdgeTAM's own storage choice and the
  stock model pays it too, but for an all-fp16 pipeline it is a free accuracy
  win to remove — it needs a patch to upstream's method.
* **Prompted frames stay in PyTorch.** They run once or twice per video, so
  this is deliberate. If you drive an interactive loop with many clicks per
  frame, a second SAM-head engine taking real point embeddings would help.
* **One CUDA graph per engine, not one per frame.** Four replays per frame
  instead of one. Fusing the SAM head and memory encoder into a single graph is
  possible (they are adjacent, with only a sigmoid between them); the memory
  attention cannot join without moving the memory-bank bookkeeping into the
  graph.
* **`high_res_multimasks` is returned as `None`.** `track_step` and
  `_use_mask_as_output` are the only callers and both discard it; materialising
  `[B, 3, 1024, 1024]` for nobody is pure waste. A future caller that wants it
  gets a `TypeError`, not a wrong answer.
* **`--opt-level 5`** has not been measured here. If the memory attention's
  fused MHA kernel is not being selected, that is the first thing to try.
