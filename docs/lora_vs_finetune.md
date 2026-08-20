# LoRA against partial fine-tuning: what the held-out split said

`src/training/finetune.py` argued that LoRA was the wrong tool for this model,
`src/training/lora.py` was written so that argument could be tested rather than
repeated, and `notebooks/06_lora_vs_finetune.ipynb` ran both under one loop.
This is the answer, and it goes against the original argument.

**Verdict: use LoRA.** `configs/edgetam_512_lora.yaml` is the 512 thermal
default, and the SAMURAI, SAM2Long and INT8 configs point at the same
checkpoint. The fine-tune is not deleted or deprecated —
`configs/edgetam_512_thermal.yaml` still runs it — because the accuracy result
is a tie and one seed is one sample.

## What was run

Both methods through `src/training/schedule.py`: same clips in the same order
from the same seed, same losses, same two stages, same one-cycle schedule, same
gradient clipping, same EMA, same validation slice, same rule for keeping a
checkpoint. `--method` changed which parameters received a gradient and how the
checkpoint was written. The learning rate was deliberately not held identical
(`tools/train_thermal.py:RATES`): LoRA starts from `B = 0` and every published
recipe gives it roughly an order of magnitude more, so matching the fine-tune's
rate would have tested a badly-tuned LoRA rather than LoRA.

60 train sequences, 16 val, `r = 16`, 400 steps per epoch, one head epoch then
two encoder epochs.

## The cost side

| | trainable | batch | peak GiB | minutes | best val |
|---|---:|---:|---:|---:|---:|
| finetune | 9.318 M | 32 | 77.6 | 36 | 0.1468 |
| lora | 1.064 M | 32 | 77.7 | 41 | 0.1461 |

131 layers adapted at `r = 16`. LoRA moved **11.42 %** of what the fine-tune
moved.

**Two of the three usual reasons to reach for LoRA did not survive contact with
this measurement, and that is worth stating plainly:**

- **It did not save memory.** Same peak, same batch size — `auto_batch_size`
  measured 32 for both. The optimiser state LoRA avoids is two moments over
  8.25 M parameters, about 63 MB. Peak memory here is activations for 32 clips
  of 8 frames at 512², and 63 MB against 77 GiB is nothing. On a GPU where
  optimiser state actually dominated, the story would be different; this is not
  that GPU.
- **It did not save time.** It cost 14 % more — the adapters add a second
  convolution beside every adapted layer in the forward pass, and the backward
  still propagates gradients through the whole trunk to reach them. Only the
  weight update is cheap, and the weight update was never the expensive part.

What is left is the third reason, and it is the one that mattered: **the
update is rank-constrained**, which is regularisation that freezing cannot
express.

## The measurement that decides

Test split, 25 sequences, seen by neither run, scored through the deployment
tracker with one box prompt on the first annotated frame
(`tools/eval_antiuav.py --mode crop`).

| | state acc | success AUC | episodes | lost frames | median len |
|---|---:|---:|---:|---:|---:|
| stock | 0.3840 | 0.3800 | 48 | 12 270 | 8.0 |
| finetune | 0.5626 | 0.5593 | 43 | 7 070 | 56.0 |
| lora | **0.5669** | **0.5650** | 54 | **6 815** | **31.0** |

### Read the first row first

Adaptation is worth **+0.183 state accuracy** over stock, and it cuts the
frames spent with the target lost by 44 %. Both methods deliver essentially all
of that. Whatever else is on this page, the choice between LoRA and fine-tuning
is a second-order decision inside a first-order win.

### The accuracy gap is a tie

0.0043 state accuracy, from one seed each. The notebook's own rule — written
before the numbers existed — is that a gap under ~0.01 needs two or three seeds
before it means anything. It has not had them. LoRA is also 0.0007 better on
validation clip loss, which is the same story: no separation.

### The dropout structure is not a tie

This is where the two methods actually differ, and it is the axis this project
cares about most. From `docs/samurai.md`: one bad frame writes `no_obj_ptr`
into the memory bank, the next `num_maskmem` frames read it back, and nothing
in the architecture un-poisons it. A long dropout is that failure; a short one
is a bad frame.

- **Total time blind:** 6 815 frames against 7 070 — 3.6 % better, marginal.
- **Median episode:** 31 frames against 56 — the target comes back in about
  1.0 s at 30 fps instead of 1.9 s.
- **Mean episode:** 126 frames against 164, so the tail is lighter too, not
  just the middle.
- **Episode count:** 54 against 43 — LoRA loses the target *more often*.

So the fine-tune fails less frequently and stays failed nearly twice as long,
while LoRA blinks more and recovers faster. For an anti-UAV pipeline, seconds
blind is the operational cost and the shorter failure is the better one. That
is a judgement about this application, not a general fact about the two
methods, and it is the reason the tie was broken in LoRA's favour.

It is also consistent with what a rank-16 update *is*: it cannot move a layer
as far as an unconstrained one can, so the adapted encoder stays closer to the
general-purpose features underneath and drifts back to them when the thermal
cues are ambiguous. That is the mechanism the tie is evidence for. It is not
independently measured here.

## What this does not license

- **One seed.** Re-run both with `--seed 1 --seed 2` before treating 0.0043 as
  a preference for LoRA on accuracy. The honest current claim is "equal
  accuracy, better failure shape, 11 % of the trainable parameters".
- **60 sequences.** The overfitting risk this comparison was built to catch is
  scene diversity, not sample count. The fine-tune has more capacity to use and
  should close the gap as scenes are added — so this verdict is specific to a
  60-scene budget and should be re-checked if that budget grows.
- **Thermal, at a 512 crop, on Anti-UAV410.** Nothing here transfers to RGB
  footage, to a different sensor, or to full-frame 640×512 input without being
  measured again.
- **`r = 16`.** Not swept. It was chosen as a common default and it worked.

## What changed in the repo because of it

| file | change |
|---|---|
| `configs/edgetam_512_lora.yaml` | the 512 thermal default |
| `configs/edgetam_512_lora_adapter.yaml` | new: stock checkpoint + adapter, merged at load |
| `configs/edgetam_samurai_512.yaml`, `configs/edgetam_sam2long_512.yaml` | point at the LoRA checkpoint |
| `configs/edgetam_trt_samurai_512.yaml`, `configs/edgetam_trt_int8_512.yaml` | same, including the export command in their headers |
| `tools/quantize_edgetam.py` | calibrates the checkpoint that will be deployed |
| `configs/edgetam_512_thermal.yaml` | kept, no longer the default |
| `src/training/finetune.py` | its docstring argued the opposite; corrected |

Nothing about the deployment path changed, and that is the point: the adapters
are merged before the checkpoint is written, so the LoRA file is an ordinary
EdgeTAM state dict that loads into the same config, exports to the same ONNX
graph and builds the same engines.

## The adapter as a separate artefact

`tools/train_thermal.py --method lora` now writes two files:

    checkpoints/edgetam_lora_512.pt            merged, 54 MB, the deployment artefact
    checkpoints/edgetam_lora_512.adapter.pt    the delta alone, a few MB

The second one is what makes "we did not disturb the original weights" a fact
about the filesystem rather than a figure of speech. Say it precisely, because
it is easy to overclaim: once merged, a LoRA checkpoint's weights are changed
exactly as much as a fine-tune's — the same tensors, in the same places. What
is constrained is the *rank* of the change, not whether there is one. Keeping
the adapter separately is a statement about what is stored, and the rank is the
statement about what was learned; they are two different arguments and both are
worth having. `configs/edgetam_512_lora_adapter.yaml`
loads stock EdgeTAM and merges the adapter at startup: same model, same masks,
milliseconds of extra load. It buys three things the merged file cannot:

- upstream's checkpoint stays byte-identical to what everyone else downloaded,
  so "what did this project change" is a few MB you can diff, archive or send;
- a second domain becomes a second adapter beside the same base, not a second
  54 MB checkpoint;
- `lora_scale` blends between stock (0.0) and the run as trained (1.0), which
  is a real knob for footage that only partly matches the adapter's domain —
  and which nothing here has measured, so treat any value but 1.0 as an
  experiment.

Export and engine builds still go through the merged file. TensorRT never sees
an adapter.

## Running these weights on footage

The point of the exercise. Nothing about the tracker, the prompt or the
pipeline changes with the adapted weights — only `--config`:

```bash
# A video file, box prompt drawn on the first frame.
python cli.py --video clip.mp4 --output out.mp4 \
    --tracker edgetam --config configs/edgetam_512_lora.yaml --prompt box

# Headless (an Orin with no display): the prompt comes from JSON.
python cli.py --video clip.mp4 --output out.mp4 \
    --config configs/edgetam_512_lora.yaml \
    --prompt file --prompt-file examples/car_box_example.json

# A frame sequence tracked in a native 512 window instead of a resize -- what
# the training saw, and what a camera that already centres the target delivers.
python cli.py --frames-dir frames/ --frame-pattern '*.tif*' --center-crop 512 \
    --output out.mp4 --config configs/edgetam_512_lora.yaml --prompt box

# The same weights with SAMURAI's memory gate on top: the recommended PyTorch
# configuration for long clips, and free (docs/samurai.md).
python cli.py --video clip.mp4 --output out.mp4 \
    --config configs/edgetam_samurai_512.yaml --prompt box

# Stock checkpoint plus the adapter, merged at load. Same masks as the first
# command; upstream's file stays untouched on disk.
python cli.py --video clip.mp4 --output out.mp4 \
    --config configs/edgetam_512_lora_adapter.yaml --prompt box

# Measure only -- no overlay, no re-read, no encode. This is the configuration
# a frame budget should be quoted from.
python cli.py --video clip.mp4 --no-video --config configs/edgetam_512_lora.yaml \
    --prompt file --prompt-file examples/car_box_example.json --fps-warmup 20
```

For targets a handful of pixels across, the window that follows the target
(`src/trackers/adaptive.py`, `tools/track_adaptive.py`) buys more than either
checkpoint does: a 10-pixel drone in a 125-pixel window arrives at the model as
a 40-pixel one, at the 512 price. That is a separate, still-unmeasured axis —
`docs/new_report.md §6` says why it is the riskiest part of the project.

**What to expect, and what not to.** The adaptation was trained on 640×512 mono
thermal video, cropped to a native 512 window, one drone per clip, 60 scenes.
That is what every number on this page describes. Colour footage, a visibly
different sensor, targets that fill the frame, a different altitude band: the
weights load and run, and whether they help there is an open question rather
than a promise. `lora_scale` in the adapter config is the cheap way to ask —
0.0 is stock, 1.0 is the thermal run, and for footage in between the answer may
be too.

The tracker names the checkpoint it loaded before the first frame, so a run
that quietly used the wrong file says so:

```
[edgetam] edgetam_lora_512.pt: lora, r=16, on Anti-UAV410, 60 sequences, trained at 512
[edgetam] lora adapter: 131 layers, 1.064 M, r=16, Anti-UAV410        (adapter config only)
```

and warns when a config runs those weights at a resolution they were not tuned
at — `configs/edgetam_768.yaml` with a 512 checkpoint, for instance.

## Reproducing it

```bash
python tools/train_thermal.py --method finetune \
    --data <anti-uav410> --labels <work>/labels \
    --out checkpoints/edgetam_thermal_512.pt --json runs/finetune.json
python tools/train_thermal.py --method lora --lora-r 16 \
    --data <anti-uav410> --labels <work>/labels \
    --out checkpoints/edgetam_lora_512.pt --json runs/lora.json

for cfg in edgetam_512 edgetam_512_thermal edgetam_512_lora; do
  python tools/eval_antiuav.py --data <anti-uav410> --split test --limit 25 \
      --tracker edgetam --config configs/$cfg.yaml --mode crop \
      --json runs/test_$cfg.json
done
```

`notebooks/06_lora_vs_finetune.ipynb` runs exactly this and draws the
per-sequence chart, which is the honest view: a method that wins the mean by
rescuing two sequences and breaking three is not the same result as one that
helps everywhere.
