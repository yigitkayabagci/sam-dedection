# Notebooks: specialising EdgeTAM for thermal drone footage

Twelve notebooks, meant for Colab. Everything they orchestrate lives in `src/`
and `tools/` and is unit-tested without a GPU — the notebooks are the recipe,
not the implementation.

| | notebook | produces | needs |
|---|---|---|---|
| 01 | `01_dataset_antiuav410.ipynb` | Anti-UAV410 clips + pseudo-mask labels | a CUDA GPU, ~20 GB of disk |
| 02 | `02_finetune_edgetam_512_thermal.ipynb` | `checkpoints/edgetam_thermal_512.pt` | **nothing** — it repeats 01 inline |
| 03 | `03_quantization_v2_int8.ipynb` | `models512_int8/` — calibrated Q/DQ graphs | 02 |
| 04 | `04_samurai_long_video.ipynb` | SAMURAI thresholds + a before/after | **nothing** |
| 05 | `05_adaptive_inference_sahi.ipynb` | whether the latency margin is worth spending | 01 for the data |
| 06 | `06_lora_vs_finetune.ipynb` | LoRA against the partial fine-tune, scored on `test` | **nothing** — it labels and trains both |
| 07 | `07_encoder_aerial_rgbt.ipynb` | `edgetam_aerial{,_lora}_512.pt` on your Drive + a held-out instance score | **nothing** — it downloads VTUAV VIS, SegFly and Kust4K itself |
| 08 | `08_encoder_vtuav_only.ipynb` | the same, from VTUAV VIS alone | **nothing** — it downloads VTUAV VIS itself |
| 09 | `09_encoder_stage_a_dronevehicle.ipynb` | the same, with stage A distilled from DroneVehicle | **nothing** — it downloads all four sets itself |
| 10 | `10_encoder_teacher_dinov3.ipynb` | the same, with stage A distilled from DINOv3 | a Hugging Face token — DINOv3 is gated |
| 11 | `11_encoder_rgb_mixed.ipynb` | the same, with VTUAV's RGB half mixed into the batches | **nothing** — the RGB frames are already in VTUAV's archives |
| 12 | `12_encoder_probe.ipynb` | whether 07–11's scores were measuring the encoder at all | **08**, for its checkpoints and its instance index |

**07 to 11 are one experiment, not five notebooks.** All five are generated
from the same source (`tools/build_notebooks.py`) and differ in a handful of
cells: the title, the build stamp, and the ones that configure the data or the
teacher. Everything else — the schedule, the losses, the seed, the
evaluation — is byte for byte the same, which is the only reason their
`test/instance_iou` numbers can be set beside each other.

Each isolates one variable against 07:

| | vs 07 | the question |
|---|---|---|
| **08** | same stage A, **VTUAV-only stage B** | what did the extra two datasets buy? |
| **09** | same stage B, **DroneVehicle stage A** | does stage A want resolution and one scene, or native pixels and variety? |
| **10** | same everything, **DINOv3 teacher** | does stage A want semantics (DINO) or class-agnostic boundaries (SAM)? |
| **11** | same everything, **RGB windows added** | is the gap modality, or is it domain — a nadir view of a 20-pixel car? |

**12 is not one of them.** It trains almost nothing and re-scores 08's four
finished checkpoints under three prompt strengths and two window scales. The
reason it exists is that 07–11 all vary the *checkpoint* while holding the
*question* fixed, and the question they hold fixed is the easiest one
available: `eval_instances.py` prompts with the ground-truth box, which on one
large isolated target states most of the mask. Stock EdgeTAM already clears
IoU 0.5 on 419 of 420 held-out instances before any training at all, so a flat
results table cannot distinguish "the encoder work bought little" from "the
prompt was giving the answer away". 12 weakens the prompt (`jitter`, then a
single centre `point`) and shrinks the targets (a wider crop resized to the
same 512) and re-reads the same four checkpoints under each. It is generated
by `tools/build_probe_notebook.py`, not by `build_notebooks.py`.

**12 has been run, and it changed 07–11.** Three findings, in order of size:

1. **Every trained checkpoint is *worse* than stock under a loose box** —
   0.057 to 0.100 IoU below it, at both window scales and under all three
   training methods, while beating it under an exact one. The cause was that
   the loop had no prompt knob at all: every prompt it ever built was a
   pixel-exact ground-truth rectangle. This is the largest single effect in the
   probe's table and it points the wrong way for stage C, where only frame 0 is
   prompted by hand and every frame after it is prompted by the memory path's
   own drifting estimate. `PROMPT = "mix"` is the fix and is now the default.
2. **The anchor was eating stage A rather than protecting it.** `distilled+ft`
   scores 0.8670 at `ANCHOR_WEIGHT = 0` against 0.8536 at 0.5 — and a plain
   fine-tune with no stage A at all scores 0.8611. So stage A did pay; the
   default is now 0.
3. **The prompt was not hiding a gain at native scale, but it was at half
   scale.** At window 512 the box and point gains match (+0.017 / +0.020 for
   the fine-tune). At window 1024, with the targets halved, point pulls away:
   +0.086 against box's +0.039. The encoder work is real and it lives on the
   scale axis — which VTUAV cannot test, because 0 of its 420 held-out
   instances fall under 32 px even halved. That question belongs to 07's
   SegFly and Kust4K.

None of it measures tracking. 12 answered what it was built to ask.

Run them on separate runtimes at once. They write to different Drive folders
(`edgetam-encoder/{all,vtuav,drone}`), and `split_frames` seeds each dataset by
name rather than by position, so **all three hold out the same VTUAV
sequences** — which is what makes the scores comparable at all. 07 is the one
to trust if you only run one: an encoder is a general feature extractor, and 14
sequences of one campus is a narrow view of the world.

**All of them are a different axis from 01–06.** Those specialise the *whole*
tracker on thermal video; these train the **image encoder alone** on static
aerial imagery, with the memory path frozen and never executed. They are stages
A and B of `docs/encoder_training_todo.md`; the architecture is laid out in
`docs/encoder_mimari.md`. They download their own data — every URL is baked
into `tools/fetch_datasets.py` and was checked against the live host, because
each of the four sets is served in a way that defeats the obvious approach.

None of them measures **tracking**, which is the metric this project is judged
on: every score is over a single prompted frame, with no memory bank in the
loop. A better encoder is a precondition for a better tracker, never evidence
of one. That is stage C, and it has not started.

## Three things to know before starting

**Notebook 02 runs end to end.** It fetches Anti-UAV410, labels it, fine-tunes
and scores the result in one runtime, with one *Run all*. Notebook 01 is the
labelling half on its own, for when you want the mask store without the
fine-tune (notebook 05) or want to look at what the teacher produced first.

**The dataset lands on local disk, never on Drive.**
`python tools/fetch_antiuav410.py --dest /content/data` pulls the 8.7 GB archive
from the authors' Google Drive, unpacks `train/`, `val/` and `test/`, and
verifies the layout. Training reads a few hundred thousand small JPEGs in random
order, and the Drive FUSE mount serves those an order of magnitude slower than
the GPU consumes them. Drive is used only for the checkpoint and the RLE label
store, which are megabytes. If the file's daily quota is spent, add it to your
own Drive and pass `--zip /content/drive/MyDrive/Anti-UAV410.zip`.

**EdgeTAM installs itself *as* the package `sam2`** — it is a fork — so Meta's
SAM 2 must never share the environment. The SAM 2.1 teacher runs through
`transformers`, which carries an independent SAM2 implementation under a
different name; that is what lets one runtime both label and train.

**Notebook 04 is independent.** SAMURAI is training-free, changes no weights and
needs no new engine. If you want the fastest measurable result, start there.

**Notebook 06 settles an argument.** `finetune.py` argues LoRA is the wrong tool
here; nothing had measured it. 06 runs both through the same loop
(`tools/train_thermal.py --method {finetune,lora}`) and scores them on the
held-out test split. LoRA's adapters merge into the base weights, so both
produce an ordinary EdgeTAM checkpoint and are evaluated identically.

## The decisions these encode

Each is argued where it is made; this is the index.

| decision | where | one line |
|---|---|---|
| semantic maps decomposed into instances, not trained on directly | 07, `src/training/aerial.py` | a promptable segmenter trained on "all cars are car" pulls two adjacent cars together |
| …but read instance masks directly where a set has them | 07, `aerial.decompose` | VTUAV's VIS split already ships the annotation the decomposition reconstructs, at 1920×1080 |
| several datasets in one batch, not one per run | `src/training/datasets.py` | the encoder carries general features, and one dataset is one sensor, one city and one set of annotation habits |
| reconstructed sets train, real-mask sets grade | `aerial.split_index`, `Source.role` | where the decomposition fused two cars its "truth" is one blob, so a model that separates them is scored wrong |
| the split is stratified per dataset | `aerial.split_index` | a 4 024-frame set beside a 100-sequence one can land almost entirely in `test`; and frame names collide across datasets |
| one encode, many prompts | 07, `src/training/image_loop.py` | the encoder is 38.7 GFLOP and does not depend on the prompt |
| the static loop goes through `track_step` | `src/training/image_loop.py` | a still image *is* frame 0 of a clip; two code paths for it would drift |
| no `object_score` term on static data | 07, `src/training/losses.py` | there is no `exist` label, and BCE against a constant 1 teaches the head to fire unconditionally |
| distillation for pretraining, not MAE | 07, `src/training/distill.py` | MAE masks tokens and needs a ViT; the student side of a distillation loss is architecture-free |
| staged A-then-B, not one summed multi-teacher loss | `docs/encoder_arastirma.md` | the two stages' data differ by two orders of magnitude; staging also removes the loss-weighting problem outright |
| an anchor term instead, during stage B | `src/training/image_loop.py` | stage A moves the encoder on 1.7 M pairs and stage B can undo it on 50 k; the reference is the model's own frozen copy, not a foundation model |
| …but off by default, because it was measured | 12, `tools/train_encoder.py` | anchor 0 reached 0.8670 where 0.5 reached 0.8536 and a plain fine-tune 0.8611 — the term meant to protect stage A was the thing eating it |
| the training prompt is mixed, not the exact box | 12, `src/training/image_loop.py` | trained on pixel-perfect boxes alone, every checkpoint fell 0.06–0.10 IoU *below stock* the moment the box was loose; in tracking, every frame after the first is prompted by a drifting estimate |
| …but validation still scores under `box` | `src/training/schedule.py` | that number selects which epoch becomes the checkpoint, and a random prompt would put a second moving part into it |
| box and jitter are mixed per instance, not per batch | `image_loop.mix_boxes` | one window contributes up to 8 prompts against a single encode; choosing per batch would correlate all of them |
| the web-pretrained DINOv3, not the satellite one | `docs/encoder_arastirma.md` | DINOv3's own results: satellite pretraining wins on metric tasks, web wins on segmentation and detection |
| Anti-UAV410, not HIT-UAV or LSOTB-TIR | 01, 02 | 640×512 mono thermal video with a per-frame `exist` flag, and a 512 crop is native pixels |
| a labelling stride, not fewer sequences | `src/training/labels.py` | a skipped frame keeps `exist` and its box; at 25 fps its mask was nearly its neighbour's |
| partial fine-tune, not LoRA | 02, `src/training/finetune.py` | the argument: 13.9 M parameters is not memory-bound, and merged LoRA is just weights |
| …but measured rather than assumed | 06, `src/training/lora.py` | the risk is 60 scenes against 9.3 M moving parameters, and only the test split can say |
| one loop, two methods | `src/training/schedule.py` | swapping the freeze and the save is the only honest way to compare them |
| the memory path stays frozen | 02, `finetune.py` | it is the write port of a recurrent loop; error there accumulates |
| training in `eval()` mode | 02, `src/training/clip_loop.py` | frozen batch-norm statistics keep the checkpoint matching its engines |
| batch size measured, not chosen | 02, `src/training/loader.py` | forward *and* backward at each candidate; the graph is what fills the card |
| the label store stays RLE in memory | `src/training/labels.py` | a 512×640 boolean frame is 328 KB and there are tens of thousands of them |
| entropy for the encoder, max for the memory encoder | 03, `docs/quantization_int8.md` | the two fail under *opposite* calibrators, measured |
| PTQ first, QAT only where it falls short | 03, `src/training/qat.py` | three of four modules already clear 0.999 in simulation |
| SAMURAI, not SAM2Long | 04, `docs/samurai.md` | memory attention is 62 % of the frame; N pathways costs N× that |
| adaptive ROI, not per-frame SAHI | 05, `src/trackers/adaptive.py` | SAHI is a detector technique; this pipeline has one prompted target |

## Verifying without Colab

Everything these notebooks call is tested on CPU with **nothing installed but
numpy and torch** — no EdgeTAM, no GPU:

```bash
python -m unittest tests.test_antiuav_dataset tests.test_accuracy tests.test_pseudo_labels tests.test_training_losses tests.test_clip_loop tests.test_quantization tests.test_samurai tests.test_adaptive tests.test_loader tests.test_fetch_antiuav410 tests.test_lora tests.test_schedule tests.test_aerial tests.test_image_loop tests.test_distill tests.test_datasets tests.test_notebooks
```

`tests.test_loader` and the end-to-end labelling case in
`tests.test_pseudo_labels` read pixels, so they skip themselves cleanly when
OpenCV is absent and run when it is there.

The three older suites need more of the environment —
`test_pipeline_smoke.py` needs OpenCV, and `test_edgetam_graphs.py` /
`test_edgetam_trt_integration.py` need pytest and EdgeTAM — so run those with
pytest, where they skip themselves cleanly when a dependency is missing:

```bash
python -m pytest tests/ -v
```
