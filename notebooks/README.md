# Notebooks: specialising EdgeTAM for thermal drone footage

Twenty-five notebooks, meant for Colab. Everything they orchestrate lives in
`src/` and `tools/` and is unit-tested without a GPU — the notebooks are the
recipe, not the implementation.

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
| 13 | `13_rgb_mask_pool.ipynb` | an aerial-RGB mask pool: VisDrone boxes → gated teacher masks, on your Drive | **nothing** — it downloads VisDrone and Kust4K's RGB half itself; SAM 3 wants an HF token |
| 14 | `14_thermal_mask_pool.ipynb` | the **thermal** mask pool: HIT-UAV + DroneVehicle boxes and RGBT234 masklets, with the thermal-vs-RGB-prompt route *measured* first | **nothing** — it downloads all four sets itself; SAM 3 wants an HF token |
| 15 | `15_dronevehicle_shared_pool.ipynb` | all of DroneVehicle in four pools: the 53 % of boxes both halves annotate identically (one RGB pass, mirrored onto thermal), plus the 33 383 thermal-only and 3 797 rgb-only targets, each prompted on the half that can see them | a copy of DroneVehicle's `train.zip` in your own Drive (set `DRIVE_DIR` in cell 1); SAM 3 wants an HF token |
| 16 | `16_vtuav_rgb_pool.ipynb` | VTUAV's **RGB** half: one box every tenth frame, prompted on the visible frame | VTUAV **RGB-T** archives in your own Drive (set `DRIVE_DIR`); runs beside 17; SAM 3 is required, there is no fallback |
| 17 | `17_vtuav_thermal_pool.ipynb` | VTUAV's **thermal** half, on `ir.txt`'s own boxes — the two halves agree on only 12 % of rows, so neither mask serves the other | the same archives; runs beside 16 on a second runtime |
| 18 | `18_kust4k_mask_pool.ipynb` | Kust4K's **drawn maps** turned into prompts: one SAM 3 pass on the RGB half mirrored onto thermal for the 71 % of frames both halves survive, then the 29 % the dataset marks broken — one modality corrupted, the manifests never say which — measured per frame and harvested on whichever half is still the scene | your own Kust4K upload under `MyDrive/datasets/kust4k` (zips or folders); SAM 3 is required, there is no fallback |
| 19 | `19_thermal_stage_b_pool.ipynb` | stage B trained on the **thermal** mask pools 13-18 produced, from stock EdgeTAM with no stage A — plus a stock-vs-trained score and a before/after panel | the pools staged under `MyDrive/edgetam-pool`, and the frames they were harvested from (a pool holds masks, not pixels) |
| 20 | `20_thermal_stage_b_pool_rgb.ipynb` | the same run with the **RGB** pools mixed into the same batches — the one variable | the same, plus the RGB pools' frames |
| 21 | `21_pool_data_readiness.ipynb` | no training: what every pool recorded, which of its frames are on disk, why the rest are not, and the exact `--pool` flags a run would take | the pools under `MyDrive/edgetam-pool`; it fetches only what `FETCH` names |
| 22 | `22_thermal_deep.ipynb` | thermal only, every thermal pool required to be present before a step is taken, the vehicle classes thinned, and the rate table inverted so the trunk learns the modality | the pools, plus VTUAV's archives in `MyDrive/VTUAV` (only the frames the pool names come out of them) |
| 23 | `23_thermal_deep_lora.ipynb` | 22 with `METHOD = "lora"` and nothing else changed — the A/B for the method | the same |
| 24 | `24_vtuav_lt_rgb_pool.ipynb` | VTUAV's **long-term** parts, RGB half, into a pool of their own (`vtuav_lt_rgb`) | the four `train_LT_*` RGB-T archives in your own Drive; SAM 3 is required, there is no fallback |
| 25 | `25_vtuav_lt_thermal_pool.ipynb` | the same for the **thermal** half (`vtuav_lt_thermal`) — this is the one to run first if thermal masks are what you want | the same archives; runs beside 24 on a second runtime |

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

**13 and 14 are a supply line, not an experiment.** Both are generated from
`tools/build_pool_notebooks.py` and neither trains anything: they turn
box-annotated datasets into `(image, box prompt, mask)` supervision through a
strong promptable teacher — `facebook/sam3` by default (gated,
transformers ≥ 5), `facebook/sam2.1-hiera-large` one string away (ungated,
Apache-2.0), `facebook/sam3.1` measured and rejected (checkpoint-only, no
transformers classes, and its Object Multiplex throughput win never applies
to a one-box-per-crop labeller). The mechanics are `labels.py`'s, reused
whole: zoom crops, the four gates, the run-length store. 13 is the RGB side
(VisDrone; the teacher on home ground). 14 is the branch this project is for,
and its first real output is a **measurement**: on Kust4K's drawn instances
it scores the teacher prompted on thermal pixels against the teacher prompted
on the registered RGB twin — per class, per size — and only then harvests
HIT-UAV and DroneVehicle by the winning route, plus RGBT234's 234 aligned
sequences as masklets (prompted on `visible.txt`, **gated on
`infrared.txt`**, which is what catches the registration's residual). The
pools are stage B's food and the masklets stage C's; wiring them into 07's
`DATASETS` is a deliberate follow-up, so a pool can be inspected and staged
once, then reused by every run after it. One contamination rule they create:
**AeroVIS is VisDrone re-labelled**, so a model trained on 13's pool cannot
be evaluated on AeroVIS — UAVScenes and MVUAV stay clean.
`docs/mask_pool_plan.md` argues the whole design.

Run them on separate runtimes at once. They write to different Drive folders
(`edgetam-encoder/{all,vtuav,drone}`), and `split_frames` seeds each dataset by
name rather than by position, so **all three hold out the same VTUAV
sequences** — which is what makes the scores comparable at all. 07 is the one
to trust if you only run one: an encoder is a general feature extractor, and 14
sequences of one campus is a narrow view of the world.

**16, 17, 24 and 25 are one generator and two splits.** All four come out of
`tools/build_vtuav_pool_notebooks.py`. 16/17 take the short-term parts, 24/25
the long-term ones, and each writes a pool of its own — `vtuav_rgb`,
`vtuav_thermal`, `vtuav_lt_rgb`, `vtuav_lt_thermal`. Separate pools rather than
four archives in one list, for three reasons: a short-term pool already
mirrored to Drive stays valid and the long-term harvest is purely additive
(`pool_reader.discover_pools` is how 19 and 20 find their food, and it picks a
new folder up with no edit anywhere); `aerial.split_index` stratifies per
dataset, so the long-term sequences get their own held-out slice instead of
being diluted into the short-term split; and the long-term parts are the ones
with real disappearances, so a pool that keeps them apart can be weighed apart.

**LT and ST are the same data.** Long-term versus short-term is about the
*tracking task*: same sensor, same 1920×1080 registered RGB-T pairs, same
`<sequence>/{rgb,ir}/` + `{rgb,ir}.txt` layout. Nothing in this pipeline reads
a tracking protocol, so the split barely matters — with two exceptions it does
have to handle. The target really leaves the frame in a long-term sequence and
those rows are marked absent, as a zero-extent box or a NaN one, and both are
dropped rather than handed to the teacher (`nan <= 0` is False, so testing only
the extent used to let a not-a-number rectangle through as a prompt). And the
stride — line k is frame 10k — was measured on `train_ST_001` and nowhere
else; a stride assumed one too small does not fail, it labels frame 9 with
frame 10's box and mirrors a pool of quietly wrong masks to Drive. So it is
derived per sequence from that sequence's own frame and row counts
(`boxes.annotated_stride`), 10 wins wherever those counts allow it (which keeps
every short-term harvest byte-identical), and a sequence whose counts allow no
single answer is skipped with its numbers printed. Cell 2 of all four notebooks
is a probe that prints exactly that — per archive: sequences, frames, rows,
implied stride, absent share — from the zip's central directory, before a byte
is extracted.

**The gate that decides is `box_iou`, and it is back at the library default.**
Four gates stand between a teacher mask and the pool: `teacher_iou` (the
teacher's own confidence — *measured* to be weak, +0.25 to +0.38 over-confident
on 12–36 px targets, so it passes almost everything), `box_iou` (does the
mask's own box agree with the annotation), `area` (mask area over box area,
0.15–1.3, which catches a fragment and background bleed alike), and `component`
(one object, not a mask scattered across the frame). `box_iou` is the
load-bearing one. 16/17/24/25 had it at 0.5, inherited from 15 where that is
argued — DroneVehicle ships oriented boxes and prompts with their axis-aligned
envelope, about twice the object's area on a 45° vehicle, so a correct mask
scores low against a box that was never tight. VTUAV has plain per-frame
`x y w h` around a 77 px median target and none of that applies, so it sits at
the 0.6 default now and 0.7 is one edit away. That edit is cheap in both
directions: the harvest writes **all four readings** into each instance's
record rather than only the verdict, and `index_pool(min_box_iou=…)` re-cuts on
the stored number — upwards only, since a pool harvested at 0.6 has no record
of what 0.5 would have kept.

**Night is a question about the teacher, and it is asked with a number.** A
promptable segmenter trained on web images reads a daylit street well and a
frame where the target is a smear around a headlight badly — and the gates do
not save you there: they catch a mask that has drifted off its box, not a
plausible one drawn around glare. The structural answer is why these are pairs
at all: each modality is prompted on **its own** pixels, so a night frame's
thermal half is 25's business and its RGB half is 24's, and no night RGB mask
is ever mirrored onto thermal. Every record now carries `luma` (the frame) and
`target_luma` (inside the boxes — an aerial night frame is mostly black with
the annotated thing under a lamp, so the frame's own mean files a well-lit
target under "night"), and cell 5 prints acceptance bucketed by the second.
`MIN_LUMA` drops frames whose targets fall below a threshold, **for the RGB arm
only**: on a thermal harvest a low reading is a *cold* target, so it defaults
to off rather than to a sensible-looking number.

**Neither stage leaves the other idle.** The harvest is two costs that used to
take turns. Unzipping is not bandwidth but seeks — `tracked_ir` keeps a
twentieth of a 15.7 GiB part, so the read is a few thousand random seeks over a
Drive mount, and `UNZIP_WORKERS` reads on that many threads with one zip handle
each. Decoding used to run on the thread that then waited for the teacher: a
1920×1080 JPEG is ~20 ms and a VTUAV frame carries **one box**, so a group of 32
put ~0.6 s of decode in front of every batch with the card doing nothing.
`label_pool(readers=…, read_ahead=…)` decodes ahead on its own threads — cv2
releases the GIL, so it is real parallelism — and the bound is in frames
because a decoded one is 6.2 MB. On a synthetic 1080p pool it takes the harvest
from 3.07 s to 1.18 s for identical output.

**No fallback teacher, in any of the four.** The load used to catch a failure
and quietly continue on `facebook/sam2.1-hiera-large`. Those four pools are
meant to be mixed into one training set, so a pool whose masks came from a
different teacher than its neighbour's is a variable nobody chose and the run
cannot see. `build_image_teacher` already fails with the gated-repo
instructions, so an unset `HF_TOKEN` says so up front instead of costing a
harvest that has to be thrown away.

**Neither split ships a mask.** VTUAV's drawn instance masks are the separate
VIS release, 100 sequences; the tracking download's 500 carry one `x y w h` per
annotated frame and nothing else. That is the whole reason these four exist.

**19 and 20 are the pools' first training run, and one experiment.** They are
generated from `tools/build_stage_b_notebooks.py`, are the comment-free shape
15–18 are, and differ in exactly one setting: 19 trains on the thermal pools,
20 adds the RGB ones to the same batches. Both start from **stock EdgeTAM with
no stage A**, which is the point — 07's number carries a distillation pass and
a fine-tune together, and these carry only what the pools bought.

Both score stock and the trained checkpoint on the same held-out split under
two prompts (`box` and `point` — 12 measured that the first cannot see an
encoder change), print the deltas by class and by modality, and then draw the
held-out instances whose IoU moved most **in both directions**: a mean can hide
a model that got better at trucks and worse at people, and the panel cannot.

Two grades, because they answer different questions. The pools' own held-out
slice is the sensitive one and its "truth" is a teacher's guess gated four
ways; Kust4K's drawn semantic maps, at `role=eval`, are the honest one. They
cannot overlap, so cell 2 **drops any pool built from the drawn set's own
frames** and says so — training on `kust4k_thermal` while grading on Kust4K
would be scoring on frames the run had seen, and the stratified split cannot
prevent that because the pool and the semantic set are separate sources with
separate permutations.

The reader that makes this possible is `src/training/pool_reader.py`, and
`--pool` is now a flag on `tools/train_encoder.py` and `tools/eval_instances.py`
beside `--dataset`. It was the follow-up `docs/mask_pool_plan.md` left open.

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
| a pool's mask is looked up, never matched | `src/training/pool_reader.py` | `Instance.label` is the box's row number, which is the key the store filed it under — so there is no matching step to get wrong |
| a pool's inset is applied when the image is read, and nowhere else | `aerial.image_origin` | DroneVehicle's 100 px band was cut before the teacher looked, so the boxes, the masks and the frame size are all the inset frame's and only the JPEG is not |
| a pool defaults to `role=train` | `pool_reader.parse_pool` | its masks are a teacher's guess gated four ways, so scoring on them measures the teacher — the same argument `split_index` makes for reconstructed semantic sets |
| the batch is measured up to 512, and the rate scales with it | 19, 20, `tools/train_encoder.py` | image mode holds no clip length and no memory bank, so it fits far more than the video path; `--steps` is fixed, so a bigger batch puts more samples behind the same number of updates |

## Verifying without Colab

Everything these notebooks call is tested on CPU with **nothing installed but
numpy and torch** — no EdgeTAM, no GPU:

```bash
python -m unittest tests.test_antiuav_dataset tests.test_accuracy tests.test_pseudo_labels tests.test_training_losses tests.test_clip_loop tests.test_quantization tests.test_samurai tests.test_adaptive tests.test_loader tests.test_fetch_antiuav410 tests.test_lora tests.test_schedule tests.test_aerial tests.test_image_loop tests.test_distill tests.test_datasets tests.test_notebooks tests.test_masklets tests.test_fetch_datasets tests.test_mask_pool tests.test_pool_reader
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
