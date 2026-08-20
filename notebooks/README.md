# Notebooks: specialising EdgeTAM for thermal drone footage

Seven notebooks, meant for Colab. Everything they orchestrate lives in `src/`
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
| 07 | `07_encoder_aerial_rgbt.ipynb` | `edgetam_aerial{,_lora}_512.pt` on your Drive + a held-out instance score | one or more aerial RGB-T datasets staged in your Drive |

**Notebook 07 is a different axis from 01–06.** Those specialise the *whole*
tracker on thermal video; 07 trains the **image encoder alone** on static
aerial imagery, with the memory path frozen and never executed. It is stage B
of `docs/encoder_training_todo.md`; the architecture it implements is laid out
in `docs/encoder_mimari.md`. Its dataset comes off your Drive rather than a
public URL, because none of these sets has one that survives a click-through.

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
| the split is stratified per dataset | `aerial.split_index` | a 4 024-frame set beside a 100-sequence one can land almost entirely in `test`; and frame names collide across datasets |
| one encode, many prompts | 07, `src/training/image_loop.py` | the encoder is 38.7 GFLOP and does not depend on the prompt |
| the static loop goes through `track_step` | `src/training/image_loop.py` | a still image *is* frame 0 of a clip; two code paths for it would drift |
| no `object_score` term on static data | 07, `src/training/losses.py` | there is no `exist` label, and BCE against a constant 1 teaches the head to fire unconditionally |
| distillation for pretraining, not MAE | 07, `src/training/distill.py` | MAE masks tokens and needs a ViT; the student side of a distillation loss is architecture-free |
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
python -m unittest tests.test_antiuav_dataset tests.test_accuracy tests.test_pseudo_labels tests.test_training_losses tests.test_clip_loop tests.test_quantization tests.test_samurai tests.test_adaptive tests.test_loader tests.test_fetch_antiuav410 tests.test_lora tests.test_schedule tests.test_aerial tests.test_image_loop tests.test_distill
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
