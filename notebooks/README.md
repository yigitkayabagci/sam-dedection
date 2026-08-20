# Notebooks: specialising EdgeTAM for thermal drone footage

Six notebooks, meant for Colab. Everything they orchestrate lives in `src/`
and `tools/` and is unit-tested without a GPU — the notebooks are the recipe,
not the implementation.

| | notebook | produces | needs |
|---|---|---|---|
| 01 | `01_dataset_antiuav410.ipynb` | Anti-UAV410 clips + pseudo-mask labels | a CUDA GPU, ~20 GB of disk |
| 02 | `02_finetune_edgetam_512_thermal.ipynb` | `checkpoints/edgetam_thermal_512.pt` | **nothing** — it repeats 01 inline |
| 03 | `03_quantization_v2_int8.ipynb` | `models512_int8/` — calibrated Q/DQ graphs | 02 or 06 |
| 04 | `04_samurai_long_video.ipynb` | SAMURAI thresholds + a before/after | **nothing** |
| 05 | `05_adaptive_inference_sahi.ipynb` | whether the latency margin is worth spending | 01 for the data |
| 06 | `06_lora_vs_finetune.ipynb` | `checkpoints/edgetam_lora_512.pt` + its adapter, and the verdict | **nothing** — it labels and trains both |

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

**Notebook 06 settled an argument, against the repo's own position.**
`finetune.py` argued LoRA was the wrong tool here; 06 ran both through the same
loop (`tools/train_thermal.py --method {finetune,lora}`) and scored them on the
held-out test split. LoRA matched the fine-tune's accuracy while training 11 %
of the parameters and recovered from dropouts in half the frames, so
`configs/edgetam_512_lora.yaml` is the 512 thermal default now. The full table
and its limits: `docs/lora_vs_finetune.md`. Both methods merge to an ordinary
EdgeTAM checkpoint, so switching between them is a config line.

## The decisions these encode

Each is argued where it is made; this is the index.

| decision | where | one line |
|---|---|---|
| Anti-UAV410, not HIT-UAV or LSOTB-TIR | 01, 02 | 640×512 mono thermal video with a per-frame `exist` flag, and a 512 crop is native pixels |
| a labelling stride, not fewer sequences | `src/training/labels.py` | a skipped frame keeps `exist` and its box; at 25 fps its mask was nearly its neighbour's |
| LoRA, not the partial fine-tune | 06, `docs/lora_vs_finetune.md` | equal accuracy on `test` for 11 % of the trainable parameters, and dropouts half as long |
| …measured, not argued | 06, `src/training/lora.py` | the repo argued the opposite first; 60 scenes against 9.3 M moving parameters is what only the test split could settle |
| the adapter is kept beside the merged file | `src/training/lora.py` | a few MB of delta leaves upstream's checkpoint untouched, and makes a second domain a second adapter |
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
python -m unittest tests.test_antiuav_dataset tests.test_accuracy tests.test_pseudo_labels tests.test_training_losses tests.test_clip_loop tests.test_quantization tests.test_samurai tests.test_adaptive tests.test_loader tests.test_fetch_antiuav410 tests.test_lora tests.test_schedule
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
