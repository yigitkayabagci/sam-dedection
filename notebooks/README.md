# Notebooks: specialising EdgeTAM for thermal drone footage

Five notebooks, meant for Colab. Everything they orchestrate lives in `src/`
and `tools/` and is unit-tested without a GPU — the notebooks are the recipe,
not the implementation.

| | notebook | produces | needs |
|---|---|---|---|
| 01 | `01_dataset_antiuav410.ipynb` | Anti-UAV410 clips + pseudo-mask labels | A100/L4, Anti-UAV410 on Drive |
| 02 | `02_finetune_edgetam_512_thermal.ipynb` | `checkpoints/edgetam_thermal_512.pt` | 01 |
| 03 | `03_quantization_v2_int8.ipynb` | `models512_int8/` — calibrated Q/DQ graphs | 02 |
| 04 | `04_samurai_long_video.ipynb` | SAMURAI thresholds + a before/after | **nothing** |
| 05 | `05_adaptive_inference_sahi.ipynb` | whether the latency margin is worth spending | 01 for the data |

## Two things to know before starting

**Notebook 01 needs its own runtime.** EdgeTAM installs itself *as* the package
`sam2` — it is a fork of it — so Meta's SAM 2 and EdgeTAM cannot coexist in one
environment. Notebook 01 runs its SAM 2.1 teacher through `transformers`, which
has an independent SAM2 implementation and does not collide; notebooks 02–05
install EdgeTAM. **The mask store on disk is the only thing that crosses
between them.**

**Notebook 04 is independent.** SAMURAI is training-free, changes no weights and
needs no new engine. If the dataset download stalls, or you just want the
fastest measurable result, start there.

## The decisions these encode

Each is argued where it is made; this is the index.

| decision | where | one line |
|---|---|---|
| Anti-UAV410, not HIT-UAV or LSOTB-TIR | 01 | 640×512 mono thermal video with a per-frame `exist` flag, and a 512 crop is native pixels |
| partial fine-tune, not LoRA | 02, `src/training/finetune.py` | 13.9 M parameters is not memory-bound, and merged LoRA is just weights |
| the memory path stays frozen | 02, `finetune.py` | it is the write port of a recurrent loop; error there accumulates |
| training in `eval()` mode | 02, `src/training/clip_loop.py` | frozen batch-norm statistics keep the checkpoint matching its engines |
| entropy for the encoder, max for the memory encoder | 03, `docs/quantization_int8.md` | the two fail under *opposite* calibrators, measured |
| PTQ first, QAT only where it falls short | 03, `src/training/qat.py` | three of four modules already clear 0.999 in simulation |
| SAMURAI, not SAM2Long | 04, `docs/samurai.md` | memory attention is 62 % of the frame; N pathways costs N× that |
| adaptive ROI, not per-frame SAHI | 05, `src/trackers/adaptive.py` | SAHI is a detector technique; this pipeline has one prompted target |

## Verifying without Colab

Everything these notebooks call is tested on CPU with **nothing installed but
numpy and torch** — no EdgeTAM, no OpenCV, no GPU:

```bash
python -m unittest tests.test_antiuav_dataset tests.test_accuracy tests.test_pseudo_labels tests.test_training_losses tests.test_clip_loop tests.test_quantization tests.test_samurai tests.test_adaptive
```

The three older suites need more of the environment —
`test_pipeline_smoke.py` needs OpenCV, and `test_edgetam_graphs.py` /
`test_edgetam_trt_integration.py` need pytest and EdgeTAM — so run those with
pytest, where they skip themselves cleanly when a dependency is missing:

```bash
python -m pytest tests/ -v
```
