#!/usr/bin/env python3
"""Build notebook 32 from the user's fixed Stage-B notebook.

The source notebook is deliberately left untouched.  This derivative keeps its
data discovery and leakage checks, adds the available VTUAV-LT pool as a hard
requirement, enables thermal contrast hardening, and replaces the unstable
large-batch LR rule with a short measured pilot followed by a conservative
OneCycle run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "22_thermal_deep_3_fixed.ipynb"
OUTPUT = ROOT / "notebooks" / "32_aerial_thermal_stage_b_stable.ipynb"


def lines(text: str) -> list[str]:
    return dedent(text).strip("\n").splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": lines(text),
    }


def replace(notebook: dict, old: str, new: str, *, count: int = 1) -> None:
    """Replace exact notebook source and fail if the template drifted."""
    found = 0
    for cell in notebook["cells"]:
        text = "".join(cell.get("source", []))
        hits = text.count(old)
        if hits:
            text = text.replace(old, new)
            cell["source"] = text.splitlines(keepends=True)
            found += hits
    if found != count:
        raise RuntimeError(f"expected {count} occurrence(s), found {found}: {old!r}")


def main() -> None:
    notebook = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    notebook["cells"].insert(0, markdown(r"""
        # 32 — Kararlı, kontrast-dayanıklı havadan termal Stage B

        Bu notebook kullanıcının `22_thermal_deep_3_fixed.ipynb` dosyasını
        **değiştirmez**; onun veri hazırlama, teacher kalite ve leakage
        kontrollerini ayrı bir deney kolunda sürdürür.

        Neden yeni kol var?

        - Drive'daki önceki `thermal_deep/run.json` koşusunda head validation
          loss `0.2723` iken encoder açıldıktan sonra `21.84` ve ardından `NaN`
          görülmüştür. Kaydedilen checkpoint head aşamasında kalmıştır.
        - O koşu `batch=128`, `lr_scale=4` ve encoder'da yüksek neck/trunk LR
          kullanmıştır. Uzun epoch bütçesi bu sayısal kararsızlığı çözmez.
        - Bu sürüm LR'yi batch boyutundan bağımsız `1.0` ölçekte tutar, trunk'ı
          en düşük hızda açar ve üç güvenli LR profilini kısa pilotta aynı split
          üzerinde karşılaştırır.
        - VTUAV-LT artık zorunludur; statik görüntü eğitiminde düşük-kontrast,
          polarity, gamma ve sensör gürültüsü augmentasyonları aktiftir.

        Ana çıktı:
        `/content/drive/MyDrive/edgetam-stage-b/aerial_thermal_stable/`
        `edgetam_pool_aerial_thermal_stable_512.pt`
    """))

    replace(notebook, 'RUN         = "thermal_deep"',
            'RUN         = "aerial_thermal_stable"')
    replace(notebook,
            'MIRROR_DIR  = "/content/drive/MyDrive/edgetam-stage-b/thermal_deep"',
            'MIRROR_DIR  = "/content/drive/MyDrive/edgetam-stage-b/aerial_thermal_stable"')
    replace(notebook,
            'REQUIRE_POOLS = {"dronevehicle_thermal": 20000, "vtuav_thermal": 20000,\n'
            '                 "hituav_thermal": 2000,\n'
            '                 "kaggle_uav_thermal": 10000}',
            'REQUIRE_POOLS = {"dronevehicle_thermal": 20000, "vtuav_thermal": 20000,\n'
            '                 "vtuav_lt_thermal": 5000, "hituav_thermal": 2000,\n'
            '                 "kaggle_uav_thermal": 10000}')
    replace(notebook,
            'CLASS_WEIGHTS = {"pool/dronevehicle_thermal": 0.45,\n'
            '                 "pool/dronevehicle_thermal_only": 0.7,\n'
            '                 "segfly": 0.85,\n'
            '                 "car": 0.7, "truck": 0.7}',
            'CLASS_WEIGHTS = {"pool/dronevehicle_thermal": 0.45,\n'
            '                 "pool/dronevehicle_thermal_only": 0.7,\n'
            '                 "pool/vtuav_thermal": 0.8,\n'
            '                 "pool/vtuav_lt_thermal": 0.8,\n'
            '                 "segfly": 0.85,\n'
            '                 "car": 0.7, "truck": 0.7}')
    # The weights this arm starts from. Empty keeps stock EdgeTAM, which is
    # what 22 always did; pointing it at 34's output is what makes a
    # pretrain -> stage B chain a chain rather than two unrelated runs.
    replace(notebook, 'DRIVE_POOLS = "/content/drive/MyDrive/edgetam-pool"',
            'DRIVE_POOLS = "/content/drive/MyDrive/edgetam-pool"\n'
            'BASE_CHECKPOINT = ""')
    replace(notebook,
            'BASE_CKPT = str(Path(EDGETAM) / "checkpoints" / "edgetam.pt")\n'
            'assert Path(BASE_CKPT).is_file(), "edgetam.pt did not download"',
            'BASE_CKPT = str(Path(EDGETAM) / "checkpoints" / "edgetam.pt")\n'
            'assert Path(BASE_CKPT).is_file(), "edgetam.pt did not download"\n'
            'if BASE_CHECKPOINT:\n'
            '    assert Path(BASE_CHECKPOINT).is_file(), (\n'
            '        f"BASE_CHECKPOINT is set to {BASE_CHECKPOINT} and nothing '
            'is there. "\n'
            '        f"Run the pretrain that writes it first, or clear the '
            'setting.")\n'
            '    BASE_CKPT = BASE_CHECKPOINT\n'
            '    _from = (Path(BASE_CHECKPOINT).stem'
            '.replace("edgetam_pool_", "").replace(f"_{SIZE}", ""))[:24]\n'
            '    RUN = f"{RUN}_from_{_from}"\n'
            '    MIRROR_DIR = f"{MIRROR_DIR.rstrip(chr(47))}_from_{_from}"\n'
            '    print("starting from", BASE_CKPT, "-- not stock EdgeTAM, so "\n'
            '          "the before/after below reports what THIS run added on "\n'
            '          "top of the pretrain rather than what both added to '
            'stock.")')
    # A pool name carries its modality only when the harvest put it there.
    # `vtuav_thermal` says so; `visdrone` does not, and VisDrone is RGB -- so
    # the template's fallback reads it as thermal, converts its colour frames
    # to grey and trains them in a thermal-only run with nothing saying it
    # happened. The generated arms got RGB_SOURCES; 32 comes from the
    # preserved template instead, so it is patched in here.
    replace(notebook,
            'def modality_of(pool):\n'
            '    if pool in POOL_MODALITIES:\n'
            '        return POOL_MODALITIES[pool]\n'
            '    GUESSED.add(pool)\n'
            '    lowered = pool.lower()\n'
            '    if lowered.endswith("_rgb") or "_rgb_" in lowered '
            'or "rgb" in lowered.split("_"):\n'
            '        return "rgb"\n'
            '    return "thermal"',
            'RGB_SOURCES = ("visdrone", "aerovis", "vtuav_vis", "rgbt234", '
            '"lasher")\n'
            '\n'
            'def modality_of(pool):\n'
            '    if pool in POOL_MODALITIES:\n'
            '        return POOL_MODALITIES[pool]\n'
            '    lowered = pool.lower()\n'
            '    if lowered.endswith("_rgb") or "_rgb_" in lowered '
            'or "rgb" in lowered.split("_"):\n'
            '        return "rgb"\n'
            '    if any(source in lowered for source in RGB_SOURCES):\n'
            '        return "rgb"\n'
            '    GUESSED.add(pool)\n'
            '    return "thermal"')
    replace(notebook, 'MIN_AREA       = 48', 'MIN_AREA       = 64')
    replace(notebook, 'MIN_SIDE       = 4', 'MIN_SIDE       = 6')
    replace(notebook, 'MAX_AREA       = 0.9', 'MAX_AREA       = 0.2')
    replace(notebook,
            'PROMPT_JITTER  = 0.3\nMETHOD',
            'PROMPT_JITTER  = 0.3\n'
            'CONTRAST_COLLAPSE = 0.40\n'
            'CONTRAST_FLOOR    = 0.15\n'
            'POLARITY_FLIP     = 0.25\n'
            'GAMMA_JITTER      = 0.30\n'
            'SENSOR_NOISE      = 5.0\n'
            'METHOD')
    replace(notebook, 'EPOCHS         = [2, 24]', 'EPOCHS         = [2, 12]')
    replace(notebook, 'STEPS          = 800', 'STEPS          = 500')
    replace(notebook, 'PATIENCE       = 4', 'PATIENCE       = 0')
    replace(notebook, 'VAL_BATCHES    = 24', 'VAL_BATCHES    = 32')
    replace(notebook, 'BATCH_CEILING  = 512', 'BATCH_CEILING  = 128')
    replace(notebook, 'BATCH_RESERVE  = 0.12', 'BATCH_RESERVE  = 0.15')
    replace(notebook,
            'LR_REFERENCE   = 16\nLR_SCALE_MAX   = 4.0\n'
            'LR_HEAD        = 0\nLR_NECK        = 1e-4\nLR_TRUNK       = 1e-4',
            'LR_SCALE       = 1.0\n'
            'RUN_LR_PILOT   = True\n'
            'LR_PILOT_STEPS = 150\n'
            'LR_PROFILES = {\n'
            '    "cautious": (2e-5, 2e-5, 5e-6),\n'
            '    "stable":   (5e-5, 5e-5, 1e-5),\n'
            '    "thermal":  (5e-5, 5e-5, 2e-5),\n'
            '}\n'
            'LR_HEAD, LR_NECK, LR_TRUNK = LR_PROFILES["stable"]')
    replace(notebook, 'NOTEBOOK = "22_thermal_deep.ipynb"',
            'NOTEBOOK = "32_aerial_thermal_stage_b_stable.ipynb"')
    replace(notebook, 'STAMP    = "029a14dfb8"', 'STAMP    = "{{STAMP}}"')

    replace(notebook,
            'COMMON += ["--index", INDEX_DIR, "--size", str(SIZE),\n'
            '           "--per-image", str(PER_IMAGE), "--max-instances", str(MAX_INSTANCES),\n'
            '           "--min-area", str(MIN_AREA), "--min-side", str(MIN_SIDE),\n'
            '           "--max-area", str(MAX_AREA), "--fill", str(FILL), "--seed", str(SEED)]',
            'COMMON += ["--index", INDEX_DIR, "--size", str(SIZE),\n'
            '           "--per-image", str(PER_IMAGE), "--max-instances", str(MAX_INSTANCES),\n'
            '           "--min-area", str(MIN_AREA), "--min-side", str(MIN_SIDE),\n'
            '           "--max-area", str(MAX_AREA), "--fill", str(FILL), "--seed", str(SEED),\n'
            '           "--contrast-collapse", str(CONTRAST_COLLAPSE),\n'
            '           "--contrast-floor", str(CONTRAST_FLOOR),\n'
            '           "--polarity-flip", str(POLARITY_FLIP),\n'
            '           "--gamma-jitter", str(GAMMA_JITTER),\n'
            '           "--sensor-noise", str(SENSOR_NOISE)]')
    replace(notebook,
            'LR_SCALE = round(min(max(BATCH / LR_REFERENCE, 1.0), LR_SCALE_MAX), 3)\n'
            'print(f"batch {BATCH} x accum {ACCUM} on {VRAM} GiB | lr-scale {LR_SCALE} "\n'
            '      f"(linear rule against {LR_REFERENCE} windows, capped at {LR_SCALE_MAX})")',
            'print(f"batch {BATCH} x accum {ACCUM} on {VRAM} GiB | "\n'
            '      f"fixed lr-scale {LR_SCALE}; large-batch linear scaling is disabled")')

    train_index = next(
        index for index, item in enumerate(notebook["cells"])
        if item["cell_type"] == "code" and
        '"tools/train_encoder.py"' in "".join(item.get("source", [])) and
        '"--mirror", MIRROR_DIR' in "".join(item.get("source", []))
    )
    notebook["cells"].insert(train_index, markdown(r"""
        ## LR pilotu ve tam eğitim bütçesi

        `epoch` burada veri setinin fiziksel olarak bir kez okunması değildir;
        `STEPS × batch` örneklemesidir. Tam encoder koşusu `12 × 500 = 6000`
        optimizer adımıdır. OneCycle bu toplam bütçeye göre kurulduğu için
        `PATIENCE=0`: en iyi validation checkpoint'i yine korunur, fakat LR
        inişi erken kesilmez.

        Pilot üç profili aynı split, aynı augmentasyon ve aynı seed ile
        `1 head + 2 encoder`, epoch başına 150 adımda sınar. Herhangi bir
        non-finite encoder loss profili elenir. Bu kısa pilot nihai accuracy
        ölçümü değildir; kararsız LR'yi pahalı tam koşudan önce elemek içindir.
    """))
    notebook["cells"].insert(train_index + 1, code(r"""
        import math

        PILOT_REPORT = {"enabled": RUN_LR_PILOT, "profiles": {}}
        if RUN_LR_PILOT:
            assert METHOD == "finetune", "LR pilotu bu notebook'ta finetune içindir"
            for _name, (_head, _neck, _trunk) in LR_PROFILES.items():
                _pilot_checkpoint = Path(WORK) / f"pilot_{_name}.pt"
                _pilot_json = Path(WORK) / f"pilot_{_name}.json"
                _flags = ["--method", METHOD,
                          "--lr-head", str(_head), "--lr-neck", str(_neck),
                          "--lr-trunk", str(_trunk)]
                subprocess.run(
                    [sys.executable, "tools/train_encoder.py", *COMMON, *_flags,
                     "--base", BASE_CKPT, "--out", str(_pilot_checkpoint),
                     "--prompt", PROMPT, "--prompt-jitter", str(PROMPT_JITTER),
                     "--jitter", str(JITTER), "--batch", str(BATCH),
                     "--accum", str(ACCUM), "--lr-scale", str(LR_SCALE),
                     "--steps", str(LR_PILOT_STEPS), "--epochs", "1", "2",
                     "--patience", "0", "--val-batches", "12",
                     "--workers", str(WORKERS), "--depth", str(DEPTH),
                     "--anchor-weight", str(ANCHOR_WEIGHT), "--device", "cuda",
                     "--json", str(_pilot_json)], check=True)
                _log = json.loads(_pilot_json.read_text())
                _encoder = [float(row["val_loss"]) for row in _log["history"]
                            if row["stage"] == "encoder"]
                _nonfinite = any(not math.isfinite(value) for value in _encoder)
                _failed = any(str(row.get("reason", "")).startswith("nonfinite")
                              for row in _log.get("stopped_early", []))
                _score = (min(_encoder) if _encoder and not _nonfinite and not _failed
                          else float("inf"))
                PILOT_REPORT["profiles"][_name] = {
                    "rates": {"head": _head, "neck": _neck, "trunk": _trunk},
                    "encoder_val": _encoder, "selection_score": _score,
                    "stopped_early": _log.get("stopped_early", []),
                }
                print(_name, PILOT_REPORT["profiles"][_name])

            _winner = min(PILOT_REPORT["profiles"],
                          key=lambda name: PILOT_REPORT["profiles"][name]["selection_score"])
            assert math.isfinite(PILOT_REPORT["profiles"][_winner]["selection_score"]), (
                "Bütün LR pilotları non-finite oldu; tam eğitimi başlatmayın.")
            LR_HEAD, LR_NECK, LR_TRUNK = LR_PROFILES[_winner]
            PILOT_REPORT["selected"] = _winner
            print("selected LR profile:", _winner,
                  {"head": LR_HEAD, "neck": LR_NECK, "trunk": LR_TRUNK})
        else:
            PILOT_REPORT["selected"] = "stable (pilot disabled)"

        _pilot_report_path = Path(WORK) / "lr_pilot.json"
        _pilot_report_path.write_text(json.dumps(PILOT_REPORT, indent=2) + "\n")
        shutil.copy2(_pilot_report_path, Path(MIRROR_DIR) / _pilot_report_path.name)
    """))

    # The full run must not be presented as a successful encoder adaptation if
    # it only preserved the earlier head checkpoint or met a non-finite value.
    train_index += 2
    text = "".join(notebook["cells"][train_index]["source"])
    needle = 'RUN_LOG = json.loads((Path(WORK) / "run.json").read_text())\n'
    guard = dedent(r"""
        RUN_LOG = json.loads((Path(WORK) / "run.json").read_text())
        _bad_stops = [row for row in RUN_LOG.get("stopped_early", [])
                      if str(row.get("reason", "")).startswith("nonfinite")]
        assert not _bad_stops, (
            f"Training non-finite oldu: {_bad_stops}. Önce lr_pilot.json'a bakın; "
            "checkpoint önceki finite en iyi aşamada korunmuştur ama encoder "
            "adaptasyonu başarılı sayılmaz.")
        _encoder_saved = [row for row in RUN_LOG["history"]
                          if row["stage"] == "encoder" and row["saved"]]
        if not _encoder_saved:
            print("!! Encoder validation'da head-only checkpoint'i geçemedi. "
                  "Dosya güvenli biçimde head aşamasında kaldı; Stage C'ye "
                  "geçmeden önce test IoU tablosunu karşılaştırın.")
    """).lstrip()
    if needle not in text:
        raise RuntimeError("training result marker not found")
    notebook["cells"][train_index]["source"] = text.replace(
        needle, guard, 1).splitlines(keepends=True)

    raw = json.dumps(notebook["cells"], ensure_ascii=False, sort_keys=True)
    stamp = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    replace(notebook, 'STAMP    = "{{STAMP}}"', f'STAMP    = "{stamp}"')
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                      encoding="utf-8")

    stamps_file = ROOT / "notebooks" / ".stamps.json"
    stamps = json.loads(stamps_file.read_text(encoding="utf-8"))
    stamps[OUTPUT.name] = stamp
    stamps_file.write_text(json.dumps(stamps, indent=1, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(f"wrote {OUTPUT} ({len(notebook['cells'])} cells, stamp {stamp})")


if __name__ == "__main__":
    main()
