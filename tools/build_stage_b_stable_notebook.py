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
    # The shape cut for merges `fill` cannot see -- two vehicles touching along
    # a full edge fill their box and pass every gate. See
    # `docs/segfly_decomposition.md` 2b; it is an upper bound, because a bus
    # from overhead lands in the same band.
    replace(notebook,
            'from src.training.aerial import (InstanceGates, rebalance, '
            'sample_windows,\n'
            '                                 save_splits, split_index)',
            'from src.training.aerial import (InstanceGates, drop_merge_profile,\n'
            '                                 rebalance, sample_windows,\n'
            '                                 resolve_quantiles, save_splits,\n'
            '                                 size_bands, split_index)')
    replace(notebook,
            'if CLASS_WEIGHTS:\n'
            '    INDEX, _balance = rebalance(INDEX, CLASS_WEIGHTS, seed=SEED)',
            'INDEX, _shape = drop_merge_profile(\n'
            '    INDEX, sources=("segfly", "segfly_thermal", "segfly_rgb"))\n'
            'if _shape["by_class"]:\n'
            '    print("\\nshape cut -- the merges `fill` cannot see "\n'
            '          "(docs/segfly_decomposition.md 2b):")\n'
            '    for _name, _row in sorted(_shape["by_class"].items()):\n'
            '        print(f"   {_name:<26}{_row[\'before\']:>7} ->"\n'
            '              f"{_row[\'after\']:>7}   dropped {_row[\'dropped\']} "\n'
            '              f"({_row[\'share\']:.1%}: {_row[\'abreast\']} abreast, "\n'
            '              f"{_row[\'end_to_end\']} end to end)")\n'
            '    print("   an upper bound, not a count of merges: a bus from "\n'
            '          "overhead is square and large too. What it buys is that "\n'
            '          "no merge survives.")\n'
            '\n'
            'if CLASS_WEIGHTS:\n'
            '    INDEX, _balance = rebalance(INDEX, CLASS_WEIGHTS, seed=SEED)')
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
            '                 "segfly:truck": 0.0,\n'
            '                 "car": 0.7, "truck": 0.7}')
    # The weights this arm starts from. Empty keeps stock EdgeTAM, which is
    # what 22 always did; pointing it at 34's output is what makes a
    # pretrain -> stage B chain a chain rather than two unrelated runs.
    # `windows_for` takes a native crop only while the frame is at least SIZE
    # on both axes; below that it resizes the whole frame, aspect ratio and
    # all. At the default 512 that boundary is exactly where the thermal sets
    # sit -- a 640x512 frame is a 512 crop with no resampling -- so raising
    # SIZE silently turns every one of them into an upsample. Nothing warns,
    # because both paths are legitimate; this counts them so the choice is made
    # on the census rather than on the number looking bigger.
    replace(notebook,
            'TRAIN, VAL, TEST = windows("train", JITTER), windows("val", 0), '
            'windows("test", 0)',
            'TRAIN, VAL, TEST = windows("train", JITTER), windows("val", 0), '
            'windows("test", 0)\n'
            '\n'
            '_native = {}\n'
            'for _entry in SPLITS["train"]:\n'
            '    _key = _entry.source.name if _entry.source else "?"\n'
            '    _fits, _seen = _native.get(_key, (0, 0))\n'
            '    _native[_key] = (_fits + int(min(_entry.size) >= SIZE), '
            '_seen + 1)\n'
            '_small = {k: v for k, v in _native.items() if v[0] < v[1]}\n'
            'if _small:\n'
            '    print(f"\\n!! too small for a native {SIZE} window -- these '
            'frames are resized\\n"\n'
            '          f"   whole, aspect ratio and all, instead of cropped:")\n'
            '    for _key, (_fits, _seen) in sorted(_small.items()):\n'
            '        print(f"     {_key:<28}{_seen - _fits:>7} of {_seen} '
            'frames")\n'
            '    print(f"   `windows_for` falls back to the whole frame when "\n'
            '          f"min(width, height) < {SIZE}. At 512 a 640x512 thermal "\n'
            '          f"frame is a crop with no resampling at all; every size "\n'
            '          f"above it turns the same frame into an upsample paying "\n'
            '          f"{(SIZE / 512) ** 2:.2f}x the token count for "\n'
            '          f"interpolated pixels. If most of the pool is listed "\n'
            '          f"here, that is the answer about {SIZE}.")')

    # The rate table is fixed and `--lr-scale 1` keeps it fixed, so the batch
    # the probe measures is the batch the rates were chosen for. Raise SIZE and
    # the probe finds a smaller one -- 768 is 2.25x the tokens of 512 -- and
    # the same rates then run on a noisier gradient with nothing said about it.
    # `EFFECTIVE_BATCH` is how a run at one size borrows another's: set it to
    # the `batch x accum` the run being compared against recorded in its
    # run.json, and accumulation makes up the difference. 0 is off, which is
    # every run taken so far.
    replace(notebook, 'BATCH_RESERVE  = 0.12',
            'BATCH_RESERVE  = 0.12\nEFFECTIVE_BATCH = 0')
    replace(notebook, 'ACCUM = 1\n',
            'ACCUM = max(1, round(EFFECTIVE_BATCH / BATCH)) '
            'if EFFECTIVE_BATCH else 1\n'
            'if ACCUM > 1:\n'
            '    print(f"effective batch held at {BATCH * ACCUM} "\n'
            '          f"(asked {EFFECTIVE_BATCH}): the probe fits {BATCH} at "\n'
            '          f"size {SIZE}, and the rate table was chosen for the "\n'
            '          f"larger number, not this one.")\n')

    # The dense-cluster term. `windows_for` already puts every other indexed
    # instance inside the window into the same sample, so the supervision for
    # "do not claim the object beside you" was in the batch and unused. 0.0
    # keeps the published objective and still prints the leak, which is the
    # order this repository asks for: find out how much of the failure the run
    # has before training against it.
    replace(notebook, 'ANCHOR_WEIGHT  = 0.0',
            'ANCHOR_WEIGHT  = 0.0\nNEIGHBOUR_WEIGHT = 0.0')
    # The report is what `tools/compare_stage_a.py` reads to decide whether two
    # arms are comparable, and a term added to the objective is not a detail
    # about how a run went -- it is what the run optimised.
    replace(notebook,
            '"gates": GATES.__dict__, "batch": BATCH, "lr_scale": LR_SCALE,',
            '"gates": GATES.__dict__, "batch": BATCH, "lr_scale": LR_SCALE,\n'
            '    "neighbour_weight": NEIGHBOUR_WEIGHT,')

    # The role stays `train`, and the audit is not a reason to change it. It
    # was `train` because a *reconstructed* target cannot score a model -- where
    # the decomposition fused two cars its ground truth says one blob and a
    # model that separates them is marked wrong. The audit fixed that for the
    # instances it could measure, and left 706 frames whose masks sit 25-50 px
    # off the vehicle, with a further 50.5 % it could not measure at all. A
    # label in the wrong *place* is the worst kind of validation error, and the
    # validation number here selects the checkpoint. So SegFly still trains and
    # still does not score.
    #
    # SegFly, hand-audited: 3 439 frames / 10 751 `vehicle` instances against
    # the raw export's 15 007, with `truck` dropped as a class, 1 162 masks
    # with no vehicle under them removed and 1 914 merged pairs split. The
    # masks are instance ids, so `labels` and not `components` -- see
    # SPECS["segfly_temiz"]. `SEGFLY_DROP` decides what to do with the one
    # finding the audit could not fix: 706 frames whose masks sit 25-50 px off
    # the vehicle. "none" trains on them, "shift" excludes them, and cell 3
    # prints the census either way.
    replace(notebook,
            'EXTRA_DATASETS = ["segfly:/content/data/SegFly:thermal:components:train"]',
            'SEGFLY_CLEAN = True\n'
            'SEGFLY_DROP  = "shift"\n'
            'EXTRA_DATASETS = (["segfly_temiz:/content/data/SegFly_temiz:'
            'thermal:labels:train"] if SEGFLY_CLEAN else\n'
            '                  ["segfly:/content/data/SegFly:thermal:'
            'components:train"])')
    replace(notebook,
            '["/content/drive/MyDrive/edgetam-pool/segfly/segfly.zip", "SegFly"],',
            '(["/content/drive/MyDrive/edgetam-pool/segfly_temiz/'
            'SegFly_temiz.zip", "SegFly_temiz"]\n'
            '     if SEGFLY_CLEAN else\n'
            '     ["/content/drive/MyDrive/edgetam-pool/segfly/segfly.zip", '
            '"SegFly"]),')
    # The weight is keyed by source name, so the rename has to carry it or the
    # arm silently trains SegFly at 1.0 while the report still lists the key.
    replace(notebook, '"segfly": 0.85,\n',
            '"segfly": 0.85, "segfly_temiz": 0.85,\n')

    # The exclusion manifest has to exist before anything globs the set, so
    # this runs before the first `build_indexes` and after `GATES`, which is
    # where the cell stops being configuration and starts reading the disk.
    replace(notebook,
            'GATES = InstanceGates(min_area=MIN_AREA, min_side=MIN_SIDE, '
            'max_area=MAX_AREA,\n                      fill=FILL)',
            'GATES = InstanceGates(min_area=MIN_AREA, min_side=MIN_SIDE, '
            'max_area=MAX_AREA,\n                      fill=FILL)\n'
            '\n'
            'if SEGFLY_CLEAN:\n'
            '    import json as _json\n'
            '    from tools.segfly_clean_manifest import (census as _census,\n'
            '                                             report as _report,\n'
            '                                             write_manifests as '
            '_write_manifests)\n'
            '    _seg_found = sorted(Path("/content/data/SegFly_temiz")'
            '.rglob("index.json"))\n'
            '    assert _seg_found, ("SEGFLY_CLEAN is on and no index.json was '
            'unpacked under "\n'
            '                        "/content/data/SegFly_temiz -- check that '
            'SegFly_temiz.zip "\n'
            '                        "reached Drive and that cell 2 staged '
            'it.")\n'
            '    _seg_root = _seg_found[0].parent\n'
            '    _seg_index = _json.loads(_seg_found[0]'
            '.read_text(encoding="utf-8"))\n'
            '    print("SegFly, hand-audited:")\n'
            '    for _line in _report(_census(_seg_index)):\n'
            '        print("   ", _line)\n'
            '    for _line in _write_manifests(_seg_root, _seg_index, '
            'SEGFLY_DROP):\n'
            '        print("   ", _line)\n'
            '    print()')

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
    # The panel calls the "before" arm stock, and it is not stock whenever
    # BASE_CHECKPOINT is set: `BASE_CKPT = BASE_CHECKPOINT` above, so red is
    # the pretrain's prediction and the top row reads "this run beat the
    # pretrain", not "this run beat EdgeTAM". Naming the file the arm actually
    # came from is the difference between reading the panel and misreading it.
    replace(notebook,
            'plt.suptitle(f"top row: stage B gained   |   '
            'bottom row: stage B lost   "\n'
            '             f"(prompt: {PANEL_PROMPT})", y=1.0)',
            '_BASE_NAME = Path(BASE_CKPT).stem\n'
            'plt.suptitle(f"top row: this run gained   |   '
            'bottom row: this run lost   "\n'
            '             f"(prompt: {PANEL_PROMPT}, base: {_BASE_NAME})", y=1.0)')
    replace(notebook,
            'print("yellow = the target\'s outline | green = only stage B found it | "\n'
            '      "red = only stock found it | blue = both agreed")',
            'print(f"yellow = the target\'s outline (this is the annotation; "\n'
            '      f"everything else is a prediction)\\n"\n'
            '      f"green = only this run found it | red = only the base '
            '({_BASE_NAME}) found it | "\n'
            '      f"blue = both agreed")')
    # A band is on the instance's longer side in source pixels, keyed like
    # CLASS_WEIGHTS -- `"pool/vtuav_thermal:car"`, most specific wins. It says
    # what MAX_AREA cannot: that gate is one fraction of the frame for every
    # source at once, and a frame here runs from 640x512 to 1920x1080, so 0.06
    # is 19 661 px in HIT-UAV and 124 416 px in VTUAV -- six times apart, for a
    # target the same size on the ground.
    replace(notebook, 'MAX_AREA       = 0.9',
            'MAX_AREA       = 0.2\nSIZE_BANDS     = {}\nSIZE_QUANTILES = {}')
    # 50 a side instead of 6. Six tiles show the extremes and nothing about
    # the shape of the tail, and the tail is where a regression that is not a
    # single broken frame lives. A 2-by-50 strip would be 155 inches wide, so
    # the two blocks are laid out as grids: gained on top, lost below.
    replace(notebook, 'PANEL_CASES    = 6',
            'PANEL_CASES    = 50\nPANEL_COLUMNS  = 10')
    replace(notebook,
            '_fig, _axes = plt.subplots(2, _half, figsize=(3.1 * _half, 7.0), '
            'squeeze=False)\n'
            'for _ax, _case in zip(_axes.ravel(), SHOWN):',
            'import math\n'
            '_cols = max(1, min(PANEL_COLUMNS, _half))\n'
            '_block = math.ceil(_half / _cols)\n'
            '_fig, _axes = plt.subplots(2 * _block, _cols,\n'
            '                           figsize=(3.1 * _cols, 3.5 * 2 * _block),\n'
            '                           squeeze=False)\n'
            'for _blank in _axes.ravel():\n'
            '    _blank.axis("off")\n'
            'for _n, _case in enumerate(SHOWN):\n'
            '    _side, _within = divmod(_n, max(_half, 1))\n'
            '    _ax = _axes[_side * _block + _within // _cols, _within % _cols]')

    # The report is printed whether or not a band applies, which is the same
    # rule `neighbour_weight` follows: the number that would justify a cut is
    # measured before anyone cuts on it. Applied after the thinning and before
    # the split, because that is the order `apply_splits` replays them in.
    replace(notebook,
            'SPLITS = split_index(INDEX, seed=SEED)',
            'if SIZE_QUANTILES:\n'
            '    _from_data = resolve_quantiles(INDEX, SIZE_QUANTILES)\n'
            '    for _key, (_lo, _hi) in sorted(_from_data.items()):\n'
            '        print(f"quantile band {_key}: "\n'
            '              f"{SIZE_QUANTILES[_key]} -> {_lo:.0f}..{_hi:.0f} px")\n'
            '    SIZE_BANDS = {**_from_data, **SIZE_BANDS}\n'
            'INDEX, _sizes = size_bands(INDEX, SIZE_BANDS, seed=SEED)\n'
            'print("\\ninstance size by source and class -- longer side in "\n'
            '      "source pixels; the grade calls under 32 small")\n'
            'print(f"{\'source:class\':<40}{\'n\':>8}{\'p10\':>7}{\'p50\':>7}"\n'
            '      f"{\'p90\':>7}{\'p99\':>7}{\'max\':>7}")\n'
            'for _key, _row in list(_sizes["sides"].items())[:18]:\n'
            '    print(f"{_key:<40}{_row[\'n\']:>8}{_row[\'p10\']:>7.0f}"\n'
            '          f"{_row[\'p50\']:>7.0f}{_row[\'p90\']:>7.0f}"\n'
            '          f"{_row[\'p99\']:>7.0f}{_row[\'max\']:>7.0f}")\n'
            'if SIZE_BANDS:\n'
            '    print(f"\\nsize bands: {_sizes[\'instances\'][\'before\']} instances "\n'
            '          f"-> {_sizes[\'instances\'][\'after\']}   dropped {_sizes[\'dropped\']}")\n'
            '    assert not _sizes["unmatched"], (\n'
            '        f"SIZE_BANDS names {_sizes[\'unmatched\']} and nothing matched "\n'
            '        f"any of them. A key is a class, a source, or `source:class` "\n'
            '        f"-- read the table above and use those names.")\n'
            'else:\n'
            '    print("\\nSIZE_BANDS is empty, so nothing was dropped by size. "\n'
            '          "The table above is what a band gets chosen from.")\n'
            'SPLITS = split_index(INDEX, seed=SEED)')

    replace(notebook,
            'PROMPT_JITTER  = 0.3\nMETHOD',
            'PROMPT_JITTER  = 0.3\n'
            # The aggressive setting -- collapse 0.40 to 0.15x, noise 5.0 --
            # was chosen on a synthetic window whose target sat at a
            # signal-to-clutter ratio of 20. HIT-UAV's real targets sit at a
            # median of 0.91, where there is no easy end to collapse towards:
            # measured, it moved the median 0.85 -> 0.79 while spending dynamic
            # range everywhere. These are the numbers that measurement chose,
            # and they are `HARDER`'s, so this arm and the pretrain it may be
            # stacked on see the same augmentation rather than two strengths.
            'CONTRAST_COLLAPSE = 0.25\n'
            'CONTRAST_FLOOR    = 0.15\n'
            'POLARITY_FLIP     = 0.25\n'
            'GAMMA_JITTER      = 0.25\n'
            'SENSOR_NOISE      = 2.0\n'
            'METHOD')
    replace(notebook, 'EPOCHS         = [2, 24]', 'EPOCHS         = [2, 30]')
    replace(notebook, 'STEPS          = 800', 'STEPS          = 500')
    replace(notebook, 'PATIENCE       = 4', 'PATIENCE       = 10')
    replace(notebook, 'VAL_BATCHES    = 24', 'VAL_BATCHES    = 32')
    replace(notebook, 'BATCH_CEILING  = 512', 'BATCH_CEILING  = 128')
    replace(notebook, 'BATCH_RESERVE  = 0.12', 'BATCH_RESERVE  = 0.15')
    replace(notebook,
            'LR_REFERENCE   = 16\nLR_SCALE_MAX   = 4.0\n'
            'LR_HEAD        = 0\nLR_NECK        = 1e-4\nLR_TRUNK       = 1e-4',
            'LR_SCALE       = 1.0\n'
            'RUN_LR_PILOT   = True\n'
            'LR_PILOT_STEPS = 150\n'
            # `thermal` (trunk 2e-5) is not here any more, and it was
            # measured out rather than argued out. The 12-epoch run
            # this notebook already took picked it in the pilot and
            # then came apart on it: encoder validation 0.1968 ->
            # 0.1944 -> 0.1949 -> 0.1908 (best, epoch 6) -> 0.1929 ->
            # 0.1958 -> 0.2721 -> 0.2261 -> 0.2155, never recovering.
            # A 150-step pilot cannot see an epoch-9 divergence, so
            # keeping the rung as a candidate for a 30-epoch budget is
            # buying the same failure at 2.5x the cost. `gentle` is the
            # new bottom rung for the same reason.
            'LR_PROFILES = {\n'
            '    "gentle":   (1e-5, 1e-5, 2e-6),\n'
            '    "cautious": (2e-5, 2e-5, 5e-6),\n'
            '    "stable":   (5e-5, 5e-5, 1e-5),\n'
            '}\n'
            'LR_HEAD, LR_NECK, LR_TRUNK = LR_PROFILES["stable"]')
    replace(notebook, 'NOTEBOOK = "22_thermal_deep.ipynb"',
            'NOTEBOOK = "32_aerial_thermal_stage_b_stable.ipynb"')
    replace(notebook, 'STAMP    = "029a14dfb8"', 'STAMP    = "{{STAMP}}"')

    # The hardening flags belong to the *training* call and to nothing else.
    # `COMMON` is also spread into every `tools/eval_instances.py` invocation,
    # which does not define them and parses strictly -- so putting them there
    # made cell 4 die with `unrecognized arguments`, after the whole download
    # and re-index had already been paid for. `HARDEN` is added to the two
    # `train_encoder.py` calls instead, which is what
    # `tools/build_stage_b_notebooks.py` already does for the generated family.
    # One working directory per run, decided after RUN has taken its `_from_`
    # suffix. `/content/work` is shared by every arm, and `score_to` returns a
    # cached `score_<tag>_<prompt>.json` if it exists -- so a second arm's
    # "before" would silently be the first arm's baseline, measured against a
    # different base checkpoint. With four arms (stock, 34, 36-grn, 36-plain)
    # that is the comparison itself being wrong rather than one number.
    replace(notebook,
            'assert METHOD in ("finetune", "lora"), '
            'f"METHOD is finetune or lora, not {METHOD!r}"',
            'assert METHOD in ("finetune", "lora"), '
            'f"METHOD is finetune or lora, not {METHOD!r}"\n'
            'WORK = f"/content/work_{RUN}"\n'
            'print("work dir", WORK, "-- named after the run so two arms do not "\n'
            '      "share splits.json or the score cache; a shared score cache "\n'
            '      "hands the second arm the first one\'s before/after.")')
    # "WHAT THIS RUN TRAINED ON" printed train+val+test. The run trained on
    # `train` -- 67199 frames where the header said 79687 -- and a summary that
    # overstates its own training set by a fifth is the one line a reader
    # quotes.
    replace(notebook,
            'print(f"  {sum(len(v) for v in SPLITS.values())} frames, "',
            'print(f"  {len(SPLITS[\'train\'])} train frames "\n'
            '      f"({sum(len(v) for v in SPLITS.values())} indexed, the rest '
            'held out), "')
    # A dead `source:class` key looked identical to a misspelling. It is not:
    # SegFly's spec does carry `truck` (id 36), so `segfly:truck` matching
    # nothing means the gates already removed those instances -- they are a
    # ninth of a vehicle's area, and MIN_AREA cuts them before rebalance ever
    # sees them. Printing what the source *did* carry answers that in the log
    # instead of leaving it to a spec read.
    replace(notebook,
            'f"misspelt class looks exactly like a class that is already rare.")',
            'f"misspelt class looks exactly like a class that is already rare.")\n'
            '        for _key in _balance["unmatched"]:\n'
            '            _src = _key.split(":")[0] if ":" in _key else _key\n'
            '            _has = sorted({_spec.name_of(_i.class_id)\n'
            '                           for _e in INDEX\n'
            '                           for _spec in [_e.source.spec] if _e.source\n'
            '                           and _e.source.name.split("/")[-1] == _src\n'
            '                           for _i in _e.instances})\n'
            '            print(f"      {_key}: {_src} carries {_has or \'nothing here\'}")')

    # The epoch table went to the kernel's stdout, not the notebook: the run is
    # a subprocess. Reading it back out of run.json is the difference between a
    # run whose curve can be read and one whose only number is its last.
    replace(notebook,
            'print("wall clock", round((time.time() - _started) / 60, 1), "min")',
            'for _row in RUN_LOG.get("history", []):\n'
            '    print(f"  {_row.get(\'stage\',\'\'):<8} epoch {_row.get(\'epoch\',\'?\'):>3}  "\n'
            '          f"train {_row.get(\'train_loss\', float(\'nan\')):.4f}  "\n'
            '          f"val {_row.get(\'val_loss\', float(\'nan\')):.4f}"\n'
            '          + ("  <- best" if _row.get("saved") else ""))\n'
            'if RUN_LOG.get("stopped_early"):\n'
            '    print("stopped early:", RUN_LOG["stopped_early"])\n'
            'print("wall clock", round((time.time() - _started) / 60, 1), "min")')

    # The whole by-class table, not its first twelve rows. The tail is where a
    # class that was already rare lives, which is the half a reader needs.
    replace(notebook,
            'for _name, (_was, _now) in list(_balance["by_class"].items())[:12]:',
            'for _name, (_was, _now) in _balance["by_class"].items():')

    # Val had no source table while train and test did -- and val is what the
    # LR pilot and the checkpoint choice are decided on.
    replace(notebook,
            'for _source, _count in TEST.sources.items():',
            'for _source, _count in VAL.sources.items():\n'
            '    print(f"  val   {_source:<34}{_count:>8}")\n'
            'for _source, _count in TEST.sources.items():')
    # `save_splits` records only which *frames* survived, so the per-instance
    # thinning `rebalance` did two lines earlier was undone the moment
    # train_encoder re-indexed: measured on a synthetic pool, a train split the
    # notebook cut to 476 instances came back as 1500. Handing it the weights
    # and the seed lets it re-apply the same decision downstream.
    replace(notebook,
            'SPLIT_FILE = str(save_splits(Path(WORK) / "splits.json", SPLITS))',
            'SPLIT_FILE = str(save_splits(Path(WORK) / "splits.json", SPLITS,\n'
            '                             CLASS_WEIGHTS, seed=SEED,\n'
            '                             bands=SIZE_BANDS))')
    replace(notebook,
            'COMMON += ["--splits", SPLIT_FILE]',
            'COMMON += ["--splits", SPLIT_FILE]\n'
            'HARDEN = ["--contrast-collapse", str(CONTRAST_COLLAPSE),\n'
            '          "--contrast-floor", str(CONTRAST_FLOOR),\n'
            '          "--polarity-flip", str(POLARITY_FLIP),\n'
            '          "--gamma-jitter", str(GAMMA_JITTER),\n'
            '          "--sensor-noise", str(SENSOR_NOISE)]')
    # And into the long run itself. Without this the hardening this notebook
    # exists to add would be configured, printed, and never applied.
    replace(notebook,
            '     "--jitter", str(JITTER), "--batch", str(BATCH), "--accum", str(ACCUM),\n'
            '     "--lr-scale", str(LR_SCALE), "--steps", str(STEPS),',
            '     "--jitter", str(JITTER), "--batch", str(BATCH), "--accum", str(ACCUM),\n'
            '     *HARDEN,\n'
            '     "--lr-scale", str(LR_SCALE), "--steps", str(STEPS),')
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
        `STEPS × batch` örneklemesidir. Tam encoder koşusu `30 × 500 = 15000`
        optimizer adımıdır. Bütçe 12'den 30'a çıktı çünkü OneCycle inişi bu
        toplama göre kuruluyor: daha uzun bütçe, daha yavaş inen bir LR ve
        kararsız bir encoder'ın ihtiyacı olan şey tam olarak budur.

        `PATIENCE=10` bir ayar knob'u değil, emniyet ağıdır. Erken durmak LR'yi
        yüksek ve inişi yarım bırakır, o yüzden eşik bütçenin üçte biri: gerçek
        bir plato kesilir, normal dalgalanma kesilmez.

        Pilot üç profili aynı split, aynı augmentasyon ve aynı seed ile
        `1 head + 2 encoder`, epoch başına 150 adımda sınar. `gentle` en alt
        basamaktır (trunk 2e-6) ve kararsızlık için oradadır.

        `thermal` (trunk 2e-5) listede yok, çünkü **ölçüldü**: bu notebook'un
        12 epoch'luk koşusu pilotta onu seçti ve encoder validation'ı
        0.1908'de (epoch 6) dip yapıp epoch 9'da **0.2721**'e fırladı, 12'ye
        kadar da toparlamadı. 150 adımlık bir pilot epoch 9'daki bir
        dağılmayı göremez; 30 epoch'luk bütçede o basamağı aday tutmak aynı
        arızayı 2.5 kat pahalıya almaktır.

        Pilot sonucunu yine de `lr_pilot.json`'dan okuyun.
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
                     *HARDEN,
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

    # Both training calls -- the pilot's and the full run's -- take the flag,
    # and the pilot cell is inserted above, so this runs once both exist.
    replace(notebook,
            '"--anchor-weight", str(ANCHOR_WEIGHT), "--device", "cuda",',
            '"--anchor-weight", str(ANCHOR_WEIGHT),\n'
            '     "--neighbour-weight", str(NEIGHBOUR_WEIGHT), "--device", "cuda",',
            count=2)

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
