#!/usr/bin/env python3
"""Track every recorded frame folder several ways and collect the comparison.

`frames/<record>/` holds one recording as an image sequence. This runs each of
them through the TensorRT backend in each input configuration and writes the
results into `frame_output/<record>/<mode>/`:

  full1024   whole frame resized to 1024x1024   the reference configuration
  crop1024   centred 1024x1024 window           native pixels, cropped scene
  full768    whole frame resized to 768x768     56% of the tokens, whole scene
  crop768    centred 768x768 window             native pixels, cropped scene
  full512    whole frame resized to 512x512     4x fewer tokens, whole scene
  crop512    centred 512x512 window             native pixels, cropped scene

The crop modes are what a camera that already centres and focuses on its
target can offer: the object is in the middle, so the outer frame is mostly
background being paid for twice -- once in the resize, once in the model.
Cropping to exactly the model input removes the resize step entirely, so it is
a speed experiment as much as an accuracy one.

Note the crop is clamped to the frame, so a source shorter than the crop is
cropped on one axis and still resized on the other: at 1280x720, `crop1024`
takes 1024x720 and the model still stretches it vertically. The summary
reports the window each run actually used.

These are not equivalent inputs and the comparison is not only FPS: 512 sees
the whole scene at half the detail, a crop sees part of the scene at full
detail. If the target leaves the centre window, the crop modes lose it -- that
is the trade being measured, and the mp4 is there to watch it happen.

768 is the midpoint, and it only earns its place on sources that carry more
than 512x512 of real detail -- on a 640x512 thermal frame it upsamples and pays
2.25x the 512 token count for interpolated pixels. See configs/edgetam_768.yaml.
Each mode needs its own engine set; `--modes` runs the subset you have built.

`--weights` is the second axis, and it is a different question from the modes:
they vary the *input*, it varies the *model*. `stock` is EdgeTAM as shipped;
`pool_deep` is notebooks/22_thermal_deep.ipynb's checkpoint, trained at 512 on
the thermal mask pools. Weights are traced into the ONNX graphs and baked into
the engines, so this selects a different models*/ directory rather than a
runtime setting -- each (weights, size) pair is its own export, and a pair with
no engines built fails immediately with the two commands that build them.

Results go to `<mode>_<weights>/`, so a stock run and a pool_deep one sit side
by side off one saved prompt: same target, same frames, one variable. A plain
stock run keeps the bare `<mode>/` name.

Prompts are picked once per record, on the full frame, and reused by every
mode (crop modes shift them automatically). They come from, in order:
`<record>/prompts.json`, a previous pick saved under the output folder,
--box, or an interactive window.

Usage:
    python tools/run_records.py --records frames --out frame_output
    python tools/run_records.py --records frames --out frame_output \\
        --prompt point --multi
    python tools/run_records.py --records frames --out frame_output \\
        --modes crop512 --box 700,300,830,430

    # Same records, half the inferences: results land in <mode>_skip2/ next to
    # the baseline, off the same saved prompt, so the two are comparable.
    python tools/run_records.py --records frames --out frame_output \\
        --modes full1024 --frame-skip

    # The thermal stage-B checkpoint against stock, same records, same prompt.
    # Pick once, then run the two: full512/ and full512_pool_deep/ end up side
    # by side under each record.
    python tools/run_records.py --records frames --out frame_output \\
        --modes full512,crop512,full768 --box 700,300,830,430 --pick-only
    python tools/run_records.py --records frames --out frame_output \\
        --modes full512,crop512,full768 --weights stock
    python tools/run_records.py --records frames --out frame_output \\
        --modes full512,crop512,full768 --weights pool_deep

`--policy` is the third axis, and the one that needs no training and no export.
The modes vary the input and `--weights` varies the model; a policy varies
neither -- it layers `samurai:` / `ego_motion:` / `guard:` onto whichever
backend YAML the pair chose (configs/policies/, merged into `config.yaml`
beside the run). So it measures what the weights already on disk can be made to
do, which is the one question that does not have to wait for a run to finish:

    # The ladder, on one checkpoint, one prompt: each row differs from the one
    # above it by exactly one thing.
    python tools/run_records.py --records frames --out frame_output \\
        --modes full512 --weights pool_deep --box 700,300,830,430 --pick-only
    for policy in plain samurai ego guard; do
        python tools/run_records.py --records frames --out frame_output \\
            --modes full512 --weights pool_deep --policy $policy
    done
    # -> full512_pool_deep/  _samurai/  _ego/  _guard/  under each record.

    # Or just the two ends, if the question is only "is the stack worth it".
    python tools/run_records.py --records frames --out frame_output \\
        --modes full512 --weights pool_deep --policy guard

Read `--policy guard`'s dropout count expecting it to go **up** where the
tracker was previously reporting a mask covering a field: the guard refuses
those and reports an empty mask, which `accuracy.dropout_episodes` counts as
the lost frame it always was. A guard run whose dropouts did not move is a
guard that never fired, not a guard that helped.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from tools.run_experiment import PY, find, run_step  # noqa: E402

# `src.io_utils` and `src.prompts` reach cv2 and the interactive picker, which
# only the prompt step needs. Imported there rather than here so `--help`, the
# mode/weights tables and the engine preflight work on a machine that has not
# installed the vision stack -- and so a missing display is an error about
# picking a target, not an import failure before the tool has read its flags.

# mode -> (model input size, centre crop size or None). A mode is an *input*
# configuration and nothing else -- which weights run inside it is the separate
# --weights axis below, because the whole point of the crop/full comparison is
# that it holds everything except the input fixed.
MODES = {
    "full1024": (1024, None),
    "crop1024": (1024, 1024),
    "full768": (768, None),
    "crop768": (768, 768),
    "full512": (512, None),
    "crop512": (512, 512),
}

# --weights -> {model input size: backend YAML}. Each entry is a set of
# engines, and engines are both shape- and weight-specific: a checkpoint is
# traced into the ONNX graphs and baked into what trtexec builds, so every
# (weights, size) pair is its own directory and its own export.
WEIGHTS = {
    "stock": {
        1024: "configs/edgetam_trt.yaml",
        768: "configs/edgetam_trt_768.yaml",
        512: "configs/edgetam_trt_512.yaml",
    },
    # notebooks/22_thermal_deep.ipynb, trained at 512 on the thermal mask pools.
    # There is deliberately no 1024 entry: the checkpoint would be running two
    # doublings off its training resolution and the number would measure the
    # resolution, not the training. 768 is already off it -- see
    # configs/edgetam_768_pool_deep.yaml, which says so in its own header.
    "pool_deep": {
        768: "configs/edgetam_trt_768_pool_deep.yaml",
        512: "configs/edgetam_trt_512_pool_deep.yaml",
    },
}

# The size each weights set was actually trained at, for the note the summary
# carries. None means "not one of ours" -- stock EdgeTAM was trained at 1024
# before this repository existed and records nothing about it.
TRAINED_AT = {"stock": 1024, "pool_deep": 512}

# --policy -> the overlay merged onto whichever backend YAML the (weights,
# size) pair chose, or None for the baseline. This is the third axis and it is
# orthogonal to the other two: `--modes` varies the input, `--weights` varies
# the model, and nothing here varies either. No engine changes shape, no
# checkpoint is touched, nothing is re-exported -- everything in
# configs/policies/ is inference-time bookkeeping this project deliberately
# left in PyTorch, which is exactly why it can be measured on the weights
# already on disk instead of waiting for a training run.
#
# They are overlays rather than whole configs because whole ones would be
# sizes x weights x policies copies of the same engine paths, and the day one
# of those paths moved most of them would be quietly stale.
POLICIES = {
    "plain": None,
    "samurai": "configs/policies/samurai.yaml",
    "ego": "configs/policies/ego.yaml",
    "guard": "configs/policies/guard.yaml",
}

# The blocks an overlay is allowed to set. A policy states runtime behaviour;
# a checkpoint or an engine path in one of these files would be that policy
# quietly changing what `--weights` means, so it is refused rather than merged.
POLICY_KEYS = ("samurai", "sam2long", "ego_motion", "guard")

# What the summary says about a policy run, so a table read later explains
# itself. Each names the rung below it, because that is the row it has to be
# read against.
POLICY_NOTE = {
    "samurai": (
        "Policy: `samurai` — motion-aware memory. A Kalman filter re-scores the "
        "three candidate masks and a frame enters the memory bank only if IoU, "
        "object score and motion agree it was a good one. **Training-free**: the "
        "checkpoint and the engines are the `plain` run's. Read against `plain`."
    ),
    "ego": (
        "Policy: `ego` — `samurai`, plus the background's measured displacement "
        "handed to that filter as a control input. On HIT-UAV's real drone "
        "footage the camera moves a median 0.9-8.6 px a frame, which on a "
        "27-pixel target is a third of its own width of apparent motion that is "
        "not the target's. **Training-free.** Read against the `samurai` row: "
        "the difference is the camera term and nothing else."
    ),
    "guard": (
        "Policy: `guard` — `ego`, plus the classical layer with no weights in "
        "it: area, aspect and travel plausibility once the camera's motion is "
        "out, hysteresis, and template re-acquisition. A refused mask is "
        "reported **empty**, never replaced by the guard's own prediction, so "
        "`dropout_episodes` counts a lost frame honestly instead of scoring a "
        "mask that covers a field as a hit. Expect dropouts to go **up** where "
        "they were being hidden. **Training-free.** Read against the `ego` row."
    ),
}


def config_for(weights: str, size: int, mode: str) -> str:
    """The YAML for one (weights, input size) pair, or a refusal that explains.

    A missing pair is not an oversight to fall back from: silently running the
    stock config here would put a row labelled `pool_deep` in the summary that
    was measured on different weights.
    """
    table = WEIGHTS[weights]
    if size in table:
        return table[size]
    have = ", ".join(str(k) for k in sorted(table))
    raise SystemExit(
        f"--weights {weights} has no {size} configuration, so mode {mode!r} "
        f"cannot run under it (it has {have}). Drop that mode, or run it "
        f"under --weights stock."
    )


def overlay_for(policy: str) -> dict:
    """The policy blocks named by `--policy`, or `{}` for the baseline.

    Refuses an overlay that reaches outside `POLICY_KEYS`: a `checkpoint:` in
    one of these files would make `--policy` a second way to change the weights,
    and the summary's `--weights` column would stop being true.
    """
    import yaml

    path = POLICIES[policy]
    if path is None:
        return {}
    body = yaml.safe_load((ROOT / path).read_text()) or {}
    stray = sorted(set(body) - set(POLICY_KEYS))
    if stray:
        raise SystemExit(
            f"{path} sets {', '.join(stray)}, which is not a policy. An overlay "
            f"may only set {', '.join(POLICY_KEYS)} -- anything else belongs in "
            f"the backend config the --weights axis selects."
        )
    return body


def staged_config(config: str, policy: str, outdir: Path) -> str:
    """The config the run actually reads: the backend YAML, plus the overlay.

    For the baseline this is the backend YAML itself, unchanged, so a `plain`
    run reads exactly the file earlier runs read. With a policy the merge is
    written to `<outdir>/config.yaml` and *that* is what cli.py is pointed at,
    which is the decisive record: the question asked later is not "which
    overlay was named" but "what did this folder run", and the answer then sits
    beside the video rather than in two files that have to be re-merged.

    The merge is per block, not per key. A policy is a complete statement of
    one -- half of `guard:` from a file and half from a backend config would be
    a setting nobody wrote.

    Relative `checkpoint:`/`*_engine:` paths are left alone: cli.py resolves
    them against the repository root, not against the config's own directory,
    so the copy works from wherever it is written.
    """
    import yaml

    blocks = overlay_for(policy)
    if not blocks:
        return config
    body = yaml.safe_load((ROOT / config).read_text()) or {}
    body.update(blocks)
    staged = outdir / "config.yaml"
    staged.write_text(
        f"# Generated by tools/run_records.py: {config}\n"
        f"# + configs/policies/{policy}.yaml (--policy {policy}).\n"
        f"# Edit those two, not this -- a re-run overwrites it.\n"
        + yaml.safe_dump(body, sort_keys=False)
    )
    return str(staged)


def cache_for(record: Path, mode: str, args) -> Path | None:
    """Where one record's decoded frames are staged, or None for the default.

    Keyed by what actually changes the bytes on disk rather than by the mode.
    The pipeline crops in the pass that writes the cache, so the cache *is* the
    cropped view: every `full*` mode stages the same source frames -- the
    resize to `image_size` happens in the model, not here -- while `crop768`
    and `crop512` stage different pixels. Sharing by crop is a space saving
    rather than a time one: the staging pass rewrites its directory on every
    run, so three full modes keep one copy of the frames between them instead
    of three, and none of them skips the transcode.

    The frame skip is in the key too, and that one is a correctness matter
    rather than a saving: a skipped run writes fewer files, the tracker indexes
    whatever the directory holds, and a leftover tail from a longer run would
    be read as extra frames.
    """
    if not args.cache_dir:
        return None
    crop = MODES[mode][1]
    parts = ["full" if crop is None else f"crop{crop}"]
    if args.frame_skip > 1:
        parts.append(f"skip{args.frame_skip}")
    return Path(args.cache_dir) / record.name / "_".join(parts)


def engines_missing(config: str) -> str | None:
    """The engine directory a config expects, when it is not there yet.

    Checked before anything is loaded: a missing engine set otherwise surfaces
    after a model build and a frame index, several minutes in.
    """
    import yaml

    body = yaml.safe_load((ROOT / config).read_text()) or {}
    engine = body.get("image_encoder_engine")
    if not engine or (ROOT / engine).exists():
        return None
    return str(Path(engine).parent)


def digest(path: Path) -> dict | None:
    """A file's size and sha256, or None when it is not there.

    The hash is what makes the record decisive. A path proves which config was
    read; only the content proves which engine ran, and two engine sets built
    from different checkpoints are otherwise indistinguishable -- same module
    names, same shapes, same everything a log could print.
    """
    import hashlib

    if not path.is_file():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hasher.hexdigest()}


def provenance(mode: str, config: str, crop: int | None, args, cmd) -> dict:
    """Everything needed to answer "which weights produced this folder".

    Written beside the video and the charts, because that question is asked
    later -- when the mp4s are being compared and the terminal that ran them is
    gone.
    """
    body = yaml.safe_load((ROOT / config).read_text()) or {}
    engines = {
        name.removesuffix("_engine"): digest(ROOT / body[name])
        for name in ("image_encoder_engine", "memory_attention_engine",
                     "memory_encoder_engine", "sam_head_engine")
        if body.get(name)
    }
    return {
        "weights": args.weights,
        "trained_at": TRAINED_AT.get(args.weights),
        # The policy by name and by value. The name alone would not survive an
        # edit to configs/policies/, and this file is read months later to
        # explain a number.
        "policy": args.policy,
        "policy_blocks": overlay_for(args.policy),
        "mode": mode,
        "image_size": int(body.get("image_size", 1024)),
        "center_crop": crop,
        "config": config,
        "checkpoint": digest(ROOT / body["checkpoint"]) if body.get("checkpoint") else None,
        "engines": engines,
        "command": [str(c) for c in cmd],
    }


def folder(mode: str, args) -> str:
    """The output folder for one run: `<mode>[_<weights>][_<policy>][_skipN]`.

    A plain `stock` run at every frame keeps the bare `<mode>` name, so results
    written before there was more than one axis stay where they are and a
    re-run lands on top of them rather than beside them. The same reasoning
    puts the policy in the name: `full512_pool_deep` and
    `full512_pool_deep_guard` are two measurements of one checkpoint, not a
    result and its replacement, and the whole point of the ladder is reading
    them side by side.
    """
    parts = [mode]
    if args.weights != "stock":
        parts.append(args.weights)
    if args.policy != "plain":
        parts.append(args.policy)
    if args.frame_skip > 1:
        parts.append(f"skip{args.frame_skip}")
    return "_".join(parts)


def resolve_prompts(record: Path, outdir: Path, args) -> Path | None:
    """One selection per record, in full-frame coordinates, reused by every mode.

    Picking interactively needs a display, so every non-interactive source is
    tried first and the pick is saved -- a re-run, or a second mode, never asks
    again.
    """
    from src.io_utils import read_first_frame_dir
    from src.prompts import BoxPrompt, PromptSet
    from src.prompts.file_source import save_prompts

    own = record / "prompts.json"
    if own.exists():
        return own
    saved = outdir / "prompts.json"
    if saved.exists() and not args.repick:
        print(f">> {record.name}: reusing {saved}")
        return saved
    if args.box:
        xyxy = tuple(float(v) for v in args.box.split(","))
        return save_prompts(PromptSet(boxes=[BoxPrompt(1, 0, xyxy)]), saved)

    from src.prompts import interactive

    first = read_first_frame_dir(record, args.pattern)
    print(f">> {record.name}: select the target ({args.pattern} frame 0)")
    if args.prompt == "box":
        picked = interactive.pick_boxes(first) if args.multi else interactive.pick_box(first)
    else:
        picked = (interactive.pick_points_multi(first) if args.multi
                  else interactive.pick_points(first))
    return save_prompts(picked, saved)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--records", default="frames",
                   help="Directory of record folders, each an image sequence.")
    p.add_argument("--out", default="frame_output", help="Where results go.")
    p.add_argument("--modes", default=",".join(MODES),
                   help=f"Comma-separated subset of {', '.join(MODES)}.")
    p.add_argument("--policy", default="plain", choices=tuple(POLICIES),
                   help="Inference-time policy layered onto the backend config: "
                        "samurai (motion-aware memory), ego (that filter given "
                        "the camera's own displacement), guard (ego plus the "
                        "classical plausibility and re-acquisition layer). None "
                        "of them trains anything or rebuilds an engine, so each "
                        "runs on the weights already on disk. Results land in "
                        "<mode>_<weights>_<policy>/ beside the baseline.")
    p.add_argument("--weights", default="stock", choices=tuple(WEIGHTS),
                   help="Which trained weights the modes run on. `stock` is "
                        "EdgeTAM as shipped; `pool_deep` is the checkpoint from "
                        "notebooks/22_thermal_deep.ipynb. Each is a separate set "
                        "of engines -- weights are baked into an engine at export "
                        "time, so this picks a different models*/ directory, not "
                        "a different runtime setting. Results land in "
                        "<mode>_<weights>/ so a stock run is never overwritten "
                        "and the two share one saved prompt.")
    p.add_argument("--pattern", default="*.tif*", help="Frame glob inside a record.")
    p.add_argument("--cache-dir", default=None,
                   help="Where the decoded frames are staged, per record. The "
                        "default is a system temp directory, which is the wrong "
                        "disk whenever the records are on an external drive: "
                        "every frame is written once as JPG before tracking "
                        "starts. Point this at that drive. Each record gets a "
                        "subdirectory per distinct staged view, so the full "
                        "modes share one copy of the frames rather than keeping "
                        "three; the staging pass still runs per mode, so this is "
                        "disk space and the right drive, not saved time. What is "
                        "written here is kept after a run and is yours to "
                        "reclaim.")
    p.add_argument("--fps", type=float, default=30.0,
                   help="Playback fps for the output mp4 (a sequence has none).")
    p.add_argument("--warmup", type=int, default=20,
                   help="Frames excluded from every reported statistic.")
    p.add_argument("--frame-skip", "--frameskip", dest="frame_skip", type=int,
                   nargs="?", const=2, default=1, metavar="N",
                   help="Infer one frame in N and hold that mask over the rest "
                        "(bare flag = 2). The clip still comes out at full length; "
                        "what halves is the inferences behind it, so each one gets N "
                        "frame periods instead of one. Results go to <mode>_skipN/ so "
                        "a baseline run is never overwritten, and the saved prompt is "
                        "shared with it -- same target, same frames, one variable.")
    p.add_argument("--prompt", choices=("box", "point"), default="box",
                   help="How to select the target interactively, when no "
                        "prompts.json and no --box is available. Needs a display.")
    p.add_argument("--multi", action="store_true",
                   help="Select several targets: multiple boxes, or points with "
                        "'n' to start the next object.")
    p.add_argument("--box", default=None,
                   help="x1,y1,x2,y2 in full-frame coordinates, applied to every "
                        "record. Skips the interactive window entirely.")
    p.add_argument("--repick", action="store_true",
                   help="Select again, replacing a pick saved by an earlier run.")
    p.add_argument("--pick-only", action="store_true",
                   help="Select each record's target, save it, and stop without "
                        "tracking. Use this when you want to run the modes "
                        "yourself, one command at a time, against one selection "
                        "-- picking separately per command would give each mode a "
                        "different box and make them incomparable.")
    p.add_argument("--strict", action="store_true",
                   help="Refuse a silent PyTorch fallback. Without it a missing "
                        "or unloadable engine prints one line and the mode "
                        "carries on in PyTorch -- the right weights on the "
                        "wrong backend, which the summary's FPS column cannot "
                        "distinguish from a TensorRT run. Pass this whenever "
                        "the numbers are going into a report.")
    p.add_argument("--no-video", action="store_true",
                   help="Measure only. The overlay and the mp4 are already "
                        "outside the reported frame budget, but they still run "
                        "on the same CPU and memory bus; drop them for a "
                        "measurement with nothing else competing.")
    args = p.parse_args(argv)

    root = Path(args.records)
    if not root.is_dir():
        raise SystemExit(f"{root} is not a directory.")
    # A folder holding frames is one record; a folder holding folders is a set
    # of them -- so --records can name the whole set or a single clip.
    records = [root] if list(root.glob(args.pattern)) else \
        sorted(d for d in root.iterdir() if d.is_dir())
    if not records:
        raise SystemExit(f"No frames matching {args.pattern!r} in {root}/, and no "
                         "record folders inside it either.")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in MODES:
            raise SystemExit(f"Unknown mode {m!r}; pick from {', '.join(MODES)}")
    # Resolve every config up front, so an unbuildable combination fails now
    # rather than after the first record has been indexed and prompted for.
    configs = {m: config_for(args.weights, MODES[m][0], m) for m in modes}
    absent = {m: d for m, d in ((m, engines_missing(c)) for m, c in configs.items()) if d}
    if absent:
        lines = [f"  {m:<10} needs {d}/" for m, d in sorted(absent.items())]
        sizes = sorted({MODES[m][0] for m in absent})
        builds = []
        for size in sizes:
            body = yaml.safe_load((ROOT / config_for(args.weights, size, "")).read_text())
            builds += [
                f"  python tools/export_edgetam_onnx.py --outdir {Path(body['image_encoder_engine']).parent}/ \\",
                f"      --image-size {size} --checkpoint {body['checkpoint']} --verify",
                f"  python tools/build_trt_engines.py --outdir {Path(body['image_encoder_engine']).parent}/ --max-batch 4",
            ]
        raise SystemExit(
            f"These modes have no engines built for --weights {args.weights}:\n"
            + "\n".join(lines)
            + "\n\nBuild them (engines are weight- and shape-specific, so each "
              "pair is its own export):\n" + "\n".join(builds)
        )

    out_root = Path(args.out)
    rows: dict[str, dict[str, dict]] = {}
    failed: list[str] = []

    for record in records:
        rows[record.name] = {}
        try:
            prompts = resolve_prompts(record, out_root / record.name, args)
        except Exception as exc:  # no display, nothing selected, unreadable frame
            print(f"!! {record.name}: no prompts ({exc}); skipped. Pass --box to "
                  "run without a display.")
            failed += [f"{record.name}/{m}" for m in modes]
            continue

        if args.pick_only:
            print(f">> {record.name}: {prompts}")
            continue

        for mode in modes:
            config, crop = configs[mode], MODES[mode][1]
            # A skipped run, and a run on other weights, are different
            # measurements of the same mode rather than replacements for it, so
            # each gets its own folder next to the baseline. `stock` with no
            # skip keeps the bare `<mode>/` name earlier runs already wrote.
            outdir = out_root / record.name / folder(mode, args)
            outdir.mkdir(parents=True, exist_ok=True)

            # After mkdir: a policy run writes the merged config into the
            # folder it is about to fill, so what ran and what it produced are
            # never in two places.
            run_config = staged_config(config, args.policy, outdir)

            cmd = [PY, "cli.py", "--tracker", "edgetam_trt", "--config", run_config,
                   "--frames-dir", record, "--frame-pattern", args.pattern,
                   "--fps", args.fps,
                   "--prompt", "file", "--prompt-file", prompts,
                   "--offload-video", "--fps-warmup", args.warmup,
                   "--fps-chart", outdir / "latency.png",
                   "--stage-chart", outdir / "stages.png"]
            cmd += ["--no-video"] if args.no_video else ["--output", outdir / "tracked.mp4"]
            cache = cache_for(record, mode, args)
            if cache is not None:
                cache.mkdir(parents=True, exist_ok=True)
                cmd += ["--frames-cache", cache]
            # The guard's own account of the run. Only a policy that carries a
            # `guard:` block produces any, and cli.py says so rather than
            # writing an empty file, so this is asked for whenever one might
            # exist: a refused frame is otherwise just a missing mask in the
            # mp4 with nothing saying which gate refused it.
            if "guard" in overlay_for(args.policy):
                cmd += ["--verdicts", outdir / "verdicts.json"]
            if args.strict:
                cmd += ["--strict"]
            if crop:
                cmd += ["--center-crop", crop]
            if args.frame_skip > 1:
                cmd += ["--frame-skip", args.frame_skip]

            (outdir / "provenance.json").write_text(
                json.dumps(provenance(mode, config, crop, args, cmd), indent=2) + "\n")
            ok, text = run_step(f"{record.name} — {mode}", cmd, outdir / "run.txt")
            if not ok:
                failed.append(f"{record.name}/{mode}")
            stages = re.search(
                r"median per frame: pre ([\d.]+) \+ inference ([\d.]+) \+ post ([\d.]+) ms",
                text,
            )
            rows[record.name][mode] = {
                "fps": find(r"avg ([\d.]+) FPS over", text) or "-",
                # Under a skip, FPS counts inferences; this is the frame rate
                # they cover, which is what has to clear the source's.
                "source_fps": find(r"\(([\d.]+) source FPS\)", text) or "-",
                "stages": "  /  ".join(stages.groups()) if stages else "-",
                "demo": find(r"overlay \+ mp4 encoding: ([\d.]+) ms/frame", text) or "-",
                # What the crop actually came out as: clamped to the frame, so
                # a source shorter than the crop still gets resized on that axis.
                "input": find(r"centre crop (\d+x\d+) at", text) or "whole frame",
            }

    if args.pick_only:
        return 1 if failed else 0

    # --------------------------------------------------------------- summary
    described = {
        "full1024": "| `full1024` | 1024x1024 | whole frame, resized |",
        "crop1024": "| `crop1024` | 1024x1024 | centred 1024x1024 window |",
        "full768": "| `full768` | 768x768 | whole frame, resized |",
        "crop768": "| `crop768` | 768x768 | centred 768x768 window |",
        "full512": "| `full512` | 512x512 | whole frame, resized |",
        "crop512": "| `crop512` | 512x512 | centred 512x512 window |",
    }
    skip = args.frame_skip
    title = (f"# Recorded clips — {len(records)} record(s), {len(modes)} mode(s), "
             f"weights `{args.weights}`")
    trained = TRAINED_AT.get(args.weights)
    off_size = sorted({MODES[m][0] for m in modes if MODES[m][0] != trained})
    lines = [
        title + (f", frame skip {skip}" if skip > 1 else ""),
        "",
        f"Weights: `{args.weights}` — "
        + (f"`{WEIGHTS[args.weights][MODES[modes[0]][0]]}` and its siblings, "
           f"trained at {trained}." if trained else "trained size unrecorded."),
        "",
        *((POLICY_NOTE[args.policy], "") if args.policy != "plain" else ()),
        "| mode | model | input |",
        "|---|---|---|",
        *(described[m] for m in modes),
        "",
        *(( [f"**{', '.join(str(s) for s in off_size)} is not the size these "
             f"weights were trained at ({trained}).** They load and run at any "
             "size -- EdgeTAM keeps no resolution in any parameter -- so nothing "
             "errors and the only symptom is a mask that is quietly worse. Read "
             f"those rows against a `--weights stock` run at the same size, never "
             f"against the {trained} row below.", ""] ) if off_size and trained else ()),
        f"`{args.warmup}` warm-up frames excluded from every number below. FPS is "
        "the real-time budget: per-frame decode + resize, the model, and masks "
        "back to source resolution. Drawing the overlay and encoding the mp4 are "
        "excluded from it and reported separately.",
        "",
    ]
    if skip > 1:
        lines += [
            f"Frame skip {skip}: one frame in {skip} is inferred and its mask held "
            f"over the rest, so the clip still comes out at full length on 1/{skip} "
            "of the inferences. FPS below counts inferences; source FPS is the frame "
            f"rate they cover. At {args.fps:g} fps in, each inference has "
            f"{1000.0 * skip / args.fps:.1f} ms instead of {1000.0 / args.fps:.1f} ms "
            "-- that threshold, not the average, is what this run is testing, so read "
            "the tail off `latency.png` rather than the median.",
            "",
        ]
    columns = ["mode", "model input from", "FPS"]
    if skip > 1:
        columns.append("source FPS")
    columns += ["median ms: pre / inference / post", "overlay + mp4 (excluded)"]
    for name, per_mode in rows.items():
        lines += [
            f"## {name}",
            "",
            "| " + " | ".join(columns) + " |",
            "|" + "---|" * len(columns),
        ]
        for mode in modes:
            r = per_mode.get(mode)
            if r is None:
                cells = ["-", "did not run"] + ["-"] * (len(columns) - 3)
            else:
                cells = [r["input"], r["fps"]]
                if skip > 1:
                    cells.append(r["source_fps"])
                cells += [r["stages"], f"{r['demo']} ms"]
            lines.append(f"| `{mode}` | " + " | ".join(cells) + " |")
        suffix = folder("", args).lstrip("_")
        lines += ["", f"Videos and charts: `{name}/<mode>"
                      f"{'_' + suffix if suffix else ''}/`", ""]

    if failed:
        lines += [f"> **Did not complete: {', '.join(failed)}** — see their `run.txt`.", ""]

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "SUMMARY.md").write_text("\n".join(lines))
    print(f"\n{'=' * 70}\n>> Everything is in {out_root}/\n{'=' * 70}")
    print((out_root / "SUMMARY.md").read_text())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
