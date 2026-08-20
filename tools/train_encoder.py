#!/usr/bin/env python3
"""Train EdgeTAM's image encoder on static aerial / RGB-T data.

Stage B of `docs/encoder_training_todo.md`. The dataset ships semantic
segmentation, which is the wrong target for a tracker, so the map is decomposed
into connected components per thing class and each component is prompted
individually -- SAM 2's own image-pretraining shape, applied to the data that
exists (`src/training/aerial.py` argues the whole case).

**One entry point for both methods, because that is the only way the comparison
between them means anything.** The frames, the split, the instance index, the
windows, the losses, the two stages, the schedule, the EMA and the validation
slice are shared code; `--method` swaps exactly two things:

    finetune   the selected weights receive gradients directly
    lora       the selected weights are frozen and gain `B @ A`; the
               checkpoint is merged before it is written

Both write a checkpoint in EdgeTAM's own `{"model": state_dict}` layout, so both
load into the same config, export to the same ONNX graph and build the same
engines. If they did not, the numbers would be comparing deployments rather
than training methods.

Usage:
    python tools/train_encoder.py --data /content/data/Kust4K --spec kust4k \\
        --out checkpoints/edgetam_aerial_512.pt --method finetune

    python tools/train_encoder.py ... --method lora --lora-r 16 \\
        --out checkpoints/edgetam_aerial_lora_512.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.aerial import (  # noqa: E402
    SPECS,
    InstanceGates,
    index_frames,
    list_frames,
    load_index,
    sample_windows,
    save_index,
    split_frames,
    summarise,
)
from src.training.finetune import Rates, apply_freeze, save_checkpoint  # noqa: E402
from src.training.schedule import Schedule, images, run_stages  # noqa: E402

INDEX_FILE = "instances.json"

# LoRA starts from B = 0, so its update has further to travel than a
# fine-tune's; an order of magnitude more learning rate is what every published
# recipe gives it, and matching the fine-tune's rate would be measuring a badly
# tuned LoRA rather than LoRA. Everything else is held identical. Same table,
# same reasoning as tools/train_thermal.py.
RATES = {
    "finetune": {"head": Rates(head=1e-4),
                 "encoder": Rates(head=5e-5, neck=5e-5, trunk=1e-5)},
    "lora": {"head": Rates(head=1e-3),
             "encoder": Rates(head=5e-4, neck=5e-4, trunk=5e-4)},
}


def build_index(data: Path, spec, modality: str, gates: InstanceGates,
                cache: Path | None, workers: int, quiet: bool = False):
    """The per-frame instance index, from cache when there is one.

    Decomposing every semantic map is a full pass over the dataset and produces
    a few hundred kilobytes of boxes. Caching it is what makes a second run --
    the LoRA one, in particular -- start training immediately, and what makes
    the two runs provably see the same instances rather than the same code.
    """
    if cache is not None and cache.is_file():
        index = load_index(cache)
        if not quiet:
            print(f"instance index reused from {cache}")
        return index

    frames = list_frames(data, spec, modality)
    index = index_frames(frames, spec, gates, workers=workers, progress=_tqdm())
    if cache is not None:
        save_index(cache, index)
        print(f"instance index written to {cache}")
    return index


def build_model(size: int, checkpoint: str, device: str):
    from sam2.build_sam import build_sam2_video_predictor

    from src.trackers._hydra_overrides import image_size_overrides

    model = build_sam2_video_predictor(
        "configs/edgetam.yaml", checkpoint, device=device,
        hydra_overrides_extra=image_size_overrides(size))
    # eval() is the policy, not an oversight: RepViT's batch-norm statistics
    # must stay frozen so the checkpoint keeps matching its exported engines.
    return model.eval()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="Dataset root.")
    p.add_argument("--spec", default="kust4k", choices=sorted(SPECS))
    p.add_argument("--modality", default="thermal", choices=("thermal", "rgb"))
    p.add_argument("--out", required=True, help="Where the checkpoint goes.")
    p.add_argument("--method", choices=("finetune", "lora"), default="finetune")
    p.add_argument("--base", default="third_party/EdgeTAM/checkpoints/edgetam.pt",
                   help="Starting weights -- the stock checkpoint, or a distilled one.")
    p.add_argument("--index", default=None, help="Cache the instance index here.")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--per-image", type=int, default=1, help="Windows sampled per frame.")
    p.add_argument("--max-instances", type=int, default=8,
                   help="Prompts per window. One encode covers all of them.")
    p.add_argument("--jitter", type=int, default=32)
    p.add_argument("--min-area", type=int, default=48)
    p.add_argument("--min-side", type=int, default=4)
    p.add_argument("--max-area", type=float, default=0.9)
    p.add_argument("--fill", type=float, default=0.25)
    p.add_argument("--batch", type=int, default=0, help="0 measures it on this GPU.")
    p.add_argument("--batch-ceiling", type=int, default=64)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--steps", type=int, default=400, help="Batches per epoch.")
    p.add_argument("--epochs", type=int, nargs=2, default=(1, 3),
                   metavar=("HEAD", "ENCODER"))
    p.add_argument("--val-batches", type=int, default=24)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=None, help="Default 2r.")
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--json", default=None, help="Also write the run log here.")
    args = p.parse_args(argv)

    import torch

    from src.training.image_loop import ImageSplit, auto_batch_size

    spec = SPECS[args.spec]
    gates = InstanceGates(min_area=args.min_area, min_side=args.min_side,
                          max_area=args.max_area, fill=args.fill)
    index = build_index(Path(args.data), spec, args.modality, gates,
                        Path(args.index) if args.index else None, args.workers)
    print(summarise(index, spec))

    # Split on frames, never on windows: two windows of one image share most of
    # their pixels, and a window-level split would put near-duplicates on both
    # sides of it and make the held-out number meaningless.
    parts = split_frames([e.frame for e in index], seed=args.seed)
    by_name = {e.frame.name: e for e in index}
    splits = {name: [by_name[f.name] for f in frames] for name, frames in parts.items()}

    gray = args.modality == "thermal"

    def windows(name, jitter):
        return ImageSplit(
            samples=sample_windows(splits[name], size=args.size,
                                   per_image=args.per_image,
                                   max_instances=args.max_instances,
                                   jitter=jitter, seed=args.seed),
            spec=spec, gates=gates, gray=gray)

    train, val = windows("train", args.jitter), windows("val", 0)
    instances = sum(len(s.instances) for s in train.samples)
    native = sum(1 for s in train.samples if s.native)
    print(f"\ntrain {len(train.samples)} windows / {instances} instances "
          f"({native / max(len(train.samples), 1):.0%} native pixels), "
          f"val {len(val.samples)} windows")

    model = build_model(args.size, args.base, args.device)
    meta = {"method": args.method, "image_size": args.size, "dataset": spec.name,
            "modality": args.modality, "base": args.base, "stage": "instances",
            "train_frames": len(splits["train"]), "train_instances": instances,
            "per_image": args.per_image, "max_instances": args.max_instances,
            "seed": args.seed}

    if args.method == "lora":
        from src.training import lora

        report = lora.inject(model, "encoder", r=args.lora_r, alpha=args.lora_alpha,
                             dropout=args.lora_dropout)
        print(lora.summarise(report))
        freeze = lora.freeze
        def save(m, extra): lora.save_merged_checkpoint(m, args.out, {**meta, **extra})
        meta |= {"lora_r": report["r"], "lora_alpha": report["alpha"],
                 "lora_parameters": report["parameters"],
                 "adapted_layers": len(report["adapted"])}
    else:
        freeze = apply_freeze
        def save(m, extra): save_checkpoint(m, args.out, {**meta, **extra})
        meta |= {"trainable_parameters":
                 sum(apply_freeze(model, "encoder").values())}

    batch = args.batch
    if batch <= 0:
        # Measure under the encoder stage, the one that keeps the trunk's
        # activations alive for the backward. LoRA's frozen base weights need
        # no gradient buffers, so it will usually measure larger -- that
        # difference is a result, not something to normalise away.
        freeze(model, "encoder")
        batch = auto_batch_size(model, train, device=args.device,
                                maximum=args.batch_ceiling)

    schedule = Schedule(
        stages=(("head", args.epochs[0], RATES[args.method]["head"]),
                ("encoder", args.epochs[1], RATES[args.method]["encoder"])),
        batch=batch, accum=args.accum, steps_per_epoch=args.steps,
        val_batches=args.val_batches, workers=args.workers, depth=args.depth,
        seed=args.seed, meta=meta)

    started = time.time()
    result = run_stages(model, train, val, schedule, freeze=freeze, save=save,
                        device=args.device, progress=_tqdm(), loop=images())
    result |= {"seconds": round(time.time() - started, 1),
               "checkpoint": str(args.out),
               "peak_gib": (torch.cuda.max_memory_allocated() / 2**30
                            if args.device.startswith("cuda") else 0.0)}

    print(f"\n{args.method}: best val instance loss {result['best_val_loss']:.4f} "
          f"in {result['seconds'] / 60:.0f} min, peak {result['peak_gib']:.1f} GiB "
          f"-> {args.out}")
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {out}")
    return 0


def _tqdm():
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - environment, not logic
        return None
    return lambda stream, total, desc: tqdm(stream, total=total, desc=desc)


if __name__ == "__main__":
    raise SystemExit(main())
