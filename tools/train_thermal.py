#!/usr/bin/env python3
"""Train EdgeTAM on Anti-UAV410 -- partially fine-tuned, or with LoRA.

One entry point for both methods, because that is the only way the comparison
between them means anything. The data, the clips, the losses, the two stages,
the schedule, the EMA and the validation slice are shared code
(`src/training/schedule.py`); `--method` swaps exactly two things:

    finetune   the selected weights receive gradients directly
    lora       the selected weights are frozen and gain `B @ A`; the
               checkpoint is merged before it is written

Both write a checkpoint in EdgeTAM's own `{"model": state_dict}` layout, so
both are scored through the same `tools/eval_antiuav.py`, the same tracker, the
same ONNX export and the same engine build. If the LoRA output needed a
different loader, the numbers would be comparing deployments rather than
training methods.

Usage:
    python tools/train_thermal.py --data /content/data/Anti-UAV410 \\
        --labels /content/work/labels --out checkpoints/edgetam_thermal_512.pt \\
        --method finetune --sequences 60 --steps 400

    python tools/train_thermal.py ... --method lora --lora-r 16 \\
        --out checkpoints/edgetam_lora_512.pt
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

from src.training import list_sequences, open_masks, sample_clips  # noqa: E402
from src.training.finetune import Rates, apply_freeze, save_checkpoint  # noqa: E402
from src.training.schedule import Schedule, Split, run_stages  # noqa: E402

MASK_STORE = "pseudo_masks.npz"

# LoRA starts from B = 0, so its updates have further to travel than a
# fine-tune's; an order of magnitude more learning rate is the usual remedy and
# is what every published recipe uses. Everything else is held identical.
RATES = {
    "finetune": {"head": Rates(head=1e-4),
                 "encoder": Rates(head=5e-5, neck=5e-5, trunk=1e-5)},
    "lora": {"head": Rates(head=1e-3),
             "encoder": Rates(head=5e-4, neck=5e-4, trunk=5e-4)},
}


def build_split(data: Path, labels: Path, split: str, limit: int | None,
                size: int, clip_len: int, clip_stride: int, jitter: int,
                frame_size: tuple[int, int], seed: int) -> Split:
    """Clips for one split, paired with the mask store notebook 01/02 wrote.

    Only sequences that were actually labelled are used: a sequence with no
    store would silently train on box supervision alone and make the two runs
    differ by which data they saw.
    """
    sequences = [s for s in list_sequences(data, split)[:limit]
                 if (labels / split / s.name / MASK_STORE).is_file()]
    if not sequences:
        raise SystemExit(
            f"no labelled sequences for split {split!r} under {labels}. "
            f"Run the labelling cells of notebook 02 (or notebook 01) first.")
    stores = {s.name: open_masks(labels / split / s.name / MASK_STORE)
              for s in sequences}
    clips = sample_clips(sequences, length=clip_len, stride=clip_stride, size=size,
                         frame_size=frame_size, jitter=jitter, seed=seed)
    return Split(clips=clips, stores=stores)


def build_model(size: int, checkpoint: str, device: str):
    from sam2.build_sam import build_sam2_video_predictor

    from src.trackers._hydra_overrides import image_size_overrides

    model = build_sam2_video_predictor(
        "configs/edgetam.yaml", checkpoint, device=device,
        hydra_overrides_extra=image_size_overrides(size))
    # eval() is the policy, not an oversight: RepViT's batch-norm statistics
    # must stay frozen so the checkpoint keeps matching its exported engines,
    # and SAM 2 withholds object_score_logits in training mode.
    return model.eval()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="Anti-UAV410 root (holds train/ val/).")
    p.add_argument("--labels", required=True, help="Pseudo-mask store root.")
    p.add_argument("--out", required=True, help="Where the checkpoint goes.")
    p.add_argument("--method", choices=("finetune", "lora"), default="finetune")
    p.add_argument("--base", default="third_party/EdgeTAM/checkpoints/edgetam.pt")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--frame-size", type=int, nargs=2, default=(640, 512))
    p.add_argument("--clip-len", type=int, default=8)
    p.add_argument("--clip-stride", type=int, default=2)
    p.add_argument("--jitter", type=int, default=32)
    p.add_argument("--sequences", type=int, default=None, help="Cap on train sequences.")
    p.add_argument("--val-sequences", type=int, default=None)
    p.add_argument("--batch", type=int, default=0, help="0 measures it on this GPU.")
    p.add_argument("--batch-ceiling", type=int, default=128)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--steps", type=int, default=400, help="Batches per epoch.")
    p.add_argument("--epochs", type=int, nargs=2, default=(1, 2),
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

    data, labels = Path(args.data), Path(args.labels)
    frame_size = tuple(args.frame_size)
    common = dict(size=args.size, clip_len=args.clip_len, clip_stride=args.clip_stride,
                  frame_size=frame_size, seed=args.seed)
    train = build_split(data, labels, "train", args.sequences, jitter=args.jitter, **common)
    val = build_split(data, labels, "val", args.val_sequences, jitter=0, **common)
    print(f"train {len(train.clips)} clips / {len(train.stores)} sequences, "
          f"val {len(val.clips)} clips / {len(val.stores)} sequences")

    model = build_model(args.size, args.base, args.device)
    meta = {"method": args.method, "image_size": args.size, "dataset": "Anti-UAV410",
            "train_sequences": sorted(train.stores), "seed": args.seed}

    adapter_path = None
    if args.method == "lora":
        from src.training import lora

        report = lora.inject(model, "encoder", r=args.lora_r, alpha=args.lora_alpha,
                             dropout=args.lora_dropout)
        print(lora.summarise(report))
        adapter_path = Path(args.out).with_suffix(".adapter.pt")
        freeze = lora.freeze

        # Two artefacts from one run, and they are not redundant. The merged
        # checkpoint is what every existing config, exporter and engine build
        # already understands. The adapter beside it is the delta alone: a few
        # MB that leave the stock checkpoint on disk exactly as upstream
        # shipped it, which is the property LoRA is usually chosen for and the
        # one a merged file quietly gives up. See
        # configs/edgetam_512_lora_adapter.yaml.
        def save(m, extra):
            lora.save_merged_checkpoint(m, args.out, {**meta, **extra})
            lora.save_adapter(m, adapter_path, {**meta, **extra})

        meta |= {"lora_r": report["r"], "lora_alpha": report["alpha"],
                 "lora_parameters": report["parameters"],
                 "adapted_layers": len(report["adapted"]),
                 "adapter": adapter_path.name}
    else:
        freeze = apply_freeze
        def save(m, extra): save_checkpoint(m, args.out, {**meta, **extra})
        meta |= {"trainable_parameters":
                 sum(apply_freeze(model, "encoder").values())}

    batch = args.batch
    if batch <= 0:
        from src.training.loader import auto_batch_size

        # Measure under the encoder stage, the one that keeps the trunk's
        # activations alive for the backward. LoRA's frozen base weights need
        # no gradient buffers, so it will usually measure larger -- that
        # difference is a result, not something to normalise away.
        freeze(model, "encoder")
        batch = auto_batch_size(model, train.clips, train.stores,
                                device=args.device, maximum=args.batch_ceiling)

    schedule = Schedule(
        stages=(("head", args.epochs[0], RATES[args.method]["head"]),
                ("encoder", args.epochs[1], RATES[args.method]["encoder"])),
        batch=batch, accum=args.accum, steps_per_epoch=args.steps,
        val_batches=args.val_batches, workers=args.workers, depth=args.depth,
        seed=args.seed, meta=meta)

    started = time.time()
    result = run_stages(model, train, val, schedule, freeze=freeze, save=save,
                        device=args.device, progress=_tqdm())
    result |= {"seconds": round(time.time() - started, 1),
               "checkpoint": str(args.out),
               "adapter": str(adapter_path) if adapter_path else None,
               "peak_gib": (torch.cuda.max_memory_allocated() / 2**30
                            if args.device.startswith("cuda") else 0.0)}

    print(f"\n{args.method}: best val clip loss {result['best_val_loss']:.4f} "
          f"in {result['seconds'] / 60:.0f} min, peak {result['peak_gib']:.1f} GiB "
          f"-> {args.out}")
    if adapter_path:
        print(f"adapter (the delta alone, base checkpoint untouched) -> {adapter_path}")
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
