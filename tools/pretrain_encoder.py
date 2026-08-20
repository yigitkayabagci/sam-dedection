#!/usr/bin/env python3
"""Pretrain EdgeTAM's image encoder by distilling an RGB teacher into it.

Stage A of `docs/encoder_training_todo.md`, and the one stage that uses **no
labels at all**: registered RGB-T pairs, the teacher on the RGB half, the
student on the thermal half. `src/training/distill.py` argues the method and
why it fits a convolutional trunk where MAE does not.

The output is an ordinary EdgeTAM checkpoint. Feed it to `train_encoder.py` as
`--base` and compare against the same run started from the stock weights --
that comparison is the only thing that can say whether this stage was worth its
GPU hours, and it is the reason the tool writes a checkpoint rather than
folding into stage B.

Usage:
    python tools/pretrain_encoder.py --data /content/data/Kust4K --spec kust4k \\
        --out checkpoints/edgetam_distilled_512.pt --steps 600

    # PEFT for the pretraining stage too -- one flag, same scope
    python tools/pretrain_encoder.py ... --method lora --lora-r 16
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

from src.training.aerial import SPECS, list_pairs  # noqa: E402
from src.training.finetune import Rates, apply_freeze, save_checkpoint  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="Dataset root holding RGB-T pairs.")
    p.add_argument("--spec", default="kust4k", choices=sorted(SPECS))
    p.add_argument("--out", required=True, help="Where the checkpoint goes.")
    p.add_argument("--method", choices=("finetune", "lora"), default="finetune")
    p.add_argument("--base", default="third_party/EdgeTAM/checkpoints/edgetam.pt")
    p.add_argument("--teacher", default="facebook/dinov2-base",
                   help="Any DINOv2-class ViT, or a facebook/sam2.1-hiera-* "
                        "checkpoint to distil SAM 2's own encoder instead. The "
                        "class is chosen from the id; see distill.py for what "
                        "each buys and costs.")
    p.add_argument("--teacher-size", type=int, default=None,
                   help="Default: 518 for a ViT teacher, 1024 for SAM 2.1. "
                        "A ViT needs a multiple of its patch size; SAM 2.1's "
                        "position embeddings are not interpolated, so lowering "
                        "this changes the teacher, not only its cost.")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--pairs", type=int, default=None, help="Cap on pairs used.")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--trunk-lr", type=float, default=5e-5)
    p.add_argument("--neck-lr", type=float, default=1e-4)
    p.add_argument("--projector-lr", type=float, default=1e-3)
    p.add_argument("--l1", type=float, default=0.0,
                   help="Weight of a smooth-L1 term beside the cosine one.")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)

    import torch

    from src.training.distill import build_teacher, pretrain
    from tools.train_encoder import _tqdm, build_model

    spec = SPECS[args.spec]
    pairs = list_pairs(Path(args.data), spec)[:args.pairs]
    print(f"{len(pairs)} registered pairs -- no labels are read at this stage")

    model = build_model(args.size, args.base, args.device)
    teacher = build_teacher(args.teacher, device=args.device, size=args.teacher_size)
    print(f"teacher {args.teacher} ({type(teacher).__name__}): {teacher.dim}-d "
          f"at stride {teacher.patch}, input {teacher.size}")

    meta = {"stage": "distill", "method": args.method, "dataset": spec.name,
            "teacher": args.teacher, "image_size": args.size, "seed": args.seed}
    if args.method == "lora":
        from src.training import lora

        report = lora.inject(model, "backbone", r=args.lora_r, alpha=args.lora_alpha)
        print(lora.summarise(report))
        freeze = lora.freeze
        def save(m, extra): lora.save_merged_checkpoint(m, args.out, {**meta, **extra})
        meta |= {"lora_r": report["r"], "lora_parameters": report["parameters"]}
    else:
        freeze = apply_freeze
        def save(m, extra): save_checkpoint(m, args.out, {**meta, **extra})

    started = time.time()
    result = pretrain(
        model, pairs, teacher, save=save, size=args.size, epochs=args.epochs,
        batch=args.batch, steps_per_epoch=args.steps,
        rates=Rates(neck=args.neck_lr, trunk=args.trunk_lr),
        projector_lr=args.projector_lr, l1_weight=args.l1, freeze=freeze,
        workers=args.workers, depth=args.depth, seed=args.seed,
        device=args.device, progress=_tqdm())
    result |= {**meta, "seconds": round(time.time() - started, 1),
               "checkpoint": str(args.out),
               "peak_gib": (torch.cuda.max_memory_allocated() / 2**30
                            if args.device.startswith("cuda") else 0.0)}

    print(f"\ndistilled to cosine distance {result['final_loss']:.4f} in "
          f"{result['seconds'] / 60:.0f} min -> {args.out}")
    print("This number is a proxy. What decides whether the stage paid for "
          "itself is stage B run from this checkpoint against stage B run from "
          "the stock one.")
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
