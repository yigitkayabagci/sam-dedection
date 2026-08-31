#!/usr/bin/env python3
"""FCMAE pretraining for EdgeTAM's trunk, on unlabelled thermal frames.

ConvNeXt V2's masked autoencoder (arXiv 2301.00808) applied to the trunk this
project already ships. `src/training/fcmae.py` holds the method and says why
each piece is shaped the way it is; this is the run around it.

**Stage A, and it reads no labels at all.** That is the point: the mask pools
this project trains on are gated -- a frame whose teacher mask failed
`MIN_BOX_IOU` is dropped -- and stage A wants every frame regardless, which on
this Drive is over an order of magnitude more data than stage B ever sees.

    python3 tools/pretrain_fcmae.py --frames /content/data --frames /content/pool \\
        --out checkpoints/edgetam_fcmae_512.pt --epochs 20 --steps 1000

**Two arms, and running both is the experiment.** `--grn` inserts Global
Response Normalization, which is the other half of the co-design; `--no-grn`
leaves it out. The paper's own ablation says the second should buy nothing
(V1 + FCMAE 83.7 against supervised 83.8) and the first should buy something
(V2 + FCMAE 84.6). Reproducing that on a different convnet, in a different
domain, is a result either way.

What comes out is an ordinary EdgeTAM state dict -- the decoder and the mask
token are dropped -- so `34` and `32` take it as `BASE_CHECKPOINT` with nothing
else changed. With `--grn` the block carries two extra vectors per expansion,
so the ONNX export has to be regenerated before that checkpoint reaches
TensorRT; without it the export is untouched.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.aerial import IMAGE_SUFFIXES, load_image, normalise  # noqa: E402
from src.training.fcmae import (  # noqa: E402
    Decoder,
    FCMAEConfig,
    fcmae_loss,
    insert_grn,
    masked_convolutions,
    patch_mask,
)
from src.training.finetune import EMA, apply_freeze  # noqa: E402


def find_frames(roots, limit: int = 0) -> list[Path]:
    """Every image under `roots`, sorted, deduplicated by resolved path.

    A plain walk rather than the pool reader: this stage has no use for a
    record, a mask or a gate, and coupling it to them would exclude exactly the
    frames it exists to use -- the ones whose teacher mask was refused.
    """
    seen: dict[Path, None] = {}
    for root in roots:
        for path in sorted(Path(root).rglob("*")):
            if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
                seen.setdefault(path.resolve(), None)
    frames = list(seen)
    return frames[:limit] if limit else frames


def collate(paths, size: int, pool) -> torch.Tensor:
    """One batch of frames at the model's input size, ImageNet-normalised.

    The same `load_image`/`normalise` pair every other stage uses. Training on
    different statistics from the ones the tracker runs at inference would be a
    domain shift of our own making.
    """
    def one(path):
        return load_image(path, (0, 0), (0, 0), size, gray=True)

    images = list(pool.map(one, paths))
    return normalise(np.stack(images), device="cpu")


def stream(frames, batch: int, size: int, seed: int, steps: int,
           device: str, workers: int, depth: int):
    from src.training.loader import batch_clips, prefetch_with

    chunks = batch_clips(frames, batch, seed=seed, limit=steps)
    return prefetch_with(chunks, lambda chunk, pool: collate(chunk, size, pool),
                         device=device, workers=workers, depth=depth)


def learning_rate(step: int, total: int, warmup: int, peak: float) -> float:
    """Linear warmup then cosine decay -- the paper's schedule."""
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def run(args) -> dict:
    from tools.train_encoder import build_model

    config = FCMAEConfig(image_size=args.size, patch=args.patch,
                         mask_ratio=args.mask_ratio,
                         decoder_dim=args.decoder_dim,
                         norm_floor=args.norm_floor)
    frames = find_frames(args.frames, args.limit)
    if len(frames) < args.batch:
        raise SystemExit(
            f"{len(frames)} frame(s) under {[str(r) for r in args.frames]} and "
            f"a batch of {args.batch}. Point --frames at the extracted image "
            f"trees, not at the pool archives.")
    print(f"[fcmae] {len(frames)} frames | {args.size}px | patch {config.patch} "
          f"| mask {config.mask_ratio:.0%} | grn {args.grn}")

    model = build_model(args.size, args.base, args.device)
    report = {"inserted": 0}
    if args.grn:
        report = insert_grn(model.image_encoder)
        if not report["inserted"]:
            raise SystemExit(
                "--grn found no dimension-expanding layer to sit behind. That "
                "is half the method missing, so this run is refused rather "
                "than started: check the trunk, or pass --no-grn deliberately.")
        model.to(args.device)
    print(f"[fcmae] GRN inserted at {report['inserted']} site(s), "
          f"{report.get('parameters', 0)} new parameters")

    counts = apply_freeze(model, "backbone")
    print(f"[fcmae] trainable: {counts} (the trunk and neck alone; the head "
          f"receives no gradient here and the memory path is frozen "
          f"everywhere)")

    channels = int(getattr(model, "hidden_dim", 256))
    decoder = Decoder(channels, config).to(args.device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    peak = args.lr * args.batch / 256.0                  # the paper's rule
    opt = torch.optim.AdamW(
        [{"params": trainable}, {"params": decoder.parameters()}],
        lr=peak, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    ema = EMA(model, decay=args.ema_decay)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    total = args.epochs * args.steps
    warmup = min(args.warmup, max(total // 10, 1))
    print(f"[fcmae] {args.epochs} x {args.steps} = {total} steps | peak lr "
          f"{peak:.2e} (base {args.lr:.2e} x batch {args.batch} / 256) | "
          f"warmup {warmup}")

    history, step, started = [], 0, time.time()
    for epoch in range(args.epochs):
        running, seen = 0.0, 0
        for images in stream(frames, args.batch, args.size, args.seed + epoch,
                             args.steps, args.device, args.workers, args.depth):
            for group in opt.param_groups:
                group["lr"] = learning_rate(step, total, warmup, peak)
            mask = patch_mask(images.shape[0], args.size, config, generator)
            mask = mask.to(images.device)
            with masked_convolutions(model.image_encoder, mask):
                from src.training.distill import encoder_features

                features = encoder_features(model, images)
            loss, terms = fcmae_loss(decoder(features, mask), images, mask, config)
            if not torch.isfinite(loss):
                raise SystemExit(
                    f"non-finite loss at step {step}. Lower --lr; the "
                    f"checkpoint on disk is the last finite one.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            ema.update(model)
            running += terms["mse"]
            seen += 1
            step += 1
        mean = running / max(seen, 1)
        history.append({"epoch": epoch, "mse": round(mean, 5),
                        "lr": round(opt.param_groups[0]["lr"], 8)})
        print(f"[fcmae] epoch {epoch}: mse {mean:.5f}")
        save(model, ema, args)

    body = {"frames": len(frames), "steps": total, "history": history,
            "grn": report, "seconds": round(time.time() - started, 1),
            "config": {"size": args.size, "patch": config.patch,
                       "mask_ratio": config.mask_ratio,
                       "decoder_dim": config.decoder_dim,
                       "norm_floor": config.norm_floor},
            "peak_lr": peak, "batch": args.batch, "seed": args.seed}
    if args.json:
        Path(args.json).write_text(json.dumps(body, indent=2) + "\n")
    return body


def save(model, ema, args) -> None:
    """The EMA weights as an ordinary EdgeTAM checkpoint, decoder discarded."""
    from src.training.finetune import save_checkpoint

    with ema.applied(model):
        save_checkpoint(model, args.out)
    if args.mirror:
        import shutil

        target = Path(args.mirror) / Path(args.out).name
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(target.suffix + ".part")
        shutil.copy(args.out, part)
        part.replace(target)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", action="append", default=[], required=True,
                   help="A directory of images, walked recursively. Repeatable.")
    p.add_argument("--limit", type=int, default=0, help="Cap the frame count.")
    p.add_argument("--base", default="third_party/EdgeTAM/checkpoints/edgetam.pt")
    p.add_argument("--out", default="checkpoints/edgetam_fcmae_512.pt")
    p.add_argument("--mirror", default=None)
    p.add_argument("--json", default=None)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--patch", type=int, default=FCMAEConfig.patch)
    p.add_argument("--mask-ratio", type=float, default=FCMAEConfig.mask_ratio)
    p.add_argument("--decoder-dim", type=int, default=FCMAEConfig.decoder_dim)
    p.add_argument("--norm-floor", type=float, default=FCMAEConfig.norm_floor)
    p.add_argument("--grn", dest="grn", action="store_true", default=True,
                   help="Insert GRN -- the co-design, and the default.")
    p.add_argument("--no-grn", dest="grn", action="store_false",
                   help="The ablation arm. The paper's table says this buys "
                        "nothing over supervised training; running it is how "
                        "that claim gets tested on this trunk.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps", type=int, default=1000, help="Batches per epoch.")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1.5e-4,
                   help="Base rate; the run scales it by batch/256, as the "
                        "paper does.")
    p.add_argument("--warmup", type=int, default=500, help="Steps.")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    body = run(args)
    print(f"[fcmae] {body['steps']} steps in {body['seconds'] / 60:.0f} min -> "
          f"{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
