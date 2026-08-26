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

**And it takes more than one dataset.** `--dataset` is repeatable and the
windows concatenate into one pool, so a batch mixes them. That matters more
here than for the head: the encoder carries general visual features, and one
dataset is one sensor, one city and one set of annotation habits. Kust4K's
4 024 frames at 640x512 are a thin thing to move a trunk with on their own.

**And it takes mask pools, on the same terms.** `--pool` points at a directory
notebooks 13-18 harvested -- box annotations turned into teacher masks -- and
its windows concatenate into the same batch as `--dataset`'s. The two differ
only in where the mask came from, so a pool run, a semantic run and a mixed run
go through one loop, one schedule and one evaluation. A pool holds masks and
not pixels, so the second field of the flag is where its frames live now.

Usage:
    python tools/train_encoder.py --dataset kust4k:/content/data/Kust4K \\
        --out checkpoints/edgetam_aerial_512.pt --method finetune

    # stage B on the mask pools alone, no stage A, thermal only
    python tools/train_encoder.py \\
        --pool /content/pool/hituav_thermal:/content/data/HIT_UAV \\
        --pool /content/pool/dronevehicle_thermal:/content/data/DroneVehicle \\
        --pool /content/pool/kust4k_thermal:/content/data/Kust4K:thermal:all \\
        --out checkpoints/edgetam_pool_512.pt --method finetune

    # higher resolution, and instance masks that need no decomposition at all
    python tools/train_encoder.py \\
        --dataset vtuav_vis:/content/data/VTUAV:thermal:labels \\
        --dataset kust4k:/content/data/Kust4K \\
        --dataset segfly:/content/data/SegFly:thermal:watershed \\
        --index /content/drive/MyDrive/edgetam-encoder/index \\
        --out checkpoints/edgetam_aerial_512.pt --method lora --lora-r 16
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
    InstanceGates,
    index_frames,
    list_frames,
    load_index,
    sample_windows,
    save_index,
    apply_splits,
    split_index,
    summarise,
)
from src.training.datasets import (  # noqa: E402
    Request,
    describe,
    describe_pools,
    index_pools,
    parse,
    parse_pool,
)
from src.training.finetune import Rates, apply_freeze, save_checkpoint  # noqa: E402
from src.training.image_loop import TRAIN_PROMPTS  # noqa: E402
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


def build_index(request: Request, cache_dir: Path | None, workers: int,
                quiet: bool = False):
    """One dataset's per-frame instance index, from cache when there is one.

    Decomposing every annotation map is a full pass over the dataset and
    produces a few hundred kilobytes of boxes. Caching it is what makes a
    second run -- the LoRA one, in particular -- start training immediately,
    and what makes the two runs provably see the same instances rather than the
    same code. The cache is keyed by the request, so a mixed run keeps one file
    per dataset and changing one dataset's mode does not invalidate the others.
    """
    source = request.source
    cache = (cache_dir / f"{request.label.replace(':', '_')}.json"
             if cache_dir else None)
    if cache is not None and cache.is_file():
        index = load_index(cache, source)
        if not quiet:
            print(f"  {request.label}: index reused from {cache}")
        return index

    frames = list_frames(request.root, source.spec, request.modality)
    index = index_frames(frames, source, workers=workers, progress=_tqdm())
    if cache is not None:
        save_index(cache, index)
        print(f"  {request.label}: index written to {cache}")
    return index


def build_indexes(requests: list[Request], cache_dir: Path | None,
                  workers: int) -> list:
    """Every dataset's index, concatenated. Order is the flag order."""
    index = []
    for request in requests:
        index.extend(build_index(request, cache_dir, workers))
    return index


def trained_size(checkpoint: str) -> int | None:
    """The input size a checkpoint was trained at, if it recorded one.

    `finetune.save` has always written `meta["image_size"]` and, until this
    function, **nothing ever read it**. That matters more than it sounds:
    EdgeTAM holds no resolution in any parameter, so a 512-trained checkpoint
    loads into a 768 build with `strict=True` and no complaint at all -- same
    982 keys, same shapes, no warning. The mismatch is undetectable at load
    time and shows up only as a model that quietly underperforms.
    """
    import torch

    try:
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception:
        return None
    meta = blob.get("meta") if isinstance(blob, dict) else None
    size = (meta or {}).get("image_size") if isinstance(meta, dict) else None
    return int(size) if isinstance(size, (int, float)) else None


def build_model(size: int, checkpoint: str, device: str):
    from sam2.build_sam import build_sam2_video_predictor

    from src.trackers._hydra_overrides import image_size_overrides

    was = trained_size(checkpoint)
    if was is not None and was != size:
        print(f"!! {checkpoint} records image_size={was}, building at {size}. "
              f"The weights load either way -- EdgeTAM keeps no resolution in "
              f"any parameter -- so this will run and may simply be worse.")

    model = build_sam2_video_predictor(
        "configs/edgetam.yaml", checkpoint, device=device,
        hydra_overrides_extra=image_size_overrides(size))
    # eval() is the policy, not an oversight: RepViT's batch-norm statistics
    # must stay frozen so the checkpoint keeps matching its exported engines.
    return model.eval()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", action="append", default=[],
                   metavar="SPEC:PATH[:MODALITY[:MODE[:ROLE]]]",
                   help="Repeatable. e.g. kust4k:/data/Kust4K or "
                        "vtuav_vis:/data/VTUAV:thermal:labels. Every dataset "
                        "given is mixed into the same batches -- see "
                        "src/training/datasets.py.")
    p.add_argument("--pool", action="append", default=[],
                   metavar="POOL_DIR[:IMAGES_ROOT[:MODALITY[:ROLE]]]",
                   help="Repeatable, and mixes into the same batches as "
                        "--dataset. A mask pool built by notebooks 13-18: "
                        "`(image, box, teacher mask)` triples in the same "
                        "run-length store, read by src/training/pool_reader.py. "
                        "IMAGES_ROOT is where that dataset's frames are now -- "
                        "the pool holds masks only, a few KB per frame, and the "
                        "path each record carries is the harvest runtime's.")
    p.add_argument("--out", required=True, help="Where the checkpoint goes.")
    p.add_argument("--method", choices=("finetune", "lora"), default="finetune")
    p.add_argument("--base", default="third_party/EdgeTAM/checkpoints/edgetam.pt",
                   help="Starting weights -- the stock checkpoint, or a distilled one.")
    p.add_argument("--index", default=None,
                   help="Directory to cache each dataset's instance index in.")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--per-image", type=int, default=1, help="Windows sampled per frame.")
    p.add_argument("--max-instances", type=int, default=8,
                   help="Prompts per window. One encode covers all of them.")
    p.add_argument("--jitter", type=int, default=32,
                   help="Pixels the 512 crop window may slide around the "
                        "instance it was placed on. This is the *window*, not "
                        "the prompt -- see --prompt.")
    p.add_argument("--prompt", default="box", choices=TRAIN_PROMPTS,
                   help="How a ground-truth box becomes the prompt the loop "
                        "trains against. `box` is the exact rectangle and is "
                        "what every run before 12_encoder_probe.ipynb used; "
                        "`mix` gives half the prompts a jittered box instead. "
                        "The probe measured 08's checkpoints at 0.057-0.100 "
                        "IoU *below* stock EdgeTAM under a loose box while "
                        "beating it under an exact one, which is what `mix` "
                        "exists to fix -- and it is the prompt distribution "
                        "stage C will actually see, since only frame 0 is "
                        "prompted by hand. Validation always scores under "
                        "`box` whatever this is set to.")
    p.add_argument("--prompt-jitter", type=float, default=0.3,
                   help="With --prompt jitter or mix: how far each edge may "
                        "move, as a fraction of its own side. Matches "
                        "eval_instances.py's default, so a model trained "
                        "against this distribution is scored against it.")
    p.add_argument("--min-area", type=int, default=48)
    p.add_argument("--min-side", type=int, default=4)
    p.add_argument("--max-area", type=float, default=0.9)
    p.add_argument("--fill", type=float, default=0.25)
    p.add_argument("--anchor-weight", type=float, default=0.0,
                   help="Penalise how far the encoder drifts from where it "
                        "started this stage. Meaningful only with --base "
                        "pointing at a pretrained checkpoint: stage A moves the "
                        "encoder over a large unlabelled set and stage B can "
                        "undo that on a set two orders of magnitude smaller. "
                        "0.1-1.0 is the range to try; costs no extra forward.")
    p.add_argument("--batch", type=int, default=0, help="0 measures it on this GPU.")
    p.add_argument("--batch-ceiling", type=int, default=64,
                   help="Upper bound on the measured batch. The default is a "
                        "16 GB card's regime; on 80 GB raise it and let the "
                        "probe find the wall -- image mode holds no clip "
                        "length and no memory bank, so it fits far more than "
                        "the video path does.")
    p.add_argument("--batch-reserve", type=float, default=0.15,
                   help="Fraction of the card the batch probe leaves free for "
                        "fragmentation, the EMA copy and the validation pass. "
                        "0.15 of an 80 GB card is 12 GB, which is generous; "
                        "lower it to push the batch further.")
    p.add_argument("--lr-scale", type=float, default=1.0,
                   help="Multiplies every learning rate. Its reason is the "
                        "batch: `--steps` is fixed, so doubling the batch "
                        "doubles the samples behind each of the same number of "
                        "updates, and the linear scaling rule says the step "
                        "should grow with it. Left at 1.0 nothing changes, "
                        "which is what every number recorded before this flag "
                        "was taken with.")
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--splits", default="",
                   help="A `save_splits` file naming the frames each "
                        "split holds. Without it the split is made "
                        "here, and any cap or overlap filter a caller "
                        "applied to its own copy of the index is lost.")
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

    # Seeded off the run's own seed, so a mixed-prompt run reproduces, and
    # created on the training device because `mix_boxes` and `jitter_boxes`
    # draw into a tensor that already sits beside the batch.
    prompts = torch.Generator(device=args.device)
    prompts.manual_seed(args.seed)

    gates = InstanceGates(min_area=args.min_area, min_side=args.min_side,
                          max_area=args.max_area, fill=args.fill)
    if not args.dataset and not args.pool:
        p.error("nothing to train on: pass --dataset, --pool, or both")
    requests = [parse(argument, gates) for argument in args.dataset]
    pools = [parse_pool(argument, gates) for argument in args.pool]
    cache = Path(args.index) if args.index else None
    if requests:
        print(describe(requests), "\n")
    if pools:
        print(describe_pools(pools), "\n")
    index = build_indexes(requests, cache, args.workers)
    index.extend(index_pools(pools, cache, args.workers, progress=_tqdm()))
    print()
    print(summarise(index))

    # Split on frames, never on windows: two windows of one image share most of
    # their pixels, and a window-level split would put near-duplicates on both
    # sides of it and make the held-out number meaningless. Stratified per
    # dataset, and on index entries rather than names -- names collide across
    # datasets.
    splits = (apply_splits(index, args.splits) if args.splits
              else split_index(index, seed=args.seed))

    def windows(name, jitter):
        return ImageSplit(sample_windows(
            splits[name], size=args.size, per_image=args.per_image,
            max_instances=args.max_instances, jitter=jitter, seed=args.seed))

    train, val = windows("train", args.jitter), windows("val", 0)
    instances = sum(len(s.instances) for s in train.samples)
    native = sum(1 for s in train.samples if s.native)
    print(f"\ntrain {len(train.samples)} windows / {instances} instances "
          f"({native / max(len(train.samples), 1):.0%} native pixels), "
          f"val {len(val.samples)} windows")
    for name, count in train.sources.items():
        print(f"  {name:<24} {count:>6} train windows")

    model = build_model(args.size, args.base, args.device)

    # Snapshot before LoRA touches anything, so the anchor is the checkpoint
    # this stage started from rather than that checkpoint plus adapters.
    anchor = None
    if args.anchor_weight:
        import copy

        anchor = copy.deepcopy(model).eval()
        for parameter in anchor.parameters():
            parameter.requires_grad_(False)
        print(f"anchoring to {args.base} at weight {args.anchor_weight}")

    meta = {"method": args.method, "image_size": args.size,
            "datasets": [r.label for r in requests] + [r.label for r in pools],
            "base": args.base,
            "stage": "instances", "train_frames": len(splits["train"]),
            "train_instances": instances, "sources": train.sources,
            "per_image": args.per_image, "max_instances": args.max_instances,
            "anchor_weight": args.anchor_weight, "seed": args.seed,
            "prompt": args.prompt,
            "prompt_jitter": (args.prompt_jitter
                              if args.prompt in ("jitter", "mix") else 0.0)}

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
                                maximum=args.batch_ceiling,
                                reserve=args.batch_reserve)

    # Scaling the rates rather than editing RATES keeps the table above the one
    # every recorded run used, and keeps `--lr-scale 1` byte-identical to it.
    def scaled(rates: Rates) -> Rates:
        if args.lr_scale == 1.0:
            return rates
        return Rates(head=rates.head * args.lr_scale,
                     neck=rates.neck * args.lr_scale,
                     trunk=rates.trunk * args.lr_scale,
                     weight_decay=rates.weight_decay)

    meta |= {"batch_reserve": args.batch_reserve, "lr_scale": args.lr_scale}
    schedule = Schedule(
        stages=(("head", args.epochs[0], scaled(RATES[args.method]["head"])),
                ("encoder", args.epochs[1],
                 scaled(RATES[args.method]["encoder"]))),
        batch=batch, accum=args.accum, steps_per_epoch=args.steps,
        val_batches=args.val_batches, workers=args.workers, depth=args.depth,
        seed=args.seed, meta=meta)

    started = time.time()
    result = run_stages(model, train, val, schedule, freeze=freeze, save=save,
                        device=args.device, progress=_tqdm(),
                        loop=images(anchor, args.anchor_weight,
                                    prompt=args.prompt,
                                    jitter=args.prompt_jitter,
                                    generator=prompts))
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
