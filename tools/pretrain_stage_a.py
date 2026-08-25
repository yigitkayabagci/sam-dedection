#!/usr/bin/env python3
"""Stage A, four arms, one artifact -- the entry point notebooks 15 and 16 call.

    # the baseline every other arm has to beat: no pretraining at all
    python tools/pretrain_stage_a.py --arm none --out ckpt/stage_a_none.pt

    # the teacher arm, on a mixed corpus described by a JSON file
    python tools/pretrain_stage_a.py --arm distil --corpora corpora.json \\
        --teacher facebook/dinov3-convnext-small-pretrain-lvd1689m \\
        --out ckpt/stage_a_convnext.pt --steps 4000

    # the mask arm, which needs no teacher and eats thermal-only data
    python tools/pretrain_stage_a.py --arm mask --corpora corpora.json \\
        --mask-mode hint --mask-ratio 0.5 --out ckpt/stage_a_mask.pt

    # and the measurement that decides whether thermal-only corpora are worth
    # feeding to the teacher arm at all -- no training, a couple of minutes
    python tools/pretrain_stage_a.py --probe --corpora corpora.json \\
        --teacher facebook/dinov3-vitb16-pretrain-lvd1689m

Whatever ran, the output is an ordinary EdgeTAM checkpoint with a
`*.stage_a.json` manifest beside it, and the next command is always the same:

    python tools/train_encoder.py --base <that checkpoint> ...

`src/training/pretrain.py` argues the design; `docs/encoder_pretrain_stage_a.md`
argues which arm to expect to win and why the baseline is not a formality.
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

from src.training.aerial import SPECS  # noqa: E402
from src.training.automask import MODES, TARGETS  # noqa: E402
from src.training.pretrain import (ARMS, LOSSES, MODALITIES,  # noqa: E402
                                   STUDENTS, Corpus)


def read_corpora(args) -> list[Corpus]:
    """The corpus list, from a JSON file, from `--corpus` strings, or both.

    JSON is the primary form and the reason is the one the notebook cares
    about: a corpus has seven fields, two of them globs, and a command line
    that can express all of them is a command line nobody can read. A file also
    survives being edited between runs, which is what "add a dataset as I find
    one" actually looks like.

        [{"name": "vtuav", "root": "/content/data/VTUAV_VIS",
          "modality": "paired", "spec": "vtuav_vis", "limit": 40000},
         {"name": "visdrone", "root": "/content/data/VisDrone",
          "modality": "rgb", "rgb": "**/images/*.jpg"}]

    The string form is for the common case where a spec already describes the
    layout: `--corpus paired:/content/data/Kust4K:kust4k`.
    """
    out: list[Corpus] = []
    if args.corpora:
        payload = json.loads(Path(args.corpora).read_text())
        if not isinstance(payload, list):
            raise SystemExit(f"{args.corpora}: expected a list of objects")
        out += [Corpus(**entry) for entry in payload]
    for entry in args.corpus or []:
        parts = entry.split(":")
        if len(parts) < 2 or parts[0] not in MODALITIES:
            raise SystemExit(
                f"--corpus {entry!r}: expected MODALITY:ROOT[:SPEC[:ROUTE]] "
                f"with MODALITY in {MODALITIES}")
        modality, root, *rest = parts
        spec = rest[0] if rest else ""
        route = rest[1] if len(rest) > 1 else ""
        if spec and spec not in SPECS:
            raise SystemExit(f"--corpus {entry!r}: no spec named {spec!r} -- "
                             f"have {sorted(SPECS)}")
        out.append(Corpus(name=spec or Path(root).name, root=root,
                          modality=modality, spec=spec, route=route))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=ARMS, default="distil")
    p.add_argument("--probe", action="store_true",
                   help="Run `modality_gap` and stop. No training, no "
                        "checkpoint -- it answers whether an RGB teacher shown "
                        "a thermal frame is describing the scene, which "
                        "decides whether thermal-only corpora can feed the "
                        "distillation arm.")
    p.add_argument("--corpora", default=None, metavar="JSON",
                   help="File describing the corpora. See read_corpora.")
    p.add_argument("--corpus", action="append", metavar="MODALITY:ROOT[:SPEC[:ROUTE]]",
                   help="One corpus, for layouts an existing spec covers. "
                        "Repeatable, and combines with --corpora.")
    p.add_argument("--out", default=None, help="Where the checkpoint goes.")
    p.add_argument("--base", default="third_party/EdgeTAM/checkpoints/edgetam.pt")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--student", choices=STUDENTS, default="thermal",
                   help="Which encoder is being pretrained. `thermal` uses "
                        "each corpus's default route; `rgb` reroutes every "
                        "corpus to same-modality distillation and drops the "
                        "thermal-only ones, which an RGB encoder cannot use.")
    p.add_argument("--method", choices=("finetune", "lora"), default="finetune")

    teacher = p.add_argument_group("the distillation arm")
    teacher.add_argument("--teacher", default=None,
                         help="Default: DINOv3 ViT-B/16. Its 16-pixel stride "
                              "makes its grid the student's grid at 512, with "
                              "no resampling anywhere. A ConvNeXt teacher is "
                              "one string away and is run at twice the input "
                              "so its stride-32 map lands on the same grid.")
    teacher.add_argument("--teacher-size", type=int, default=None,
                         help="Default: derived from the teacher's measured "
                              "stride so its map is never coarser than the "
                              "student's. Setting it changes the teacher, not "
                              "only its cost.")
    teacher.add_argument("--loss", choices=LOSSES, default="cosine")
    teacher.add_argument("--layers", type=int, default=1,
                         help="Align the student against the teacher's last N "
                              "blocks rather than only its last.")
    teacher.add_argument("--gram", type=float, default=0.0,
                         help="Weight of the relational (Gram) term. Robust to "
                              "misregistration in a way a positional loss is "
                              "not; DINOv3 uses weight 2 for its own.")
    teacher.add_argument("--jitter", type=int, default=0,
                         help="Pixels of random offset between the teacher's "
                              "window and the student's. Train under the "
                              "misalignment these archives actually have.")
    teacher.add_argument("--tolerance", type=int, default=0)
    teacher.add_argument("--moments", type=float, default=0.0)
    teacher.add_argument("--cutoff", type=float, default=0.5)
    teacher.add_argument("--high-weight", type=float, default=0.1)
    teacher.add_argument("--beta", type=float, default=2.0)
    teacher.add_argument("--projector-lr", type=float, default=1e-3)

    mask = p.add_argument_group("the mask arm")
    mask.add_argument("--mask-mode", choices=MODES, default="hint")
    mask.add_argument("--mask-ratio", type=float, default=0.5)
    mask.add_argument("--mask-hint", type=float, default=0.1)
    mask.add_argument("--mask-block", type=int, default=2)
    mask.add_argument("--saliency", choices=("distinct", "norm"), default="distinct")
    mask.add_argument("--target", choices=TARGETS, default="ema")
    mask.add_argument("--target-decay", type=float, default=0.999)

    sched = p.add_argument_group("schedule")
    sched.add_argument("--batch", type=int, default=8)
    sched.add_argument("--steps", type=int, default=400)
    sched.add_argument("--epochs", type=int, default=1)
    sched.add_argument("--trunk-lr", type=float, default=5e-5)
    sched.add_argument("--neck-lr", type=float, default=1e-4)
    sched.add_argument("--limit", type=int, default=None,
                       help="Cap on samples per corpus, spread across it.")
    sched.add_argument("--probe-samples", type=int, default=256)
    sched.add_argument("--workers", type=int, default=8)
    sched.add_argument("--depth", type=int, default=2)
    sched.add_argument("--seed", type=int, default=0)
    sched.add_argument("--device", default="cuda")
    sched.add_argument("--lora-r", type=int, default=16)
    sched.add_argument("--lora-alpha", type=float, default=None)
    sched.add_argument("--json", default=None)
    args = p.parse_args(argv)

    import torch

    from src.training import pretrain
    from src.training.pretrain import for_student as pretrain_module_for_student
    from src.training.distill import DINO_ID, build_teacher
    from src.training.finetune import Rates, apply_freeze, save_checkpoint

    corpora = pretrain_module_for_student(read_corpora(args), args.student)
    if args.arm != "none" or args.probe:
        if not corpora:
            raise SystemExit(
                "no corpora given -- pass --corpora FILE.json or --corpus "
                "MODALITY:ROOT[:SPEC]. Only --arm none can run without data.")
        if args.limit is not None:
            corpora = [c for c in corpora]
            from dataclasses import replace as _replace
            corpora = [_replace(c, limit=args.limit) if c.limit is None else c
                       for c in corpora]

    # -- the baseline ------------------------------------------------------
    if args.arm == "none" and not args.probe:
        if not args.out:
            raise SystemExit("--arm none still needs --out; the baseline is a "
                             "file so stage B loads it the same way.")
        written = pretrain.baseline(args.base, args.out,
                                    {"image_size": args.size, "seed": args.seed})
        print(f"stage A arm 'none': the stock weights copied to {written}")
        print("Nothing was trained. This is the number every other arm has to "
              "beat, and it is measured in stage B, not here.")
        print("\n" + pretrain.stage_b_command(written))
        return 0

    items = pretrain.index(corpora, size=args.size, seed=args.seed)
    if not items:
        raise SystemExit("no samples found under any corpus root")
    print()
    print(pretrain.summarise(items))
    print()

    # -- the measurement ---------------------------------------------------
    if args.probe:
        model_id = args.teacher or DINO_ID
        teach = build_teacher(model_id, device=args.device, size=args.teacher_size)
        print(f"teacher {model_id}: {teach.dim}-d at stride {teach.patch}, "
              f"input {teach.size}, grid {teach.size // teach.patch}\n")
        gap = pretrain.modality_gap(teach, items, size=args.size,
                                    samples=args.probe_samples,
                                    batch=args.batch, seed=args.seed,
                                    device=args.device)
        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(json.dumps(gap, indent=2) + "\n")
            print(f"\nwrote {args.json}")
        return 0

    if not args.out:
        raise SystemExit("--out is required for a training arm")

    from tools.train_encoder import _tqdm, build_model

    model = build_model(args.size, args.base, args.device)
    meta = {"stage": "pretrain", "arm": args.arm, "method": args.method,
            "student": args.student, "image_size": args.size, "seed": args.seed,
            "corpora": [{"name": c.name, "modality": c.modality,
                         "route": c.routing, "root": str(c.root)}
                        for c in corpora]}

    if args.method == "lora":
        from src.training import lora

        report = lora.inject(model, "backbone", r=args.lora_r, alpha=args.lora_alpha)
        print(lora.summarise(report))
        freeze = lora.freeze
        def save(m, extra): lora.save_merged_checkpoint(m, args.out, {**meta, **extra})
        meta |= {"lora_r": report["r"]}
    else:
        freeze = apply_freeze
        def save(m, extra): save_checkpoint(m, args.out, {**meta, **extra})

    shared = dict(save=save, size=args.size, epochs=args.epochs,
                  batch=args.batch, steps_per_epoch=args.steps,
                  rates=Rates(neck=args.neck_lr, trunk=args.trunk_lr),
                  freeze=freeze, workers=args.workers, depth=args.depth,
                  seed=args.seed, device=args.device, progress=_tqdm())

    started = time.time()
    if args.arm in ("distil", "both"):
        model_id = args.teacher or DINO_ID
        teach = build_teacher(model_id, device=args.device, size=args.teacher_size)
        print(f"teacher {model_id}: {teach.dim}-d at stride {teach.patch}, "
              f"input {teach.size}, grid {teach.size // teach.patch} "
              f"(the student's is {args.size // 16})")
        result = pretrain.distil(
            model, items, teach, loss=args.loss, beta=args.beta,
            cutoff=args.cutoff, high_weight=args.high_weight,
            gram_weight=args.gram, layers=args.layers, jitter=args.jitter,
            moment_weight=args.moments, tolerance=args.tolerance,
            projector_lr=args.projector_lr, **shared)
        del teach
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    if args.arm in ("mask", "both"):
        from src.training.automask import pretrain_masked, summarise_collapse

        result = pretrain_masked(
            model, items, ratio=args.mask_ratio, mode=args.mask_mode,
            score=args.saliency, hint=args.mask_hint, block=args.mask_block,
            target=args.target, target_decay=args.target_decay, **shared)
        print()
        print(summarise_collapse(result["history"]))

    result |= {**meta, "seconds": round(time.time() - started, 1),
               "checkpoint": str(args.out),
               "peak_gib": (torch.cuda.max_memory_allocated() / 2**30
                            if args.device.startswith("cuda") else 0.0)}
    manifest = pretrain.write_manifest(args.out, result)

    print(f"\nstage A arm {args.arm!r}: final objective {result['final_loss']:.4f} "
          f"in {result['seconds'] / 60:.0f} min -> {args.out}")
    print(f"manifest -> {manifest}")
    print("That number is a proxy. What decides whether this stage was worth "
          "its GPU hours is stage B run from this checkpoint against stage B "
          "run from the 'none' arm's.")
    print("\n" + pretrain.stage_b_command(args.out))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2, default=str) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
