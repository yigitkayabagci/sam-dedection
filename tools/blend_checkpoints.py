#!/usr/bin/env python3
"""Interpolate between the checkpoint stage B started from and the one it wrote.

Stage B moves 9.3 M of EdgeTAM's 13.9 M parameters over masks a teacher
produced, and the first thermal run's paired comparison says what that costs:
1 222 of 1 707 held-out instances came back better under a point prompt and
**281 came back worse**. Those are instances the base model already handled,
and no amount of training data removes the category -- a fine-tune that moves
the encoder at all will move some of them the wrong way.

A blend is the cheapest thing that buys most of them back:

    theta = (1 - alpha) * base + alpha * tuned

Both checkpoints are the same architecture key for key -- a LoRA run merges its
adapters before saving, so this works on either method's output -- and the
average of two points in weight space that are a short fine-tune apart is a
valid checkpoint rather than a broken one. It is the published WiSE-FT recipe
and it needs no retraining: sweep alpha, score each one on **val**, and keep
the alpha that holds the gain while giving back the regressions. `alpha = 1`
returns the tuned checkpoint exactly, `alpha = 0` the base.

Choosing alpha on the test split would be choosing a hyper-parameter on the
grade, so the notebook sweeps on val and reports the winner on test.

Usage:
    python tools/blend_checkpoints.py --base third_party/EdgeTAM/checkpoints/edgetam.pt \\
        --tuned checkpoints/edgetam_pool_thermal_512.pt --alpha 0.7 \\
        --out /content/work/blend_070.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def state_dict_of(payload):
    """The weights out of a checkpoint written either way."""
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    return payload


def blend(base: dict, tuned: dict, alpha: float) -> dict:
    """`(1 - alpha) * base + alpha * tuned`, key for key.

    Non-floating tensors -- batch-norm's `num_batches_tracked`, and anything
    integral a future module adds -- are taken from `tuned` rather than
    averaged: an average of two counters is not a counter, and they carry no
    signal a blend could preserve.

    A key either side is missing raises. The two files are meant to be the
    same architecture, and a silent `strict=False`-style merge is how a
    blend that quietly dropped the mask decoder gets shipped.
    """
    import torch

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    missing = sorted(set(base) ^ set(tuned))
    if missing:
        raise ValueError(
            f"the two checkpoints do not hold the same {len(base)} keys -- "
            f"{len(missing)} differ, e.g. {missing[:3]}. They are not the same "
            f"architecture, so nothing between them is a checkpoint.")
    out = {}
    for key, tuned_value in tuned.items():
        base_value = base[key]
        if not torch.is_tensor(tuned_value) or not tuned_value.is_floating_point():
            out[key] = tuned_value
            continue
        if base_value.shape != tuned_value.shape:
            raise ValueError(
                f"{key}: {tuple(base_value.shape)} in the base and "
                f"{tuple(tuned_value.shape)} in the tuned checkpoint")
        out[key] = torch.lerp(base_value.to(torch.float32),
                              tuned_value.to(torch.float32),
                              alpha).to(tuned_value.dtype)
    return out


def main(argv: list[str] | None = None) -> int:
    import torch

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True,
                   help="What the run started from -- stage A's output, or "
                        "stock EdgeTAM where there was no stage A.")
    p.add_argument("--tuned", required=True, help="What the run wrote.")
    p.add_argument("--alpha", type=float, required=True,
                   help="1.0 is the tuned checkpoint, 0.0 the base.")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    tuned = torch.load(args.tuned, map_location="cpu", weights_only=False)
    merged = blend(state_dict_of(base), state_dict_of(tuned), args.alpha)
    meta = dict(tuned.get("meta", {})) if isinstance(tuned, dict) else {}
    meta |= {"blend_alpha": args.alpha, "blend_base": str(args.base),
             "blend_tuned": str(args.tuned)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": merged, "meta": meta}, out)
    print(f"alpha {args.alpha:g}: {len(merged)} tensors -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
