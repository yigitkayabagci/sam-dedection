#!/usr/bin/env python3
"""Where does an EdgeTAM frame's compute actually go?

Counts FLOPs and parameters per module at the real inference resolution, using
`torch.utils.flop_counter`, so the "which module is worth accelerating"
question is answered by measurement rather than by intuition. Runs on CPU; no
GPU or TensorRT needed.

It also traces the memory bank across a real tracking run and reports how the
memory-attention key sequence length changes frame to frame -- the property
that makes that module awkward to put behind a fixed-shape engine.

Usage:
    python tools/analyze_cost.py
    python tools/analyze_cost.py --trace-frames 24 --json outputs/cost.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trackers._trt_layout import MemoryLayout, build_memory_mask  # noqa: E402
from tools.edgetam_graphs import (  # noqa: E402
    ImageEncoderGraph,
    MemoryAttentionGraph,
    MemoryEncoderGraph,
    SamHeadGraph,
)

MODULE_ORDER = ("image_encoder", "memory_attention", "memory_encoder", "sam_head")


def _sdpa_flops(query, key, value, *_args, **_kwargs) -> int:
    """QK^T plus AV, the same formula torch's own sdpa counter uses."""
    batch, heads, q_len, head_dim = query.shape
    k_len = key.shape[-2]
    v_dim = value.shape[-1]
    return batch * heads * q_len * k_len * (head_dim + v_dim) * 2


class FlopCounter(torch.utils._python_dispatch.TorchDispatchMode):
    """Count matmul/conv/attention FLOPs for one forward pass.

    Uses torch's own `flop_registry` but not `FlopCounterMode`, whose module
    tracking registers autograd hooks and so cannot run in an inference
    context.

    Two traps this avoids, both of which silently report zero:

    * Under `torch.inference_mode()` composite ops never decompose before the
      Python dispatch key, so the mode sees `aten.matmul`/`aten.conv2d` rather
      than the `aten.mm`/`aten.convolution` the registry is keyed on. Counting
      happens under `no_grad` instead.
    * CPU scaled-dot-product attention lands on
      `aten._scaled_dot_product_flash_attention_for_cpu`, which is not in the
      registry at all — and attention is the single largest term in EdgeTAM's
      memory attention, so missing it would invert the conclusion.

    Only the operations that dominate are counted; elementwise ops,
    normalisations and data movement are not. That is the right lens here:
    those are what fp16 tensor cores and TensorRT's fused kernels change.
    `unmatched` reports anything compute-shaped that still went uncounted, so a
    heavyweight cannot hide.
    """

    def __init__(self) -> None:
        super().__init__()
        from torch.utils.flop_counter import flop_registry

        self._registry = flop_registry
        self.flops = 0
        self.by_op: dict[str, int] = {}
        self.unmatched: set[str] = set()

    def _record(self, name: str, flops: int) -> None:
        self.flops += flops
        self.by_op[name] = self.by_op.get(name, 0) + flops

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        packet = getattr(func, "_overloadpacket", func)
        name = str(packet)
        counter = self._registry.get(packet) or self._registry.get(func)
        if counter is not None:
            try:
                self._record(name, counter(*args, **kwargs, out_val=out))
            except TypeError:  # older registries take `out=`
                self._record(name, counter(*args, **kwargs, out=out))
            return out
        if "scaled_dot_product" in name:
            self._record(name, _sdpa_flops(*args))
            return out
        if any(k in name for k in ("mm", "conv", "einsum", "matmul")):
            self.unmatched.add(name)
        return out


def count_flops(module, args) -> tuple[int, dict[str, int], set[str]]:
    counter = FlopCounter()
    with torch.no_grad(), counter:
        module(*args)
    return counter.flops, counter.by_op, counter.unmatched


def module_cases(model, layout: MemoryLayout, batch: int = 1):
    side = model.image_size // model.backbone_stride
    dim, mem_dim = model.hidden_dim, model.mem_dim
    size = model.image_size
    randn = torch.randn

    return {
        "image_encoder": (
            ImageEncoderGraph(model),
            (randn(batch, 3, size, size),),
        ),
        "memory_attention": (
            MemoryAttentionGraph(model, layout),
            (
                randn(batch, dim, side, side),
                randn(batch, dim, side, side),
                randn(batch, layout.total_tokens, mem_dim),
                randn(batch, layout.total_tokens, mem_dim),
                build_memory_mask(layout, layout.num_slots, layout.ptr_tokens, batch),
            ),
        ),
        "memory_encoder": (
            MemoryEncoderGraph(model),
            (randn(batch, dim, side, side), torch.rand(batch, 1, size, size) * 20 - 10),
        ),
        "sam_head": (
            SamHeadGraph(model),
            (
                randn(batch, dim, side, side),
                randn(batch, dim // 8, side * 4, side * 4),
                randn(batch, dim // 4, side * 2, side * 2),
            ),
        ),
    }


def trace_memory_shapes(model, frames: int, size: int = 256):
    """Record the memory-attention key length on every tracked frame.

    Runs at a reduced source resolution so this stays minutes rather than
    hours on CPU; the memory bank's bookkeeping does not depend on how big the
    frames are, only on how many have been tracked.
    """
    import cv2
    import numpy as np

    from src.prompts import BoxPrompt, PromptSet
    from src.trackers.edgetam_tracker import EdgeTAMTracker

    tmp = Path(tempfile.mkdtemp(prefix="trace_"))
    rng = np.random.default_rng(0)
    background = rng.integers(30, 90, size=(size, size, 3), dtype=np.uint8)
    for t in range(frames):
        frame = background.copy()
        cv2.circle(frame, (size // 4 + 2 * t, size // 2), size // 10, (230, 60, 60), -1)
        cv2.imwrite(str(tmp / f"{t:05d}.jpg"), frame)

    trace = []
    original = model.memory_attention.forward

    def traced(curr, memory, curr_pos=None, memory_pos=None,
               num_obj_ptr_tokens=0, num_spatial_mem=-1):
        trace.append(
            {
                "keys": int(memory.shape[0]),
                "spatial_memories": int(num_spatial_mem),
                "pointer_tokens": int(num_obj_ptr_tokens),
            }
        )
        return original(
            curr=curr, curr_pos=curr_pos, memory=memory, memory_pos=memory_pos,
            num_obj_ptr_tokens=num_obj_ptr_tokens, num_spatial_mem=num_spatial_mem,
        )

    tracker = EdgeTAMTracker(
        model_cfg="configs/edgetam.yaml", checkpoint="unused",
        device="cpu", precision="float32",
    )
    tracker._predictor = model
    try:
        model.memory_attention.forward = traced
        tracker.prepare(tmp)
        tracker.set_prompts(
            PromptSet(boxes=[BoxPrompt(1, 0, (size / 4 - 25, size / 2 - 25,
                                              size / 4 + 25, size / 2 + 25))])
        )
        for _ in tracker.propagate():
            pass
    finally:
        model.memory_attention.forward = original
        tracker.reset()
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    return trace


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", default="third_party/EdgeTAM/checkpoints/edgetam.pt")
    p.add_argument("--model-cfg", default="configs/edgetam.yaml")
    p.add_argument("--batch", type=int, default=1, help="Objects tracked at once.")
    p.add_argument("--max-ptr-tokens", type=int, default=None)
    p.add_argument("--trace-frames", type=int, default=24,
                   help="Frames to trace the memory bank over. 0 to skip.")
    p.add_argument("--json", default=None, help="Also write the numbers here.")
    args = p.parse_args(argv)

    from sam2.build_sam import build_sam2_video_predictor

    checkpoint = args.checkpoint if Path(args.checkpoint).exists() else None
    if checkpoint is None:
        print(f"WARNING: {args.checkpoint} not found; FLOPs are architecture-only "
              "and unaffected, but run with real weights when you can.")
    model = build_sam2_video_predictor(args.model_cfg, checkpoint, device="cpu").eval()
    layout = MemoryLayout.from_model(model, ptr_tokens=args.max_ptr_tokens)

    print(f"\nEdgeTAM @ {model.image_size}x{model.image_size}, "
          f"{args.batch} object(s), {sum(q.numel() for q in model.parameters()) / 1e6:.1f} M params")
    print(f"memory bank: {layout.num_slots} slots x {layout.tokens_per_slot} tokens "
          f"+ {layout.ptr_tokens} pointer tokens = {layout.total_tokens} keys")

    cases = module_cases(model, layout, args.batch)
    results = {}
    unmatched: set[str] = set()
    for name in MODULE_ORDER:
        module, inputs = cases[name]
        flops, by_op, missed = count_flops(module.eval(), inputs)
        unmatched |= missed
        params = sum(q.numel() for q in module.parameters())
        attention = sum(v for k, v in by_op.items() if "scaled_dot_product" in k)
        results[name] = {
            "gflops": flops / 1e9,
            "attention_gflops": attention / 1e9,
            "params_m": params / 1e6,
        }

    total = sum(r["gflops"] for r in results.values())
    print(f"\n{'module':<20}{'GFLOP/frame':>13}{'share':>9}{'of which attn':>15}{'params (M)':>13}")
    print("-" * 71)
    for name in MODULE_ORDER:
        r = results[name]
        print(f"{name:<20}{r['gflops']:>13.1f}{r['gflops'] / total * 100:>8.1f}%"
              f"{r['attention_gflops']:>15.1f}{r['params_m']:>13.2f}")
    print("-" * 71)
    attn_total = sum(r["attention_gflops"] for r in results.values())
    print(f"{'total':<20}{total:>13.1f}{100.0:>8.1f}%{attn_total:>15.1f}")
    if unmatched:
        print(f"\nNOTE: uncounted compute-shaped ops: {sorted(unmatched)}")
    encoder_share = results["image_encoder"]["gflops"] / total * 100
    print(f"\nAccelerating only the image encoder covers {encoder_share:.1f}% of the "
          f"per-frame compute; {100 - encoder_share:.1f}% stayed in PyTorch.")

    trace = []
    if args.trace_frames > 0:
        print(f"\ntracing the memory bank over {args.trace_frames} frames...")
        trace = trace_memory_shapes(model, args.trace_frames)
        print(f"\n{'frame':>6}{'spatial mems':>14}{'ptr tokens':>12}{'key length':>12}")
        print("-" * 44)
        for idx, entry in enumerate(trace):
            print(f"{idx + 1:>6}{entry['spatial_memories']:>14}"
                  f"{entry['pointer_tokens']:>12}{entry['keys']:>12}")
        lengths = sorted({e["keys"] for e in trace})
        print(f"\n{len(lengths)} distinct key lengths over {len(trace)} frames: "
              f"{lengths[0]} ... {lengths[-1]}")
        print("A CUDA graph fixes one shape for its lifetime, so each of these "
              "would need its own capture -- or one padded shape plus a mask.")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"modules": results, "layout": layout.__dict__, "memory_trace": trace},
            indent=2) + "\n")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
