#!/usr/bin/env python3
"""Check the built engines against PyTorch: numerics first, then speed.

Run this on the Orin after `build_trt_engines.py`, before trusting a tracking
run. It answers two questions per module:

  1. How far does the fp16 engine drift from fp32 PyTorch? The report also
     shows how far *PyTorch's own* fp16 autocast drifts, because that is the
     baseline you were already running -- an engine within the same ballpark
     is not a precision regression, it is the same precision with better
     kernels.

  2. How much faster is it? Timed three ways -- PyTorch autocast, TensorRT with
     per-call enqueue, TensorRT replayed from a CUDA graph -- so the graph's
     contribution is visible separately from TensorRT's.

Usage:
    python tools/check_trt_parity.py --outdir models/ \\
        --checkpoint third_party/EdgeTAM/checkpoints/edgetam.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trackers._trt_layout import MemoryLayout, build_memory_mask  # noqa: E402
from src.trackers._trt_runtime import TRTEngine  # noqa: E402
from tools.edgetam_graphs import (  # noqa: E402
    ImageEncoderGraph,
    MemoryAttentionGraph,
    MemoryEncoderGraph,
    SamHeadGraph,
)

MODULES = ("image_encoder", "memory_attention", "memory_encoder", "sam_head")


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def make_case(name: str, model, layout: MemoryLayout, batch: int, device: str):
    """(wrapper, {input_name: tensor}, [output_name, ...]) for one module.

    Input names and order must match what the exporter declared; the engine
    looks them up by name, so a mismatch surfaces as a KeyError, not a wrong
    answer.
    """
    side = model.image_size // model.backbone_stride
    dim, mem_dim = model.hidden_dim, model.mem_dim
    randn = lambda *s: torch.randn(*s, device=device)  # noqa: E731

    if name == "image_encoder":
        return (
            ImageEncoderGraph(model),
            {"image": randn(batch, 3, model.image_size, model.image_size)},
            ["backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2"],
        )
    if name == "memory_attention":
        # A full memory bank: the steady state after ~7 tracked frames, and the
        # shape every frame runs at once the bank has filled.
        mask = build_memory_mask(
            layout,
            layout.num_slots,
            layout.ptr_tokens,
            batch=batch,
            device=device,
        )
        return (
            MemoryAttentionGraph(model, layout),
            {
                "curr": randn(batch, dim, side, side),
                "curr_pos": randn(batch, dim, side, side),
                "memory": randn(batch, layout.total_tokens, mem_dim),
                "memory_pos": randn(batch, layout.total_tokens, mem_dim),
                "memory_mask": mask,
            },
            ["pix_feat"],
        )
    if name == "memory_encoder":
        scale, bias = model.sigmoid_scale_for_mem_enc, model.sigmoid_bias_for_mem_enc
        return (
            MemoryEncoderGraph(model),
            {
                "pix_feat": randn(batch, dim, side, side),
                "mask_for_mem": torch.rand(
                    batch, 1, model.image_size, model.image_size, device=device
                )
                * scale
                + bias,
            },
            ["maskmem", "maskmem_pos"],
        )
    if name == "sam_head":
        return (
            SamHeadGraph(model),
            {
                "pix_feat": randn(batch, dim, side, side),
                "high_res_0": randn(batch, dim // 8, side * 4, side * 4),
                "high_res_1": randn(batch, dim // 4, side * 2, side * 2),
            },
            [
                "low_res_multimasks",
                "ious",
                "low_res_masks",
                "high_res_masks",
                "obj_ptr",
                "object_score_logits",
            ],
        )
    raise ValueError(name)


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


def error(reference: torch.Tensor, other: torch.Tensor) -> tuple[float, float]:
    """(max absolute error, relative L2) between two tensors."""
    ref = reference.detach().float()
    got = other.detach().float().reshape(ref.shape)
    diff = (ref - got).abs()
    denom = ref.norm().item()
    rel = (diff.norm().item() / denom) if denom > 0 else float(diff.norm().item())
    return diff.max().item(), rel


def timed(fn, iters: int, warmup: int = 10) -> float:
    """Median milliseconds per call. Median, not mean: on a Jetson a DVFS or
    thermal blip during the run skews a mean and leaves the median alone."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    samples.sort()
    return samples[len(samples) // 2]


def check_module(
    name: str, engine_path: Path, model, layout, batch: int, device: str, iters: int
) -> dict | None:
    wrapper, inputs, output_names = make_case(name, model, layout, batch, device)
    wrapper = wrapper.to(device).eval()

    print(f"\n=== {name}")
    try:
        engine = TRTEngine(engine_path, name=name, use_cuda_graph=True)
    except Exception as exc:
        print(f"  engine unavailable: {exc}")
        return None

    shapes = {n: tuple(t.shape) for n, t in inputs.items()}
    if not engine.accepts(shapes):
        print(f"  engine profile does not accept {shapes}")
        print("  " + engine.describe().replace("\n", "\n  "))
        for n in engine.input_names:
            print(f"    {n}: profile min/opt/max {engine.profile_shape(n)}")
        return None

    with torch.inference_mode():
        reference = wrapper(*inputs.values())
        if isinstance(reference, torch.Tensor):
            reference = (reference,)

        autocast_out = None
        try:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                autocast_out = wrapper(*inputs.values())
            if isinstance(autocast_out, torch.Tensor):
                autocast_out = (autocast_out,)
        except Exception as exc:
            print(f"  (PyTorch fp16 autocast reference unavailable: {exc})")

        trt_out = engine.run(inputs)

    print(f"  {'output':<22}{'TRT fp16 max|err|':>18}{'rel L2':>10}"
          f"{'torch fp16 rel L2':>20}{'ratio':>8}")
    worst_rel = 0.0
    rows = []
    for idx, out_name in enumerate(output_names):
        if reference[idx] is None:
            continue
        max_err, rel = error(reference[idx], trt_out[out_name])
        worst_rel = max(worst_rel, rel)
        baseline = ratio_text = ""
        ratio = None
        if autocast_out is not None and autocast_out[idx] is not None:
            _, base_rel = error(reference[idx], autocast_out[idx])
            baseline = f"{base_rel:>20.2e}"
            if base_rel > 0:
                ratio = rel / base_rel
                ratio_text = f"{ratio:>7.1f}x"
        rows.append({"output": out_name, "rel": rel, "ratio": ratio})
        print(f"  {out_name:<22}{max_err:>18.3e}{rel:>10.2e}{baseline}{ratio_text}")

    # Speed. The engine call includes the host-side copies into its persistent
    # buffers, which is what a real frame pays too.
    binding = engine.binding(shapes)
    for n, t in inputs.items():
        binding.inputs[n].copy_(t)

    def torch_fp16():
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            wrapper(*inputs.values())

    def trt_enqueue():
        binding.enqueue()

    def trt_graph():
        binding.run()

    ms_torch = timed(torch_fp16, iters) if autocast_out is not None else float("nan")
    ms_enqueue = timed(trt_enqueue, iters)
    ms_graph = timed(trt_graph, iters)
    graphed = binding.graph is not None

    print(f"  torch fp16 autocast   {ms_torch:8.3f} ms")
    print(f"  tensorrt enqueue      {ms_enqueue:8.3f} ms   ({ms_torch / ms_enqueue:.2f}x)")
    if graphed:
        print(
            f"  tensorrt + cuda graph {ms_graph:8.3f} ms   "
            f"({ms_torch / ms_graph:.2f}x vs torch, "
            f"{ms_enqueue / ms_graph:.2f}x vs enqueue)"
        )
    else:
        print("  tensorrt + cuda graph      n/a       (capture failed; see warning above)")

    return {
        "module": name,
        "worst_rel": worst_rel,
        "outputs": rows,
        "torch_ms": ms_torch,
        "trt_ms": ms_graph if graphed else ms_enqueue,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--outdir", default="models")
    p.add_argument("--checkpoint", default="third_party/EdgeTAM/checkpoints/edgetam.pt")
    p.add_argument("--model-cfg", default="configs/edgetam.yaml")
    p.add_argument("--module", default="all", choices=("all",) + MODULES)
    p.add_argument("--batch", type=int, default=1, help="Object count to test at.")
    p.add_argument("--iters", type=int, default=50)
    # Neither threshold works alone, and the measurements say why.
    #
    #   image_encoder     rel 2.4e-2, ratio 2.9x   -- 132 conv/linear layers
    #   memory_attention  rel 9.0e-4, ratio 7.8x   -- autocast keeps softmax fp32
    #
    # An absolute threshold flags the encoder for being deep. A ratio threshold
    # flags the attention for having an unusually clean PyTorch baseline. Both
    # engines are fine. Only an output that is *both* numerically significant
    # and much worse than PyTorch's own fp16 is worth stopping for.
    p.add_argument(
        "--rel-tol",
        type=float,
        default=2e-2,
        help="Relative L2 above which an output counts as numerically "
        "significant (default 2e-2).",
    )
    p.add_argument(
        "--max-ratio",
        type=float,
        default=5.0,
        help="Multiple of PyTorch's own fp16 autocast error above which an "
        "output counts as a regression (default 5). A module fails only when "
        "one output trips this AND --rel-tol; either alone flags engines that "
        "are working correctly.",
    )
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("This tool needs CUDA; run it on the Orin.")

    from sam2.build_sam import build_sam2_video_predictor

    checkpoint = args.checkpoint if args.checkpoint and Path(args.checkpoint).exists() else None
    if checkpoint is None:
        print(f"WARNING: {args.checkpoint} not found; comparing with random weights.")
    model = build_sam2_video_predictor(args.model_cfg, checkpoint, device="cuda").eval()

    outdir = Path(args.outdir)
    wanted = MODULES if args.module == "all" else (args.module,)

    layout = None
    mem_engine = outdir / "edgetam_memory_attention.engine"
    if mem_engine.exists():
        try:
            probe = TRTEngine(mem_engine, name="probe", use_cuda_graph=False)
            layout = MemoryLayout.from_model_and_engine(
                model, probe.engine_shape("memory")[1]
            )
            del probe
        except Exception as exc:
            print(f"Could not read the memory layout off the engine: {exc}")
    if layout is None:
        layout = MemoryLayout.from_model(model)
    print(f"Memory layout: {layout} ({layout.total_tokens} tokens)")

    results = []
    for name in wanted:
        engine_path = outdir / f"edgetam_{name}.engine"
        if not engine_path.exists():
            print(f"\n=== {name}\n  no engine at {engine_path}")
            continue
        result = check_module(
            name, engine_path, model, layout, args.batch, "cuda", args.iters
        )
        if result:
            results.append(result)

    if results:
        torch_total = sum(r["torch_ms"] for r in results)
        trt_total = sum(r["trt_ms"] for r in results)
        print("\n=== summary (accelerated modules only, batch "
              f"{args.batch})")
        for r in results:
            print(
                f"  {r['module']:<18}{r['torch_ms']:8.3f} ms -> {r['trt_ms']:8.3f} ms"
                f"   ({r['torch_ms'] / r['trt_ms']:.2f}x)"
            )
        print(f"  {'total':<18}{torch_total:8.3f} ms -> {trt_total:8.3f} ms"
              f"   ({torch_total / trt_total:.2f}x)")
        print(
            "  End-to-end FPS will land below this: frame decode, the mask "
            "overlay and the memory bookkeeping are not in these numbers."
        )

    suspect = []
    for r in results:
        for row in r["outputs"]:
            ratio = row["ratio"]
            if row["rel"] > args.rel_tol and (ratio is None or ratio > args.max_ratio):
                suspect.append((r["module"], row))

    loud = [r for r in results if r["worst_rel"] > args.rel_tol]
    if loud and not suspect:
        print(
            f"\nAbove --rel-tol {args.rel_tol:g}, but within {args.max_ratio:g}x of "
            "PyTorch's own fp16: "
            + ", ".join(f"{r['module']} ({r['worst_rel']:.2e})" for r in loud)
            + "\n  Absolute fp16 error grows with depth -- the image encoder runs "
            "132 conv/linear\n  layers to the memory attention's 20 -- so this "
            "number on its own does not mean\n  the engine is wrong, only that "
            "the module is deep. Masks settle it:\n    python "
            "tools/compare_backends.py --frames 500 --offload-video"
        )

    if suspect:
        print(
            f"\nFAILED: error is both above --rel-tol {args.rel_tol:g} and more than "
            f"{args.max_ratio:g}x PyTorch's own fp16 --"
        )
        for module, row in suspect:
            ratio = "no baseline" if row["ratio"] is None else f"{row['ratio']:.1f}x"
            print(f"  {module}.{row['output']}: rel {row['rel']:.2e}, {ratio}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
