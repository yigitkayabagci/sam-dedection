#!/usr/bin/env python3
"""Is the image encoder compute-bound or launch-bound?

The image_size sweep produced a result that decides how the rest of the
optimization work should be spent: cutting the input from 1024 to 512 --
four times fewer pixels, so four times fewer FLOPs -- made the encoder only
~14% faster. That is the signature of a model whose GPU is idling between
kernels while the CPU issues them, not one whose GPU is saturated.

This isolates the encoder from the rest of the pipeline and tests it
directly, on two axes:

    input size   If time scales with pixels, compute dominates.
                 If it barely moves, launch overhead dominates.
    CUDA graph   Replaying a captured graph issues the whole kernel sequence
                 with one command instead of hundreds. A large speedup means
                 the CPU was the bottleneck; a small one means it was not.

Why it matters: quantization and smaller inputs both reduce how much
arithmetic each kernel does. Neither reduces how many kernels there are. If
this comes back launch-bound, INT8 will disappoint and the leverage is in
fusion (TensorRT) and graph capture instead -- worth knowing before building
a calibration set.

    python3 tools/encoder_probe.py
    python3 tools/encoder_probe.py --sizes 1024,512 --iters 100
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402


def load_encoder(config_path: Path, device: str, image_size: int):
    """Build the predictor and hand back just its image encoder."""
    import torch  # noqa: F401

    from src.trackers import build_tracker

    cfg = yaml.safe_load(config_path.read_text())
    checkpoint = Path(cfg["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = (ROOT / checkpoint).resolve()
    tracker = build_tracker(
        "edgetam",
        model_cfg=cfg["model_cfg"],
        checkpoint=str(checkpoint),
        device=device,
        precision=cfg.get("precision", "bfloat16"),
        image_size=image_size,
    )
    tracker._ensure_predictor()
    return tracker, tracker._predictor.image_encoder


def time_eager(encoder, sample, iters: int, warmup: int, ctx) -> list[float]:
    import torch

    with ctx():
        for _ in range(warmup):
            encoder(sample)
        torch.cuda.synchronize()

        times = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            encoder(sample)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    return times


def time_graph(encoder, sample, iters: int, warmup: int, ctx) -> tuple[list[float], str]:
    """Capture one encoder call into a CUDA graph and replay it."""
    import torch

    try:
        with ctx():
            static_in = sample.clone()
            # Capture must not race with lazy cuBLAS/cuDNN init or the caching
            # allocator's first-touch work, so warm up on a side stream first.
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(max(3, warmup)):
                    encoder(static_in)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                encoder(static_in)
            torch.cuda.synchronize()

            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()

            times = []
            for _ in range(iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                graph.replay()
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000.0)
        return times, ""
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/edgetam.yaml")
    p.add_argument("--sizes", default="1024,768,512",
                   help="Comma-separated image_size values to probe.")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--no-graph", action="store_true", help="Skip the CUDA graph arm.")
    p.add_argument("--json", default=None, help="Write the raw results here.")
    args = p.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("This probe needs CUDA.")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    rows = []
    for size in sizes:
        print(f"\n=== image_size {size} ===")
        tracker, encoder = load_encoder(config_path, "cuda", size)
        sample = torch.randn(1, 3, size, size, device="cuda")

        # Same weights at every size: RepViT's trunk is convolutional, so the
        # kernels slide over whatever spatial extent they are given. Printing
        # the parameter count and the feature shapes makes that checkable
        # rather than something to take on trust.
        params = sum(p.numel() for p in encoder.parameters())
        with tracker._inference_ctx():
            feats = encoder(sample)["backbone_fpn"]
        shapes = [tuple(f.shape[-2:]) for f in feats]
        print(f"  params     {params / 1e6:7.2f} M   features {shapes}")

        eager = time_eager(encoder, sample, args.iters, args.warmup,
                           tracker._inference_ctx)
        row = {"image_size": size, "pixels": size * size,
               "params": params, "feature_shapes": [list(s) for s in shapes],
               "eager_ms": statistics.median(eager)}
        print(f"  eager      {row['eager_ms']:7.2f} ms")

        if not args.no_graph:
            graph, err = time_graph(encoder, sample, args.iters, args.warmup,
                                    tracker._inference_ctx)
            if graph:
                row["graph_ms"] = statistics.median(graph)
                row["graph_speedup"] = row["eager_ms"] / row["graph_ms"]
                print(f"  cuda graph {row['graph_ms']:7.2f} ms "
                      f"({row['graph_speedup']:.2f}x)")
            else:
                row["graph_error"] = err
                print(f"  cuda graph failed: {err}")

        rows.append(row)
        del tracker, encoder, sample
        torch.cuda.empty_cache()

    # --- verdict -----------------------------------------------------------
    print("\n" + "=" * 66)
    print(f"{'image_size':>10} {'pixels':>8} {'eager ms':>9} {'graph ms':>9} "
          f"{'speedup':>8}")
    for row in rows:
        rel = row["pixels"] / rows[0]["pixels"]
        print(f"{row['image_size']:>10} {rel:>7.2f}x {row['eager_ms']:>9.2f} "
              f"{row.get('graph_ms', float('nan')):>9.2f} "
              f"{row.get('graph_speedup', float('nan')):>7.2f}x")

    if len(rows) >= 2:
        first, last = rows[0], rows[-1]
        pixel_ratio = first["pixels"] / last["pixels"]
        time_ratio = first["eager_ms"] / last["eager_ms"]
        # Perfectly compute-bound would make these ratios equal; perfectly
        # launch-bound would leave the time ratio at 1.0 however far the
        # pixel count fell.
        scaling = (time_ratio - 1) / (pixel_ratio - 1) if pixel_ratio > 1 else 0.0
        print(f"\n{pixel_ratio:.1f}x fewer pixels bought {time_ratio:.2f}x less time "
              f"-> {scaling * 100:.0f}% of ideal compute scaling")
        if scaling < 0.35:
            print("VERDICT: launch-bound. The GPU is waiting on the CPU to issue "
                  "kernels, so INT8 and smaller inputs both attack the wrong "
                  "thing. Spend the effort on fusion (TensorRT) and CUDA graphs.")
        elif scaling > 0.7:
            print("VERDICT: compute-bound. Arithmetic is the bottleneck, so "
                  "precision reduction should pay off as expected.")
        else:
            print("VERDICT: mixed. Both kernel count and arithmetic matter; "
                  "measure fp16 and int8 rather than predicting them.")

    speedups = [r["graph_speedup"] for r in rows if "graph_speedup" in r]
    if speedups:
        best = max(speedups)
        print(f"CUDA graph best speedup {best:.2f}x — "
              + ("large, which confirms launch overhead was the cost."
                 if best > 1.25 else
                 "small, so kernel launches were not the bottleneck."))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"\nraw -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
