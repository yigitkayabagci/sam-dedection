#!/usr/bin/env python3
"""Prove the drop-stale queue in ms, against the real class, no GPU needed.

`src/realtime.py`'s `RealtimeSource`/`LatestFrameQueue` are what `--realtime-fps`
runs on top of in `cli.py`. This drives the real classes with a stand-in
"tracker" that just costs a fixed number of milliseconds, so the mechanism can
be checked before spending a GPU run on it: does staleness at pickup actually
stay under one source period, at the timings this machine actually sees?

The number that matters is "frame staleness at pickup" -- how old the frame
handed to the tracker already was the moment it was handed over. A one-slot
queue's item is always the most recent one produced before pickup, so that age
is bounded by one source period (1000/fps ms) no matter how far behind the
tracker's *throughput* falls -- only the drop rate changes, not the staleness.
If staleness ever exceeds the period, the queue itself is broken, not just
busy; `cli.py --realtime-fps` prints the same check against real per-frame
timings once the model is actually running.

Usage:
    # "my measured worst-case frame is 41 ms; is 30 FPS still safe?"
    python tools/bench_realtime_queue.py --fps 30 --work-ms 41

    # sweep a few candidate frame costs against the same source rate
    python tools/bench_realtime_queue.py --fps 30 --work-ms 33,36,40,45
"""
from __future__ import annotations

import argparse
import time

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.realtime import RealtimeSource  # noqa: E402


def _percentile(ordered_ms: list[float], q: float) -> float:
    return ordered_ms[min(len(ordered_ms) - 1, int(q * len(ordered_ms)))] if ordered_ms else 0.0


def run_once(frames: int, fps: float, work_ms: float, jitter_ms: float) -> dict:
    """Drive the real RealtimeSource with a `work_ms`-per-frame stand-in tracker."""
    import random

    work_s = work_ms / 1000.0
    jitter_s = jitter_ms / 1000.0
    tracked = 0
    with RealtimeSource(frames, fps) as source:
        for _ in source:
            tracked += 1
            time.sleep(max(0.0, work_s + random.uniform(-jitter_s, jitter_s)))

    period_ms = 1000.0 / fps
    lag_ms = sorted(s * 1000.0 for s in source.lag_s)
    offered = source.delivered + source.dropped
    return {
        "work_ms": work_ms,
        "period_ms": period_ms,
        "tracked": tracked,
        "offered": offered,
        "dropped": source.dropped,
        "drop_pct": 100.0 * source.dropped / offered if offered else 0.0,
        "lag_p50": _percentile(lag_ms, 0.50),
        "lag_p99": _percentile(lag_ms, 0.99),
        "lag_max": lag_ms[-1] if lag_ms else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--fps", type=float, default=30.0, help="Simulated source rate.")
    p.add_argument("--work-ms", default="36",
                   help="Comma-separated per-frame cost(s) to simulate, e.g. 33,36,40,45.")
    p.add_argument("--frames", type=int, default=300, help="Frames the source offers.")
    p.add_argument("--jitter-ms", type=float, default=0.0,
                   help="Uniform +/- jitter added to each simulated frame, for a less "
                        "artificially flat worst case than a fixed --work-ms.")
    args = p.parse_args(argv)

    print(f"[bench] real RealtimeSource/LatestFrameQueue, {args.frames} frames "
          f"@ {args.fps:g} FPS (period {1000.0 / args.fps:.1f} ms)\n")
    header = (f"{'work ms':>8}  {'tracked':>8}  {'dropped':>8}  {'drop %':>7}  "
              f"{'lag p50 ms':>10}  {'lag p99 ms':>10}  {'lag max ms':>10}  verdict")
    print(header)
    print("-" * len(header))
    for work_ms in (float(v) for v in args.work_ms.split(",")):
        r = run_once(args.frames, args.fps, work_ms, args.jitter_ms)
        ok = r["lag_max"] <= r["period_ms"] + 1.0
        verdict = "bounded (OK)" if ok else "OVER PERIOD -- queue misbehaving"
        print(f"{r['work_ms']:7.1f}  {r['tracked']:8d}  {r['dropped']:8d}  "
              f"{r['drop_pct']:6.1f}%  {r['lag_p50']:9.1f}  {r['lag_p99']:9.1f}  "
              f"{r['lag_max']:9.1f}  {verdict}")

    print("\n[bench] 'lag' is staleness at pickup: how old the handed-over frame "
          "already was. It must stay <= the source period regardless of work_ms --\n"
          "        that bound, not the drop count, is what proves the one-slot queue "
          "is actually dropping instead of falling behind.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
