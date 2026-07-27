from __future__ import annotations

import statistics
from pathlib import Path


def fps_summary(per_frame_dt: list[float], warmup: int = 0) -> dict:
    """Compute FPS stats from per-frame durations, optionally dropping the
    first `warmup` frames (which include model load / CUDA warm-up).

    Two different averages come out of this, and they are not interchangeable:

      avg_fps_post_warmup   frames / total_seconds -- actual throughput, the
                            number the device really sustains.
      fps_mean              mean of the per-frame 1/dt values. Fast frames drag
                            it upward, so it reads higher than the throughput
                            and describes a rate the device never achieved.

    For dt = [0.1, 0.1, 1.0]: throughput is 2.5 FPS, the mean of instantaneous
    FPS is 7.0. Report the throughput; `fps_mean` is here for completeness and
    because the gap between the two is itself a useful jitter signal.
    """
    n = len(per_frame_dt)
    total = sum(per_frame_dt)
    out = {
        "frames": n,
        "total_s": total,
        "avg_fps_all": (n / total) if total > 0 else 0.0,
        "warmup": 0,
        "avg_fps_post_warmup": (n / total) if total > 0 else 0.0,
        "kept_frames": n,
    }
    w = max(0, min(warmup, n - 1)) if n > 1 else 0
    kept = per_frame_dt[w:]
    ksum = sum(kept)
    out["warmup"] = w
    out["kept_frames"] = len(kept)
    out["avg_fps_post_warmup"] = (len(kept) / ksum) if ksum > 0 else 0.0

    instant = [1.0 / dt for dt in kept if dt > 0]
    out["fps_min"] = min(instant) if instant else 0.0
    out["fps_max"] = max(instant) if instant else 0.0
    out["fps_mean"] = statistics.fmean(instant) if instant else 0.0
    out["fps_median"] = statistics.median(instant) if instant else 0.0
    out["fps_stdev"] = statistics.stdev(instant) if len(instant) > 1 else 0.0
    return out


def write_fps_chart(per_frame_dt: list[float], out_path: str | Path, warmup: int = 0):
    """Save a per-frame FPS line chart (PNG). Returns the path, or None if
    matplotlib is unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless: no display needed
        import matplotlib.pyplot as plt
    except ImportError:
        print("[metrics] matplotlib not installed; skipping FPS chart "
              "(pip install matplotlib).")
        return None

    fps = [(1.0 / dt) if dt > 0 else 0.0 for dt in per_frame_dt]
    idx = list(range(len(fps)))
    stats = fps_summary(per_frame_dt, warmup=warmup)
    avg = stats["avg_fps_post_warmup"]
    w = stats["warmup"]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lo, hi = stats["fps_min"], stats["fps_max"]

    plt.figure(figsize=(10, 4.4))
    # min-max envelope first, so the trace and reference lines sit on top of it.
    if hi > lo:
        plt.axhspan(lo, hi, color="#1f77b4", alpha=0.07, zorder=0)
    plt.plot(idx, fps, linewidth=1.0, color="#1f77b4", label="instant FPS", zorder=3)

    if w > 0:
        plt.axvspan(-0.5, w - 0.5, color="orange", alpha=0.15, zorder=1,
                    label=f"warmup (first {w}, excluded)")
        # Scale the y-axis to the post-warmup data. Model load and cuDNN
        # algorithm selection make the first frames extreme outliers, and
        # letting them set the range flattens the part you actually came to
        # read. Warm-up frames stay plotted, just possibly off-scale.
        pad = max(1.0, (hi - lo) * 0.18)
        plt.ylim(max(0.0, lo - pad), hi + pad)

    for value, label in ((hi, "max"), (lo, "min")):
        if value > 0:
            plt.axhline(value, color="#5b6b7a", linestyle=":", linewidth=1.0, zorder=2,
                        label=f"{label} = {value:.1f} FPS")
    if avg > 0:
        # Throughput, not the mean of the instantaneous values -- see fps_summary.
        plt.axhline(avg, color="#c0392b", linestyle="--", linewidth=1.4, zorder=4,
                    label=f"avg = {avg:.1f} FPS  (frames / total time)")

    plt.xlabel("frame index")
    plt.ylabel("FPS")
    plt.title("Per-frame tracking throughput")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower right", fontsize=8, framealpha=0.92, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    return out_path.resolve()
