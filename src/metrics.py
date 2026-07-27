from __future__ import annotations

from pathlib import Path


def fps_summary(per_frame_dt: list[float], warmup: int = 0) -> dict:
    """Compute FPS stats from per-frame durations, optionally dropping the
    first `warmup` frames (which include model load / CUDA warm-up)."""
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

    plt.figure(figsize=(10, 4))
    plt.plot(idx, fps, linewidth=1.0, color="#1f77b4", label="instant FPS")
    if w > 0:
        plt.axvspan(-0.5, w - 0.5, color="orange", alpha=0.15,
                    label=f"warmup (first {w}, excluded)")
        # Scale the y-axis to the post-warmup data. Model load and cuDNN
        # algorithm selection make the first frames extreme outliers, and
        # letting them set the range flattens the part you actually came to
        # read. Warm-up frames stay plotted, just possibly off-scale.
        kept_fps = fps[w:]
        if kept_fps:
            lo, hi = min(kept_fps), max(kept_fps)
            pad = max(1.0, (hi - lo) * 0.15)
            plt.ylim(max(0.0, lo - pad), hi + pad)
    if avg > 0:
        plt.axhline(avg, color="red", linestyle="--", linewidth=1.2,
                    label=f"avg (post-warmup) = {avg:.1f} FPS")
    plt.xlabel("frame index")
    plt.ylabel("FPS")
    plt.title("Per-frame tracking throughput")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    return out_path.resolve()
