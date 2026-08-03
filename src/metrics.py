from __future__ import annotations

from pathlib import Path

# Categorical slots 1-3 from the shared palette (blue/orange/aqua): the fixed
# order that clears colourblind-safety checks for a 3-series chart, so pre /
# inference / post keep the same colour every time this is rendered.
_BLUE = "#2a78d6"
_ORANGE = "#eb6834"
_AQUA = "#1baf7a"

_SURFACE = "#fcfcfb"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"


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


def format_benchmark_note(
    *,
    tracker_name: str | None = None,
    precision: str | None = None,
    source_size: tuple[int, int] | None = None,
    model_size: int | None = None,
    batch: int | None = None,
    extra: str | None = None,
) -> str:
    """A short, factual caption for a chart footer: what was actually run.

    Only reports what the caller passes in -- nothing here is inferred or
    assumed, so a missing piece (no CUDA, precision not tracked) is silently
    omitted rather than guessed.
    """
    parts = []
    try:
        import torch

        parts.append(f"PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            if torch.version.cuda:
                parts.append(f"CUDA {torch.version.cuda}")
            try:
                parts.append(torch.cuda.get_device_name(0))
            except Exception:
                pass
    except ImportError:
        pass

    if source_size and model_size:
        parts.append(f"{source_size[0]}x{source_size[1]} -> {model_size}x{model_size}")
    elif model_size:
        parts.append(f"{model_size}x{model_size}")
    if precision:
        parts.append(str(precision))
    if batch:
        parts.append(f"batch {batch}")
    if tracker_name:
        parts.append(tracker_name)
    if extra:
        parts.append(extra)
    return "  ·  ".join(parts)


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display needed
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        print("[metrics] matplotlib not installed; skipping chart "
              "(pip install matplotlib).")
        return None


def _style_axes(ax) -> None:
    ax.set_facecolor(_SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_AXIS)
        ax.spines[spine].set_linewidth(1.0)
    ax.tick_params(colors=_INK_MUTED, labelsize=9, length=0)
    ax.yaxis.grid(True, color=_GRID, linewidth=1.0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def _warmup_span(ax, warmup: int) -> None:
    """Shade the excluded frames. No in-plot label: at any data value the line
    can reach, text placed inside the axes collides with it sooner or later --
    the span itself plus the subtitle in `_finish` is the label."""
    if warmup <= 0:
        return
    ax.axvspan(-0.5, warmup - 0.5, color=_AXIS, alpha=0.15, linewidth=0, zorder=1)


def _finish(
    fig, ax, out_path, title: str, note: str | None, warmup: int = 0
) -> Path:
    ax.set_xlabel("frame", color=_INK_MUTED, fontsize=9)
    ax.set_ylabel("ms", color=_INK_MUTED, fontsize=9)
    ax.set_title(title, fontsize=10.5, color=_INK_PRIMARY, loc="left", pad=14)
    if warmup > 0:
        # Above the axes (y > 1 in axes-fraction), so it can never sit on top
        # of a data line -- only the plot area itself spans fraction [0, 1].
        ax.text(
            0.0, 1.02, f"shaded: first {warmup} frames (warm-up, excluded)",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, color=_INK_MUTED, clip_on=False,
        )
    if note:
        fig.text(0.01, 0.01, note, fontsize=7.5, color=_INK_MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 0.90))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=_SURFACE)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return out_path.resolve()


def write_latency_chart(
    per_frame_ms: list[float],
    out_path: str | Path,
    warmup: int = 0,
    note: str | None = None,
    label: str | None = None,
):
    """Per-frame latency in ms: the line, plus max/min/avg -- nothing else.

    Takes milliseconds directly (matching write_stage_chart) -- callers that
    only have second-resolution durations convert before calling.

    Max and min are labelled directly on the point they belong to (the two
    frames that were actually slowest and fastest); avg gets a reference line
    across the post-warmup region. All three exclude the warm-up frames, same
    as the printed FPS summary, so the chart and the console agree.
    """
    plt = _matplotlib()
    if plt is None:
        return None
    ms = list(per_frame_ms)
    n = len(ms)
    if n < 2:
        return None
    w = max(0, min(warmup, n - 1))
    kept = ms[w:]
    if not kept:
        return None
    avg = sum(kept) / len(kept)
    worst = max(kept)
    best = min(kept)
    worst_i = w + kept.index(worst)
    best_i = w + kept.index(best)

    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=140)
    fig.patch.set_facecolor(_SURFACE)
    idx = list(range(n))
    ax.plot(idx, ms, linewidth=2.0, color=_BLUE, solid_capstyle="round", zorder=3)
    _style_axes(ax)
    _warmup_span(ax, w)

    # Headroom for the max/min annotations: on a short/flat clip the default
    # autoscale margin can put an extreme point close enough to the axes edge
    # that its offset label pokes into the subtitle above or the tick labels
    # below.
    ymin, ymax = ax.get_ylim()
    pad = 0.12 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.hlines(avg, w, n - 1, colors=_INK_SECONDARY, linestyles=(0, (4, 3)),
              linewidth=1.2, zorder=2)
    ax.text(n - 1, avg, f"  avg {avg:.1f}", fontsize=9, color=_INK_SECONDARY,
            ha="left", va="center")

    for i, v, tag, dy in ((worst_i, worst, "max", 1), (best_i, best, "min", -1)):
        ax.scatter([i], [v], s=34, color=_BLUE, edgecolors=_SURFACE,
                   linewidths=1.5, zorder=4)
        ax.annotate(f"{tag} {v:.1f}", (i, v), textcoords="offset points",
                    xytext=(0, 11 * dy), ha="center", fontsize=8.5,
                    color=_INK_SECONDARY)

    title = "per-frame latency" + (f" — {label}" if label else "")
    return _finish(fig, ax, out_path, title, note, warmup=w)


def write_stage_chart(
    pre_ms: list[float],
    infer_ms: list[float],
    post_ms: list[float],
    out_path: str | Path,
    warmup: int = 0,
    note: str | None = None,
    label: str | None = None,
):
    """Where each frame's time goes: decode, model inference, render/overlay.

    Three lines rather than a stack, so each stage's own value stays directly
    readable instead of requiring the reader to subtract band edges.
    """
    plt = _matplotlib()
    if plt is None:
        return None
    n = min(len(pre_ms), len(infer_ms), len(post_ms))
    if n < 2:
        return None
    idx = list(range(n))

    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=140)
    fig.patch.set_facecolor(_SURFACE)
    for name, values, color in (
        ("pre (decode)", pre_ms[:n], _BLUE),
        ("inference", infer_ms[:n], _ORANGE),
        ("post (render)", post_ms[:n], _AQUA),
    ):
        ax.plot(idx, values, linewidth=2.0, color=color, label=name,
                solid_capstyle="round", zorder=3)

    w = max(0, min(warmup, n - 1))
    _style_axes(ax)
    _warmup_span(ax, w)
    ax.legend(loc="upper right", frameon=False, fontsize=9,
              labelcolor=_INK_SECONDARY, handlelength=1.4, borderaxespad=0.0)

    title = "per-frame stage breakdown" + (f" — {label}" if label else "")
    return _finish(fig, ax, out_path, title, note, warmup=w)
