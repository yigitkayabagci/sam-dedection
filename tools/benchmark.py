#!/usr/bin/env python3
"""Benchmark harness for the EdgeTAM tracking backends on Jetson Orin AGX.

Differences from `cli.py --fps-chart`, and why they matter:

  * No rendering, no video writing, no per-frame imread. `cli.py` measures the
    whole application; this measures the model.
  * Frames are decoded to a JPG cache ONCE and every variant reuses it, so
    decode cost is identical across the matrix.
  * The image encoder is timed separately with CUDA events, so you can see
    how much of the frame budget TensorRT can actually address (Amdahl).
  * Repeats with a median, and a warm-up window that is dropped.
  * Optional per-frame mask dump for accuracy comparison against the FP32
    PyTorch baseline (see tools/compare_masks.py). Speed numbers without an
    accuracy column cannot justify a quantization choice.

Usage
-----
    # Phase 1 baseline
    python3 tools/benchmark.py --video samples/synthetic_car.mp4 \\
        --prompt-file samples/synthetic_car_box.json \\
        --variants pytorch_fp32,pytorch_fp16,pytorch_bf16 \\
        --out runs/phase1 --dump-masks-dir runs/phase1

    # Everything defined in configs/bench_matrix.yaml that is runnable
    python3 tools/benchmark.py --video ... --prompt-file ... --variants all

    # Re-render the table from previously written result JSONs
    python3 tools/benchmark.py --report runs/ --out runs/final_report.md
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

DEFAULT_MATRIX = ROOT / "configs" / "bench_matrix.yaml"


# --------------------------------------------------------------------------
# Variant definitions
# --------------------------------------------------------------------------

def load_matrix(path: Path) -> tuple[dict, dict]:
    data = yaml.safe_load(path.read_text())
    return data.get("defaults", {}) or {}, data.get("variants", {}) or {}


def resolve_path(value: str) -> str:
    p = Path(value)
    return str(p if p.is_absolute() else (ROOT / p).resolve())


def variant_kwargs(defaults: dict, spec: dict) -> tuple[str, dict]:
    merged = {**defaults, **spec}
    tracker = merged.pop("tracker", "edgetam")
    merged.pop("note", None)
    for key in ("checkpoint", "engine_path"):
        if merged.get(key):
            merged[key] = resolve_path(merged[key])
    return tracker, merged


def variant_runnable(tracker: str, kwargs: dict) -> tuple[bool, str]:
    """Skip variants whose prerequisites are missing, with a real reason."""
    ckpt = kwargs.get("checkpoint")
    if ckpt and not Path(ckpt).exists():
        return False, f"checkpoint missing: {ckpt} (run scripts/setup_edgetam.sh)"
    if tracker == "edgetam_trt":
        engine = kwargs.get("engine_path")
        if not engine or not Path(engine).exists():
            return False, f"engine missing: {engine} (build it, Phase 3/4)"
    return True, ""


# --------------------------------------------------------------------------
# Frame cache
# --------------------------------------------------------------------------

def build_frame_cache(video: str | None, frames_dir: str | None,
                      pattern: str, cache_dir: Path, max_frames: int | None) -> Path:
    """Produce a 00000.jpg... directory once; every variant reuses it."""
    import cv2

    from src.io_utils import list_frame_files, load_frame_rgb8

    existing = sorted(cache_dir.glob("*.jpg")) if cache_dir.is_dir() else []
    if existing:
        print(f"[bench] reusing frame cache {cache_dir} ({len(existing)} frames)")
        return cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    if video:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise SystemExit(f"Could not open video: {video}")
        try:
            while max_frames is None or n < max_frames:
                ok, bgr = cap.read()
                if not ok:
                    break
                cv2.imwrite(str(cache_dir / f"{n:05d}.jpg"), bgr)
                n += 1
        finally:
            cap.release()
    else:
        files = list_frame_files(frames_dir, pattern)
        if max_frames is not None:
            files = files[:max_frames]
        for path in files:
            rgb = load_frame_rgb8(path)
            cv2.imwrite(str(cache_dir / f"{n:05d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            n += 1
    if n == 0:
        raise SystemExit("No frames decoded — check --video / --frames-dir.")
    print(f"[bench] frame cache {cache_dir}: {n} frames")
    return cache_dir


# --------------------------------------------------------------------------
# Instrumentation
# --------------------------------------------------------------------------

# The blocks worth attributing time to. `image_encoder` is the one already
# moved to TensorRT; everything after it is what sets the Amdahl ceiling, so
# knowing which of them is expensive is what decides whether exporting more
# modules is worth the effort. Names that a given build does not have are
# skipped, so this survives EdgeTAM/SAM2 structure differences.
PROFILE_BLOCKS = (
    "image_encoder",
    "memory_attention",
    "memory_encoder",
    "sam_prompt_encoder",
    "sam_mask_decoder",
)


class ModuleTimer:
    """Time submodule forwards with CUDA events (no per-call host sync).

    Events are recorded on the stream and only read back in `stop()`, so the
    measurement perturbs the run by a couple of microseconds per call rather
    than serialising it the way a `torch.cuda.synchronize()` per block would.
    """

    def __init__(self, predictor, device: str, names=PROFILE_BLOCKS) -> None:
        # torch is only needed for the CUDA event path, so import it there —
        # that keeps the timer usable (and testable) on a plain CPU box.
        self._torch = None
        self._cuda = False
        if device == "cuda":
            import torch

            self._torch = torch
            self._cuda = torch.cuda.is_available()
        self._orig: dict[str, tuple] = {}
        self._events: dict[str, list] = {}
        self._cpu_ms: dict[str, list[float]] = {}
        self._mark: dict[str, int] = {}

        for name in names:
            module = getattr(predictor, name, None)
            if module is None or not hasattr(module, "forward"):
                continue
            self._install(name, module)

    def _install(self, name: str, module) -> None:
        torch = self._torch
        original = module.forward
        self._orig[name] = (module, original)
        self._events[name] = []
        self._cpu_ms[name] = []
        events, cpu_ms = self._events[name], self._cpu_ms[name]

        if self._cuda:
            def timed(*args, **kwargs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                out = original(*args, **kwargs)
                end.record()
                events.append((start, end))
                return out
        else:
            def timed(*args, **kwargs):
                t0 = time.perf_counter()
                out = original(*args, **kwargs)
                cpu_ms.append((time.perf_counter() - t0) * 1000.0)
                return out

        module.forward = timed

    def mark(self) -> None:
        """Warm-up boundary: only calls after this point are reported."""
        for name in self._orig:
            done = self._events[name] if self._cuda else self._cpu_ms[name]
            self._mark[name] = len(done)

    def stop(self) -> dict[str, list[float]]:
        """Restore the original forwards and return post-mark call times (ms)."""
        for module, original in self._orig.values():
            module.forward = original
        if self._cuda and any(self._events.values()):
            self._torch.cuda.synchronize()
        out: dict[str, list[float]] = {}
        for name in self._orig:
            start = self._mark.get(name, 0)
            if self._cuda:
                out[name] = [s.elapsed_time(e) for s, e in self._events[name][start:]]
            else:
                out[name] = self._cpu_ms[name][start:]
        return out


class TegrastatsSampler:
    """Sample tegrastats in the background: GPU utilisation, temps, power."""

    _RE_RAM = re.compile(r"RAM (\d+)/(\d+)MB")
    _RE_GPU = re.compile(r"GR3D_FREQ (\d+)%")
    _RE_TEMP = re.compile(r"(\w+)@([\d.]+)C")
    _RE_PWR = re.compile(r"(VDD_\w+|VIN_\w+) (\d+)mW/(\d+)mW")

    def __init__(self, interval_ms: int = 500) -> None:
        self.proc = None
        self.samples: list[dict] = []
        self._interval = interval_ms

    def __enter__(self):
        if not shutil.which("tegrastats"):
            return self
        try:
            self.proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self._interval)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
        except OSError:
            self.proc = None
        return self

    def __exit__(self, *_exc) -> None:
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            out, _ = self.proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            return
        for line in (out or "").splitlines():
            sample: dict = {}
            if (m := self._RE_RAM.search(line)):
                sample["ram_used_mb"] = int(m.group(1))
            if (m := self._RE_GPU.search(line)):
                sample["gpu_pct"] = int(m.group(1))
            temps = {k: float(v) for k, v in self._RE_TEMP.findall(line)}
            if temps:
                sample["temp_max_c"] = max(temps.values())
            power = sum(int(cur) for _, cur, _ in self._RE_PWR.findall(line))
            if power:
                sample["power_mw"] = power
            if sample:
                self.samples.append(sample)

    def summary(self) -> dict:
        if not self.samples:
            return {}
        def avg(key):
            vals = [s[key] for s in self.samples if key in s]
            return round(statistics.fmean(vals), 1) if vals else None
        def peak(key):
            vals = [s[key] for s in self.samples if key in s]
            return max(vals) if vals else None
        return {
            "gpu_util_pct_mean": avg("gpu_pct"),
            "power_mw_mean": avg("power_mw"),
            "temp_max_c": peak("temp_max_c"),
            "ram_used_mb_peak": peak("ram_used_mb"),
        }


# --------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------

def run_variant(name: str, tracker_name: str, kwargs: dict, frames_dir: Path,
                prompts, warmup: int, max_frames: int | None,
                dump_masks: Path | None, use_tegrastats: bool) -> dict:
    import numpy as np
    import torch

    from src.trackers import build_tracker

    device = kwargs.get("device", "cuda")
    cuda = device == "cuda" and torch.cuda.is_available()
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        free_before, total = torch.cuda.mem_get_info()

    tracker = build_tracker(tracker_name, **kwargs)
    t_load = time.perf_counter()
    tracker.prepare(frames_dir)
    load_s = time.perf_counter() - t_load

    if tracker_name == "edgetam_trt" and not getattr(tracker, "trt_active", False):
        raise RuntimeError(
            "TensorRT engine was NOT used — the tracker fell back to PyTorch. "
            "Set strict: true on this variant so the failure is loud instead of "
            "silently reporting PyTorch numbers under a TRT label."
        )

    timer = ModuleTimer(tracker._predictor, device)
    tracker.set_prompts(prompts)

    per_frame_dt: list[float] = []
    packed_masks: list[np.ndarray] = []
    frame_ids: list[int] = []
    obj_ids: list[int] = []
    height = width = 0

    with TegrastatsSampler() if use_tegrastats else _NullCtx() as tegra:
        t_prev = time.perf_counter()
        for result in tracker.propagate():
            now = time.perf_counter()
            per_frame_dt.append(now - t_prev)
            t_prev = now
            if len(per_frame_dt) == warmup:
                timer.mark()  # block timings start counting after warm-up too
            if dump_masks is not None and result.masks:
                ids = sorted(result.masks)
                if not obj_ids:
                    obj_ids = ids
                    first = result.masks[ids[0]]
                    height, width = first.shape
                stack = np.stack([result.masks[i] for i in obj_ids])
                packed_masks.append(np.packbits(stack, axis=-1))
                frame_ids.append(result.frame_idx)
            if max_frames is not None and len(per_frame_dt) >= max_frames:
                break

    block_calls = timer.stop()
    tracker.reset()

    peak_mb = device_mb = None
    if cuda:
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        free_after, _ = torch.cuda.mem_get_info()
        device_mb = (free_before - free_after) / 1e6

    del tracker
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()

    if dump_masks is not None and packed_masks:
        dump_masks.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dump_masks,
            masks=np.stack(packed_masks),
            frames=np.array(frame_ids, dtype=np.int32),
            obj_ids=np.array(obj_ids, dtype=np.int32),
            width=np.int32(width),
            height=np.int32(height),
            variant=str(name),
        )
        print(f"[bench] masks -> {dump_masks} ({dump_masks.stat().st_size / 1e6:.1f} MB)")

    kept = per_frame_dt[warmup:] if len(per_frame_dt) > warmup else per_frame_dt
    total_s = sum(kept)
    latencies_ms = sorted(dt * 1000 for dt in kept)
    frame_mean_ms = statistics.fmean(latencies_ms) if latencies_ms else 0.0
    n_measured = max(1, len(kept))

    # Per-block cost is reported per FRAME, not per call: a block invoked twice
    # a frame costs the frame both invocations.
    blocks_ms = {name: sum(calls) / n_measured
                 for name, calls in block_calls.items() if calls}
    # None rather than 0.0 when the hook never fired (e.g. a wrapper we could
    # not intercept) — a zero would read as "the encoder costs nothing".
    enc_mean = blocks_ms.get("image_encoder")
    accounted = sum(blocks_ms.values())

    return {
        "frames_total": len(per_frame_dt),
        "frames_measured": len(kept),
        "warmup": warmup,
        "fps": (len(kept) / total_s) if total_s > 0 else 0.0,
        "latency_p50_ms": _percentile(latencies_ms, 50),
        "latency_p95_ms": _percentile(latencies_ms, 95),
        "frame_mean_ms": frame_mean_ms,
        "encoder_ms": enc_mean,
        # What TensorRT does NOT cover today — the Amdahl ceiling in one number.
        "rest_ms": None if enc_mean is None else max(0.0, frame_mean_ms - enc_mean),
        "encoder_share_pct": (None if enc_mean is None or not frame_mean_ms
                              else 100 * enc_mean / frame_mean_ms),
        "mem_attn_ms": blocks_ms.get("memory_attention"),
        "mem_enc_ms": blocks_ms.get("memory_encoder"),
        "mask_dec_ms": blocks_ms.get("sam_mask_decoder"),
        "prompt_enc_ms": blocks_ms.get("sam_prompt_encoder"),
        # Everything no timed block explains: the mask device->host copy, the
        # Python loop, and EdgeTAM's state bookkeeping.
        "unaccounted_ms": max(0.0, frame_mean_ms - accounted) if blocks_ms else None,
        "blocks_ms": {k: round(v, 3) for k, v in blocks_ms.items()},
        "peak_torch_mb": peak_mb,
        "device_used_mb": device_mb,
        "load_seconds": load_s,
        **(tegra.summary() if use_tegrastats else {}),
    }


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def summary(self) -> dict:
        return {}


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * pct / 100.0
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

_COLUMNS = [
    ("variant", "variant", "{}"),
    ("fps", "fps_median", "{:.2f}"),
    ("p50 ms", "latency_p50_ms", "{:.2f}"),
    ("p95 ms", "latency_p95_ms", "{:.2f}"),
    ("enc ms", "encoder_ms", "{:.2f}"),
    ("memattn ms", "mem_attn_ms", "{:.2f}"),
    ("memenc ms", "mem_enc_ms", "{:.2f}"),
    ("dec ms", "mask_dec_ms", "{:.2f}"),
    ("other ms", "unaccounted_ms", "{:.2f}"),
    ("rest ms", "rest_ms", "{:.2f}"),
    ("enc %", "encoder_share_pct", "{:.0f}"),
    ("torch MB", "peak_torch_mb", "{:.0f}"),
    ("dev MB", "device_used_mb", "{:.0f}"),
    ("GPU %", "gpu_util_pct_mean", "{:.0f}"),
    ("mW", "power_mw_mean", "{:.0f}"),
    ("engine MB", "engine_mb", "{:.1f}"),
]


def markdown_table(rows: list[dict]) -> str:
    active = [c for c in _COLUMNS if any(r.get(c[1]) is not None for r in rows)]
    lines = ["| " + " | ".join(c[0] for c in active) + " |",
             "|" + "|".join("---" for _ in active) + "|"]
    for row in sorted(rows, key=lambda r: -(r.get("fps_median") or 0)):
        cells = []
        for _, key, fmt in active:
            value = row.get(key)
            cells.append("-" if value is None else
                         (fmt.format(value) if not isinstance(value, str) else value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def latex_table(rows: list[dict]) -> str:
    """A booktabs tabular fragment for docs/report/report.tex to \\input.

    Emitted as a fragment rather than a full table environment so the report
    owns the caption, label and placement.
    """
    active = [c for c in _COLUMNS if any(r.get(c[1]) is not None for r in rows)]
    escape = lambda s: str(s).replace("_", r"\_").replace("%", r"\%")  # noqa: E731
    header = " & ".join(escape(c[0]) for c in active) + r" \\"
    lines = [r"\begin{tabular}{l" + "r" * (len(active) - 1) + "}",
             r"\toprule", header, r"\midrule"]
    for row in sorted(rows, key=lambda r: -(r.get("fps_median") or 0)):
        cells = []
        for _, key, fmt in active:
            value = row.get(key)
            if value is None:
                cells.append("--")
            elif isinstance(value, str):
                cells.append(escape(value))
            else:
                cells.append(fmt.format(value))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def csv_table(rows: list[dict]) -> str:
    keys = sorted({k for r in rows for k in r if not isinstance(r.get(k), (dict, list))})
    keys = ["variant"] + [k for k in keys if k != "variant"]
    out = [",".join(keys)]
    for row in rows:
        out.append(",".join("" if row.get(k) is None else str(row.get(k)) for k in keys))
    return "\n".join(out)


def collect_results(directory: Path) -> list[dict]:
    rows = []
    for path in sorted(directory.rglob("*.result.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except ValueError:
            print(f"[bench] skipping unreadable {path}")
    return rows


def attach_engine_metadata(row: dict, kwargs: dict) -> None:
    engine = kwargs.get("engine_path")
    if not engine:
        return
    meta_path = Path(engine + ".json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except ValueError:
            return
        row["engine_mb"] = meta.get("engine_mb")
        row["build_seconds"] = meta.get("build_seconds")
        row["trt_precision"] = meta.get("precision")


def write_report(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    table = markdown_table(rows)
    header = (f"# EdgeTAM benchmark\n\n"
              f"host: `{platform.node()}` / `{platform.machine()}`  \n"
              f"variants: {len(rows)}\n\n")
    out.write_text(header + table + "\n")
    csv_path = out.with_suffix(".csv")
    csv_path.write_text(csv_table(rows) + "\n")
    tex_path = out.with_suffix(".tex")
    tex_path.write_text(latex_table(rows) + "\n")
    print(f"\n{table}\n")
    print(f"[bench] report -> {out}")
    print(f"[bench] csv    -> {csv_path}")
    print(f"[bench] latex  -> {tex_path}  (\\input this from docs/report/report.tex)")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report", metavar="DIR",
                   help="Aggregate existing *.result.json files and exit.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--video", help="Input video.")
    src.add_argument("--frames-dir", help="Directory of frames.")
    p.add_argument("--frame-pattern", default="*.tif*")
    p.add_argument("--prompt-file", help="JSON prompts (headless; see examples/).")
    p.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    p.add_argument("--variants", default="all",
                   help="Comma-separated variant names from the matrix, or 'all'.")
    p.add_argument("--out", default="runs/bench", help="Output directory (or .md path with --report).")
    p.add_argument("--frames-cache", default=None,
                   help="Where to keep the shared JPG cache (default: <out>/frames_cache).")
    p.add_argument("--repeat", type=int, default=1, help="Runs per variant; median reported.")
    p.add_argument("--warmup", type=int, default=10, help="Frames dropped from each run.")
    p.add_argument("--max-frames", type=int, default=None, help="Cap frames per run.")
    p.add_argument("--dump-masks-dir", default=None,
                   help="Save <variant>.npz masks here for tools/compare_masks.py.")
    p.add_argument("--tegrastats", action="store_true",
                   help="Sample GPU utilisation / power / temperature during each run.")
    p.add_argument("--keep-cache", action="store_true", help="Do not delete the JPG cache.")
    p.add_argument("--skip-missing", action="store_true", default=True,
                   help="Skip variants whose engine/checkpoint is absent (default).")
    p.add_argument("--no-skip-missing", dest="skip_missing", action="store_false",
                   help="Fail instead of skipping unbuildable variants.")
    args = p.parse_args()

    if args.report:
        rows = collect_results(Path(args.report))
        if not rows:
            raise SystemExit(f"No *.result.json under {args.report}")
        out = Path(args.out)
        if out.suffix != ".md":
            out = out / "report.md"
        write_report(rows, out)
        return 0

    if not (args.video or args.frames_dir):
        raise SystemExit("Pass --video or --frames-dir (or --report DIR).")
    if not args.prompt_file:
        raise SystemExit(
            "--prompt-file is required: benchmarks must be headless and every "
            "variant must get identical prompts. Create one with "
            "`python3 cli.py --prompt box ...` on a machine with a display, or "
            "copy examples/car_box_example.json.")

    from src.prompts.file_source import load_prompts

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.frames_cache) if args.frames_cache else out_dir / "frames_cache"

    defaults, matrix = load_matrix(Path(args.matrix))
    if args.variants == "all":
        names = list(matrix)
    else:
        names = [n.strip() for n in args.variants.split(",") if n.strip()]
        unknown = [n for n in names if n not in matrix]
        if unknown:
            raise SystemExit(f"Unknown variant(s): {unknown}. Available: {list(matrix)}")

    prompts = load_prompts(args.prompt_file)
    frames_dir = build_frame_cache(args.video, args.frames_dir, args.frame_pattern,
                                   cache_dir, args.max_frames)

    rows: list[dict] = []
    for name in names:
        tracker_name, kwargs = variant_kwargs(defaults, matrix[name])
        ok, reason = variant_runnable(tracker_name, kwargs)
        if not ok:
            if args.skip_missing:
                print(f"\n[bench] SKIP {name}: {reason}")
                continue
            raise SystemExit(f"{name}: {reason}")

        print(f"\n[bench] ===== {name} ({tracker_name}) =====")
        runs, error = [], None
        for i in range(args.repeat):
            dump = None
            if args.dump_masks_dir and i == 0:
                dump = Path(args.dump_masks_dir) / f"{name}.npz"
            try:
                run = run_variant(name, tracker_name, kwargs, frames_dir, prompts,
                                  args.warmup, args.max_frames, dump, args.tegrastats)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                print(f"[bench] FAILED {name}: {error}")
                break
            runs.append(run)
            split = ("enc n/a" if run["encoder_ms"] is None else
                     f"enc {run['encoder_ms']:.2f} ms, rest {run['rest_ms']:.2f} ms")
            print(f"[bench] run {i + 1}/{args.repeat}: {run['fps']:.2f} FPS ({split})")

        row: dict = {"variant": name, "tracker": tracker_name,
                     "precision": kwargs.get("precision"), "repeats": len(runs)}
        if error:
            row["error"] = error
        if runs:
            numeric = {k for r in runs for k, v in r.items() if isinstance(v, (int, float))}
            for key in numeric:
                vals = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
                if vals:
                    row[f"{key}_median" if key == "fps" else key] = round(
                        statistics.median(vals), 4)
            # Kept verbatim from the first run: the full per-block breakdown is
            # a dict, so the numeric median pass above skips it.
            row["blocks_ms"] = runs[0].get("blocks_ms")
            attach_engine_metadata(row, kwargs)
        rows.append(row)
        (out_dir / f"{name}.result.json").write_text(json.dumps(row, indent=2))

    if not args.keep_cache and args.frames_cache is None:
        shutil.rmtree(cache_dir, ignore_errors=True)

    write_report(rows, out_dir / "report.md")
    failed = [r["variant"] for r in rows if "error" in r]
    if failed:
        print(f"[bench] {len(failed)} variant(s) failed: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
