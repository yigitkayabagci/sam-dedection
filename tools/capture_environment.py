#!/usr/bin/env python3
"""Record exactly what this machine is, so a number can be traced to a setup.

A benchmark figure without its environment is not reproducible and not
reviewable: "26 ms/frame" means nothing without the board, the power mode, the
clock state, and the TensorRT version that produced it. This writes all of it
to JSON and to a markdown table ready to paste into a report.

Everything is probed, nothing is assumed. A field that cannot be read is
recorded as null rather than guessed, so a gap in the record is visible as a
gap.

Usage:
    python tools/capture_environment.py --out results/environment
    python tools/capture_environment.py --out results/1024/environment --engines models/
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run(cmd: list[str], timeout: int = 15) -> str | None:
    """Stdout of `cmd`, or None if it is missing, fails, or hangs."""
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
        return None
    text = (out.stdout or out.returncode == 0 and out.stderr or "").strip()
    return text or None


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeDecodeError):
        return None


def jetson_info() -> dict:
    """Board model, L4T/JetPack, power mode and whether clocks are pinned.

    Power mode and clock state matter more than they look: without
    `jetson_clocks` the same command re-run can differ by tens of percent, which
    is larger than most of the effects being measured.
    """
    info = {
        "model": _read("/sys/firmware/devicetree/base/model"),
        "l4t_release": _read("/etc/nv_tegra_release"),
        "jetpack": None,
        "nvpmodel": _run(["nvpmodel", "-q"]),
        "jetson_clocks": _run(["jetson_clocks", "--show"]),
    }
    # JetPack version lives in the nvidia-l4t-core package version.
    pkg = _run(["dpkg-query", "--showformat=${Version}", "--show", "nvidia-l4t-core"])
    if pkg:
        info["jetpack"] = pkg
    if info["model"]:
        info["model"] = info["model"].replace("\x00", "").strip()
    return info


def python_stack() -> dict:
    stack: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": None,
        "torch_cuda": None,
        "cudnn": None,
        "torchvision": None,
        "tensorrt": None,
        "onnx": None,
        "onnxruntime": None,
        "opencv": None,
        "numpy": None,
        "gpu_name": None,
        "gpu_capability": None,
        "cuda_available": False,
    }
    try:
        import torch

        stack["torch"] = torch.__version__
        stack["torch_cuda"] = torch.version.cuda
        stack["cudnn"] = getattr(torch.backends.cudnn, "version", lambda: None)()
        stack["cuda_available"] = bool(torch.cuda.is_available())
        if stack["cuda_available"]:
            stack["gpu_name"] = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            stack["gpu_capability"] = f"sm_{major}{minor}"
    except Exception:
        pass
    for name, key in (
        ("torchvision", "torchvision"), ("tensorrt", "tensorrt"), ("onnx", "onnx"),
        ("onnxruntime", "onnxruntime"), ("cv2", "opencv"), ("numpy", "numpy"),
    ):
        try:
            stack[key] = __import__(name).__version__
        except Exception:
            pass
    return stack


def repo_state() -> dict:
    def git(*args):
        return _run(["git", "-C", str(ROOT), *args])

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "describe": git("describe", "--always", "--dirty"),
    }


def engine_info(engines_dir: Path | None) -> dict | None:
    """Size, mtime and profile of each built engine.

    An engine is specific to the GPU and TensorRT version that built it, so
    recording which files were used is part of recording the measurement.
    """
    if engines_dir is None or not engines_dir.is_dir():
        return None
    out: dict = {}
    for path in sorted(engines_dir.glob("*.engine")):
        entry = {
            "bytes": path.stat().st_size,
            "modified": datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
        }
        spec = path.with_suffix("").with_suffix(".spec.json")
        if not spec.exists():
            spec = engines_dir / (path.stem + ".spec.json")
        if spec.exists():
            try:
                data = json.loads(spec.read_text())
                entry["inputs"] = {
                    io["name"]: io["shape"] for io in data.get("inputs", [])
                }
                entry["dynamic_batch"] = data.get("dynamic_batch")
            except Exception:
                pass
        out[path.name] = entry
    return out or None


def to_markdown(record: dict) -> str:
    j, s, r = record["jetson"], record["stack"], record["repo"]
    rows = [
        ("Board", j.get("model")),
        ("L4T / JetPack", (j.get("l4t_release") or "").splitlines()[0] if j.get("l4t_release") else None),
        ("JetPack package", j.get("jetpack")),
        ("GPU", f"{s.get('gpu_name')} ({s.get('gpu_capability')})" if s.get("gpu_name") else None),
        ("CUDA (torch)", s.get("torch_cuda")),
        ("cuDNN", s.get("cudnn")),
        ("TensorRT", s.get("tensorrt")),
        ("PyTorch", s.get("torch")),
        ("torchvision", s.get("torchvision")),
        ("ONNX / onnxruntime", f"{s.get('onnx')} / {s.get('onnxruntime')}"),
        ("OpenCV / NumPy", f"{s.get('opencv')} / {s.get('numpy')}"),
        ("Python", s.get("python")),
        ("OS", s.get("platform")),
        ("Repo commit", f"`{(r.get('commit') or '')[:12]}`"
                        f"{' (dirty)' if r.get('dirty') else ''}" if r.get("commit") else None),
        ("Branch", r.get("branch")),
        ("Captured", record["captured_utc"]),
    ]
    lines = ["| | |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows if v]

    power = (j.get("nvpmodel") or "").replace("\n", " ").strip()
    clocks_pinned = _clocks_look_pinned(j)
    lines += [
        "",
        "### Clock state",
        "",
        f"- Power mode: `{power}`" if power else "- Power mode: not readable",
        f"- Clocks pinned (`jetson_clocks`): **{clocks_pinned}**",
    ]
    if clocks_pinned == "no":
        lines.append(
            "- ⚠️ Without pinned clocks the same command re-run can differ by tens "
            "of percent — larger than most effects being measured here."
        )
    if record.get("engines"):
        lines += ["", "### Engines used", "", "| file | MB | built (UTC) |", "|---|---|---|"]
        for name, e in record["engines"].items():
            lines.append(f"| `{name}` | {e['bytes'] / 1e6:.1f} | {e['modified'][:19]} |")
    return "\n".join(lines) + "\n"


def _clocks_look_pinned(jetson: dict) -> str:
    """"yes" / "no" / "unknown" from jetson_clocks --show.

    When clocks are pinned every governor reads `userspace`; the default
    `schedutil` (or `interactive`) means they are still free to move.
    """
    show = jetson.get("jetson_clocks")
    if not show:
        return "unknown"
    lowered = show.lower()
    if "userspace" in lowered:
        return "yes"
    if "schedutil" in lowered or "interactive" in lowered or "ondemand" in lowered:
        return "no"
    return "unknown"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", default="results/environment",
                   help="Path stem; writes <out>.json and <out>.md.")
    p.add_argument("--engines", default=None,
                   help="Engine directory to record alongside the environment.")
    args = p.parse_args(argv)

    record = {
        "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "jetson": jetson_info(),
        "stack": python_stack(),
        "repo": repo_state(),
        "engines": engine_info(Path(args.engines) if args.engines else None),
        "tools": {
            "trtexec": shutil.which("trtexec") or (
                "/usr/src/tensorrt/bin/trtexec"
                if Path("/usr/src/tensorrt/bin/trtexec").is_file() else None
            ),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    markdown = to_markdown(record)
    out.with_suffix(".md").write_text(markdown)

    print(markdown)
    print(f">> wrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
