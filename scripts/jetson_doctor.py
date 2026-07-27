#!/usr/bin/env python3
"""Diagnose a Jetson Orin AGX environment for the EdgeTAM / TensorRT pipeline.

Run this FIRST, before anything else, and again after every fix. It is
dependency-free on purpose (stdlib only) so it still runs on a half-broken
environment — that is exactly when you need it.

    python3 scripts/jetson_doctor.py            # human-readable report
    python3 scripts/jetson_doctor.py --json     # machine-readable
    python3 scripts/jetson_doctor.py --quiet    # only WARN/BLOCKER lines

Exit codes:
    0  no blockers (warnings may exist)
    1  at least one BLOCKER — fix it before moving to the next phase
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

OK, WARN, BLOCK, INFO = "OK", "WARN", "BLOCKER", "INFO"

# Orin AGX GPU is Ampere, compute capability 8.7.
EXPECTED_ARCH = "sm_87"


class Report:
    def __init__(self) -> None:
        self.items: list[dict] = []
        self.section = "general"

    def start(self, name: str) -> None:
        self.section = name

    def add(self, level: str, name: str, detail: str = "", fix: str = "") -> None:
        self.items.append(
            {"section": self.section, "level": level, "name": name,
             "detail": detail, "fix": fix}
        )

    def blockers(self) -> list[dict]:
        return [i for i in self.items if i["level"] == BLOCK]

    def warnings(self) -> list[dict]:
        return [i for i in self.items if i["level"] == WARN]


def run(cmd: list[str] | str, timeout: int = 20) -> tuple[int, str]:
    """Run a command, never raise. Returns (returncode, combined output)."""
    try:
        proc = subprocess.run(
            cmd,
            shell=isinstance(cmd, str),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, (proc.stdout + proc.stderr).strip()
    except FileNotFoundError:
        return 127, "not found"
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except Exception as exc:  # pragma: no cover - defensive
        return 1, f"{type(exc).__name__}: {exc}"


def read_text(path: str) -> str:
    try:
        return Path(path).read_text(errors="replace").strip()
    except OSError:
        return ""


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def check_platform(r: Report) -> None:
    r.start("platform")
    machine = platform.machine()
    r.add(INFO, "arch", machine)
    if machine != "aarch64":
        r.add(
            WARN, "not-a-jetson",
            f"machine={machine}, expected aarch64 — this looks like a dev box, "
            "so the Jetson-specific checks below are not meaningful here.",
        )

    # /proc/device-tree/model is NUL-terminated.
    model = read_text("/proc/device-tree/model").replace("\x00", "").strip()
    if model:
        r.add(INFO, "board", model)
        if "orin" not in model.lower():
            r.add(WARN, "board-mismatch", f"model={model!r}, expected an Orin board.")
    elif machine == "aarch64":
        r.add(WARN, "board", "could not read /proc/device-tree/model")

    l4t = read_text("/etc/nv_tegra_release")
    if l4t:
        r.add(INFO, "l4t", l4t.splitlines()[0])
    rc, out = run(["dpkg-query", "-W", "-f=${Version}", "nvidia-jetpack"])
    if rc == 0 and out:
        r.add(INFO, "jetpack", out)
    elif machine == "aarch64":
        r.add(
            WARN, "jetpack",
            "nvidia-jetpack package not found (partial flash / non-standard image?)",
            "sudo apt update && sudo apt install nvidia-jetpack",
        )

    osr = read_text("/etc/os-release")
    m = re.search(r'PRETTY_NAME="([^"]+)"', osr)
    if m:
        r.add(INFO, "os", m.group(1))
    r.add(INFO, "python", f"{sys.version.split()[0]} @ {sys.executable}")


def check_power_mode(r: Report) -> None:
    """Read the power mode and GPU clocks WITHOUT root.

    `nvpmodel -q` and `jetson_clocks --show` both want root on most JetPack
    images, but everything they report is also exposed through sysfs, which is
    world-readable. Diagnosis should never need sudo — only *changing* the
    power mode does.
    """
    r.start("power")
    if not shutil.which("nvpmodel") and not Path("/var/lib/nvpmodel").exists():
        r.add(INFO, "nvpmodel", "not present (not a Jetson?)")
        return

    # /var/lib/nvpmodel/status holds e.g. "pmode:0000 fmode:fan_mode_quiet".
    status = read_text("/var/lib/nvpmodel/status")
    mode_id = None
    if (m := re.search(r"pmode:\s*(\d+)", status)):
        mode_id = int(m.group(1))
    elif status:
        r.add(INFO, "nvpmodel-status", status[:80])
    if mode_id is None:
        rc, out = run(["nvpmodel", "-q"])  # last resort; usually needs root
        if rc == 0 and (m := re.search(r"^\s*(\d+)\s*$", out, re.MULTILINE)):
            mode_id = int(m.group(1))

    if mode_id is None:
        r.add(WARN, "power-mode", "could not determine nvpmodel mode",
              "sudo nvpmodel -q   (and set MAXN with: sudo nvpmodel -m 0)")
    else:
        names = {}
        for line in read_text("/etc/nvpmodel.conf").splitlines():
            if (m := re.match(r"<\s*POWER_MODEL\s+ID=(\d+)\s+NAME=(\S+)\s*>", line.strip())):
                names[int(m.group(1))] = m.group(2)
        label = names.get(mode_id, "?")
        r.add(INFO, "power-mode", f"id={mode_id} ({label})")
        # Mode 0 is MAXN on every Orin SKU; anything else caps clocks/cores.
        if mode_id != 0:
            r.add(WARN, "power-mode-not-maxn",
                  f"nvpmodel id={mode_id} ({label}); benchmarks will not be comparable.",
                  "sudo nvpmodel -m 0 && sudo jetson_clocks")

    # GPU devfreq: current vs max frequency tells us whether clocks are locked.
    for node in sorted(Path("/sys/class/devfreq").glob("*")) if \
            Path("/sys/class/devfreq").is_dir() else []:
        name = node.name
        if "gpu" not in name and "gv11b" not in name and "ga10b" not in name:
            continue
        cur = read_text(str(node / "cur_freq"))
        mx = read_text(str(node / "max_freq"))
        if not (cur.isdigit() and mx.isdigit()):
            continue
        r.add(INFO, "gpu-clock", f"{name}: {int(cur) / 1e6:.0f}/{int(mx) / 1e6:.0f} MHz")
        # Idle GPUs legitimately clock down, so this is a hint, not a verdict.
        if int(cur) < int(mx):
            r.add(WARN, "clocks-not-locked",
                  f"GPU at {int(cur) / 1e6:.0f} MHz of {int(mx) / 1e6:.0f} MHz max",
                  "sudo jetson_clocks   (lock clocks before benchmarking; an idle "
                  "GPU also reads low, so re-check while a run is in flight)")
        break


def check_cuda(r: Report) -> None:
    r.start("cuda")
    cuda_home = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    if Path(cuda_home).exists():
        real = os.path.realpath(cuda_home)
        r.add(INFO, "cuda-home", f"{cuda_home} -> {real}")
    else:
        r.add(WARN, "cuda-home", f"{cuda_home} missing",
              "JetPack installs CUDA at /usr/local/cuda; check the flash.")

    nvcc = shutil.which("nvcc") or f"{cuda_home}/bin/nvcc"
    rc, out = run([nvcc, "--version"])
    if rc == 0:
        m = re.search(r"release ([\d.]+)", out)
        r.add(INFO, "nvcc", m.group(1) if m else out.splitlines()[-1])
    else:
        r.add(WARN, "nvcc", "not found",
              f"export PATH={cuda_home}/bin:$PATH  (nvcc is only needed to build, "
              "not to run)")

    rc, out = run("ldconfig -p | grep -c libcudart")
    if rc == 0 and out.strip().isdigit() and int(out.strip()) > 0:
        r.add(OK, "libcudart", f"{out.strip()} entries in ldconfig")
    else:
        r.add(
            WARN, "libcudart", "not in ldconfig cache",
            f"export LD_LIBRARY_PATH={cuda_home}/lib64:$LD_LIBRARY_PATH",
        )

    # cuDNN version header (JetPack installs it under /usr/include/aarch64-linux-gnu)
    for header in ("/usr/include/cudnn_version.h",
                   "/usr/include/aarch64-linux-gnu/cudnn_version.h"):
        txt = read_text(header)
        if txt:
            major = re.search(r"#define CUDNN_MAJOR (\d+)", txt)
            minor = re.search(r"#define CUDNN_MINOR (\d+)", txt)
            patch = re.search(r"#define CUDNN_PATCHLEVEL (\d+)", txt)
            if major:
                v = ".".join(g.group(1) for g in (major, minor, patch) if g)
                r.add(INFO, "cudnn", v)
            break
    else:
        rc, out = run(["dpkg-query", "-W", "-f=${Version}", "libcudnn9-cuda-12"])
        if rc == 0 and out:
            r.add(INFO, "cudnn", out)


def check_tensorrt(r: Report) -> None:
    r.start("tensorrt")
    # Python module — the one the runtime actually needs.
    code = "import tensorrt as t; print(t.__version__); print(t.__file__)"
    rc, out = run([sys.executable, "-c", code])
    if rc == 0:
        version, path = (out.splitlines() + ["", ""])[:2]
        r.add(OK, "python-module", f"tensorrt {version} ({path})")
        major = int(version.split(".")[0]) if version and version[0].isdigit() else 0
        if major and major < 10:
            r.add(
                WARN, "trt-version",
                f"TensorRT {version}: this repo targets the TRT 10 API "
                "(set_input_shape / execute_async_v3).",
                "Upgrade JetPack, or adapt src/trackers/_trt_runtime.py to the 8.x API.",
            )
    else:
        in_venv = sys.prefix != sys.base_prefix
        fix = ("JetPack installs TensorRT into the SYSTEM dist-packages via apt. "
               "A plain venv cannot see it.\n"
               "     FIX: python3 -m venv --system-site-packages .venv  "
               "(recreate the venv)") if in_venv else \
              "sudo apt install python3-libnvinfer python3-libnvinfer-dev"
        r.add(BLOCK, "python-module", "import tensorrt failed", fix)

    # trtexec
    trtexec = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
    if Path(trtexec).exists():
        rc, out = run([trtexec, "--help"], timeout=30)
        r.add(OK, "trtexec", trtexec)
    else:
        r.add(WARN, "trtexec", "not found",
              "Jetson ships it at /usr/src/tensorrt/bin/trtexec — add to PATH. "
              "tools/build_trt_engine.py does not need it (pure Python API).")

    rc, out = run(["dpkg-query", "-W", "-f=${Version}", "libnvinfer10"])
    if rc == 0 and out:
        r.add(INFO, "libnvinfer", out)


# A missing CUDA shared library -> how to supply it. JetPack installs CUDA as
# ~15 separate apt packages, and a partial flash routinely leaves the math
# libraries out. torch links ALL of them into one libtorch_cuda.so whether it
# calls them or not, so the dynamic linker refuses to load it and `import torch`
# dies on line one — long before you would ever call torch.linalg or cuRAND.
_CUDA_LIB_FIX = {
    "libcurand": ("libcurand-12-6", "nvidia-curand-cu12"),
    "libcusolver": ("libcusolver-12-6", "nvidia-cusolver-cu12"),
    "libcusparse": ("libcusparse-12-6", "nvidia-cusparse-cu12"),
    "libcublas": ("libcublas-12-6", "nvidia-cublas-cu12"),
    "libcufft": ("libcufft-12-6", "nvidia-cufft-cu12"),
    "libnvrtc": ("cuda-nvrtc-12-6", "nvidia-cuda-nvrtc-cu12"),
    "libcupti": ("cuda-cupti-12-6", "nvidia-cuda-cupti-cu12"),
    "libcufile": ("libcufile-12-6", "nvidia-cufile-cu12"),
    "libnvjitlink": ("libnvjitlink-12-6", "nvidia-nvjitlink-cu12"),
    "libcudart": ("cuda-cudart-12-6", "nvidia-cuda-runtime-cu12"),
    "libcudnn": ("libcudnn9-cuda-12", None),
    "libcusparseLt": ("libcusparselt0", None),
    "libcudss": ("libcudss0-cuda-12", None),
}


def _find_torch_lib_dir() -> Path | None:
    """Locate torch/lib WITHOUT importing torch — the whole point is that the
    import is what's broken."""
    for base in sys.path:
        candidate = Path(base) / "torch" / "lib"
        if candidate.is_dir():
            return candidate
    return None


def check_torch_shared_libs(r: Report) -> None:
    r.start("pytorch-libs")
    lib_dir = _find_torch_lib_dir()
    if lib_dir is None:
        r.add(INFO, "torch-libs", "torch not installed yet")
        return
    if not shutil.which("ldd"):
        r.add(INFO, "torch-libs", "ldd not available; skipping")
        return

    missing: set[str] = set()
    for so in sorted(lib_dir.glob("*.so")):
        rc, out = run(["ldd", str(so)], timeout=60)
        for line in out.splitlines():
            if "not found" in line:
                missing.add(line.split("=>")[0].strip())
    if not missing:
        r.add(OK, "shared-libs", f"all resolved ({lib_dir})")
        return

    apt, pip = [], []
    for soname in sorted(missing):
        stem = soname.split(".so")[0]
        apt_pkg, pip_pkg = _CUDA_LIB_FIX.get(stem, (None, None))
        if apt_pkg:
            apt.append(apt_pkg)
        if pip_pkg:
            pip.append(pip_pkg)
    fix = f"sudo apt install -y {' '.join(sorted(set(apt)))}" if apt else \
        "identify the providing CUDA package and install it"
    if pip:
        fix += ("\nor, without root, into the venv:\n"
                f"pip install {' '.join(sorted(set(pip)))}\n"
                "then prepend those site-packages/nvidia/*/lib dirs to "
                "LD_LIBRARY_PATH (keep /usr/local/cuda/targets/aarch64-linux/lib "
                "FIRST so JetPack's Tegra builds still win).")
    r.add(BLOCK, "shared-libs", f"unresolved: {', '.join(sorted(missing))}", fix)


def check_torch(r: Report) -> None:
    r.start("pytorch")
    code = (
        "import json, torch;"
        "d=dict(version=torch.__version__, cuda=torch.version.cuda,"
        "avail=torch.cuda.is_available(), file=torch.__file__,"
        "archs=torch.cuda.get_arch_list());"
        "d['name']=torch.cuda.get_device_name(0) if d['avail'] else None;"
        "d['cap']=list(torch.cuda.get_device_capability(0)) if d['avail'] else None;"
        "print(json.dumps(d))"
    )
    rc, out = run([sys.executable, "-c", code], timeout=120)
    if rc != 0:
        r.add(BLOCK, "import", f"import torch failed: {out.splitlines()[-1] if out else ''}",
              "Install the JetPack-matched wheel from "
              "https://developer.download.nvidia.com/compute/redist/jp/ "
              "(NOT `pip install torch` — the PyPI aarch64 wheel is CPU-only).")
        return

    try:
        d = json.loads(out.splitlines()[-1])
    except (ValueError, IndexError):
        r.add(WARN, "probe", f"unexpected output: {out[:200]}")
        return

    r.add(INFO, "version", f"{d['version']} (built with CUDA {d['cuda']})")
    if not d["cuda"]:
        r.add(
            BLOCK, "cpu-only-wheel",
            "torch.version.cuda is None — this is a CPU-only build.",
            "pip uninstall -y torch torchvision && install the NVIDIA Jetson wheel "
            "matching your JetPack.",
        )
        return
    if not d["avail"]:
        r.add(
            BLOCK, "cuda-unavailable",
            "torch.cuda.is_available() == False despite a CUDA build.",
            "Check: (1) wheel matches JetPack, (2) LD_LIBRARY_PATH has "
            "/usr/local/cuda/lib64, (3) user is in the `video` group, "
            "(4) `sudo systemctl status nvargus-daemon`.",
        )
        return

    r.add(OK, "cuda", f"{d['name']} cc {d['cap'][0]}.{d['cap'][1]}")
    archs = d.get("archs") or []
    r.add(INFO, "arch-list", ", ".join(archs) or "(empty)")
    if archs and EXPECTED_ARCH not in archs:
        r.add(
            BLOCK, "wrong-arch-wheel",
            f"torch was not compiled for {EXPECTED_ARCH} (Orin). arch_list={archs}",
            "This is an SBSA/x86 CUDA wheel, not a Jetson wheel. Kernels will fail "
            "at launch. Install the NVIDIA Jetson wheel.",
        )

    rc, out = run([sys.executable, "-c",
                   "import torchvision, torch; print(torchvision.__version__); "
                   "import torchvision.ops as o; print(bool(o.nms))"], timeout=120)
    if rc == 0:
        r.add(OK, "torchvision", out.splitlines()[0])
    else:
        tail = out.splitlines()[-1] if out else ""
        r.add(WARN, "torchvision", tail,
              "torchvision must match the torch version exactly. Build from source "
              "against the installed torch, or use the matching Jetson wheel.")


def _version_tuple(text: str) -> tuple:
    parts = []
    for chunk in re.split(r"[.\-+]", text):
        if chunk.isdigit():
            parts.append(int(chunk))
        else:
            break
    return tuple(parts)


def check_python_deps(r: Report) -> None:
    r.start("python-deps")
    required = {
        "numpy": BLOCK,
        "cv2": BLOCK,
        "yaml": BLOCK,
        "tqdm": BLOCK,
        "PIL": BLOCK,
        "timm": BLOCK,
        "matplotlib": WARN,
        "onnx": WARN,
        "onnxruntime": WARN,
        "decord": INFO,
    }
    # Minimums from requirements.txt. In a --system-site-packages venv these are
    # easy to miss: pip accepts an ancient apt-installed package as satisfying
    # the requirement and never puts a current one in the venv.
    minimums = {
        "numpy": ("1.24", "numpy"),
        "cv2": ("4.8", "opencv-python"),
        "yaml": ("6.0", "pyyaml"),
        "tqdm": ("4.66", "tqdm"),
        "PIL": ("10.0", "Pillow"),
        "timm": ("1.0", "timm"),
        "matplotlib": ("3.7", "matplotlib"),
    }
    for mod, level in required.items():
        rc, out = run([sys.executable, "-c",
                       f"import {mod}; print(getattr({mod}, '__version__', '?'));"
                       f"print({mod}.__file__)"],
                      timeout=90)
        if rc == 0:
            lines = out.splitlines()
            version = lines[-2] if len(lines) >= 2 else lines[-1]
            path = lines[-1] if len(lines) >= 2 else ""
            from_system = "dist-packages" in path
            r.add(OK, mod, f"{version}{'  (system dist-packages)' if from_system else ''}")

            floor, pip_name = minimums.get(mod, (None, None))
            if floor and _version_tuple(version) < _version_tuple(floor):
                where = ("It is being picked up from the system dist-packages, so pip "
                         "considered the requirement already met and never installed a "
                         "current one into the venv. " if from_system else "")
                r.add(WARN, f"{mod}-too-old",
                      f"{version} < {floor} required by requirements.txt",
                      f"{where}pip install --force-reinstall '{pip_name}>={floor}'")
            continue
        if mod == "decord":
            r.add(INFO, "decord",
                  "absent — pipeline falls back to JPG mode automatically "
                  "(no aarch64 wheel upstream).")
        elif mod == "onnxruntime":
            r.add(WARN, "onnxruntime", "absent",
                  "Needed for `export_image_encoder_onnx.py --verify`, which is the "
                  "gate that proves the exported graph still matches PyTorch before "
                  "you spend build time on TensorRT.\n"
                  "     pip install onnxruntime   <- the CPU wheel, which is all the "
                  "parity check uses and which PyPI does ship for aarch64.\n"
                  "     Only onnxruntime-GPU needs NVIDIA's Jetson-specific build.")
        else:
            hint = {"cv2": "pip install opencv-python  (or sudo apt install python3-opencv)",
                    "PIL": "pip install Pillow",
                    "yaml": "pip install pyyaml"}.get(mod, f"pip install {mod}")
            r.add(level, mod, "not importable", hint)

    # numpy 2.x vs Jetson torch wheels compiled against numpy 1.x
    rc, out = run([sys.executable, "-c", "import numpy; print(numpy.__version__)"])
    if rc == 0 and out.startswith("2."):
        r.add(WARN, "numpy-2x", f"numpy {out}",
              "Jetson torch wheels are usually built against numpy 1.x. If you see "
              "'_ARRAY_API not found' or ABI errors: pip install 'numpy<2'")


def check_venv(r: Report) -> None:
    r.start("venv")
    in_venv = sys.prefix != sys.base_prefix
    if not in_venv:
        r.add(INFO, "venv", "not in a virtualenv (using system python)")
        return
    r.add(INFO, "venv", sys.prefix)
    sees_system = any("dist-packages" in p for p in sys.path)
    if sees_system:
        r.add(OK, "system-site-packages", "system dist-packages visible")
    else:
        r.add(
            WARN, "system-site-packages",
            "venv does NOT see /usr/lib/python3/dist-packages",
            "TensorRT (and often cv2) are apt-installed there. Recreate with:\n"
            "     python3 -m venv --system-site-packages .venv",
        )


def check_project(r: Report) -> None:
    r.start("project")
    root = Path(__file__).resolve().parent.parent
    rc, out = run([sys.executable, "-c",
                   "import sam2; print(sam2.__file__)"], timeout=120)
    if rc == 0:
        r.add(OK, "sam2 (EdgeTAM)", out.splitlines()[-1])
    else:
        r.add(BLOCK, "sam2 (EdgeTAM)", "import sam2 failed",
              "bash scripts/setup_edgetam.sh")

    ckpt = root / "third_party/EdgeTAM/checkpoints/edgetam.pt"
    if ckpt.exists():
        r.add(OK, "checkpoint", f"{ckpt} ({ckpt.stat().st_size / 1e6:.0f} MB)")
    else:
        r.add(BLOCK, "checkpoint", f"missing: {ckpt}",
              "bash scripts/setup_edgetam.sh  (downloads via EdgeTAM's "
              "checkpoints/download_ckpts.sh)")

    onnx = root / "models/edgetam_image_encoder.onnx"
    r.add(INFO, "onnx", str(onnx) if onnx.exists() else "not exported yet (Phase 2)")
    engines = sorted((root / "models").glob("*.engine")) if (root / "models").is_dir() else []
    r.add(INFO, "engines", ", ".join(e.name for e in engines) or "none built yet (Phase 3)")

    # Verify the setup script's two source patches actually landed.
    timm_py = root / "third_party/EdgeTAM/sam2/modeling/backbones/timm.py"
    if timm_py.exists() and "pretrained=True" in read_text(str(timm_py)):
        r.add(WARN, "repvit-pretrained-patch", "pretrained=True still present",
              "Re-run scripts/setup_edgetam.sh — the RepViT trunk will try to "
              "download ImageNet weights at build time (fails offline / behind a proxy).")
    perceiver = root / "third_party/EdgeTAM/sam2/modeling/perceiver.py"
    if perceiver.exists() and "expand(B, -1, -1).view(" in read_text(str(perceiver)):
        r.add(WARN, "perceiver-multiobj-patch", "unpatched .view() present",
              "Re-run scripts/setup_edgetam.sh — multi-object tracking (B>1) "
              "will raise 'view size is not compatible'.")


def check_resources(r: Report) -> None:
    r.start("resources")
    meminfo = read_text("/proc/meminfo")
    m = re.search(r"MemTotal:\s+(\d+) kB", meminfo)
    a = re.search(r"MemAvailable:\s+(\d+) kB", meminfo)
    if m:
        total_gb = int(m.group(1)) / 1024 / 1024
        avail_gb = int(a.group(1)) / 1024 / 1024 if a else 0
        r.add(INFO, "ram", f"{total_gb:.1f} GB total, {avail_gb:.1f} GB available")
        if avail_gb < 4:
            r.add(WARN, "ram-low", f"{avail_gb:.1f} GB free",
                  "TRT engine builds are memory-hungry. Close other apps or add swap: "
                  "sudo fallocate -l 8G /swapfile && sudo mkswap /swapfile && "
                  "sudo swapon /swapfile")
    sw = re.search(r"SwapTotal:\s+(\d+) kB", meminfo)
    if sw:
        r.add(INFO, "swap", f"{int(sw.group(1)) / 1024 / 1024:.1f} GB")

    try:
        usage = shutil.disk_usage(str(Path(__file__).resolve().parent.parent))
        free_gb = usage.free / 1e9
        r.add(INFO, "disk-free", f"{free_gb:.1f} GB")
        # Rough budget for the remaining phases: ONNX ~0.1 GB, ~10 engines at
        # 30-60 MB each plus their build scratch, and a JPG frame cache that
        # scales with clip length (a few hundred 1080p frames is ~0.2 GB).
        # Running out mid-build wastes a long TensorRT run, so warn early.
        if free_gb < 10:
            r.add(WARN, "disk-low", f"{free_gb:.1f} GB free",
                  "The remaining phases want ~3-4 GB (ONNX, ~10 engines, frame "
                  "caches, mask dumps). Reclaim some first: pip cache purge, "
                  "delete old frame caches under runs/, or drop an unused conda "
                  "install.")
    except OSError:
        pass


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

_COLOR = {OK: "\033[32m", WARN: "\033[33m", BLOCK: "\033[31m", INFO: "\033[90m"}
_RESET = "\033[0m"


def render(r: Report, quiet: bool) -> None:
    use_color = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    section = None
    for item in r.items:
        if quiet and item["level"] in (OK, INFO):
            continue
        if item["section"] != section:
            section = item["section"]
            print(f"\n── {section} " + "─" * max(0, 60 - len(section)))
        tag = item["level"]
        colored = f"{_COLOR[tag]}{tag:<7}{_RESET}" if use_color else f"{tag:<7}"
        line = f"  {colored} {item['name']}"
        if item["detail"]:
            line += f": {item['detail']}"
        print(line)
        if item["fix"]:
            for i, fixline in enumerate(item["fix"].splitlines()):
                prefix = "          FIX: " if i == 0 else "               "
                print(f"{prefix}{fixline.strip()}")

    blockers, warns = r.blockers(), r.warnings()
    print("\n" + "=" * 64)
    if blockers:
        print(f"{len(blockers)} BLOCKER(S) — fix these before continuing:")
        for b in blockers:
            print(f"  - [{b['section']}] {b['name']}: {b['detail']}")
    else:
        print("No blockers. Environment is ready for the next phase.")
    if warns:
        print(f"{len(warns)} warning(s) — see above; usually non-fatal.")
    print("=" * 64)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", action="store_true", help="Emit JSON instead of a report.")
    p.add_argument("--quiet", action="store_true", help="Only show WARN/BLOCKER lines.")
    args = p.parse_args()

    r = Report()
    for check in (check_platform, check_power_mode, check_cuda, check_tensorrt,
                  check_venv, check_torch_shared_libs, check_torch,
                  check_python_deps, check_project, check_resources):
        try:
            check(r)
        except Exception as exc:  # a broken check must not hide the others
            r.add(WARN, f"{check.__name__}-crashed", f"{type(exc).__name__}: {exc}")

    if args.json:
        print(json.dumps({
            "items": r.items,
            "blockers": len(r.blockers()),
            "warnings": len(r.warnings()),
        }, indent=2))
    else:
        render(r, args.quiet)
    return 1 if r.blockers() else 0


if __name__ == "__main__":
    sys.exit(main())
