#!/usr/bin/env python3
"""Build TensorRT engines from the ONNX graphs `export_edgetam_onnx.py` wrote.

Reads each `<name>.spec.json` next to its .onnx, so shapes and optimisation
profiles are never retyped by hand.

Builds through `trtexec` when it is on the machine, and through TensorRT's
Python API when it is not -- JetPack does not always install the samples
package that ships trtexec, even though `import tensorrt` works. Both paths
produce the same engine; `--builder` forces one.

Engines are hardware- and TensorRT-version-specific. Build them on the Orin
you will run on; do not copy from another machine.

Usage:
    python tools/build_trt_engines.py --outdir models/
    python tools/build_trt_engines.py --outdir models/ --max-batch 8
    python tools/build_trt_engines.py --outdir models/ --module memory_attention

Precision
---------
Default is --fp16, with engine inputs/outputs in fp16 too. That second part
matters more than it looks: with the default fp32 boundary TensorRT inserts
reformat kernels around every engine, and the image encoder alone would write
~45 MB of fp32 feature maps per frame. Pass --io-fp32 to get the fp32 boundary
back if you need to compare against a previous build.

--precision bf16 and --precision fp32 build the same engines at other
precisions, for comparison. bf16 is not expected to win on an Orin: it uses the
same tensor cores at the same rate as fp16 while carrying 8 bits of significand
to fp16's 11, and TensorRT has fewer bf16 tactics. Its one advantage is dynamic
range, which only matters if activations overflow fp16 -- and
tools/analyze_precision.py measures that EdgeTAM's do not.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Running this as a script puts tools/ on sys.path, not the repo root, so the
# sibling Python-API builder would not import.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULES = ("image_encoder", "memory_attention", "memory_encoder", "sam_head")


def find_trtexec(explicit: str | None) -> str | None:
    """Locate trtexec, or None if it is not installed."""
    if explicit:
        return explicit
    found = shutil.which("trtexec")
    if found:
        return found
    # JetPack installs it here and does not put it on PATH.
    jetson = Path("/usr/src/tensorrt/bin/trtexec")
    return str(jetson) if jetson.is_file() else None


def python_api_available() -> bool:
    try:
        import tensorrt  # noqa: F401
    except Exception:
        return False
    return True


def shape_tuples(spec: dict, lo: int, opt: int, hi: int) -> dict:
    """{input name: (min, opt, max)} with the batch axis substituted."""
    out = {}
    for io in spec["inputs"]:
        def at(batch):
            shape = list(io["shape"])
            shape[io["batch_axis"]] = batch
            return tuple(shape)

        out[io["name"]] = (at(lo), at(opt), at(hi))
    return out


def shape_arg(spec: dict, batch: int) -> str:
    """`name:d0xd1x...` for every input, with the batch axis overridden."""
    parts = []
    for io in spec["inputs"]:
        shape = list(io["shape"])
        shape[io["batch_axis"]] = batch
        parts.append(f"{io['name']}:" + "x".join(str(d) for d in shape))
    return ",".join(parts)


# trtexec flag and IO dtype per precision. bf16 needs TensorRT >= 9 and an
# Ampere-class GPU or newer; Orin (sm_87) qualifies.
PRECISION_FLAGS = {
    "fp16": (["--fp16"], "fp16"),
    "bf16": (["--bf16"], "bf16"),
    "fp32": ([], "fp32"),
}


def build_one(
    spec_path: Path,
    trtexec: str | None,
    *,
    precision: str,
    verbose: bool,
    min_batch: int,
    opt_batch: int,
    max_batch: int,
    workspace_mb: int,
    opt_level: int,
    io_reduced: bool,
    timing_cache: Path | None,
    extra: list[str],
    dry_run: bool,
) -> None:
    spec = json.loads(spec_path.read_text())
    outdir = spec_path.parent
    onnx = outdir / spec["onnx"]
    engine = outdir / spec["engine"]
    if not onnx.exists():
        raise SystemExit(f"{spec['module']}: ONNX missing at {onnx}; re-run the exporter.")

    print(f"\n>> {spec['module']} [{precision}]: {onnx.name} -> {engine.name}")
    if spec.get("dynamic_batch"):
        print(f"   batch profile min={min_batch} opt={opt_batch} max={max_batch}")
    else:
        print(f"   static batch={spec['batch']}")

    if trtexec is None:
        from tools._trt_build_python import build_engine

        print("   builder: TensorRT Python API (trtexec not installed)")
        if dry_run:
            return
        build_engine(
            onnx,
            engine,
            shapes=shape_tuples(spec, min_batch, opt_batch, max_batch),
            precision=precision,
            io_reduced=io_reduced,
            workspace_mb=workspace_mb,
            opt_level=opt_level,
            timing_cache=timing_cache,
            verbose=verbose,
        )
        print(f"   built {engine} ({engine.stat().st_size / 1e6:.1f} MB)")
        return

    precision_flags, io_dtype = PRECISION_FLAGS[precision]
    cmd = [
        trtexec,
        f"--onnx={onnx}",
        f"--saveEngine={engine}",
        *precision_flags,
        f"--memPoolSize=workspace:{workspace_mb}",
        f"--builderOptimizationLevel={opt_level}",
    ]
    if spec.get("dynamic_batch"):
        cmd += [
            f"--minShapes={shape_arg(spec, min_batch)}",
            f"--optShapes={shape_arg(spec, opt_batch)}",
            f"--maxShapes={shape_arg(spec, max_batch)}",
        ]
    if io_reduced:
        cmd += [
            "--inputIOFormats=" + ",".join([f"{io_dtype}:chw"] * len(spec["inputs"])),
            "--outputIOFormats=" + ",".join([f"{io_dtype}:chw"] * len(spec["outputs"])),
        ]
    if timing_cache is not None:
        cmd.append(f"--timingCacheFile={timing_cache}")
    cmd += extra

    if verbose:
        cmd.append("--verbose")
    print("   builder: trtexec")
    print("   " + " ".join(cmd))
    if dry_run:
        return

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(
            f"{spec['module']}: trtexec failed (exit {result.returncode}).\n"
            "If the error mentions IO formats or tensor formats, retry with "
            "--io-fp32; if it mentions workspace, raise --workspace."
        )
    print(f"   built {engine} ({engine.stat().st_size / 1e6:.1f} MB)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--outdir", default="models", help="Where the .onnx/.spec.json live.")
    p.add_argument("--module", default="all", choices=("all",) + MODULES)
    p.add_argument(
        "--max-batch",
        type=int,
        default=4,
        help="Largest object count the engines must accept (default: 4). "
        "Tracking more targets than this falls back to PyTorch.",
    )
    p.add_argument(
        "--opt-batch",
        type=int,
        default=1,
        help="Object count TensorRT tunes kernels for (default: 1).",
    )
    p.add_argument(
        "--image-encoder-batch",
        type=int,
        default=1,
        help="Batch profile for the image encoder (default: 1). EdgeTAM runs the "
        "encoder once per frame and broadcasts the result across objects, so a "
        "wider profile here buys nothing.",
    )
    p.add_argument("--workspace", type=int, default=4096, help="Workspace pool, MiB.")
    p.add_argument(
        "--opt-level",
        type=int,
        default=3,
        help="trtexec --builderOptimizationLevel (0-5). 5 searches harder and "
        "builds much slower; worth trying once for the memory attention.",
    )
    p.add_argument(
        "--precision",
        default="fp16",
        choices=tuple(PRECISION_FLAGS),
        help="Engine precision (default: fp16). On Orin, bf16 runs on the same "
        "tensor cores at the same rate as fp16 but carries 8 bits of significand "
        "to fp16's 11, and TensorRT has fewer bf16 tactics -- so it is a "
        "strictly worse trade here unless activations overflow fp16, which "
        "tools/analyze_precision.py says EdgeTAM's do not.",
    )
    p.add_argument(
        "--io-fp32",
        action="store_true",
        help="Keep engine inputs/outputs in fp32 instead of the engine precision.",
    )
    p.add_argument(
        "--no-timing-cache",
        action="store_true",
        help="Do not reuse trtexec's kernel timing cache between builds.",
    )
    p.add_argument("--trtexec", default=None, help="Path to trtexec.")
    p.add_argument(
        "--builder",
        default="auto",
        choices=("auto", "trtexec", "python"),
        help="auto (default) uses trtexec when present and the TensorRT Python "
        "API otherwise. Both produce the same engine.",
    )
    p.add_argument("--verbose", action="store_true", help="Full TensorRT builder log.")
    p.add_argument("--dry-run", action="store_true", help="Print commands, build nothing.")
    p.add_argument(
        "extra", nargs="*", help="Extra arguments appended verbatim to every trtexec call."
    )
    args = p.parse_args(argv)

    outdir = Path(args.outdir)
    if not outdir.is_dir():
        raise SystemExit(f"{outdir} does not exist; run tools/export_edgetam_onnx.py first.")

    wanted = MODULES if args.module == "all" else (args.module,)
    specs = []
    for name in wanted:
        spec_path = outdir / f"edgetam_{name}.spec.json"
        if spec_path.exists():
            specs.append(spec_path)
        elif args.module != "all":
            raise SystemExit(f"No spec at {spec_path}; export {name} first.")
    if not specs:
        raise SystemExit(
            f"No edgetam_*.spec.json found in {outdir}. Run:\n"
            f"  python tools/export_edgetam_onnx.py --outdir {outdir}"
        )

    trtexec = None if args.builder == "python" else find_trtexec(args.trtexec)
    if trtexec is None:
        if args.builder == "trtexec":
            raise SystemExit(
                "trtexec not found. It ships with TensorRT's samples package "
                "(JetPack puts it at /usr/src/tensorrt/bin/trtexec). Pass "
                "--trtexec to point at it, or use --builder python."
            )
        if not python_api_available():
            raise SystemExit(
                "Neither trtexec nor the TensorRT Python bindings are available. "
                "Install TensorRT, or run this on the Orin."
            )
        print("trtexec not found; building through the TensorRT Python API instead.")
    timing_cache = None if args.no_timing_cache else outdir / "trt_timing.cache"

    for spec_path in specs:
        module = spec_path.stem.replace("edgetam_", "").replace(".spec", "")
        if module == "image_encoder":
            lo = opt = hi = args.image_encoder_batch
        else:
            lo, opt, hi = 1, args.opt_batch, args.max_batch
        build_one(
            spec_path,
            trtexec,
            precision=args.precision,
            verbose=args.verbose,
            min_batch=lo,
            opt_batch=opt,
            max_batch=hi,
            workspace_mb=args.workspace,
            opt_level=args.opt_level,
            io_reduced=not args.io_fp32,
            timing_cache=timing_cache,
            extra=list(args.extra),
            dry_run=args.dry_run,
        )

    print(
        "\n>> Done. Check numerics and speed with:\n"
        f"   python tools/check_trt_parity.py --outdir {outdir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
