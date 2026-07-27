#!/usr/bin/env python3
"""Build a TensorRT engine from the EdgeTAM image-encoder ONNX file.

Why a Python builder instead of just `trtexec`? Because INT8 needs a real
calibrator fed with real, correctly-preprocessed frames, and `trtexec` can
only *read* a calibration cache, not produce one. Everything else (fp16,
bf16, sparsity, DLA, timing cache, layer-precision overrides) is exposed
here too so the whole quantization matrix comes from one command.

Precision modes
---------------
    fp32       true FP32 (TF32 explicitly disabled -- an honest baseline)
    tf32       TF32 allowed (this is TensorRT's default on Ampere/Orin)
    fp16       FP16 tensor cores; the main win on Orin
    bf16       BF16; same speed class as FP16, wider dynamic range
    int8       INT8 PTQ, requires --calib-video/--calib-dir or --calib-cache
    int8-fp16  INT8 + FP16, TensorRT picks per layer
    best       everything on, TensorRT picks per layer

Orin AGX is Ampere (sm_87): FP8 and INT4 are not available on this hardware.

Examples
--------
    python3 tools/build_trt_engine.py \\
        --onnx models/edgetam_image_encoder.onnx \\
        --engine models/enc_fp16.engine --precision fp16

    python3 tools/build_trt_engine.py \\
        --onnx models/edgetam_image_encoder.onnx \\
        --engine models/enc_int8.engine --precision int8 \\
        --calib-video samples/synthetic_car.mp4 --calib-num 200 \\
        --calib-cache models/enc_int8.calib

    python3 tools/build_trt_engine.py --inspect models/enc_fp16.engine
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PRECISIONS = ("fp32", "tf32", "fp16", "bf16", "int8", "int8-fp16", "best")


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()[:16]


def _set_flag(trt, config, name: str) -> bool:
    """Set a BuilderFlag if this TensorRT version has it. Returns success."""
    flag = getattr(trt.BuilderFlag, name, None)
    if flag is None:
        return False
    config.set_flag(flag)
    return True


def _apply_precision(trt, builder, config, precision: str, report: dict) -> None:
    enabled: list[str] = []

    if precision == "fp32":
        # TF32 is ON by default on Ampere. Leaving it on and calling the result
        # "FP32" is the classic way to under-report the speedup of FP16.
        tf32 = getattr(trt.BuilderFlag, "TF32", None)
        if tf32 is not None:
            config.clear_flag(tf32)
            enabled.append("TF32=off")
    elif precision == "tf32":
        enabled.append("TF32=on (default)")
    else:
        if precision in ("fp16", "int8-fp16", "best"):
            if not builder.platform_has_fast_fp16:
                print("[build] WARNING: platform reports no fast FP16")
            enabled.append("FP16" if _set_flag(trt, config, "FP16") else "FP16(unavailable)")
        if precision in ("bf16", "best"):
            if _set_flag(trt, config, "BF16"):
                enabled.append("BF16")
            else:
                print("[build] WARNING: this TensorRT build has no BF16 flag; skipping")
        if precision in ("int8", "int8-fp16", "best"):
            if not builder.platform_has_fast_int8:
                print("[build] WARNING: platform reports no fast INT8")
            enabled.append("INT8" if _set_flag(trt, config, "INT8") else "INT8(unavailable)")

    report["flags"] = enabled
    print(f"[build] precision={precision} flags={enabled}")


def _apply_layer_overrides(trt, network, config, fp16_re: str | None,
                           fp32_re: str | None, report: dict) -> None:
    """Pin specific layers to a precision (INT8 accuracy rescue valve)."""
    if not fp16_re and not fp32_re:
        return
    rules = [(re.compile(fp16_re), trt.float16, "fp16")] if fp16_re else []
    if fp32_re:
        rules.append((re.compile(fp32_re), trt.float32, "fp32"))

    _set_flag(trt, config, "OBEY_PRECISION_CONSTRAINTS")
    pinned: list[str] = []
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        for pattern, dtype, label in rules:
            if pattern.search(layer.name):
                layer.precision = dtype
                for o in range(layer.num_outputs):
                    layer.set_output_type(o, dtype)
                pinned.append(f"{layer.name}={label}")
                break
    report["pinned_layers"] = pinned
    print(f"[build] pinned {len(pinned)} layer(s) to an explicit precision")
    if not pinned:
        print("[build] WARNING: no layer names matched — check the regex against "
              "`--dry-run` output")


def _add_profile_if_dynamic(trt, builder, network, config, shape, report: dict) -> None:
    dynamic = []
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        if any(d < 0 for d in tensor.shape):
            dynamic.append(tensor)
    if not dynamic:
        return
    profile = builder.create_optimization_profile()
    for tensor in dynamic:
        profile.set_shape(tensor.name, shape, shape, shape)
        print(f"[build] dynamic input {tensor.name}{tuple(tensor.shape)} "
              f"-> fixed profile {tuple(shape)}")
    config.add_optimization_profile(profile)
    report["profile"] = {"shape": list(shape)}


def _load_timing_cache(trt, config, path: Path | None) -> object | None:
    if path is None:
        return None
    buf = path.read_bytes() if path.exists() else b""
    cache = config.create_timing_cache(buf)
    if cache is None:
        print("[build] timing cache unsupported by this TensorRT version")
        return None
    config.set_timing_cache(cache, ignore_mismatch=False)
    print(f"[build] timing cache: {path} ({len(buf)} bytes in)")
    return cache


def build(args) -> int:
    import tensorrt as trt

    onnx_path = Path(args.onnx)
    engine_path = Path(args.engine)
    if not onnx_path.exists():
        raise SystemExit(f"ONNX not found: {onnx_path}\n"
                         "Run tools/export_image_encoder_onnx.py first (Phase 2).")
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    logger = trt.Logger(trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)

    # EXPLICIT_BATCH is implicit (and deprecated) in TRT 10, still required in 8.x.
    flags = 0
    explicit = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    if explicit is not None and int(trt.__version__.split(".")[0]) < 10:
        flags |= 1 << int(explicit)
    network = builder.create_network(flags)

    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        print(f"[build] ONNX parse FAILED ({parser.num_errors} error(s)):")
        for i in range(parser.num_errors):
            print(f"  [{i}] {parser.get_error(i)}")
        raise SystemExit(
            "Parse failure usually means the opset is newer than this TensorRT. "
            "Re-export with a lower --opset, or check the doctor's TRT version."
        )

    print(f"[build] parsed {onnx_path.name}: {network.num_layers} layers, "
          f"{network.num_inputs} input(s), {network.num_outputs} output(s)")
    for i in range(network.num_inputs):
        t = network.get_input(i)
        print(f"        in  {t.name} {tuple(t.shape)} {t.dtype}")
    for i in range(network.num_outputs):
        t = network.get_output(i)
        print(f"        out {t.name} {tuple(t.shape)} {t.dtype}")

    if args.dry_run:
        for i in range(network.num_layers):
            print(f"  layer[{i:4d}] {network.get_layer(i).name}")
        return 0

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace * (1 << 20))
    report: dict = {}
    _apply_precision(trt, builder, config, args.precision, report)

    if args.opt_level is not None and hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = args.opt_level
        print(f"[build] builder_optimization_level={args.opt_level}")

    if args.sparsity:
        if _set_flag(trt, config, "SPARSE_WEIGHTS"):
            report["sparsity"] = True
            print("[build] 2:4 structured sparsity enabled (no gain unless the "
                  "weights were trained sparse — measure, don't assume)")

    if args.dla_core is not None:
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = args.dla_core
        if args.allow_gpu_fallback:
            _set_flag(trt, config, "GPU_FALLBACK")
        # DLA only runs FP16/INT8; without one of them the build fails.
        if args.precision in ("fp32", "tf32"):
            _set_flag(trt, config, "FP16")
            print("[build] DLA requires FP16/INT8; FP16 force-enabled")
        report["dla_core"] = args.dla_core
        print(f"[build] targeting DLA core {args.dla_core} "
              f"(gpu_fallback={args.allow_gpu_fallback})")

    input_shape = [1, 3, args.input_size, args.input_size]
    _add_profile_if_dynamic(trt, builder, network, config, input_shape, report)
    _apply_layer_overrides(trt, network, config, args.fp16_layers, args.fp32_layers, report)

    calibrator = None
    if "int8" in args.precision or args.precision == "best":
        calibrator = _make_calibrator(trt, args, input_shape, report)
        if calibrator is not None:
            config.int8_calibrator = calibrator

    timing_cache = _load_timing_cache(
        trt, config, Path(args.timing_cache) if args.timing_cache else None)

    print("[build] building engine — this can take several minutes...")
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_s = time.perf_counter() - t0
    if serialized is None:
        raise SystemExit(
            f"Engine build FAILED after {build_s:.1f}s. Re-run with --verbose for "
            "the TensorRT log. Common causes: workspace too small (--workspace), "
            "out of system memory, or an unsupported layer/precision combination."
        )

    engine_path.write_bytes(memoryview(serialized))
    if timing_cache is not None and args.timing_cache:
        Path(args.timing_cache).parent.mkdir(parents=True, exist_ok=True)
        Path(args.timing_cache).write_bytes(memoryview(timing_cache.serialize()))

    size_mb = engine_path.stat().st_size / 1e6
    print(f"[build] OK  {engine_path}  {size_mb:.1f} MB  in {build_s:.1f}s")

    meta = {
        "engine": str(engine_path),
        "onnx": str(onnx_path),
        "onnx_sha256_16": _sha256(onnx_path),
        "precision": args.precision,
        "input_shape": input_shape,
        "workspace_mib": args.workspace,
        "build_seconds": round(build_s, 2),
        "engine_mb": round(size_mb, 2),
        "tensorrt": trt.__version__,
        "machine": platform.machine(),
        "calibration": report.get("calibration"),
        **{k: v for k, v in report.items() if k != "calibration"},
    }
    meta_path = engine_path.with_suffix(engine_path.suffix + ".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[build] metadata -> {meta_path}")
    return 0


def _make_calibrator(trt, args, input_shape, report: dict):
    from tools.calibration import calibration_batches, make_entropy_calibrator

    cache = Path(args.calib_cache) if args.calib_cache else None
    has_data = bool(args.calib_video or args.calib_dir)

    if not has_data:
        if cache and cache.exists():
            print(f"[build] no calibration data given; using cache {cache}")
            report["calibration"] = {"source": "cache", "cache": str(cache)}
            return make_entropy_calibrator(trt, iter(()), cache)
        raise SystemExit(
            "INT8 requires calibration. Pass --calib-video <mp4> or --calib-dir "
            "<frames/> (optionally with --calib-cache to reuse it later).\n"
            "Building with --int8 and no calibrator gives every tensor a dynamic "
            "range of 1.0 and the engine will output garbage."
        )

    batches = calibration_batches(
        video=args.calib_video,
        frames_dir=args.calib_dir,
        size=args.input_size,
        limit=args.calib_num,
        stride=args.calib_stride,
        batch_size=input_shape[0],
    )
    source = args.calib_video or args.calib_dir
    print(f"[build] calibrating on {args.calib_num} frames from {source} "
          f"(stride {args.calib_stride})")
    report["calibration"] = {
        "source": str(source),
        "num_frames": args.calib_num,
        "stride": args.calib_stride,
        "cache": str(cache) if cache else None,
        "algorithm": "IInt8EntropyCalibrator2",
    }
    return make_entropy_calibrator(trt, batches, cache)


def inspect(engine_path: str) -> int:
    import tensorrt as trt

    path = Path(engine_path)
    if not path.exists():
        raise SystemExit(f"Engine not found: {path}")
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise SystemExit(
            f"Could not deserialize {path}. Engines are specific to the GPU AND "
            "the TensorRT version that built them — rebuild on this device."
        )
    print(f"engine: {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"tensorrt: {trt.__version__}")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = "IN " if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "OUT"
        print(f"  {mode} {name} {tuple(engine.get_tensor_shape(name))} "
              f"{engine.get_tensor_dtype(name)}")
    meta = path.with_suffix(path.suffix + ".json")
    if meta.exists():
        print("\nbuild metadata:")
        print(meta.read_text())
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inspect", metavar="ENGINE",
                   help="Print an existing engine's I/O and metadata, then exit.")
    p.add_argument("--onnx", default="models/edgetam_image_encoder.onnx")
    p.add_argument("--engine", default="models/edgetam_image_encoder.engine")
    p.add_argument("--precision", default="fp16", choices=PRECISIONS)
    p.add_argument("--input-size", type=int, default=1024)
    p.add_argument("--workspace", type=int, default=4096, help="Workspace pool, MiB.")
    p.add_argument("--opt-level", type=int, default=None,
                   help="builderOptimizationLevel 0-5 (higher = slower build, "
                        "possibly faster engine).")
    p.add_argument("--sparsity", action="store_true", help="Enable 2:4 sparse weights.")
    p.add_argument("--dla-core", type=int, default=None, help="Target a DLA core (0/1).")
    p.add_argument("--allow-gpu-fallback", action="store_true",
                   help="With --dla-core: let unsupported layers run on the GPU "
                        "(effectively mandatory).")
    p.add_argument("--timing-cache", default="models/timing.cache",
                   help="Reuse tactic timings across builds; big build-time saver.")
    p.add_argument("--fp16-layers", default=None,
                   help="Regex of layer names to pin to FP16 (rescue INT8 accuracy).")
    p.add_argument("--fp32-layers", default=None,
                   help="Regex of layer names to pin to FP32.")
    p.add_argument("--calib-video", default=None, help="Video to calibrate INT8 from.")
    p.add_argument("--calib-dir", default=None, help="Frame folder to calibrate from.")
    p.add_argument("--calib-num", type=int, default=200, help="Calibration frames.")
    p.add_argument("--calib-stride", type=int, default=1,
                   help="Take every Nth frame (spreads samples over the clip).")
    p.add_argument("--calib-cache", default=None,
                   help="Read/write the calibration cache here. Delete it after "
                        "changing the ONNX.")
    p.add_argument("--dry-run", action="store_true",
                   help="Parse the ONNX and list layers without building.")
    p.add_argument("--verbose", action="store_true", help="TensorRT VERBOSE logging.")
    args = p.parse_args()

    try:
        import tensorrt  # noqa: F401
    except ImportError:
        raise SystemExit(
            "import tensorrt failed. On Jetson TensorRT comes from apt and lives in "
            "the system dist-packages — a plain venv cannot see it. Recreate the "
            "venv with `python3 -m venv --system-site-packages .venv`, or run "
            "`python3 scripts/jetson_doctor.py` for the full diagnosis."
        )

    if args.inspect:
        return inspect(args.inspect)
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
