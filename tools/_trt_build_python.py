"""Build a TensorRT engine through the Python API instead of `trtexec`.

`trtexec` lives in TensorRT's samples package, which JetPack does not always
install even though the Python bindings are present. Everything trtexec does
for us -- precision flags, an optimisation profile, IO formats, a workspace
limit, a timing cache -- has a direct Python equivalent, so a missing binary
is no reason to stop.

The Python path is arguably the better one anyway: shapes come straight from
the exporter's spec as tuples rather than being formatted into a command-line
string and re-parsed, and parser errors arrive as text instead of an exit code.

Target: TensorRT 10 (JetPack 6). Calls that moved between 8.x and 10 are
guarded so this still works on JetPack 5.
"""
from __future__ import annotations

from pathlib import Path


def _network_flags(trt):
    """Explicit batch: a flag on TensorRT 8/9, the only option on 10."""
    flags = getattr(trt, "NetworkDefinitionCreationFlag", None)
    if flags is not None and hasattr(flags, "EXPLICIT_BATCH"):
        return 1 << int(flags.EXPLICIT_BATCH)
    return 0


def _io_dtype(trt, precision: str):
    return {
        "fp16": trt.float16,
        "bf16": getattr(trt, "bfloat16", trt.float16),
        "fp32": trt.float32,
    }[precision]


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    *,
    shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
    precision: str = "fp16",
    io_reduced: bool = True,
    workspace_mb: int = 4096,
    opt_level: int = 3,
    timing_cache: Path | None = None,
    verbose: bool = False,
) -> None:
    """Parse `onnx_path` and serialise an engine to `engine_path`.

    `shapes` maps each input name to (min, opt, max) shape tuples.
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(_network_flags(trt))
    parser = trt.OnnxParser(network, logger)

    blob = onnx_path.read_bytes()
    if not parser.parse(blob, str(onnx_path.parent)):
        errors = "\n  ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise SystemExit(f"Failed to parse {onnx_path.name}:\n  {errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = opt_level

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "bf16":
        if not hasattr(trt.BuilderFlag, "BF16"):
            raise SystemExit(
                "This TensorRT build has no BF16 flag; it needs TensorRT 9 or newer. "
                "Use --precision fp16."
            )
        config.set_flag(trt.BuilderFlag.BF16)

    # Optimisation profile: one profile covering the batch range.
    profile = builder.create_optimization_profile()
    declared = {network.get_input(i).name for i in range(network.num_inputs)}
    missing = declared - set(shapes)
    if missing:
        raise SystemExit(f"{onnx_path.name}: no shape given for input(s) {sorted(missing)}")
    for name in declared:
        lo, opt, hi = shapes[name]
        profile.set_shape(name, lo, opt, hi)
    config.add_optimization_profile(profile)

    # Force the IO precision so TensorRT does not wrap every engine in fp32
    # reformat kernels. LINEAR is trtexec's "chw".
    if io_reduced and precision != "fp32":
        dtype = _io_dtype(trt, precision)
        linear = 1 << int(trt.TensorFormat.LINEAR)
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            tensor.dtype = dtype
            tensor.allowed_formats = linear
        for i in range(network.num_outputs):
            tensor = network.get_output(i)
            tensor.dtype = dtype
            tensor.allowed_formats = linear

    cache = None
    if timing_cache is not None and hasattr(config, "create_timing_cache"):
        previous = timing_cache.read_bytes() if timing_cache.exists() else b""
        cache = config.create_timing_cache(previous)
        if cache is not None:
            config.set_timing_cache(cache, ignore_mismatch=False)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit(
            f"{onnx_path.name}: TensorRT returned no engine. Re-run with --verbose "
            "for the builder log; if it mentions memory, raise --workspace."
        )

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))

    if cache is not None and timing_cache is not None:
        try:
            timing_cache.write_bytes(bytes(cache.serialize()))
        except Exception:
            pass  # a lost cache only costs build time on the next run

    # Deserialise once so a broken engine fails here rather than mid-tracking.
    runtime = trt.Runtime(logger)
    if runtime.deserialize_cuda_engine(engine_path.read_bytes()) is None:
        raise SystemExit(f"{engine_path.name} was written but will not deserialise.")
