"""TensorRT runtime with CUDA-graph replay.

Everything TensorRT-specific is imported lazily, so the rest of the codebase
stays importable on machines without TRT (dev laptops, CI).

Why CUDA graphs matter here
---------------------------
A tracked EdgeTAM frame runs four engines plus a few dozen PyTorch ops. Each
TensorRT engine is itself tens to hundreds of kernels. On an Orin AGX the CPU
spends roughly 5-10 us launching each one, and the Cortex-A78AE cores are not
fast enough to stay ahead of the GPU once the per-kernel work drops into the
tens-of-microseconds range -- the GPU goes idle waiting for launches. A CUDA
graph records the whole launch sequence once and replays it with a single
call, which removes that ceiling.

The price is rigidity: a captured graph has one fixed set of device pointers
and one fixed set of shapes, forever. So this wrapper owns persistent input and
output buffers, hands TensorRT their addresses once, and asks callers to copy
into those buffers rather than passing fresh tensors. Different shapes get
their own buffers, context and graph, cached by shape -- object count is
constant for a whole video, so in practice exactly one is ever built.

Target: TensorRT 10.x (JetPack 6). The 8.x binding API differs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Collection, Iterable, Mapping

# Names in `run(clone=...)` are cloned before returning. Cloning is the safe
# default because the alternative -- handing back a view of a buffer the next
# replay overwrites -- fails silently and late.
CLONE_ALL = "__all__"


def tensorrt_available() -> bool:
    try:
        import tensorrt  # noqa: F401
    except Exception:
        return False
    return True


def _torch_dtype(trt, torch, trt_dtype):
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT8: torch.int8,
        trt.DataType.BOOL: torch.bool,
        trt.DataType.UINT8: torch.uint8,
    }
    for attr, torch_dtype in (("BF16", "bfloat16"), ("INT64", "int64")):
        if hasattr(trt.DataType, attr):
            mapping[getattr(trt.DataType, attr)] = getattr(torch, torch_dtype)
    if trt_dtype not in mapping:
        raise RuntimeError(f"Unsupported TensorRT tensor dtype: {trt_dtype}")
    return mapping[trt_dtype]


class _Binding:
    """Persistent buffers, execution context and captured graph for one shape."""

    def __init__(self, engine: "TRTEngine", context, inputs: dict, outputs: dict) -> None:
        self._engine = engine
        self._context = context
        self.inputs = inputs
        self.outputs = outputs
        self.graph = None

    def enqueue(self) -> None:
        torch = self._engine._torch
        stream = torch.cuda.current_stream().cuda_stream
        if not self._context.execute_async_v3(stream):
            raise RuntimeError(f"{self._engine.name}: execute_async_v3 failed.")

    def run(self) -> None:
        if self.graph is not None:
            self.graph.replay()
        else:
            self.enqueue()


class TRTEngine:
    """A deserialised engine plus its persistent IO buffers and CUDA graphs.

    Typical use, when the caller wants to fill buffers itself (e.g. writing
    only the live prefix of a padded memory bank):

        binding = engine.binding({"curr": (1, 256, 64, 64), ...})
        binding.inputs["curr"].copy_(feat)
        binding.run()
        out = binding.outputs["pix_feat"]      # valid until the next run()

    Or, for the simple case:

        out = engine.run({"curr": feat, ...})  # cloned outputs, safe to keep
    """

    def __init__(
        self,
        engine_path: str | Path,
        *,
        name: str | None = None,
        use_cuda_graph: bool = True,
        warmup: int = 3,
        capture_error_mode: str = "thread_local",
        device: str = "cuda",
    ) -> None:
        import tensorrt as trt
        import torch

        engine_path = Path(engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(f"TRT engine not found: {engine_path}")

        self._trt = trt
        self._torch = torch
        self.name = name or engine_path.stem
        self.path = engine_path
        self.device = device
        self.use_cuda_graph = use_cuda_graph
        self.warmup = max(1, warmup)
        self.capture_error_mode = capture_error_mode

        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        blob = engine_path.read_bytes()
        self._engine = self._runtime.deserialize_cuda_engine(blob)
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        self.input_names: list[str] = []
        self.output_names: list[str] = []
        self._dtypes: dict[str, "torch.dtype"] = {}
        self._engine_shapes: dict[str, tuple[int, ...]] = {}
        for i in range(self._engine.num_io_tensors):
            tensor = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(tensor) == trt.TensorIOMode.INPUT:
                self.input_names.append(tensor)
            else:
                self.output_names.append(tensor)
            self._dtypes[tensor] = _torch_dtype(
                trt, torch, self._engine.get_tensor_dtype(tensor)
            )
            self._engine_shapes[tensor] = tuple(self._engine.get_tensor_shape(tensor))

        self._bindings: dict[tuple, _Binding] = {}

    # -------------------------------------------------------------- info

    def dtype(self, name: str):
        return self._dtypes[name]

    def engine_shape(self, name: str) -> tuple[int, ...]:
        """Declared shape; -1 marks a dynamic axis."""
        return self._engine_shapes[name]

    def profile_shape(self, name: str):
        """(min, opt, max) shapes of an input under optimisation profile 0."""
        if name not in self.input_names:
            return None
        try:
            bounds = self._engine.get_tensor_profile_shape(name, 0)
        except Exception:
            return None
        return tuple(tuple(b) for b in bounds)

    def accepts(self, shapes: Mapping[str, Iterable[int]]) -> bool:
        """Whether these input shapes fit the engine's optimisation profile.

        Callers use this to decide between the engine and a PyTorch fallback,
        so it answers rather than raises: an object count outside the profile
        is a routine condition, not an error.
        """
        if set(shapes) != set(self.input_names):
            return False
        for name, shape in shapes.items():
            bounds = self.profile_shape(name)
            if bounds is None:
                continue
            lo, _, hi = bounds
            shape = tuple(shape)
            if len(shape) != len(lo):
                return False
            if any(s < a or s > b for s, a, b in zip(shape, lo, hi)):
                return False
        return True

    def describe(self) -> str:
        parts = [f"{self.name}:"]
        for n in self.input_names:
            parts.append(f"  in  {n:<20} {self._engine_shapes[n]} {self._dtypes[n]}")
        for n in self.output_names:
            parts.append(f"  out {n:<20} {self._engine_shapes[n]} {self._dtypes[n]}")
        return "\n".join(parts)

    # ---------------------------------------------------------- bindings

    def binding(self, shapes: Mapping[str, Iterable[int]]) -> _Binding:
        """Get (building on first use) the buffers and graph for these shapes."""
        key = tuple(sorted((n, tuple(s)) for n, s in shapes.items()))
        cached = self._bindings.get(key)
        if cached is None:
            cached = self._build_binding(dict(key))
            self._bindings[key] = cached
        return cached

    def _build_binding(self, shapes: dict[str, tuple[int, ...]]) -> _Binding:
        torch = self._torch
        missing = set(self.input_names) - set(shapes)
        if missing:
            raise KeyError(f"{self.name}: no shape given for input(s) {sorted(missing)}")

        # A context per shape: TensorRT bakes shape-dependent state into the
        # context, and a captured graph must keep seeing the state it was
        # captured with.
        context = self._engine.create_execution_context()
        if context is None:
            raise RuntimeError(f"{self.name}: could not create an execution context.")

        # Buffers must outlive every replay, so keep them out of any
        # inference-mode/graph allocator pool.
        with torch.inference_mode(False):
            inputs = {}
            for name in self.input_names:
                shape = shapes[name]
                if not context.set_input_shape(name, shape):
                    raise RuntimeError(
                        f"{self.name}: input '{name}' shape {shape} is outside the "
                        f"engine's optimisation profile (declared "
                        f"{self._engine_shapes[name]}). Rebuild the engine with a "
                        f"profile that covers it."
                    )
                inputs[name] = torch.zeros(
                    shape, dtype=self._dtypes[name], device=self.device
                )

            # Deprecated in TensorRT 10 but still the clearest signal that an
            # engine wants shape tensors we are not providing.
            if not getattr(context, "all_shape_inputs_specified", True):
                raise RuntimeError(f"{self.name}: engine has unset shape inputs.")

            outputs = {}
            for name in self.output_names:
                shape = tuple(context.get_tensor_shape(name))
                outputs[name] = torch.zeros(
                    shape, dtype=self._dtypes[name], device=self.device
                )

        for name, tensor in list(inputs.items()) + list(outputs.items()):
            if not context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"{self.name}: could not bind tensor '{name}'.")

        binding = _Binding(self, context, inputs, outputs)
        if self.use_cuda_graph:
            try:
                binding.graph = self._capture(binding)
            except Exception as exc:
                # A failed capture costs speed, not correctness: plain enqueues
                # still produce the same results. Say so and carry on rather
                # than taking down a tracking run.
                print(
                    f"[trt] {self.name}: CUDA graph capture failed ({exc}); "
                    "falling back to per-call enqueue."
                )
                binding.graph = None
        return binding

    def _capture(self, binding: _Binding):
        """Warm up on a side stream, then record one enqueue into a graph.

        The warm-up is not optional: TensorRT defers some per-context setup to
        the first enqueue, and anything it does lazily during capture either
        fails the capture or gets baked in wrongly.
        """
        torch = self._torch
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self.warmup):
                binding.enqueue()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        with torch.inference_mode(False):
            graph = torch.cuda.CUDAGraph()
            try:
                ctx = torch.cuda.graph(graph, capture_error_mode=self.capture_error_mode)
            except TypeError:  # torch too old for capture_error_mode
                ctx = torch.cuda.graph(graph)
            with ctx:
                binding.enqueue()
        return graph

    # --------------------------------------------------------------- run

    def run(
        self,
        inputs: Mapping[str, "torch.Tensor"],
        clone: "str | Collection[str]" = CLONE_ALL,
    ) -> dict[str, "torch.Tensor"]:
        """Copy inputs in, replay, return outputs.

        Input tensors are copied into the persistent buffers, converting dtype
        and layout as needed, so callers may pass fp32 views of anything.

        `clone` names the outputs to copy out. The default copies everything,
        because the buffers are reused on the next call; pass an explicit
        (possibly empty) collection for outputs the caller consumes before then.
        """
        binding = self.binding({n: tuple(t.shape) for n, t in inputs.items()})
        for name, tensor in inputs.items():
            binding.inputs[name].copy_(tensor, non_blocking=True)
        binding.run()
        if clone == CLONE_ALL:
            return {n: t.clone() for n, t in binding.outputs.items()}
        wanted = set(clone)
        return {
            n: (t.clone() if n in wanted else t) for n, t in binding.outputs.items()
        }
