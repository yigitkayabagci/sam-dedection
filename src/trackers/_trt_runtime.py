"""Thin TensorRT runtime wrapper for the EdgeTAM image encoder engine.

Only loaded when the edgetam_trt backend is actually built. Everything
TensorRT-specific is imported lazily so the rest of the codebase remains
usable on machines without TRT (development, CI, this conversation's
sandbox, etc.).

Target: TensorRT 10.x as shipped with JetPack 6 on Orin AGX. The 8.x API
differs (set_binding_shape vs set_input_shape) — adjust if you must run
on JetPack 5.

Two execution modes:

  default        Bind the caller's input tensor, allocate fresh output
                 tensors per call, enqueue. Output tensors are owned by the
                 caller — no aliasing hazards. Torch's caching allocator
                 makes the per-call allocation ~free after warm-up.

  CUDA graph     Persistent input/output buffers at fixed addresses, the
                 enqueue captured once and replayed. Removes TensorRT's
                 per-kernel launch overhead, which matters on Orin because
                 its CPU is comparatively slow. Outputs are cloned out of
                 the persistent buffers before returning (frame N's features
                 stay alive in EdgeTAM's feature cache while frame N+1 is
                 encoded, so handing back the buffers themselves would let
                 the next frame overwrite them). That clone is the price of
                 graph mode — benchmark both, don't assume.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence


def _build_dtype_map(trt, torch) -> dict:
    """Map TensorRT dtypes to torch dtypes, skipping ones this build lacks.

    Written defensively because the set of DataType members differs across
    TensorRT versions (BF16 arrived in 8.6, FP8/INT4 later and not on Orin).
    A missing entry used to silently degrade to float32, which corrupts
    output for any non-FP32 engine.
    """
    pairs = (
        ("FLOAT", "float32"),
        ("HALF", "float16"),
        ("BF16", "bfloat16"),
        ("INT8", "int8"),
        ("INT32", "int32"),
        ("INT64", "int64"),
        ("BOOL", "bool"),
        ("UINT8", "uint8"),
        ("FP8", "float8_e4m3fn"),
    )
    out = {}
    for trt_name, torch_name in pairs:
        trt_dtype = getattr(trt.DataType, trt_name, None)
        torch_dtype = getattr(torch, torch_name, None)
        if trt_dtype is not None and torch_dtype is not None:
            out[trt_dtype] = torch_dtype
    return out


class TRTImageEncoder:
    """Runs a single-input, multi-output TRT engine using torch CUDA tensors.

    Zero-copy: PyTorch allocates the input/output buffers on the GPU and we
    hand TensorRT their raw device pointers. No `pycuda` needed.
    """

    def __init__(
        self,
        engine_path: str | Path,
        use_cuda_graph: bool = False,
        synchronize: bool = True,
    ) -> None:
        import tensorrt as trt
        import torch

        engine_path = Path(engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(f"TRT engine not found: {engine_path}")

        self._trt = trt
        self._torch = torch
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        with open(engine_path, "rb") as f:
            self._engine = self._runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(
                f"Failed to deserialize engine: {engine_path}. Engines are tied to "
                f"the GPU and the TensorRT version that built them "
                f"(this process has TensorRT {trt.__version__}) — rebuild it here."
            )

        self._context = self._engine.create_execution_context()
        self._synchronize = synchronize
        self._use_cuda_graph = use_cuda_graph
        self._graph = None
        self._static_in = None
        self._static_out: list = []

        # Sort tensor names into (input_name, [output_names]).
        self._input_name: str | None = None
        self._output_names: list[str] = []
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            mode = self._engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                if self._input_name is not None:
                    raise RuntimeError("Expected exactly one input.")
                self._input_name = name
            else:
                self._output_names.append(name)
        if self._input_name is None:
            raise RuntimeError("Engine has no input tensor.")

        self._dtype_map = _build_dtype_map(trt, torch)
        self._input_dtype = self._torch_dtype(self._input_name)

    # -- helpers -----------------------------------------------------------

    def _torch_dtype(self, name: str):
        trt_dtype = self._engine.get_tensor_dtype(name)
        torch_dtype = self._dtype_map.get(trt_dtype)
        if torch_dtype is None:
            raise RuntimeError(
                f"Engine tensor {name!r} has TensorRT dtype {trt_dtype}, which has "
                "no torch equivalent in this build. Rebuild the engine with "
                "standard FP32 I/O (do not pass --inputIOFormats/--outputIOFormats)."
            )
        return torch_dtype

    def output_specs(self) -> list[tuple[str, tuple[int, ...], "torch.dtype"]]:
        return [
            (name, tuple(self._engine.get_tensor_shape(name)), self._torch_dtype(name))
            for name in self._output_names
        ]

    def _alloc_outputs(self) -> list:
        torch = self._torch
        return [
            torch.empty(
                tuple(self._context.get_tensor_shape(name)),
                dtype=self._torch_dtype(name),
                device="cuda",
            )
            for name in self._output_names
        ]

    def _bind(self, image, outputs) -> None:
        self._context.set_tensor_address(self._input_name, image.data_ptr())
        for name, tensor in zip(self._output_names, outputs):
            self._context.set_tensor_address(name, tensor.data_ptr())

    def _enqueue(self, stream) -> None:
        if not self._context.execute_async_v3(stream):
            raise RuntimeError(
                "TRT execute_async_v3 failed. Check that the input shape was set "
                "and every I/O address is bound."
            )

    # -- CUDA graph --------------------------------------------------------

    def _setup_graph(self, image) -> bool:
        """Capture one enqueue into a CUDA graph. Returns False on failure."""
        torch = self._torch
        try:
            self._static_in = torch.empty_like(image)
            self._static_in.copy_(image)
            self._context.set_input_shape(self._input_name, tuple(image.shape))
            self._static_out = self._alloc_outputs()
            self._bind(self._static_in, self._static_out)

            # Warm up on a side stream first: capture must not race with lazy
            # cuBLAS/cuDNN init or the caching allocator's first-touch work.
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._enqueue(side.cuda_stream)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._enqueue(torch.cuda.current_stream().cuda_stream)
            self._graph = graph
            print("[trt] CUDA graph captured")
            return True
        except Exception as exc:
            print(f"[trt] CUDA graph capture failed ({exc}); using normal enqueue.")
            self._graph = None
            self._static_in = None
            self._static_out = []
            self._use_cuda_graph = False
            return False

    # -- inference ---------------------------------------------------------

    def infer(self, image: "torch.Tensor") -> Sequence["torch.Tensor"]:
        torch = self._torch

        if not image.is_cuda:
            image = image.cuda()
        # An engine built with non-default I/O formats expects a specific dtype;
        # binding a float32 pointer to a half input reads garbage rather than
        # raising, so convert explicitly.
        if image.dtype != self._input_dtype:
            image = image.to(self._input_dtype)
        image = image.contiguous()

        if self._use_cuda_graph:
            if self._graph is None and not self._setup_graph(image):
                return self.infer(image)  # capture failed; fall through to normal path
            self._static_in.copy_(image)
            self._graph.replay()
            if self._synchronize:
                torch.cuda.current_stream().synchronize()
            # Clone: EdgeTAM keeps the previous frame's features alive while the
            # next frame is encoded, and these buffers get overwritten in place.
            return [t.clone() for t in self._static_out]

        # Communicate input shape to dynamic-shape engines (no-op for static).
        self._context.set_input_shape(self._input_name, tuple(image.shape))
        outputs = self._alloc_outputs()
        self._bind(image, outputs)

        stream = torch.cuda.current_stream().cuda_stream
        self._enqueue(stream)
        if self._synchronize:
            # Not strictly required — TensorRT enqueues on the caller's stream, so
            # later torch ops on that stream are already ordered after it. Kept on
            # by default so a stray cross-stream read cannot produce silently wrong
            # masks; set synchronize=False to measure without it.
            torch.cuda.current_stream().synchronize()
        return outputs
