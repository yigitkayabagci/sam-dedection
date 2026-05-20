"""Thin TensorRT runtime wrapper for the EdgeTAM image encoder engine.

Only loaded when the edgetam_trt backend is actually built. Everything
TensorRT-specific is imported lazily so the rest of the codebase remains
usable on machines without TRT (development, CI, this conversation's
sandbox, etc.).

Target: TensorRT 10.x as shipped with JetPack 6 on Orin AGX. The 8.x API
differs (set_binding_shape vs set_input_shape) — adjust if you must run
on JetPack 5.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence


class TRTImageEncoder:
    """Runs a single-input, multi-output TRT engine using torch CUDA tensors.

    Zero-copy: PyTorch allocates the input/output buffers on the GPU and we
    hand TensorRT their raw device pointers. No `pycuda` needed.
    """

    def __init__(self, engine_path: str | Path) -> None:
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
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        self._context = self._engine.create_execution_context()

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

        self._dtype_map = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.INT32: torch.int32,
        }

    def output_specs(self) -> list[tuple[str, tuple[int, ...], "torch.dtype"]]:
        out = []
        for name in self._output_names:
            shape = tuple(self._engine.get_tensor_shape(name))
            dtype = self._dtype_map.get(
                self._engine.get_tensor_dtype(name), self._torch.float32
            )
            out.append((name, shape, dtype))
        return out

    def infer(self, image: "torch.Tensor") -> Sequence["torch.Tensor"]:
        torch = self._torch

        if not image.is_cuda:
            image = image.cuda()
        image = image.contiguous()

        # Communicate input shape to dynamic-shape engines (no-op for static).
        self._context.set_input_shape(self._input_name, tuple(image.shape))

        # Allocate output tensors based on the engine's resolved shapes.
        outputs: list[torch.Tensor] = []
        for name in self._output_names:
            shape = tuple(self._context.get_tensor_shape(name))
            dtype = self._dtype_map.get(
                self._engine.get_tensor_dtype(name), torch.float32
            )
            outputs.append(torch.empty(shape, dtype=dtype, device="cuda"))

        # Bind device pointers.
        self._context.set_tensor_address(self._input_name, image.data_ptr())
        for name, t in zip(self._output_names, outputs):
            self._context.set_tensor_address(name, t.data_ptr())

        stream = torch.cuda.current_stream().cuda_stream
        if not self._context.execute_async_v3(stream):
            raise RuntimeError("TRT execute_async_v3 failed.")
        # Caller-side .cpu()/.cuda() ops will synchronize; for safety:
        torch.cuda.current_stream().synchronize()
        return outputs
