"""PyTorch stand-ins for the TensorRT engines, behind the same runtime API.

Each one runs the exact wrapper module the ONNX was exported from
(`tools/edgetam_graphs.py`), but presents the slice of `TRTEngine` the tracker
uses. That makes the whole TensorRT integration path runnable anywhere —
no GPU, no TensorRT, no built engines — which is what lets the rewrite be
separated from TensorRT when something looks wrong:

    engines run, masks differ  -> the rewrite or the wiring is at fault
    engines run, masks match   -> look at the engine build or fp16

Output buffers are reused across calls, exactly as a replayed CUDA graph
reuses them. That is deliberate: it turns "the tracker forgot to copy an
output that outlives the frame" from a rare heisenbug on the Orin into a
deterministic wrong answer here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trackers._trt_layout import MemoryLayout  # noqa: E402
from src.trackers._trt_runtime import CLONE_ALL  # noqa: E402
from tools.edgetam_graphs import (  # noqa: E402
    ImageEncoderGraph,
    MemoryAttentionGraph,
    MemoryEncoderGraph,
    SamHeadGraph,
)

OUTPUT_NAMES = {
    "image_encoder": ["backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2"],
    "memory_attention": ["pix_feat"],
    "memory_encoder": ["maskmem", "maskmem_pos"],
    "sam_head": [
        "low_res_multimasks",
        "ious",
        "low_res_masks",
        "high_res_masks",
        "obj_ptr",
        "object_score_logits",
    ],
}


class _Binding:
    def __init__(self, graph, input_names, output_names, shapes):
        self._graph = graph
        self._input_names = list(input_names)
        self._output_names = list(output_names)
        self.inputs = {n: torch.zeros(shapes[n]) for n in self._input_names}
        self.outputs = {n: t.clone() for n, t in zip(self._output_names, self._call())}
        self.graph = object()  # stands in for a captured CUDA graph

    def _call(self):
        with torch.no_grad():
            out = self._graph(*[self.inputs[n] for n in self._input_names])
        return (out,) if isinstance(out, torch.Tensor) else out

    def enqueue(self):
        self.run()

    def run(self):
        for name, value in zip(self._output_names, self._call()):
            self.outputs[name].copy_(value)


class ReferenceEngine:
    """The subset of `TRTEngine` the tracker touches, running PyTorch instead."""

    def __init__(self, name, graph, reference_inputs, output_names):
        self.name = name
        self._graph = graph.eval()
        self.input_names = list(reference_inputs)
        self.output_names = list(output_names)
        self._reference = {n: tuple(t.shape) for n, t in reference_inputs.items()}
        with torch.no_grad():
            out = self._graph(*reference_inputs.values())
        out = (out,) if isinstance(out, torch.Tensor) else out
        self._shapes = dict(self._reference)
        self._shapes.update({n: tuple(t.shape) for n, t in zip(output_names, out)})
        self._bindings: dict[tuple, _Binding] = {}

    def dtype(self, name):
        return torch.float32

    def engine_shape(self, name):
        return self._shapes[name]

    def profile_shape(self, name):
        return None

    def accepts(self, shapes):
        if set(shapes) != set(self.input_names):
            return False
        # Batch is free; every other axis must match what the graph was built for.
        return all(
            tuple(shapes[n])[1:] == self._reference[n][1:] for n in self.input_names
        )

    def binding(self, shapes):
        key = tuple(sorted((n, tuple(s)) for n, s in shapes.items()))
        if key not in self._bindings:
            self._bindings[key] = _Binding(
                self._graph, self.input_names, self.output_names, dict(key)
            )
        return self._bindings[key]

    def run(self, inputs, clone=CLONE_ALL):
        binding = self.binding({n: tuple(t.shape) for n, t in inputs.items()})
        for name, tensor in inputs.items():
            binding.inputs[name].copy_(tensor)
        binding.run()
        if clone == CLONE_ALL:
            return {n: t.clone() for n, t in binding.outputs.items()}
        wanted = set(clone)
        return {n: (t.clone() if n in wanted else t) for n, t in binding.outputs.items()}


def build_reference_engines(model, batch: int = 1, ptr_tokens: int | None = None):
    """One ReferenceEngine per module, shaped the way the exporter would.

    `model` must be a *separate* instance from the one the tracker will patch:
    a real engine is a frozen snapshot taken at export time, so sharing live
    modules would have a graph call back into its own patched forward.
    """
    layout = MemoryLayout.from_model(model, ptr_tokens=ptr_tokens)
    side = model.image_size // model.backbone_stride
    dim, mem_dim = model.hidden_dim, model.mem_dim
    size = model.image_size
    zeros = torch.zeros

    return {
        "image_encoder": ReferenceEngine(
            "image_encoder",
            ImageEncoderGraph(model),
            {"image": zeros(1, 3, size, size)},
            OUTPUT_NAMES["image_encoder"],
        ),
        "memory_attention": ReferenceEngine(
            "memory_attention",
            MemoryAttentionGraph(model, layout),
            {
                "curr": zeros(batch, dim, side, side),
                "curr_pos": zeros(batch, dim, side, side),
                "memory": zeros(batch, layout.total_tokens, mem_dim),
                "memory_pos": zeros(batch, layout.total_tokens, mem_dim),
                "memory_mask": zeros(batch, 1, 1, layout.total_tokens),
            },
            OUTPUT_NAMES["memory_attention"],
        ),
        "memory_encoder": ReferenceEngine(
            "memory_encoder",
            MemoryEncoderGraph(model),
            {
                "pix_feat": zeros(batch, dim, side, side),
                "mask_for_mem": zeros(batch, 1, size, size),
            },
            OUTPUT_NAMES["memory_encoder"],
        ),
        "sam_head": ReferenceEngine(
            "sam_head",
            SamHeadGraph(model),
            {
                "pix_feat": zeros(batch, dim, side, side),
                "high_res_0": zeros(batch, dim // 8, side * 4, side * 4),
                "high_res_1": zeros(batch, dim // 4, side * 2, side * 2),
            },
            OUTPUT_NAMES["sam_head"],
        ),
    }
