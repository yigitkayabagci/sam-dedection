"""Capture real engine inputs, so INT8 scales are calibrated on real data.

The obvious way to calibrate the image encoder is easy -- feed it thermal
frames. The other three engines are the problem: they take `pix_feat`, a padded
`memory` bank, and a sigmoid-scaled `mask_for_mem`. Nobody can write down those
distributions, and calibrating on random noise sets scales for a network that
does not exist. This project has already measured what a badly chosen scale
does to EdgeTAM: min/max calibration on the image encoder alone took mean IoU
from 0.9999 to **0.8170**, losing the object outright on 5 frames of 30
(`docs/tensorrt_fp16.md`).

So calibration data is *recorded from a real tracking run*, at the engine
boundary, in exactly the layout the deployed engine receives -- padded memory
slots, additive mask and all. `RecordingEngine` wraps any engine that presents
the `TRTEngine` interface, which means the same recorder works against:

  * `tests/reference_engines.py` -- PyTorch, no TensorRT, no GPU. The default:
    calibration has to happen *before* an INT8 engine exists.
  * real fp16 engines on the Orin, for a second opinion taken on the device
    that will run them.

It records inputs only. Outputs are the quantised network's problem.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ._trt_runtime import CLONE_ALL


class CalibrationStore:
    """Up to `limit` samples per input tensor, written out as one `.npz`.

    Samples are taken every `stride`-th call rather than from the first N
    frames. A tracked clip is not stationary -- the memory bank fills over the
    first ~16 frames and the object's appearance drifts after that -- so the
    opening frames alone would calibrate for a transient the deployed engine
    spends almost none of its time in.
    """

    def __init__(self, limit: int = 512, stride: int = 1) -> None:
        self.limit = limit
        self.stride = max(int(stride), 1)
        self.samples: dict[str, list[np.ndarray]] = {}
        self.calls = 0

    @property
    def full(self) -> bool:
        return bool(self.samples) and all(
            len(v) >= self.limit for v in self.samples.values()
        )

    def record(self, inputs: dict) -> None:
        self.calls += 1
        if self.calls % self.stride:
            return
        for name, tensor in inputs.items():
            bucket = self.samples.setdefault(name, [])
            if len(bucket) < self.limit:
                # `.clone()` is load-bearing, not defensive. The memory-attention
                # path records persistent buffers that the next frame overwrites,
                # and on CPU `.numpy()` shares storage with the tensor -- so
                # without it every sample would end up holding the last frame's
                # values. Rows are independent samples: a batch of N tracked
                # objects is N observations of this tensor, not one.
                bucket.append(tensor.detach().float().cpu().clone().numpy())

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: np.concatenate(chunks, axis=0)[: self.limit]
                for name, chunks in self.samples.items() if chunks}

    def save(self, path: str | Path) -> Path:
        arrays = self.arrays()
        if not arrays:
            raise RuntimeError(
                "nothing was recorded -- the engine was never called. Check that "
                "the module is actually on the accelerated path for this clip."
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
        return path

    def summary(self) -> str:
        return ", ".join(
            f"{name} {tuple(array.shape)}" for name, array in sorted(self.arrays().items())
        )


class _RecordingBinding:
    """Snapshot a binding's input buffers at the moment it is run.

    The memory-attention path writes into persistent buffers and then replays a
    captured graph, so the inputs only exist as *the buffers' current contents*
    -- there is no argument to intercept. They have to be copied here, before
    `run()` hands them to the engine.
    """

    def __init__(self, binding, store: CalibrationStore) -> None:
        self._binding = binding
        self._store = store

    def __getattr__(self, name):
        return getattr(self._binding, name)

    @property
    def inputs(self):
        return self._binding.inputs

    @property
    def outputs(self):
        return self._binding.outputs

    def run(self) -> None:
        self._store.record(self._binding.inputs)
        self._binding.run()

    def enqueue(self) -> None:
        self._store.record(self._binding.inputs)
        self._binding.enqueue()


class RecordingEngine:
    """An engine that also keeps what it was given.

    Everything not overridden here passes straight through, so the tracker
    cannot tell the difference and the recorded run is the real one -- not a
    reconstruction of it.
    """

    def __init__(self, engine, store: CalibrationStore | None = None) -> None:
        self._engine = engine
        self.store = store or CalibrationStore()
        self._bindings: dict[int, _RecordingBinding] = {}

    def __getattr__(self, name):
        return getattr(self._engine, name)

    def binding(self, shapes):
        binding = self._engine.binding(shapes)
        wrapped = self._bindings.get(id(binding))
        if wrapped is None:
            wrapped = self._bindings[id(binding)] = _RecordingBinding(binding, self.store)
        return wrapped

    def run(self, inputs, clone=CLONE_ALL):
        self.store.record(inputs)
        return self._engine.run(inputs, clone=clone)


def wrap_engines(engines: dict, limit: int = 512, stride: int = 1) -> dict[str, CalibrationStore]:
    """Wrap every engine in place; return each one's store.

    Mutates the dict the tracker holds, which is the whole trick: no code path
    downstream has to know calibration is happening.
    """
    stores: dict[str, CalibrationStore] = {}
    for name, engine in list(engines.items()):
        recorder = RecordingEngine(engine, CalibrationStore(limit, stride))
        engines[name] = recorder
        stores[name] = recorder.store
    return stores
