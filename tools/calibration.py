"""INT8 calibration data plumbing for the EdgeTAM image encoder.

The single most common way an INT8 TensorRT build goes wrong is calibration:

  * No calibrator at all -> TRT assigns a dynamic range of 1.0 to every tensor
    and the engine emits garbage (empty or random masks). `--int8` alone is
    never correct.
  * Calibration data preprocessed differently from inference -> the collected
    activation histograms describe a distribution the model never sees, so the
    scales are wrong even though the build "succeeds".

So `preprocess_rgb` below mirrors EdgeTAM/SAM2's own `_load_img_as_tensor` +
`load_video_frames` exactly: PIL resize to (1024, 1024), /255, then subtract
mean and divide by std. Feeding a `.jpg` directory goes through PIL end to
end, which is bit-for-bit what the model receives at inference time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

# SAM 2 / EdgeTAM normalization constants (sam2/utils/misc.py defaults).
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def preprocess_pil(img, size: int) -> np.ndarray:
    """PIL.Image -> normalized CHW float32 array, matching SAM2 exactly."""
    arr = np.asarray(img.convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr -= IMG_MEAN[:, None, None]
    arr /= IMG_STD[:, None, None]
    return np.ascontiguousarray(arr, dtype=np.float32)


def preprocess_rgb(img_rgb: np.ndarray, size: int) -> np.ndarray:
    """HWC uint8 RGB array -> normalized CHW float32 array."""
    from PIL import Image

    return preprocess_pil(Image.fromarray(img_rgb), size)


def iter_frames_from_dir(
    frames_dir: str | Path, size: int, limit: int, stride: int = 1
) -> Iterator[np.ndarray]:
    from PIL import Image

    d = Path(frames_dir)
    files = sorted(p for p in d.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No images found in {d}")
    taken = 0
    for i, path in enumerate(files):
        if i % stride:
            continue
        if taken >= limit:
            return
        yield preprocess_pil(Image.open(path), size)
        taken += 1


def iter_frames_from_video(
    video_path: str | Path, size: int, limit: int, stride: int = 1
) -> Iterator[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        idx = taken = 0
        while taken < limit:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield preprocess_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), size)
                taken += 1
            idx += 1
    finally:
        cap.release()
    if taken == 0:
        raise RuntimeError(f"No frames decoded from {video_path}")


def calibration_batches(
    *,
    video: str | Path | None = None,
    frames_dir: str | Path | None = None,
    size: int = 1024,
    limit: int = 200,
    stride: int = 1,
    batch_size: int = 1,
) -> Iterator[np.ndarray]:
    """Yield NCHW float32 calibration batches from a video or a frame folder."""
    if (video is None) == (frames_dir is None):
        raise ValueError("Pass exactly one of video= / frames_dir=")
    src = (iter_frames_from_video(video, size, limit, stride) if video is not None
           else iter_frames_from_dir(frames_dir, size, limit, stride))
    buf: list[np.ndarray] = []
    for frame in src:
        buf.append(frame)
        if len(buf) == batch_size:
            yield np.ascontiguousarray(np.stack(buf), dtype=np.float32)
            buf = []
    # A short trailing group is dropped on purpose: TRT calibrators must see a
    # fixed batch size across every get_batch() call.


class _DeviceBuffer:
    """A CUDA device allocation we can hand TensorRT as a raw pointer.

    Prefers a torch CUDA tensor (always present on a working Jetson setup) and
    falls back to cuda-python, so no pycuda dependency is introduced.
    """

    def __init__(self, nbytes: int) -> None:
        self._holder = None
        self._free = None
        try:
            import torch

            self._holder = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
            self.ptr = int(self._holder.data_ptr())
            self._backend = "torch"
            return
        except Exception:
            pass
        try:
            from cuda import cudart

            err, ptr = cudart.cudaMalloc(nbytes)
            if int(err) != 0:
                raise RuntimeError(f"cudaMalloc failed: {err}")
            self.ptr = int(ptr)
            self._free = lambda: cudart.cudaFree(ptr)
            self._backend = "cuda-python"
            return
        except ImportError as exc:
            raise RuntimeError(
                "INT8 calibration needs a CUDA allocator. Install the Jetson "
                "torch wheel (preferred) or `pip install cuda-python`."
            ) from exc

    def copy_from(self, host: np.ndarray) -> None:
        if self._backend == "torch":
            import torch

            flat = torch.from_numpy(host.reshape(-1).view(np.uint8))
            self._holder.copy_(flat)  # copy_ handles the host->device hop
        else:
            from cuda import cudart

            err = cudart.cudaMemcpy(
                self.ptr, host.ctypes.data, host.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )[0]
            if int(err) != 0:
                raise RuntimeError(f"cudaMemcpy failed: {err}")

    def close(self) -> None:
        if self._free is not None:
            self._free()
            self._free = None
        self._holder = None


def make_entropy_calibrator(
    trt,
    batches: Sequence[np.ndarray] | Iterator[np.ndarray],
    cache_file: str | Path | None = None,
    progress: bool = True,
):
    """Build an IInt8EntropyCalibrator2 over an iterable of NCHW batches.

    Defined as a factory rather than a module-level class so that importing
    this module never requires tensorrt (dev machines, CI, ONNX-only work).
    """

    class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self) -> None:
            super().__init__()
            self._iter = iter(batches)
            self._cache = Path(cache_file) if cache_file else None
            self._buf: _DeviceBuffer | None = None
            self._batch_size = 1
            self._count = 0

        def get_batch_size(self) -> int:
            return self._batch_size

        def get_batch(self, names):  # noqa: ARG002 - single input engine
            try:
                batch = next(self._iter)
            except StopIteration:
                if self._buf is not None:
                    self._buf.close()
                    self._buf = None
                if progress:
                    print(f"[calib] done, {self._count} batches consumed")
                return None
            batch = np.ascontiguousarray(batch, dtype=np.float32)
            self._batch_size = int(batch.shape[0])
            if self._buf is None:
                self._buf = _DeviceBuffer(batch.nbytes)
            self._buf.copy_from(batch)
            self._count += 1
            if progress and self._count % 25 == 0:
                print(f"[calib] {self._count} batches...", flush=True)
            return [self._buf.ptr]

        def read_calibration_cache(self):
            if self._cache and self._cache.exists():
                print(f"[calib] reusing cache {self._cache} "
                      "(delete it to force a fresh calibration)")
                return self._cache.read_bytes()
            return None

        def write_calibration_cache(self, cache) -> None:
            if self._cache:
                self._cache.parent.mkdir(parents=True, exist_ok=True)
                self._cache.write_bytes(bytes(cache))
                print(f"[calib] wrote cache -> {self._cache}")

    return EntropyCalibrator()
