"""`open_video_writer` must not just throw when mp4v won't open.

Regression coverage for a real failure: a headless Jetson build of OpenCV's
FFmpeg has no software MPEG-4/H.264 encoder and no accessible /dev/video
encode node, so `cv2.VideoWriter(..., 'mp4v', ...).isOpened()` is False with
nothing on stderr to explain why -- confirmed on-device, `mp4v`/`avc1`/`H264`
all fail there while `MJPG`/`XVID` in an `.avi` container both open. This
fakes that exact failure (mp4v refuses, everything else behaves normally) so
the fallback chain is exercised without needing that hardware.

Run with:  python -m pytest tests/test_video_writer_fallback.py
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from src.io_utils.video import open_video_writer


class _Mp4vRefusesToOpen:
    """Stands in for cv2.VideoWriter: mp4v never opens, other fourccs do."""

    _real = cv2.VideoWriter

    def __init__(self, path, fourcc, fps, size):
        mp4v = cv2.VideoWriter_fourcc(*"mp4v")
        self._inner = None if fourcc == mp4v else self._real(path, fourcc, fps, size)

    def isOpened(self):
        return self._inner.isOpened() if self._inner else False

    def write(self, frame):
        if self._inner:
            self._inner.write(frame)

    def release(self):
        if self._inner:
            self._inner.release()


class TestVideoWriterFallback(unittest.TestCase):
    def test_falls_back_to_a_container_that_actually_opens(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("cv2.VideoWriter", _Mp4vRefusesToOpen):
            requested = Path(tmp) / "tracked.mp4"
            with open_video_writer(requested, 30.0, (64, 48)) as write:
                # The path a caller reports "wrote X" from must be the file
                # that actually has bytes in it, not the one that was asked
                # for and never got created.
                self.assertNotEqual(write.path, requested)
                self.assertEqual(write.path.suffix, ".avi")
                write(np.zeros((48, 64, 3), dtype=np.uint8))
            self.assertFalse(requested.exists())
            self.assertTrue(write.path.exists())
            self.assertGreater(write.path.stat().st_size, 0)

    def test_the_common_case_is_untouched(self):
        # No monkeypatch: on a normal build mp4v opens first try, exactly as
        # before this fallback existed.
        with tempfile.TemporaryDirectory() as tmp:
            requested = Path(tmp) / "tracked.mp4"
            with open_video_writer(requested, 30.0, (64, 48)) as write:
                self.assertEqual(write.path, requested)
                write(np.zeros((48, 64, 3), dtype=np.uint8))
            self.assertTrue(requested.exists())

    def test_raises_when_nothing_will_open(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("cv2.VideoWriter") as fake:
            fake.return_value.isOpened.return_value = False
            with self.assertRaises(RuntimeError):
                with open_video_writer(Path(tmp) / "tracked.mp4", 30.0, (64, 48)):
                    pass


if __name__ == "__main__":
    unittest.main()
