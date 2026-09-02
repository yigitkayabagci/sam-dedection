"""A frame sequence has to be one size, and the run has to say so early.

The failure this guards against is real and was hit on a 331-frame folder: one
odd frame in a directory produced `IndexError: boolean index did not match
indexed array along dimension 0; dimension is 768 but corresponding boolean
dimension is 1896`, from `visualize.overlay_masks`, twenty seconds in and after
the engines were loaded. Two integers and no filename.

The cause is that nothing between `frames_metadata` and the overlay checks:
frame 0 sets the geometry, `rgb[y0:y0 + ch, x0:x0 + cw]` clips silently rather
than raising, and a smaller frame therefore arrives as a smaller array. So what
is pinned here is that the check happens on the *headers* -- cheap enough to run
over every frame before anything is built -- and that its message names the
files rather than the shapes.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import frame_size, odd_sized_frames  # noqa: E402


def write(path: Path, width: int, height: int) -> Path:
    from PIL import Image

    Image.new("RGB", (width, height), (0, 0, 0)).save(path)
    return path


class FrameSize(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.dir, ignore_errors=True))

    def test_reads_the_size_without_decoding(self):
        self.assertEqual(frame_size(write(self.dir / "a.png", 1280, 768)), (1280, 768))

    def test_an_unreadable_file_is_not_this_function_s_error(self):
        """`decode_frame` raises with its own message; guessing here would
        replace a good error with a worse one."""
        (self.dir / "b.png").write_bytes(b"not an image")
        self.assertIsNone(frame_size(self.dir / "b.png"))
        self.assertIsNone(frame_size(self.dir / "missing.png"))


class OddSizedFrames(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.dir, ignore_errors=True))

    def files(self, *sizes):
        return [write(self.dir / f"{i:03d}.png", w, h)
                for i, (w, h) in enumerate(sizes)]

    def test_one_size_throughout_is_nothing_to_report(self):
        files = self.files((1280, 768), (1280, 768), (1280, 768))
        self.assertEqual(odd_sized_frames(files, 1280, 768), [])

    def test_names_the_frame_that_differs_and_its_size(self):
        files = self.files((1280, 768), (960, 768), (1280, 768))
        odd = odd_sized_frames(files, 1280, 768)
        self.assertEqual([(p.name, size) for p, size in odd],
                         [("001.png", (960, 768))])

    def test_a_taller_frame_counts_too(self):
        """The reported failure was this one: frame 0 at 1896 tall against a
        768-tall neighbour, which the crop clips into silently."""
        files = self.files((1280, 1896), (1280, 768))
        self.assertEqual(len(odd_sized_frames(files, 1280, 1896)), 1)

    def test_stops_at_the_limit_rather_than_listing_a_whole_directory(self):
        files = self.files(*([(960, 768)] * 40))
        self.assertEqual(len(odd_sized_frames(files, 1280, 768, limit=5)), 5)

    def test_an_unreadable_file_is_not_reported_as_odd(self):
        """It is a different failure with a different message, and reporting it
        here would send someone looking for a resize that is not the problem."""
        files = self.files((1280, 768))
        broken = self.dir / "zzz.png"
        broken.write_bytes(b"not an image")
        self.assertEqual(odd_sized_frames(files + [broken], 1280, 768), [])


if __name__ == "__main__":
    unittest.main()
