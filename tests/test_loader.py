"""Batching, prefetching and the batch-size search.

No GPU here, and no JPEGs: `prefetch` is tested against real `Clip` objects
backed by images written to a temp directory, and `auto_batch_size`'s search is
tested through the pure function it delegates to. What matters is that the
loader hands the training thread *the same batches in the same order* it would
have built itself -- a prefetcher that reorders silently turns a reproducible
run into an unreproducible one.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.antiuav import list_sequences, sample_clips  # noqa: E402
from src.training.loader import (  # noqa: E402
    CANDIDATES,
    batch_clips,
    largest_that_fits,
    prefetch,
)

SIZE = 32

try:
    import cv2  # noqa: F401 -- only the pixel-reading tests need it
    HAVE_CV2 = True
except ImportError:  # pragma: no cover - environment, not logic
    HAVE_CV2 = False


def write_sequence(root: Path, name: str, frames: int = 12) -> None:
    """A tiny 64x48 sequence on disk: enough for the crop geometry to be real."""
    import cv2

    folder = root / "train" / name
    folder.mkdir(parents=True)
    for i in range(frames):
        image = np.full((48, 64), 10 + i, dtype=np.uint8)
        image[20:26, 30:38] = 240
        cv2.imwrite(str(folder / f"{i}.jpg"), image)
    (folder / "IR_label.json").write_text(json.dumps({
        "exist": [1] * frames,
        "gt_rect": [[30, 20, 8, 6]] * frames,
    }))


class TestBatchClips(unittest.TestCase):
    def setUp(self):
        self.clips = list(range(23))

    def test_batches_are_full_and_drop_the_remainder(self):
        batches = list(batch_clips(self.clips, 4, seed=0))
        self.assertEqual(len(batches), 5)          # 23 // 4
        self.assertTrue(all(len(b) == 4 for b in batches))

    def test_every_clip_appears_at_most_once_in_a_pass(self):
        seen = [c for b in batch_clips(self.clips, 4, seed=0) for c in b]
        self.assertEqual(len(seen), len(set(seen)))

    def test_the_same_seed_gives_the_same_pass(self):
        a = list(batch_clips(self.clips, 4, seed=7))
        b = list(batch_clips(self.clips, 4, seed=7))
        self.assertEqual(a, b)

    def test_a_different_seed_gives_a_different_pass(self):
        a = list(batch_clips(self.clips, 4, seed=1))
        b = list(batch_clips(self.clips, 4, seed=2))
        self.assertNotEqual(a, b)

    def test_limit_bounds_an_epoch(self):
        # 60000 overlapping clips is not an epoch anyone waits for; a capped,
        # reshuffled pass is.
        batches = list(batch_clips(self.clips, 2, seed=0, limit=3))
        self.assertEqual(len(batches), 3)

    def test_the_short_last_batch_is_kept_only_when_asked_for(self):
        self.assertEqual(len(list(batch_clips(self.clips, 4, seed=0, drop_last=False))), 6)


@unittest.skipUnless(HAVE_CV2, "reading clip pixels needs OpenCV")
class TestPrefetch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        for name in ("s0", "s1"):
            write_sequence(root, name)
        self.sequences = list_sequences(root, "train")
        self.clips = sample_clips(self.sequences, length=3, stride=1, size=SIZE,
                                  frame_size=(64, 48), seed=0)
        mask = np.zeros((48, 64), dtype=bool)
        mask[20:26, 30:38] = True
        self.stores = {s.name: {i: mask for i in range(len(s))} for s in self.sequences}

    def tearDown(self):
        self.tmp.cleanup()

    def test_yields_every_batch_in_order(self):
        chunks = list(batch_clips(self.clips, 2, seed=0, limit=4))
        batches = list(prefetch(chunks, self.stores, device="cpu", workers=3))

        self.assertEqual(len(batches), len(chunks))
        for chunk, batch in zip(chunks, batches):
            self.assertEqual([c.indices for c in batch.clips], [c.indices for c in chunk])

    def test_the_batches_are_what_collate_would_have_built(self):
        from src.training.clip_loop import collate

        chunk = next(batch_clips(self.clips, 2, seed=0))
        expected = collate(chunk, [self.stores[c.sequence.name] for c in chunk], "cpu")
        got = next(iter(prefetch([chunk], self.stores, device="cpu", workers=2)))

        self.assertTrue(np.allclose(expected.images.numpy(), got.images.numpy()))
        np.testing.assert_allclose(expected.boxes.numpy(), got.boxes.numpy())
        np.testing.assert_array_equal(expected.exist.numpy(), got.exist.numpy())
        self.assertEqual(got.images.shape, (2, 3, 3, SIZE, SIZE))

    def test_an_empty_stream_is_not_an_error(self):
        self.assertEqual(list(prefetch([], self.stores, device="cpu")), [])

    def test_a_consumer_that_stops_early_does_not_hang(self):
        chunks = list(batch_clips(self.clips, 2, seed=0, limit=6))
        stream = prefetch(chunks, self.stores, device="cpu", workers=2)
        self.assertIsNotNone(next(stream))
        stream.close()

    def test_masks_are_decoded_lazily_from_a_store(self):
        # The real store is a MaskStore, which decodes on __getitem__. The
        # loader must go through Mapping.get like clip_masks does.
        from src.training.labels import MaskStore, rle_encode

        mask = np.zeros((48, 64), dtype=bool)
        mask[20:26, 30:38] = True
        runs = rle_encode(mask)
        store = MaskStore((48, 64), runs, np.array([0, len(runs)]), np.array([1]))
        stores = {name: store for name in self.stores}

        chunk = next(batch_clips(self.clips, 1, seed=0))
        batch = next(iter(prefetch([chunk], stores, device="cpu", workers=2)))
        self.assertEqual(len(batch.masks), 3)


class TestLargestThatFits(unittest.TestCase):
    def test_returns_the_last_size_that_ran(self):
        self.assertEqual(largest_that_fits(lambda n: n <= 7, (1, 2, 4, 8, 16)), 4)

    def test_stops_at_the_first_refusal_rather_than_probing_past_it(self):
        tried = []

        def trial(n):
            tried.append(n)
            return n <= 4

        largest_that_fits(trial, (1, 2, 4, 8, 16, 32))
        self.assertEqual(tried, [1, 2, 4, 8])

    def test_everything_fitting_gives_the_largest_candidate(self):
        self.assertEqual(largest_that_fits(lambda n: True, CANDIDATES), max(CANDIDATES))

    def test_nothing_fitting_is_an_error_not_a_zero(self):
        with self.assertRaises(RuntimeError):
            largest_that_fits(lambda n: False, (1, 2, 4))


class TestAutoBatchSizeWithoutCuda(unittest.TestCase):
    def test_falls_back_to_one_on_cpu(self):
        from src.training.loader import auto_batch_size

        self.assertEqual(auto_batch_size(None, [], {}, device="cpu"), 1)


if __name__ == "__main__":
    unittest.main()
