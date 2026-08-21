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

    def test_a_limit_is_met_even_when_the_pool_is_smaller_than_it(self):
        """The bug that silently voided the notebooks' headline comparison.

        A single pass used to end the epoch when the pool ran out, with no
        error and nothing in the logs but a short loss curve. VTUAV alone is
        ~1 400 training windows, so at batch 64 an epoch that asked for 400
        steps stopped after 21 -- and `OneCycleLR`, built for 400, never left
        its warm-up. The three-dataset run had ten times the windows and filled
        its 400, so "same schedule, only the data differs" compared 400 steps
        against 21.
        """
        pool = list(range(1400))
        self.assertEqual(sum(1 for _ in batch_clips(pool, 64, seed=0, limit=400)),
                         400)

    def test_every_batch_is_still_full_when_the_pool_is_cycled(self):
        batches = list(batch_clips(list(range(23)), 4, seed=0, limit=20))
        self.assertEqual(len(batches), 20)
        self.assertTrue(all(len(b) == 4 for b in batches))

    def test_a_reused_pool_is_reshuffled_rather_than_repeated(self):
        # Cycling in the same order would make every pass after the first an
        # exact repeat, which is worse for training than seeing fewer steps.
        batches = list(batch_clips(list(range(20)), 4, seed=0, limit=10))
        self.assertNotEqual(batches[:5], batches[5:])

    def test_a_pool_larger_than_the_limit_is_unaffected(self):
        batches = list(batch_clips(list(range(9999)), 64, seed=0, limit=10))
        self.assertEqual(len(batches), 10)

    def test_an_empty_pool_ends_instead_of_cycling_forever(self):
        self.assertEqual(list(batch_clips([], 4, seed=0, limit=400)), [])

    def test_without_a_limit_it_is_still_a_single_pass(self):
        # `limit=None` means "one epoch over the data", and cycling that would
        # never return.
        self.assertEqual(len(list(batch_clips(list(range(23)), 4, seed=0))), 5)

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

    def test_an_executor_collates_the_same_batch_a_plain_loop_would(self):
        from concurrent.futures import ThreadPoolExecutor

        from src.training.clip_loop import collate

        chunk = next(batch_clips(self.clips, 3, seed=0))
        stores = [self.stores[c.sequence.name] for c in chunk]
        serial = collate(chunk, stores, "cpu")
        with ThreadPoolExecutor(max_workers=4) as pool:
            parallel = collate(chunk, stores, "cpu", pool)

        np.testing.assert_array_equal(serial.images.numpy(), parallel.images.numpy())
        np.testing.assert_array_equal(serial.boxes.numpy(), parallel.boxes.numpy())
        for a, b in zip(serial.masks, parallel.masks):
            self.assertEqual(a is None, b is None)
            if a is not None:
                np.testing.assert_array_equal(a.numpy(), b.numpy())

    def test_only_depth_batches_are_ever_assembled_ahead(self):
        # The bound is batches, not threads. A batch of 64 clips is 1.6 GB, so
        # a queue sized by the worker count is how the host runs out of memory.
        import threading

        live, peak, lock = 0, 0, threading.Lock()

        def counted(chunk, stores, device="cpu", executor=None):
            nonlocal live, peak
            from src.training.clip_loop import collate as real
            with lock:
                live += 1
                peak = max(peak, live)
            try:
                return real(chunk, stores, device, executor)
            finally:
                with lock:
                    live -= 1

        import src.training.loader as loader
        real_collate = loader.collate
        loader.collate = counted
        try:
            chunks = list(batch_clips(self.clips, 2, seed=0, limit=6))
            list(prefetch(chunks, self.stores, device="cpu", workers=8, depth=2))
        finally:
            loader.collate = real_collate
        self.assertLessEqual(peak, 2)

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
