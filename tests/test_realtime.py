"""Drop-stale frame delivery. Pure stdlib -- no EdgeTAM, no torch, no cv2.

Run with:  python -m pytest tests/test_realtime.py
      or:  python -m unittest tests.test_realtime
"""
from __future__ import annotations

import threading
import time
import unittest

from src.realtime import LatestFrameQueue, RealtimeSource


class TestLatestFrameQueue(unittest.TestCase):
    def test_put_overwrites_and_counts_the_drop(self):
        q = LatestFrameQueue()
        q.put(1)
        q.put(2)
        q.put(3)
        self.assertEqual(q.get(), 3)  # the newest, not the oldest
        self.assertEqual(q.dropped, 2)

    def test_nothing_is_dropped_when_the_consumer_keeps_up(self):
        q = LatestFrameQueue()
        for i in range(5):
            q.put(i)
            self.assertEqual(q.get(), i)
        self.assertEqual(q.dropped, 0)

    def test_get_returns_none_once_closed_and_drained(self):
        q = LatestFrameQueue()
        q.put("last")
        q.close()
        self.assertEqual(q.get(), "last")
        self.assertIsNone(q.get())

    def test_get_blocks_until_a_frame_arrives(self):
        q = LatestFrameQueue()
        threading.Timer(0.05, q.put, args=("late",)).start()
        start = time.perf_counter()
        self.assertEqual(q.get(), "late")
        self.assertGreater(time.perf_counter() - start, 0.02)

    def test_close_releases_a_waiting_consumer(self):
        q = LatestFrameQueue()
        got = []
        reader = threading.Thread(target=lambda: got.append(q.get()))
        reader.start()
        q.close()
        reader.join(timeout=1.0)
        self.assertFalse(reader.is_alive())
        self.assertEqual(got, [None])


class TestRealtimeSource(unittest.TestCase):
    def test_a_fast_consumer_sees_every_frame(self):
        with RealtimeSource(frame_count=6, fps=200.0) as source:
            self.assertEqual(list(source), list(range(6)))
        self.assertEqual(source.dropped, 0)
        self.assertEqual(source.delivered, 6)

    def test_a_slow_consumer_skips_ahead_instead_of_falling_behind(self):
        # 40 frames at 400 FPS = 100 ms of source; ~10 ms per frame of work
        # means roughly every fourth frame, and never a backlog.
        seen = []
        with RealtimeSource(frame_count=40, fps=400.0) as source:
            for idx in source:
                seen.append(idx)
                time.sleep(0.01)
        self.assertLess(len(seen), 40)
        self.assertEqual(seen, sorted(seen))          # still chronological
        self.assertEqual(len(seen) + source.dropped, source.frame_count)
        # The point of dropping: the last frame tracked is the last frame the
        # source produced, not something from the start of the backlog.
        self.assertEqual(seen[-1], 39)

    def test_waiting_for_the_source_is_recorded_separately(self):
        wait_s: list[float] = []
        with RealtimeSource(frame_count=4, fps=100.0, wait_s=wait_s) as source:
            list(source)
        # Four frames at 100 FPS: the consumer does nothing, so nearly all of
        # the ~30 ms it spent iterating was idle and must not be charged to it.
        self.assertEqual(len(wait_s), 5)  # one per frame, plus the closing read
        self.assertGreater(sum(wait_s), 0.02)

    def test_start_skips_frames_before_the_prompt(self):
        with RealtimeSource(frame_count=5, fps=200.0, start=2) as source:
            self.assertEqual(list(source), [2, 3, 4])

    def test_leaving_early_stops_the_producer(self):
        source = RealtimeSource(frame_count=10_000, fps=100.0)
        with source:
            for idx in source:
                break
        self.assertFalse(source._thread.is_alive())

    def test_rejects_a_nonpositive_rate(self):
        with self.assertRaises(ValueError):
            RealtimeSource(frame_count=1, fps=0.0)


if __name__ == "__main__":
    unittest.main()
