"""Drop-stale frame delivery, for when the tracker cannot keep up with the source.

A camera does not wait. It produces a frame every 1/fps seconds whether or not
anything is ready to look at one, and a tracker slower than that has to decide
what to do with the backlog. Buffering every frame is the option that quietly
fails: at 15 FPS on a 30 FPS source the tracker falls a second behind within a
minute and never recovers, because each frame it finishes is followed by one
that is older still. The masks stay accurate for a scene that has already
happened.

Dropping instead bounds the lag at one frame time. Whatever is current when the
model comes free is what it looks at; everything that arrived while it was busy
is gone. The cost is frames never processed -- and in a memory-based tracker
like EdgeTAM, a memory bank with holes in it -- which is the trade this module
exists to make measurable.
"""
from __future__ import annotations

import threading
import time
from typing import Iterator


class LatestFrameQueue:
    """A one-slot mailbox: a newly delivered frame replaces the waiting one.

    `put` never blocks and the queue never grows past a single frame -- that is
    the whole mechanism. `get` blocks until a frame is waiting, so a consumer
    faster than the source idles instead of spinning, and a consumer slower than
    the source always picks up the newest frame rather than the next one in line.

    Single producer, single consumer. `dropped` counts frames overwritten before
    anyone read them.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._item = None  # None means empty; the producer never puts None
        self._closed = False
        self.dropped = 0

    def put(self, item) -> None:
        with self._cond:
            if self._item is not None:
                self.dropped += 1
            self._item = item
            self._cond.notify()

    def get(self):
        """The freshest frame. None once the producer has closed and drained."""
        with self._cond:
            while self._item is None and not self._closed:
                self._cond.wait()
            item, self._item = self._item, None
            return item

    def close(self) -> None:
        """No more frames are coming; release a consumer blocked in `get`."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()


class RealtimeSource:
    """Frame indices of a clip, delivered on a clock instead of on demand.

    A producer thread pushes index `start`, `start + 1`, ... into a
    `LatestFrameQueue` one every `1 / fps` seconds, and iterating this object
    pulls from the other end. Consume it faster than `fps` and iteration blocks
    until the next frame is due; consume it slower and the indices come back
    with holes, because the frames in between were overwritten before anyone
    asked for them.

    The thread is what makes this a source rather than a schedule. The same
    indices could be computed from the wall clock in three lines, but then the
    pacing would be an assumption baked into the consumer instead of a property
    of the producer -- and swapping a real camera in later would mean rewriting
    the consumer rather than replacing the thread.

    Seconds spent blocked in `get` are appended to `wait_s` so the caller can
    take them back out of its frame budget: waiting for a camera is not work the
    model did.

    Seconds of staleness at pickup -- how long ago the frame handed over was
    actually due -- go into `lag_s`. This is the number that proves the
    mechanism rather than just describing it: the item sitting in a one-slot
    queue is always the most recent one produced before pickup, so its age at
    pickup is bounded by one source period no matter how far behind the
    consumer's *throughput* is. A model twice, five times, ten times too slow
    still only ever picks up a frame less than `1/fps` old; only the drop rate
    changes. `lag_s` climbing past that bound on a real run is the one number
    that says the mechanism itself is broken, as opposed to merely dropping a
    lot of frames, which is expected.
    """

    def __init__(self, frame_count: int, fps: float, start: int = 0,
                 wait_s: list[float] | None = None,
                 lag_s: list[float] | None = None) -> None:
        if fps <= 0:
            raise ValueError(f"realtime fps must be positive, got {fps}")
        self.frame_count = frame_count
        self.fps = fps
        self.start = start
        self.wait_s = wait_s if wait_s is not None else []
        self.lag_s = lag_s if lag_s is not None else []
        self.delivered = 0
        self._queue = LatestFrameQueue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._produce, name="frame-source",
                                        daemon=True)
        # Set in __enter__, before the producer thread starts, so both threads
        # agree on frame 0's due time without a lock: the main thread finishes
        # writing it before the producer (or a consumer calling __iter__) can
        # possibly read it.
        self._t0 = 0.0

    @property
    def dropped(self) -> int:
        return self._queue.dropped

    def _produce(self) -> None:
        period = 1.0 / self.fps
        for offset, idx in enumerate(range(self.start, self.frame_count)):
            # Against `_t0` rather than the previous frame, so a late wake-up
            # does not push every frame after it late as well.
            due = self._t0 + offset * period
            # `Event.wait` doubles as an interruptible sleep: True means the
            # consumer left early and there is nobody to deliver to.
            if self._stop.wait(max(due - time.perf_counter(), 0.0)):
                break
            self._queue.put(idx)
        self._queue.close()

    def __enter__(self) -> "RealtimeSource":
        self._t0 = time.perf_counter()
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._queue.close()
        self._thread.join(timeout=1.0)

    def __iter__(self) -> Iterator[int]:
        period = 1.0 / self.fps
        while True:
            waiting_from = time.perf_counter()
            idx = self._queue.get()
            now = time.perf_counter()
            self.wait_s.append(now - waiting_from)
            if idx is None:
                return
            self.delivered += 1
            due = self._t0 + (idx - self.start) * period
            self.lag_s.append(max(now - due, 0.0))
            yield idx
