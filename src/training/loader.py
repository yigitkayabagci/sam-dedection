"""Keeping the GPU fed, and choosing how many clips to feed it.

Two things stand between the clip loop and the hardware it runs on.

**The data is small files, and a lot of them.** One batch of 16 clips is 128
JPEGs to decode, crop and normalise. On the training thread that is ~0.4 s of
pure Python and OpenCV against ~0.15 s of compute, so the accelerator idles for
two thirds of every step and a bigger GPU makes the run *no* faster. `prefetch`
moves that work onto threads -- `cv2.imread` and the numpy arithmetic around it
both release the GIL, so this is real parallelism, not concurrency theatre --
and hands the training thread batches that are already assembled.

**The batch size is a property of the machine, not of the recipe.** Clip-mode
activation memory scales with batch x clip length x resolution, and the numbers
that fit differ by an order of magnitude between a 16 GB card and a 90 GB one.
`auto_batch_size` measures it: a real forward and backward at each candidate
size, the largest one that both runs and leaves headroom. Guessing gets you
either an OOM three hours in or a GPU at 20 % utilisation.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .antiuav import Clip
from .clip_loop import Batch, collate

# Doubling to 16 and then widening the steps: the interesting range on a 90 GB
# card is 16-64, and probing 5, 6, 7 there wastes minutes for nothing.
CANDIDATES = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128)


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------


def batch_clips(
    clips: Sequence[Clip],
    size: int,
    seed: int | None = None,
    limit: int | None = None,
    drop_last: bool = True,
) -> Iterator[list[Clip]]:
    """Shuffle `clips` reproducibly and cut them into batches of `size`.

    `limit` caps the number of batches, which is how an epoch is bounded on a
    dataset where every frame is a clip start: 60 sequences of 1000 frames is
    ~60000 clips, and an "epoch" over all of them is a day. A capped, reshuffled
    pass sees a different random subset each time, which is what is wanted
    anyway -- consecutive clip starts overlap in 7 of their 8 frames.
    """
    order = np.random.default_rng(seed).permutation(len(clips))
    batch: list[Clip] = []
    produced = 0
    for i in order:
        batch.append(clips[int(i)])
        if len(batch) < size:
            continue
        yield batch
        produced += 1
        batch = []
        if limit is not None and produced >= limit:
            return
    if batch and not drop_last and (limit is None or produced < limit):
        yield batch


def prefetch(
    chunks: Iterable[list[Clip]],
    stores: Mapping[str, Mapping[int, np.ndarray]],
    device: str = "cuda",
    workers: int = 6,
    depth: int = 2,
) -> Iterator[Batch]:
    """Collate `chunks` on worker threads; yield them, in order, on `device`.

    At most `workers * depth` batches are ever in flight, so a slow training
    step cannot make the loader read the whole dataset into RAM ahead of it.
    Order is preserved: the futures are consumed from a queue rather than as
    they complete, which keeps a run reproducible from its seed.
    """
    source = iter(chunks)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: deque = deque()

        def submit() -> bool:
            chunk = next(source, None)
            if chunk is None:
                return False
            pending.append(pool.submit(
                collate, chunk, [stores[c.sequence.name] for c in chunk], "cpu"
            ))
            return True

        for _ in range(max(workers * depth, 1)):
            if not submit():
                break
        while pending:
            batch = pending.popleft().result()
            submit()
            yield batch.to(device)


# --------------------------------------------------------------------------
# Batch size
# --------------------------------------------------------------------------


def largest_that_fits(
    trial: Callable[[int], bool], candidates: Sequence[int] = CANDIDATES
) -> int:
    """The largest candidate `trial` accepts, stopping at the first refusal.

    Monotonicity is assumed, and it holds: a batch that does not fit never
    starts fitting at a larger size. Separating the search from what it
    measures is what makes it testable without a GPU.
    """
    best = 0
    for n in candidates:
        if not trial(n):
            break
        best = n
    if best == 0:
        raise RuntimeError(
            "not even the smallest candidate batch fits. Reduce the clip length "
            "or the input size before training."
        )
    return best


def auto_batch_size(
    model,
    clips: Sequence[Clip],
    stores: Mapping[str, Mapping[int, np.ndarray]],
    device: str = "cuda",
    maximum: int | None = None,
    reserve: float = 0.15,
    candidates: Sequence[int] = CANDIDATES,
    verbose: bool = True,
) -> int:
    """Measure the largest clip batch that trains on this GPU.

    Each candidate gets a real forward *and backward* -- the backward is where
    the activation graph is actually held, so a forward-only probe would report
    a size that OOMs on the first optimiser step. `reserve` keeps a fraction of
    the card free: fragmentation, the EMA copy and the evaluation pass all want
    memory that the probe never asks for.

    Call this **after** `apply_freeze`: which parameters require gradients is
    most of what decides the answer.
    """
    import torch

    from .clip_loop import clip_losses

    if not torch.cuda.is_available() or not device.startswith("cuda"):
        return 1

    total = torch.cuda.get_device_properties(device).total_memory
    budget = total * (1.0 - reserve)
    pool = list(clips)
    ceiling = min(maximum or len(pool), len(pool))

    def trial(n: int) -> bool:
        if n > ceiling:
            return False
        batch = collate(pool[:n], [stores[c.sequence.name] for c in pool[:n]], "cpu")
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = clip_losses(model, batch.to(device))
            loss.backward()
            peak = torch.cuda.max_memory_allocated(device)
        except torch.cuda.OutOfMemoryError:
            if verbose:
                print(f"  batch {n:>3}: out of memory")
            return False
        finally:
            model.zero_grad(set_to_none=True)
            del batch
            torch.cuda.empty_cache()
        if verbose:
            print(f"  batch {n:>3}: peak {peak / 2**30:5.1f} GiB "
                  f"of {total / 2**30:.0f} GiB"
                  f"{'  <- over budget' if peak > budget else ''}")
        return peak <= budget

    best = largest_that_fits(trial, candidates)
    if verbose:
        print(f"batch size {best}")
    return best
