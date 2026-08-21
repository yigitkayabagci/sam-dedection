"""Keeping the GPU fed, and choosing how many clips to feed it.

Two things stand between the clip loop and the hardware it runs on.

**The data is small files, and a lot of them.** One batch of 16 clips is 128
JPEGs to decode, crop and normalise. On the training thread that is ~0.4 s of
pure Python and OpenCV against ~0.15 s of compute, so the accelerator idles for
two thirds of every step and a bigger GPU makes the run *no* faster. `prefetch`
moves that work onto threads -- `cv2.imread` and the numpy arithmetic around it
both release the GIL, so this is real parallelism, not concurrency theatre --
and hands the training thread batches that are already assembled. The threads
work *inside* a batch rather than on several at once, because host memory is
the thing that runs out first: 64 clips of 8 frames at 512x512 float32 is
1.6 GB, and a queue of those sized by the thread count is tens of gigabytes.

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
    clips: Sequence,
    size: int,
    seed: int | None = None,
    limit: int | None = None,
    drop_last: bool = True,
) -> Iterator[list]:
    """Shuffle `clips` reproducibly and cut them into batches of `size`.

    Named for clips and used for windows too (`aerial.Sample`): it never looks
    inside an item, and having one shuffle means a clip-mode run and an
    image-mode run are reproducible from their seed in exactly the same way.

    `limit` caps the number of batches, which is how an epoch is bounded on a
    dataset where every frame is a clip start: 60 sequences of 1000 frames is
    ~60000 clips, and an "epoch" over all of them is a day. A capped, reshuffled
    pass sees a different random subset each time, which is what is wanted
    anyway -- consecutive clip starts overlap in 7 of their 8 frames.

    **`limit` is a floor as well as a ceiling**, and it was not always. A single
    pass over a pool smaller than `limit * size` used to end the epoch early and
    say nothing: VTUAV alone is ~1 400 training windows, so at batch 64 an epoch
    asked for 400 steps and stopped after 21. Everything downstream then reads
    as if the run happened -- there is no error, the loss curve is just short --
    and `OneCycleLR`, built for 400 steps, never leaves its warm-up, so the run
    trains at a fraction of the intended rate.

    That also silently voided the comparison this repo is built around: a
    three-dataset run filled its 400 steps while the VTUAV-only run did 21, so
    "same schedule, only the data differs" was not true. The pool is now cycled
    with a fresh permutation per pass until `limit` batches exist, which is what
    "400 steps of batch 64" already meant everywhere it was written down.

    A pool smaller than one batch shrinks the batch instead of filling it by
    repeating items. That case is the validation slice, not training: scoring
    one batch that holds the same window three times weights the mean by pool
    size and nothing else, and the checkpoint rule reads that mean.
    """
    if not clips:
        return
    rng = np.random.default_rng(seed)
    if limit is not None:
        size = min(size, len(clips))
    batch: list[Clip] = []
    produced = 0
    while True:
        for i in rng.permutation(len(clips)):
            batch.append(clips[int(i)])
            if len(batch) < size:
                continue
            yield batch
            produced += 1
            batch = []
            if limit is not None and produced >= limit:
                return
        # One pass done. Without a limit that is the epoch; with one, keep
        # drawing -- a fresh permutation each time, so a small pool is reused
        # in a different order rather than in the same order.
        if limit is None:
            break
    if batch and not drop_last and (limit is None or produced < limit):
        yield batch


def prefetch_with(
    chunks: Iterable[list],
    assemble: Callable[[list, ThreadPoolExecutor], object],
    device: str = "cuda",
    workers: int = 8,
    depth: int = 2,
) -> Iterator:
    """Assemble `chunks` on worker threads; yield them, in order, on `device`.

    **`workers` and `depth` are different knobs on purpose.** `workers` is how
    many items are read at once, and wants to be around the core count.
    `depth` is how many *batches* may exist in host memory ahead of the
    training step, and wants to stay at two or three whatever `workers` is: a
    batch of 64 clips is 1.6 GB, so tying the queue length to the thread count
    would put 25 GB of pinned-down JPEG decode in front of a card that is
    perfectly happy with two.

    Two pools rather than one because a task that submits to the pool it is
    running on deadlocks the moment that pool is full. Batch assembly runs on
    its own small pool and blocks on the item pool, which submits nothing.

    Order is preserved -- futures are consumed from a queue rather than as they
    complete -- so a run stays reproducible from its seed.

    `assemble(chunk, pool)` is the only thing that knows what a batch is made
    of, which is what lets clip mode and image mode share this: one reads eight
    JPEGs and a run-length store per row, the other one image and a semantic
    map, and neither difference belongs in a prefetcher.
    """
    source = iter(chunks)
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as item_pool, \
            ThreadPoolExecutor(max_workers=max(depth, 1)) as batch_pool:
        pending: deque = deque()

        def submit() -> bool:
            chunk = next(source, None)
            if chunk is None:
                return False
            pending.append(batch_pool.submit(assemble, chunk, item_pool))
            return True

        for _ in range(max(depth, 1)):
            if not submit():
                break
        while pending:
            batch = pending.popleft().result()
            submit()
            yield batch.to(device)


def prefetch(
    chunks: Iterable[list[Clip]],
    stores: Mapping[str, Mapping[int, np.ndarray]],
    device: str = "cuda",
    workers: int = 8,
    depth: int = 2,
) -> Iterator[Batch]:
    """`prefetch_with`, wired to the clip collate and its mask stores."""
    return prefetch_with(
        chunks,
        lambda chunk, pool: collate(
            chunk, [stores[c.sequence.name] for c in chunk], "cpu", pool),
        device=device, workers=workers, depth=depth,
    )


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


def measure_batch_size(
    model,
    step: Callable[[int], object],
    device: str = "cuda",
    maximum: int | None = None,
    reserve: float = 0.15,
    candidates: Sequence[int] = CANDIDATES,
    verbose: bool = True,
) -> int:
    """The largest candidate `step(n)` runs in, measured rather than guessed.

    `step(n)` must build a batch of `n` and return its **loss with the graph
    still attached**; the backward is run here, because the backward is where
    the activation graph is actually held and a forward-only probe would report
    a size that OOMs on the first optimiser step.

    `reserve` keeps a fraction of the card free: fragmentation, the EMA copy
    and the evaluation pass all want memory the probe never asks for.

    Call this **after** the freeze: which parameters require gradients is most
    of what decides the answer.
    """
    import torch

    if not torch.cuda.is_available() or not device.startswith("cuda"):
        return 1

    total = torch.cuda.get_device_properties(device).total_memory
    budget = total * (1.0 - reserve)
    ceiling = maximum or max(candidates)

    def trial(n: int) -> bool:
        if n > ceiling:
            return False
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = step(n)
            loss.backward()
            peak = torch.cuda.max_memory_allocated(device)
        except torch.cuda.OutOfMemoryError:
            if verbose:
                print(f"  batch {n:>3}: out of memory")
            return False
        finally:
            model.zero_grad(set_to_none=True)
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
    """`measure_batch_size` for clip mode: the largest clip batch that trains here.

    Clip-mode activation memory scales with batch x clip length x resolution,
    and the numbers that fit differ by an order of magnitude between a 16 GB
    card and a 90 GB one.
    """
    from .clip_loop import clip_losses

    pool = list(clips)

    def step(n: int):
        batch = collate(pool[:n], [stores[c.sequence.name] for c in pool[:n]], "cpu")
        return clip_losses(model, batch.to(device))[0]

    return measure_batch_size(model, step, device,
                              maximum=min(maximum or len(pool), len(pool)),
                              reserve=reserve, candidates=candidates,
                              verbose=verbose)
