"""SAM2Long: keep several memory pathways instead of betting on one.

An **experimental variant**, offered next to SAMURAI rather than instead of it.
SAMURAI is the recommended answer to memory poisoning because it costs
essentially nothing; this costs real milliseconds and is here so the accuracy
question can be answered with a measurement instead of an argument.

The idea (Ding et al., 2024) attacks the same failure from the other side.
SAM 2 commits to one mask per frame and writes it to memory; if that choice was
wrong, the memory is wrong and there is no way back. SAM2Long instead carries
`pathways` competing hypotheses, each with its own memory, and only decides at
the end which one to believe.

How it fits here without a new inference loop
---------------------------------------------
SAM 2 batches *objects*, and every row of that batch already carries its own
memory bank, its own object pointer and its own object score -- the attention
never mixes rows. So **N pathways are N replicated objects**: the tracker
prompts the same box N times, the rows diverge because they take different
candidate masks, and the output collapses back to the best-scoring row. No new
propagation loop, no engine change beyond `--max-batch >= N x objects`.

Divergence is deliberately rare. On a frame where the candidates are clearly
separated, every pathway takes the same one and they differ only through the
memory each has already accumulated. They split only when the head is
genuinely unsure -- which is the paper's constrained tree search, and what
stops the hypotheses fanning out into noise.

What this does not reproduce
----------------------------
The paper also refines *which* memories a pathway keeps using object-occurrence
statistics. That is not implemented; each pathway uses SAM 2's stock memory
selection (or SAMURAI's, if that is also switched on -- they compose, since one
chooses within a row and the other filters across frames).

The cost, stated plainly: memory attention is **61.9 % of a frame's arithmetic**
(`docs/tensorrt_fp16.md`) and it runs once per pathway. Three pathways puts a
10 ms frame near 22 ms. That fits a 20-30 ms budget and spends most of it here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Internal object ids are `user_id * PATHWAY_STRIDE + pathway`. A stride far
# above any plausible object count keeps the two recoverable by arithmetic
# rather than by a side table that could fall out of step with the state.
PATHWAY_STRIDE = 1000


@dataclass(frozen=True)
class Sam2LongConfig:
    """How many hypotheses, and when they are allowed to disagree.

    `pathways`            competing memory banks per tracked object. 3 is the
                          paper's setting; every one of them costs another pass
                          of the memory path.
    `uncertainty_margin`  a frame is uncertain when the best two candidates are
                          within this of each other. Below it there is nothing
                          to disagree about and branching would only add noise.
    `object_score_floor`  a frame is also uncertain when the object score is
                          this low -- the case where SAM 2 is about to write
                          `no_obj_ptr` and being wrong is most expensive.
    `score_weight`        how much the object score counts towards a pathway's
                          running total, against mask quality.
    """

    enabled: bool = False
    pathways: int = 3
    uncertainty_margin: float = 0.05
    object_score_floor: float = 1.0
    score_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.pathways < 1:
            raise ValueError(f"pathways must be >= 1, got {self.pathways}")


def expand_ids(obj_id: int, pathways: int) -> list[int]:
    """The internal object ids standing in for one user object's pathways."""
    return [obj_id * PATHWAY_STRIDE + p for p in range(pathways)]


def split_id(internal: int) -> tuple[int, int]:
    """`(user object id, pathway rank)` from an internal id."""
    return divmod(int(internal), PATHWAY_STRIDE)


class Sam2Long:
    """Per-row pathway state: which candidate to take, and how each is doing.

    `set_rows` has to be called once the tracker knows the batch order, because
    the mask decoder sees rows while everything else here is keyed by object
    id. Reading the order off the inference state rather than assuming it
    follows insertion is what keeps the two from silently diverging.
    """

    def __init__(self, config: Sam2LongConfig | None = None) -> None:
        self.config = config or Sam2LongConfig()
        self.ranks: list[int] = []
        self.ids: list[int] = []
        self.scores = np.zeros(0)
        self.branches = 0

    def set_rows(self, internal_ids) -> None:
        self.ids = [int(i) for i in internal_ids]
        self.ranks = [split_id(i)[1] for i in self.ids]
        self.scores = np.zeros(len(self.ids))

    def reset(self) -> None:
        self.ranks, self.ids = [], []
        self.scores = np.zeros(0)
        self.branches = 0

    # -- the branching decision -------------------------------------------

    def assign(self, iou_pred, object_score_logits):
        """Rewrite the IoU predictions so each pathway takes its own candidate.

        The same lever SAMURAI uses, for a different purpose: SAM 2 derives the
        mask, the object pointer and the memory write from one `argmax` over
        these values, so steering it steers all three together and nothing has
        to be recomputed or kept consistent by hand.
        """
        import torch

        if iou_pred.shape[-1] < 2:
            # The prompted frame can be single-mask: there is nothing to
            # choose between, and `_uncertain` needs a runner-up to compare to.
            return iou_pred
        ious = iou_pred.detach().float().cpu().numpy()
        scores = object_score_logits.detach().float().cpu().numpy().reshape(-1)
        if not self.ranks or len(self.ranks) != ious.shape[0]:
            return iou_pred  # rows not registered yet; behave as stock SAM 2

        out = iou_pred.clone()
        candidates = ious.shape[1]
        for row in range(ious.shape[0]):
            order = np.argsort(-ious[row])
            chosen = order[0]
            if self._uncertain(ious[row], scores[row]):
                chosen = order[min(self.ranks[row], candidates - 1)]
                if self.ranks[row] > 0:
                    self.branches += 1
            # Only the winner has to come out on top; leaving the rest as they
            # were keeps the reported values meaningful for anything that reads
            # them, and SAM 2 itself only looks at the argmax.
            rewritten = ious[row].copy()
            rewritten[chosen] = float(ious[row].max()) + 1.0
            out[row] = torch.as_tensor(rewritten, dtype=out.dtype, device=out.device)
            self.scores[row] += (float(ious[row][chosen])
                                 + self.config.score_weight * _sigmoid(scores[row]))
        return out

    def _uncertain(self, ious: np.ndarray, object_score: float) -> bool:
        top = np.sort(ious)[::-1]
        return (bool(top[0] - top[1] < self.config.uncertainty_margin)
                or bool(object_score < self.config.object_score_floor))

    # -- collapsing back --------------------------------------------------

    def winners(self) -> dict[int, int]:
        """`{user object id: winning row}` by cumulative score.

        Ties go to the lowest pathway rank, which is the one that always took
        the head's own first choice -- so with everything else equal, the
        answer is stock SAM 2's.
        """
        best: dict[int, int] = {}
        for row, internal in enumerate(self.ids):
            user = split_id(internal)[0]
            current = best.get(user)
            if current is None or self.scores[row] > self.scores[current]:
                best[user] = row
        return best


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0))))


def install(predictor, config: Sam2LongConfig | None = None) -> Sam2Long:
    """Patch the mask decoder so pathways can disagree. Returns the state."""
    state = Sam2Long(config)
    if not state.config.enabled:
        return state

    decoder = predictor.sam_mask_decoder
    decode = decoder.forward

    def wrapped(*args, **kwargs):
        masks, iou_pred, tokens, object_score_logits = decode(*args, **kwargs)
        return (masks, state.assign(iou_pred, object_score_logits), tokens,
                object_score_logits)

    decoder.forward = wrapped
    cfg = state.config
    print(f"[sam2long] {cfg.pathways} memory pathways per object "
          f"(branch when the top two are within {cfg.uncertainty_margin} or the "
          f"object score is below {cfg.object_score_floor}). Engines need "
          f"--max-batch >= {cfg.pathways} x objects.")
    return state
