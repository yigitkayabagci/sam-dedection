"""The mask arm: hide the parts of the frame the model itself finds informative.

Masked image modelling on a **convolutional** trunk, which is the constraint
that shapes everything here. `docs/encoder_training_todo.md` closed this route
with one sentence -- MAE masks tokens and needs a ViT, EdgeTAM's trunk is a
RepViT -- and that sentence is right about MAE and wrong about masking. What
MAE actually needs a ViT for is *dropping* the masked tokens, which is how it
gets its 3x speedup; ConvNeXt V2 (arXiv 2301.00808) had to reach for sparse
convolution to recover the same trick, and reported that a dense convolution
over a masked image **leaks** -- the kernel straddles the boundary and copies
the hole's shape forward, so the model can learn where the mask was instead of
what was under it.

Neither of those is a reason a convolutional trunk cannot be trained on a
masked input. They are reasons it cannot be trained *cheaply* and that the
objective has to be chosen so leakage does not pay. So:

* **No sparse convolution.** The student runs a full dense forward on the
  masked image. That forfeits MAE's speedup and nothing else; at this scale the
  encoder is 38.7 GFLOP and the run is minutes, not days.
* **No pixel reconstruction.** The target is a *feature map*, not the missing
  pixels. Reconstructing pixels is where leakage pays -- a model that knows
  where the hole is can hedge -- and it is also the part of MAE that later work
  kept moving away from: CAPI predicts latent clusters, data2vec predicts the
  teacher's own latents, and the direction of travel has been away from pixels
  for four years.
* **No decoder.** The prediction is read straight off the encoder's stride-16
  map, which is the tensor stage B and the deployment both consume.

**One real cost of skipping sparse convolution, stated plainly.** RepViT's
later stages contain squeeze-and-excitation blocks, and an SE gate is a global
average over the spatial map. Averaged over a masked input that average is
biased by the visible fraction -- and at stage B and at deployment there is no
mask, so the gate sees a distribution this stage never trained it on. SparK's
answer is to patch every pooling and normalisation layer to count only visible
positions; that is the correct fix and it is roughly a hundred lines against a
trunk this module does not own. What is done instead is to keep the bias small
and measurable: the fill value is the **mean pixel** (zero after
normalisation), which is the least-biasing choice available, and the default
mask ratio is 0.5 rather than MAE's 0.75, which halves the bias. If this arm is
ever the winner, patching the pooling is the first thing to try next.

**And it is the arm least likely to win, which is worth knowing before the GPU
hours are spent.** The measured record for masked pretraining at this scale is
poor: ConvNeXt V2's own table shows that on an *unmodified* architecture its
FCMAE bought -0.1 to +0.1 top-1, the +1.0 in the headline coming from
co-designing the GRN layer alongside it; TinyMIM reports MAE pretraining making
a 5.7 M ViT **worse** than training it from scratch (-0.6). And the specific
claim that a learned mask beats a random one is smaller than it reads: HPM's
own isolation ablation attributes +0.26 of its +0.72 to simply adding the
auxiliary predictor, leaving +0.46 for the mask itself, and MILAN's
semantic-aware sampling is worth +0.3 against random with everything else
fixed. None of it was measured below 22 M parameters. So this arm is run
because "we did not try" is a bad answer in a report and because the corpora it
can eat are ones the other arm cannot -- not because it is expected to win.

## Where the mask comes from, which is the point

The published default is uniform random masking, and there is a body of work
arguing it can be beaten by masking what *matters*: AttMask (ECCV 2022) masks
the patches a ViT's class token attends to most, and reports that hiding the
easy background teaches less than hiding the object. A convolutional trunk has
no attention map to read, but it has the thing attention was standing in for --
how far a position's feature is from the image's own average feature. A cell
that looks like everything else in the frame is background; a cell that does
not is where the content is. That is `saliency`, and it makes the mask a
function of the model's current state rather than of a random number generator,
which is what "the model produces its own mask" means here.

**And it is not obviously better, which is why it is a setting.** AttMask's own
lesson is that masking *only* the most salient cells can make the task
unsolvable rather than hard, so it reveals a slice of the very top back as a
hint; `mode="hint"` is that, `mode="attentive"` is the version without it, and
`mode="random"` is the baseline both have to beat. On a corpus of aerial
thermal frames -- mostly uniform ground with a few small hot targets --
saliency-driven masking hides the targets and leaves the ground, which is
either exactly right or a way to spend every gradient on the 2 % of the frame
that is hardest. That is an experiment, not a preference, and the three modes
run through the same loop so the comparison is fair.

## What supervises it

The target is the model's own map on the **unmasked** frame, from a copy of
itself. Two choices, and the difference is not cosmetic:

* `target="ema"` -- an exponential moving average of the student, updated every
  step. This is the data2vec / BootMAE shape and it is the one that can
  actually *improve* the representation, because the target moves with the
  model. It can also collapse: nothing stops student and target from agreeing
  on a constant. The defences are the ones the literature uses -- score only
  the masked positions, so the trivially-copyable ones give no gradient;
  centre the target across channels; start from trained weights rather than
  noise -- and `collapse` measures whether they held rather than assuming it.
* `target="frozen"` -- the initial weights, never updated. Cannot collapse, by
  construction: the objective becomes "recover the stock encoder's clean
  features from a masked frame", which is an occlusion-robustness objective and
  a genuinely useful one for a tracker. Its ceiling is the stock encoder, which
  is the honest thing against it.

Both write the same artifact and are graded the same way -- by stage B, against
the arm that pretrained nothing at all.
"""
from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .distill import encoder_features
from .pretrain import Batch, Item, centre, run

MODES = ("random", "attentive", "hint", "block", "green")
TARGETS = ("ema", "frozen")


# --------------------------------------------------------------------------
# Saliency and masks
# --------------------------------------------------------------------------


def saliency(features: torch.Tensor, mode: str = "distinct") -> torch.Tensor:
    """`[B, h, w]` -- how much each cell differs from the rest of its own frame.

    `distinct` is the convolutional stand-in for AttMask's attention map: the
    cosine distance between a position's feature and the frame's mean feature
    direction. A ViT's class token attends to the positions that are unlike the
    background, and this is that quantity computed directly. It is
    scale-invariant, which matters because feature magnitude varies across a
    map for reasons that have nothing to do with content.

    `norm` is the cruder alternative -- feature magnitude alone -- and is here
    because it is what "feature energy" means in the papers that use it, and
    because it costs one line to be able to say which of the two the run used.
    """
    maps = features.float()
    if mode == "norm":
        return maps.norm(dim=1)
    if mode == "distinct":
        unit = F.normalize(maps, dim=1)
        mean = F.normalize(unit.mean(dim=(2, 3), keepdim=True), dim=1)
        return 1.0 - (unit * mean).sum(dim=1)
    raise ValueError(f"saliency mode must be 'distinct' or 'norm', got {mode!r}")


def green_noise(shape: tuple[int, int, int], low: float = 0.15,
                high: float = 0.45, generator: torch.Generator | None = None,
                device: str = "cpu") -> torch.Tensor:
    """A band-pass-filtered noise field -- ColorMAE's "green noise" (ECCV 2024).

    The cheapest thing in this module and the one hardest to argue against.
    ColorMAE (arXiv 2407.13036) generates masks by filtering white noise in the
    frequency domain rather than learning anything: low-pass gives large blobs
    ("red"), high-pass gives scattered single cells ("blue"), band-pass gives
    medium clusters ("green"), and green was its best. It has **no parameters
    and no extra forward pass**, and in its own controlled comparison it
    matched or beat the learned maskers of the previous three years.

    That is the reason it is here beside `attentive` and `hint`: those two are
    what "the model makes its own mask" means, and this is the control that
    says whether making one was worth anything. A run where green noise ties
    the saliency-driven mask is a run that has learned something real about
    this data, and cheaply.

    The band is in normalised radial frequency, 0 at DC and 1 at the corner.
    """
    batch, height, width = shape
    field = torch.randn(batch, height, width, device=device, generator=generator)
    spectrum = torch.fft.fft2(field)
    fy = torch.fft.fftfreq(height, device=device)[:, None]
    fx = torch.fft.fftfreq(width, device=device)[None, :]
    radius = (fy * fy + fx * fx).sqrt() / (0.5 * 2 ** 0.5)
    band = ((radius >= low) & (radius <= high)).to(spectrum.real.dtype)
    return torch.fft.ifft2(spectrum * band).real


def sample_mask(scores: torch.Tensor, ratio: float = 0.5, mode: str = "hint",
                hint: float = 0.1, block: int = 2,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """`[B, h, w]` boolean -- True where the input will be blanked.

    Exactly `round(ratio * h * w)` cells are masked in every mode and every
    image, so the four modes differ in *which* cells and in nothing else. A
    ratio that varied per image would put a second moving part into the
    comparison and into the loss, which is averaged over masked positions.

    * `random` -- uniform. The baseline.
    * `green` -- the top `ratio` of a band-pass-filtered noise field, which
      produces medium-sized clusters without looking at the image at all. The
      *other* baseline, and the one that matters: it costs nothing and it is
      what a learned mask has to beat to have earned its extra forward pass.
    * `attentive` -- the `ratio` most salient cells. AttMask's harsh variant.
    * `hint` -- the most salient cells **except** the very top `hint` fraction,
      which stay visible. AttMask's own remedy for a task that becomes
      unsolvable rather than hard: something of the object is always left.
    * `block` -- contiguous `block`x`block` squares, chosen at random. SimMIM's
      shape, and the one that a convolution can least easily see around: a
      scattered single-cell mask is 16 pixels wide and a 3x3 kernel two layers
      up already spans it.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"ratio must be in (0, 1), got {ratio}")
    batch, height, width = scores.shape
    cells = height * width
    wanted = max(int(round(ratio * cells)), 1)
    device = scores.device

    if mode == "block":
        coarse_h = max(height // block, 1)
        coarse_w = max(width // block, 1)
        coarse = max(int(round(ratio * coarse_h * coarse_w)), 1)
        noise = torch.rand(batch, coarse_h * coarse_w, device=device,
                           generator=generator)
        picked = noise.argsort(dim=1)[:, :coarse]
        flat = torch.zeros(batch, coarse_h * coarse_w, dtype=torch.bool, device=device)
        flat.scatter_(1, picked, True)
        grid = flat.reshape(batch, 1, coarse_h, coarse_w).float()
        grown = F.interpolate(grid, size=(height, width), mode="nearest")
        return grown[:, 0] > 0.5

    if mode in ("random", "green"):
        field = (torch.rand(batch, cells, device=device, generator=generator)
                 if mode == "random"
                 else green_noise((batch, height, width), generator=generator,
                                  device=device).reshape(batch, cells))
        chosen = field.argsort(dim=1, descending=True)[:, :wanted]
    else:
        order = scores.reshape(batch, cells).argsort(dim=1, descending=True)
        start = int(round(hint * wanted)) if mode == "hint" else 0
        start = min(start, max(cells - wanted, 0))
        chosen = order[:, start:start + wanted]

    mask = torch.zeros(batch, cells, dtype=torch.bool, device=device)
    mask.scatter_(1, chosen, True)
    return mask.reshape(batch, height, width)


def blank(images: torch.Tensor, mask: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    """Replace the masked cells' pixels, at the ratio the feature grid implies.

    `fill=0.0` is not black. The images are ImageNet-normalised, so zero is the
    **mean pixel** -- the least informative value available, and the one every
    masked-modelling paper uses for the same reason. Filling with black would
    be filling with a strong, learnable signal.
    """
    scale = images.shape[-1] // mask.shape[-1]
    if scale < 1 or images.shape[-1] != scale * mask.shape[-1]:
        raise ValueError(
            f"a {mask.shape[-2]}x{mask.shape[-1]} mask does not tile a "
            f"{images.shape[-2]}x{images.shape[-1]} image")
    grown = F.interpolate(mask[:, None].float(), scale_factor=scale, mode="nearest")
    return images * (1.0 - grown) + fill * grown


# --------------------------------------------------------------------------
# The loss, and the thing that has to be watched while it runs
# --------------------------------------------------------------------------


def masked_loss(student: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor, whiten: bool = True
                ) -> tuple[torch.Tensor, dict[str, float]]:
    """Cosine distance on the masked cells, and the unmasked ones as a check.

    Only the masked positions carry gradient. The unmasked ones are the
    positions whose pixels the student can see, so agreeing there is a copy
    rather than an inference -- data2vec scores masked positions only, and the
    same argument applies with more force here, where a dense convolution can
    see across a mask boundary and would otherwise be rewarded for it.

    The unmasked cosine is still *reported*, because it is the cheapest
    available check that nothing is broken: it should sit near 1 and stay
    there. A run where it falls is a run where the student is drifting away
    from its own target everywhere, not learning to fill holes.
    """
    if whiten:
        target = centre(target)
    if student.shape[-2:] != target.shape[-2:]:
        raise ValueError(
            f"student map {tuple(student.shape[-2:])} and target map "
            f"{tuple(target.shape[-2:])} disagree")
    cosine = 1.0 - (F.normalize(student.float(), dim=1)
                    * F.normalize(target.float(), dim=1)).sum(dim=1)
    hidden = mask.float()
    seen = 1.0 - hidden
    loss = (cosine * hidden).sum() / hidden.sum().clamp(min=1.0)
    terms = {"masked": float(loss.detach()),
             "visible": float(((cosine * seen).sum()
                               / seen.sum().clamp(min=1.0)).detach())}
    return loss, terms


@torch.no_grad()
def collapse(features: torch.Tensor, positions: int = 256,
             seed: int = 0) -> dict[str, float]:
    """Two numbers that say whether the representation is still a representation.

    `similarity` is the mean cosine between pairs of *different* positions in
    the same frame. A healthy dense map has positions that disagree -- that is
    what makes it dense -- so this sits well below 1. Approaching 1 is collapse:
    every position has become the same vector and the map carries no spatial
    information for the decoder to read.

    `channel_std` is the mean spread of each channel across positions. Falling
    toward 0 is the same failure seen from the other side, and it is the one
    `distill.moment_loss` was written to penalise.

    Sampled at `positions` cells rather than computed over all of them: the
    pairwise matrix is quadratic, and a 32x32 map has 1 024 of them.
    """
    unit = F.normalize(features.float(), dim=1).flatten(2)
    count = unit.shape[-1]
    if count > positions:
        pick = torch.randperm(count, device=unit.device,
                              generator=torch.Generator(device=unit.device)
                              .manual_seed(seed))[:positions]
        unit = unit[..., pick]
    gram = torch.einsum("bcn,bcm->bnm", unit, unit)
    n = gram.shape[-1]
    off = (gram.sum(dim=(1, 2)) - n) / max(n * (n - 1), 1)
    return {"similarity": float(off.mean()),
            "channel_std": float(features.float().flatten(2).std(dim=2).mean())}


# --------------------------------------------------------------------------
# The target model
# --------------------------------------------------------------------------


class Target:
    """A second copy of the encoder, holding what the masked student aims at.

    Kept as a whole model rather than as a shadow of the parameters (which is
    what `finetune.EMA` is) because it has to be *run*, not just averaged, and
    running it needs the buffers and the module graph too. It is never
    optimised and never saved.
    """

    def __init__(self, model, decay: float = 0.999, mode: str = "ema") -> None:
        if mode not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, got {mode!r}")
        self.mode = mode
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def features(self, images: torch.Tensor) -> torch.Tensor:
        return encoder_features(self.model, images)

    @torch.no_grad()
    def update(self, model) -> None:
        """Move the target a step toward the student, or not at all if frozen."""
        if self.mode == "frozen":
            return
        theirs = dict(model.named_parameters())
        for name, param in self.model.named_parameters():
            other = theirs.get(name)
            if other is not None:
                param.lerp_(other.detach(), 1.0 - self.decay)


# --------------------------------------------------------------------------
# The arm
# --------------------------------------------------------------------------


def pretrain_masked(model, items: Sequence[Item], *, save, size: int = 512,
                    ratio: float = 0.5, mode: str = "hint",
                    score: str = "distinct", hint: float = 0.1,
                    block: int = 2, target: str = "ema",
                    target_decay: float = 0.999, whiten: bool = True,
                    watch_every: int = 50, device: str = "cuda",
                    seed: int = 0, log=print, **kwargs) -> dict:
    """Train the encoder to recover its own clean features from a masked frame.

    Needs no teacher and no registered pair, which is the whole reason this arm
    exists beside the distillation one: the corpora it can eat are the
    thermal-only ones, which are the large ones and the ones in the modality
    the deployment actually sees.
    """
    watcher = Target(model, decay=target_decay, mode=target)
    log(f"target: {target} copy of the encoder"
        + (f", EMA decay {target_decay}" if target == "ema" else ", frozen")
        + f" | mask {mode} at {ratio:.0%}"
        + (f" (score {score}, hint {hint:.0%})" if mode in ("attentive", "hint") else "")
        + (f" (block {block})" if mode == "block" else ""))

    generator = torch.Generator(device=device).manual_seed(seed)
    state = {"step": 0, "watch": {}}

    def step(chunk: Batch):
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=device.startswith("cuda")):
            clean = watcher.features(chunk.student)
        # `random` and `block` ignore the scores; computing them anyway would
        # be a per-step cost for a number nothing reads, so the shape is
        # borrowed from the map instead and `sample_mask` never looks at it.
        scores = (saliency(clean, score) if mode in ("attentive", "hint")
                  else clean[:, 0].float())
        hidden = sample_mask(scores.detach(), ratio=ratio, mode=mode, hint=hint,
                             block=block, generator=generator)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=device.startswith("cuda")):
            guess = encoder_features(model, blank(chunk.student, hidden))
        loss, terms = masked_loss(guess, clean, hidden, whiten=whiten)

        # Recomputed every `watch_every` steps and reported on every one, so
        # the epoch mean is a held value rather than a sample mean. That is the
        # intent: the collapse numbers are a level to read, not a curve, and a
        # quadratic Gram matrix on every step is not worth what it buys.
        state["step"] += 1
        if watch_every and state["step"] % watch_every == 1:
            state["watch"] = collapse(guess.detach(), seed=seed)
        terms |= state["watch"]
        watcher.update(model)
        return loss, terms

    result = run(model, items, step, save=save, size=size, device=device,
                 want_teacher=False, seed=seed, log=log,
                 meta={"arm": "mask", "mask_mode": mode, "mask_ratio": ratio,
                       "saliency": score, "hint": hint, "block": block,
                       "target": target, "target_decay": target_decay,
                       "whiten": whiten},
                 **kwargs)
    del watcher
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result | {"arm": "mask", "mask_mode": mode, "mask_ratio": ratio}


def summarise_collapse(history: Sequence[dict]) -> str:
    """Whether the representation held up, read off the epoch rows.

    Stated as a verdict because the failure it names is one that a loss curve
    hides completely: a collapsing EMA self-distillation has a loss that falls
    beautifully, all the way to zero, while the map behind it becomes constant.
    """
    rows = [r for r in history if "similarity" in r]
    if not rows:
        return "no collapse statistics recorded"
    first, last = rows[0], rows[-1]
    lines = [f"{'epoch':>5}  {'masked':>8}  {'visible':>8}  "
             f"{'similarity':>10}  {'channel_std':>11}"]
    for row in rows:
        lines.append(f"{row['epoch']:>5}  {row.get('masked', float('nan')):>8.4f}  "
                     f"{row.get('visible', float('nan')):>8.4f}  "
                     f"{row['similarity']:>10.4f}  {row['channel_std']:>11.4f}")
    moved = last["similarity"] - first["similarity"]
    if last["similarity"] > 0.9:
        verdict = ("COLLAPSED -- positions in the same frame are nearly "
                   "identical vectors. This checkpoint is worse than doing "
                   "nothing; lower the mask ratio, raise the EMA decay, or run "
                   "target='frozen'.")
    elif moved > 0.15:
        verdict = ("drifting toward collapse -- position similarity rose "
                   f"{moved:+.3f} over the run. Watch it, and do not extend "
                   f"the schedule without re-reading this table.")
    else:
        verdict = "stable -- the map still distinguishes positions."
    return "\n".join(lines + ["", f"  -> {verdict}"])
