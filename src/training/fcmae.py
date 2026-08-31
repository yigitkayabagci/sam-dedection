"""FCMAE: masked-autoencoder pretraining for a convolutional trunk.

**Why this exists, and what it overturns.** `distill.py` and notebook 07 both
record the same refusal: "MAE masks *tokens* and is built for a ViT, while
EdgeTAM's trunk is a convolutional RepViT with no token grid to mask." That was
right about token masking and wrong as a general claim, and ConvNeXt V2 (Woo et
al., arXiv 2301.00808) is the paper that says why: a convnet can be masked in
*pixels* rather than tokens, provided the masked positions never leak into the
visible ones -- and provided the block is changed so the features do not
collapse under the sparse signal.

**The paper's own numbers are the reason to read this carefully before
spending GPU hours on it.** Two ablations decide the design:

    dense (naive) masking   79.3        sparse masking          83.7
    V1 + supervised  83.8   V1 + FCMAE  83.7   V2 + FCMAE  84.6

The second line is the one that matters here: **FCMAE alone does not beat
supervised training.** The gain comes from the co-design -- the masked
autoencoder *and* the Global Response Normalization layer together. A run that
takes the masking and skips GRN is, by the authors' own table, expected to buy
nothing. So `insert_grn` is not an optional extra in this module; it is half
the method, and the arm without it exists to reproduce that finding rather than
to be shipped.

**What is faithful here and what is an approximation.** The paper's encoder
uses submanifold sparse convolution so a masked position contributes nothing at
any depth. It also names the alternative this module implements: "simulating
sparse encoding with the masked dense convolution, which can be easily
implemented by applying binary masks before and after the standard convolution
operation". That is what `masked_convolutions` does, for every convolution in
the trunk, by hook. It costs the same FLOPs as a dense forward -- the saving
sparse convolution buys is speed, not correctness -- and it is the difference
between the 79.3 row and the 83.7 row only if the mask is genuinely re-applied
at every layer, which is why the hook goes on every `Conv2d` rather than on
stage boundaries.

**What comes out.** An ordinary EdgeTAM state dict -- same keys, same loader --
with a differently-trained trunk and neck inside it, exactly the contract
`distill.py` established, so `34`/`32` can take it as `BASE_CHECKPOINT`. The
decoder and the mask token are scaffolding and are thrown away. GRN is the one
exception: it adds parameters to the block, so a checkpoint pretrained with it
carries them and the ONNX export has to be regenerated.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# The paper's masking, and both numbers are load-bearing: 32-pixel patches at a
# 60 % ratio. A finer patch makes the task easy -- a convnet inpaints a 4-pixel
# hole from its neighbours without learning anything about the scene -- and a
# lower ratio does the same.
PATCH = 32
MASK_RATIO = 0.6


@dataclass(frozen=True)
class FCMAEConfig:
    """Everything that changes what the pretraining measures.

    Defaults are the paper's, except `image_size`, which is this project's
    (512, the size every other stage trains at).
    """

    image_size: int = 512
    patch: int = PATCH
    mask_ratio: float = MASK_RATIO
    # The decoder is deliberately tiny: the paper's ablation puts a single
    # ConvNeXt block at 83.7 % against 83.5-83.7 % for a UNet decoder, at 7.7
    # hours of pretraining against 12.9. A heavier decoder does the
    # reconstruction the encoder was supposed to learn to support.
    decoder_dim: int = 512
    decoder_depth: int = 1
    # Patch-wise normalisation of the reconstruction target, as in MAE. On
    # thermal footage this is the setting to think about rather than inherit:
    # a flat patch has a standard deviation near zero, and dividing by it turns
    # sensor noise into a target. `norm_floor` is what stops that.
    norm_pixels: bool = True
    norm_floor: float = 1e-3


# --------------------------------------------------------------------------
# Masking
# --------------------------------------------------------------------------


def patch_mask(batch: int, size: int, config: FCMAEConfig,
               generator: torch.Generator | None = None,
               device=None) -> torch.Tensor:
    """`[B, h, w]` boolean mask at patch resolution. True means **removed**.

    Exactly `round(ratio * h * w)` patches per image rather than an
    independent coin per patch: a binomial draw gives some images in the batch
    far more signal than others, and the loss is then dominated by whichever
    images happened to keep the most.
    """
    grid = size // config.patch
    total = grid * grid
    keep = total - int(round(config.mask_ratio * total))
    noise = torch.rand(batch, total, generator=generator, device=device)
    # The `keep` smallest go visible; argsort twice turns the ranking into a
    # per-position rank, which is MAE's own trick for a fixed-count mask.
    rank = noise.argsort(dim=1).argsort(dim=1)
    return (rank >= keep).reshape(batch, grid, grid)


def expand_mask(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """The patch mask at a feature map's resolution, as a `[B, 1, H, W]` float.

    Nearest-neighbour on purpose. A bilinear resize would put fractional values
    on the boundary between a kept and a removed patch, and a position that is
    30 % masked is a position that leaks.
    """
    return F.interpolate(mask.float()[:, None], size=(height, width),
                         mode="nearest")


class masked_convolutions:
    """Re-apply the mask to the output of every convolution, for a forward.

    The paper's stated alternative to submanifold sparse convolution: "applying
    binary masks before and after the standard convolution operation". Done by
    hook rather than by editing the trunk, for the same reason the rest of this
    project patches rather than forks -- the trunk is upstream code, the
    masking is only true during pretraining, and a fine-tuning forward has to
    be the unmodified one.

    Every `Conv2d`, not every stage. A mask applied only at stage boundaries
    lets a 7x7 depthwise kernel pull masked values into visible positions
    inside the stage, and by the next boundary the leak is already in the
    features. That leak is what separates the paper's 79.3 % row from its
    83.7 % one.

    Used as a context manager so the hooks cannot outlive the forward:

        with masked_convolutions(trunk, mask):
            features = trunk(images)
    """

    def __init__(self, module: nn.Module, mask: torch.Tensor) -> None:
        self.module = module
        self.mask = mask
        self.handles: list = []
        self.touched = 0

    def _clear(self, tensor: torch.Tensor) -> torch.Tensor:
        keep = 1.0 - expand_mask(self.mask, *tensor.shape[-2:])
        return tensor * keep.to(dtype=tensor.dtype, device=tensor.device)

    def _before(self, _module, inputs):
        if not inputs or not torch.is_tensor(inputs[0]) or inputs[0].dim() != 4:
            return None
        return (self._clear(inputs[0]), *inputs[1:])

    def _after(self, _module, _inputs, output):
        if not torch.is_tensor(output) or output.dim() != 4:
            return output
        self.touched += 1
        return self._clear(output)

    def __enter__(self) -> "masked_convolutions":
        for child in self.module.modules():
            if isinstance(child, nn.Conv2d):
                # Both sides, and the "before" is not decoration. Masking only
                # the output zeroes the positions whose centre sits under the
                # mask -- but a kernel that straddles a patch boundary has
                # already pulled real pixels from the masked side into a
                # *visible* position, and that value survives. Zeroing the
                # input first is what makes the output invariant to whatever is
                # under the mask, which is the property the whole method rests
                # on. `tests/test_fcmae.py` fails without this line.
                self.handles.append(child.register_forward_pre_hook(self._before))
                self.handles.append(child.register_forward_hook(self._after))
        if not self.handles:
            raise ValueError(
                "no Conv2d found to mask. FCMAE's encoder-side masking is "
                "defined on convolutions; a trunk with none of them needs the "
                "token masking this method exists to replace.")
        return self

    def __exit__(self, *exc) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


# --------------------------------------------------------------------------
# Global Response Normalization -- the other half of the co-design
# --------------------------------------------------------------------------


class GRN(nn.Module):
    """Global Response Normalization, on `[B, C, H, W]`.

    Three steps, as the paper states them: aggregate each channel to its L2
    norm over space, normalise those norms against each other so channels
    compete, and scale each channel by its own share.

        G(X)_i = ||X_i||          N(G)_i = G_i / mean_j(G_j)
        Y_i    = gamma_i * (X_i * N(G)_i) + beta_i + X_i

    The residual and the two learnable vectors are the paper's; they start at
    zero, so an untrained GRN is the identity and inserting it cannot change a
    checkpoint's behaviour until it has been trained.

    **What it is for**, and the reason it is not optional: under a 60 % mask
    the surviving activations are few and a convnet's channels drift towards
    redundancy -- many channels responding to the same thing. Dividing by the
    mean response makes a channel's gain depend on how loud the *others* are,
    which is a competition, and competition is what stops the collapse. The
    paper measures the difference: FCMAE without it reaches 83.7 % against
    supervised training's 83.8 %, and with it 84.6 %.
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norms = torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True)
        share = norms / (norms.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * share) + self.beta + x


class GRNAfter(nn.Sequential):
    """`module` with a GRN behind it, so a swap is one assignment.

    A wrapper rather than an edit to the block's `forward`: the trunk is
    upstream code, and the checkpoint has to stay loadable by the same loader
    with the same keys plus these two vectors.
    """

    def __init__(self, module: nn.Module, channels: int) -> None:
        super().__init__(module, GRN(channels))


def _expansion_width(module: nn.Module) -> int:
    """The output width of an expansion layer, or 0 if this is not one."""
    if isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1):
        return module.out_channels if module.out_channels > module.in_channels else 0
    if isinstance(module, nn.Linear):
        return module.out_features if module.out_features > module.in_features else 0
    return 0


def insert_grn(model: nn.Module, ratio: float = 2.0) -> dict:
    """Put a GRN after every dimension-expanding MLP layer. Returns a report.

    The paper places GRN "after the dimension-expansion MLP layer" and drops
    LayerScale, which becomes unnecessary. Found by shape rather than by name:
    an expansion is a 1x1 convolution or a linear layer whose output is at
    least `ratio` times its input, which is what an inverted-bottleneck FFN
    looks like in every convnet this could be run on, RepViT included.

    Returns what it changed rather than printing it, so a notebook can assert
    on the count -- a discovery pass that silently finds nothing would be a
    pretraining run with half the method missing and no sign of it.
    """
    found: dict[str, int] = {}
    for name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            width = _expansion_width(child)
            if width and width >= ratio * (
                    child.in_channels if isinstance(child, nn.Conv2d)
                    else child.in_features):
                setattr(parent, child_name, GRNAfter(child, width))
                found[f"{name}.{child_name}".lstrip(".")] = width
    return {"inserted": len(found), "channels": found,
            "parameters": sum(2 * width for width in found.values())}


# --------------------------------------------------------------------------
# The decoder, the target, and the loss
# --------------------------------------------------------------------------


class ConvNeXtBlock(nn.Module):
    """One ConvNeXt block with GRN, which is the ConvNeXt V2 block.

    7x7 depthwise, LayerNorm, a 4x expansion, GELU, GRN, projection back, and
    a residual. LayerScale is absent for the reason the paper gives: with GRN
    it is unnecessary.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.expand = nn.Conv2d(dim, 4 * dim, 1)
        self.grn = GRN(4 * dim)
        self.project = nn.Conv2d(4 * dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.project(self.grn(F.gelu(self.expand(self.norm(
            self.depthwise(x))))))
        return x + out


class Decoder(nn.Module):
    """Encoder features -> the pixels of every patch.

    Thrown away when pretraining ends, and deliberately small: one block at
    `decoder_dim`, then a 1x1 convolution that emits a whole patch per
    position. Its output grid is the patch grid, so the encoder's stride-16
    map is pooled to stride-`patch` first -- the reconstruction is defined per
    patch, and predicting at a finer grid would let the decoder do work the
    loss never asks for.

    `mask_token` is the value the masked positions carry into the decoder. The
    encoder never saw them, so leaving its zeros there would make "masked" and
    "black" the same input; a learned token is what tells the decoder which
    positions it is being asked to invent.
    """

    def __init__(self, channels: int, config: FCMAEConfig) -> None:
        super().__init__()
        self.config = config
        self.mask_token = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.project = nn.Conv2d(channels, config.decoder_dim, 1)
        self.blocks = nn.Sequential(*[ConvNeXtBlock(config.decoder_dim)
                                      for _ in range(config.decoder_depth)])
        self.head = nn.Conv2d(config.decoder_dim, config.patch ** 2 * 3, 1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        grid = mask.shape[-1]
        if features.shape[-1] != grid:
            features = F.adaptive_avg_pool2d(features, (grid, grid))
        removed = expand_mask(mask, grid, grid).to(features.dtype)
        features = features * (1.0 - removed) + self.mask_token * removed
        out = self.head(self.blocks(self.project(features)))
        return out.reshape(out.shape[0], 3, self.config.patch,
                           self.config.patch, grid, grid)


def patchify(images: torch.Tensor, patch: int) -> torch.Tensor:
    """`[B, 3, S, S]` -> `[B, 3, p, p, h, w]`, the shape the decoder emits."""
    batch, channels, height, width = images.shape
    out = images.reshape(batch, channels, height // patch, patch,
                         width // patch, patch)
    return out.permute(0, 1, 3, 5, 2, 4)


def normalise_patches(patches: torch.Tensor, floor: float) -> torch.Tensor:
    """Each patch to zero mean and unit variance, with a floor on the spread.

    MAE's `norm_pix_loss`, and the floor is this project's addition rather
    than the paper's. On thermal footage a patch of empty ground has a
    standard deviation near zero; dividing by it rescales sensor noise into a
    unit-variance target and the reconstruction loss then spends its capacity
    predicting noise. The floor is in the units of the normalised input, so it
    is the same number whatever the frame's own range was.
    """
    dims = (1, 2, 3)
    mean = patches.mean(dim=dims, keepdim=True)
    spread = patches.std(dim=dims, keepdim=True)
    return (patches - mean) / spread.clamp(min=floor)


def fcmae_loss(prediction: torch.Tensor, images: torch.Tensor,
               mask: torch.Tensor, config: FCMAEConfig) -> tuple:
    """Mean squared error on the **masked patches only**.

    Scoring the visible patches too would reward copying, which a convolution
    can do perfectly and which teaches nothing. Returns `(loss, terms)` in the
    shape the schedule logs.
    """
    target = patchify(images, config.patch)
    if config.norm_pixels:
        target = normalise_patches(target, config.norm_floor)
    error = (prediction - target).pow(2).mean(dim=(1, 2, 3))
    removed = mask.to(error.dtype)
    total = removed.sum().clamp(min=1.0)
    loss = (error * removed).sum() / total
    return loss, {"mse": float(loss.detach()),
                  "masked": float(removed.mean().detach())}
