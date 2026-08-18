"""The resolution knob: what makes a size valid, and that the shipped ones are.

Running EdgeTAM at a size other than the checkpoint's 1024 is one config field,
`image_size`, and every derived shape in the export path, the graph wrappers
and the TensorRT tracker is computed from it. That only holds while the size
divides cleanly through the architecture, so the arithmetic is pinned here --
it needs neither torch nor EdgeTAM, which is what makes it cheap enough to
guard every config in the repo rather than only the one being worked on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trackers._hydra_overrides import BACKBONE_STRIDE, image_size_overrides  # noqa: E402

# The deepest stride-2 chain the model puts an input through: the trunk down to
# the stride-16 feature map, and the memory encoder's mask downsampler doing
# the same to the mask. A size not divisible by this hits an odd intermediate,
# and the FPN's top-down 2x upsample then lands one pixel off its skip.
DEEPEST_DOWNSAMPLE = 16


def _configs():
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        body = yaml.safe_load(path.read_text()) or {}
        if "image_size" in body:
            yield path.name, int(body["image_size"])


def test_no_override_when_size_is_unset():
    """No `image_size` in the YAML means the Hydra config's own 1024 stands."""
    assert image_size_overrides(None) == []
    assert image_size_overrides(0) == []


@pytest.mark.parametrize("size,side", [(256, 16), (512, 32), (768, 48), (1024, 64)])
def test_q_sizes_tracks_the_feature_map(size, side):
    """The query RoPE table is sized from image_size, never written by hand.

    `RoPEAttentionv2` precomputes its rotary table once at construction and
    never revisits it, so a stale `q_sizes` is either a shape error at export
    (`_CrossAttentionGraph` compares the two and raises) or, without that
    guard, a silently wrong rotation.
    """
    assert image_size_overrides(size) == [
        f"++model.image_size={size}",
        f"model.memory_attention.layer.cross_attention.q_sizes=[{side},{side}]",
    ]
    assert side == size // BACKBONE_STRIDE


@pytest.mark.parametrize("size", [256, 512, 768, 1024])
def test_size_divides_through_the_architecture(size):
    """Every shape the export derives from `image_size` comes out an integer.

    768 = 3 * 2^8 passes for the same reason 512 = 2^9 and 1024 = 2^10 do; a
    size like 720 does not, and this is where that would be caught.
    """
    assert size % DEEPEST_DOWNSAMPLE == 0
    side = size // BACKBONE_STRIDE
    # `_SelfAttentionGraph` rejects a token count that is not a perfect square.
    assert int(round((side * side) ** 0.5)) ** 2 == side * side
    # The SAM head's two high-res levels are strides 8 and 4 of the input.
    assert (size // 8, size // 4) == (side * 2, side * 4)


def test_shipped_configs_declare_a_usable_size():
    """Every `image_size:` in configs/ satisfies the same constraint."""
    declared = dict(_configs())
    assert declared, "no config declares image_size -- the glob is wrong"
    bad = {name: size for name, size in declared.items()
           if size % DEEPEST_DOWNSAMPLE or size <= 0}
    assert not bad, f"image_size not divisible by {DEEPEST_DOWNSAMPLE}: {bad}"


def test_engine_paths_match_the_resolution_they_declare():
    """A TRT config's engines are shape-specific: models512/ must not serve 768.

    Mixing them is the one failure this whole file exists to prevent, and it
    would otherwise surface as a TensorRT profile error deep in a run.
    """
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        body = yaml.safe_load(path.read_text()) or {}
        engine = body.get("image_encoder_engine")
        if not engine or "image_size" not in body:
            continue
        size = int(body["image_size"])
        directory = Path(engine).parent.name
        assert str(size) in directory, (
            f"{path.name} runs at {size} but takes engines from {directory}/"
        )
