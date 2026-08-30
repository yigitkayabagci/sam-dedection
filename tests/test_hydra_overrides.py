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

from src.trackers._hydra_overrides import (  # noqa: E402
    BACKBONE_STRIDE,
    WINDOW_ALIGNMENT,
    check_image_size,
    image_size_overrides,
)

# What actually binds is the **spatial perceiver**, and the constant now lives
# with the check that enforces it (`_hydra_overrides.WINDOW_ALIGNMENT`) rather
# than being restated here. It partitions the stride-16 feature map into exactly
# 16x16 windows with a partition that does not pad, so the feature side has to
# be a multiple of 16 -- and the feature side is size/16. Hence 256, not 16.
#
# This comment used to blame the FPN's 2x top-down upsample and the constant
# was 16, which let sizes through that crash at the first forward pass:
#
#     640  RuntimeError: size of tensor a (256) must match tensor b (400)
#     896  RuntimeError: shape '[1,18,3,18,3,64]' is invalid for input of size 200704
#
# 640/16 = 40 and 896/16 = 56, neither divisible by 16. 512, 768, 1024 and 1280
# all are, which is why nothing had caught it.


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
    assert size % WINDOW_ALIGNMENT == 0
    side = size // BACKBONE_STRIDE
    # `_SelfAttentionGraph` rejects a token count that is not a perfect square.
    assert int(round((side * side) ** 0.5)) ** 2 == side * side
    # The SAM head's two high-res levels are strides 8 and 4 of the input.
    assert (size // 8, size // 4) == (side * 2, side * 4)


@pytest.mark.parametrize("size", [256, 512, 768, 1024, 1280])
def test_the_check_admits_every_size_this_repo_builds(size):
    assert check_image_size(size) is None


@pytest.mark.parametrize("size", [640, 896, 720, 1000, 0, -512])
def test_the_check_refuses_the_sizes_that_crash(size):
    """640 and 896 are the two that were measured crashing; the check has to
    name a reason rather than merely returning False, because the reason is
    what tells a caller whether to pick 768 or 1024."""
    refusal = check_image_size(size)
    assert refusal and str(size) in refusal


def test_an_inadmissible_size_is_refused_before_anything_is_built():
    """`image_size_overrides` is the one door every backend goes through --
    the tracker, the TRT tracker and the ONNX exporter all call it -- so
    refusing here is refusing everywhere, at config-read time rather than in
    the first forward pass."""
    with pytest.raises(ValueError, match="640"):
        image_size_overrides(640)


def test_shipped_configs_declare_a_usable_size():
    """Every `image_size:` in configs/ satisfies the same constraint."""
    declared = dict(_configs())
    assert declared, "no config declares image_size -- the glob is wrong"
    bad = {name: check_image_size(size) for name, size in declared.items()
           if check_image_size(size)}
    assert not bad, f"configs declare a size the architecture cannot build: {bad}"


def test_every_config_key_is_a_knob_some_tracker_takes():
    """A key nothing reads is a setting that silently does nothing.

    `build_tracker(name, **cfg)` hands the whole YAML to a constructor, so a
    misspelled or stale key is a `TypeError` at best and, for a key that merely
    never existed, a config that looks configured and is not. Checked here
    rather than at run time because it needs no GPU, no engines and no
    checkpoint -- only the two signatures.
    """
    import inspect

    from src.trackers.edgetam_tracker import EdgeTAMTracker
    from src.trackers.edgetam_trt_tracker import EdgeTAMTRTTracker

    plain = set(inspect.signature(EdgeTAMTracker.__init__).parameters)
    accelerated = set(inspect.signature(EdgeTAMTRTTracker.__init__).parameters)
    unknown = {}
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        body = yaml.safe_load(path.read_text()) or {}
        if "model_cfg" not in body:
            continue          # a Hydra model config, not a tracker config
        takes = accelerated if "trt" in path.name else plain
        extra = sorted(key for key in body if key not in takes)
        if extra:
            unknown[path.name] = extra
    assert not unknown, f"config keys no tracker accepts: {unknown}"


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
