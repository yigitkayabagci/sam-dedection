from __future__ import annotations

from typing import Any

from .base import VideoTracker

_BUILDERS = {}


def register(name: str):
    def deco(cls):
        _BUILDERS[name] = cls
        cls.name = name
        return cls
    return deco


def available_trackers() -> list[str]:
    # Force import so decorators run.
    from . import edgetam_tracker  # noqa: F401
    from . import efficientsam3_tracker  # noqa: F401
    return sorted(_BUILDERS.keys())


def build_tracker(name: str, **kwargs: Any) -> VideoTracker:
    available_trackers()  # ensure registrations
    if name not in _BUILDERS:
        raise KeyError(f"Unknown tracker '{name}'. Available: {sorted(_BUILDERS)}")
    return _BUILDERS[name](**kwargs)
