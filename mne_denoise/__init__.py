"""Narrow-band artefact suppression utilities integrated with MNE-Python."""

from __future__ import annotations

__version__ = "0.0.1"

from .core import PyZaplinePlus, zapline_plus
from .noise_detection import find_next_noisefreq

__all__ = [
    "PyZaplinePlus",
    "zapline_plus",
    "find_next_noisefreq",
    "__version__",
]

# Optional MNE helper
try:  # pragma: no cover - optional dependency
    from ._mne import apply_zapline_to_raw  # type: ignore[F401]

    __all__.append("apply_zapline_to_raw")
except Exception:  # pragma: no cover - ignore missing dependency
    pass
