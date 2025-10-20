"""
Human-authored rewrite of the PyZaplinePlus toolkit.

The module exposes a friendly functional API via :func:`zapline_plus`
and an object-oriented interface via :class:`PyZaplinePlus`.
"""

from __future__ import annotations

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("pyzaplineplus")
except Exception:  # pragma: no cover - editable installs
    __version__ = "0.0.0"

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

