"""Narrow-band artefact suppression utilities integrated with MNE-Python."""

from __future__ import annotations

__version__ = "0.0.1"

# Future-facing scaffolding
from .dss import DSS, DSSConfig
from .zapline import ZapLineConfig, ZapLineSummary, zapline_plus
from .utils import chunk_signal, compute_psd, identify_peaks

# Legacy implementation retained for reference and backwards compatibility
from .core import PyZaplinePlus  # noqa: E402
from .noise_detection import find_next_noisefreq  # noqa: E402

__all__ = [
    "__version__",
    "DSS",
    "DSSConfig",
    "ZapLineConfig",
    "ZapLineSummary",
    "zapline_plus",
    "chunk_signal",
    "compute_psd",
    "identify_peaks",
    "PyZaplinePlus",
    "find_next_noisefreq",
]

try:  # pragma: no cover - optional dependency
    from ._mne import apply_zapline_to_raw  # type: ignore[F401]
except Exception:  # pragma: no cover - ignore missing dependency
    apply_zapline_to_raw = None  # type: ignore[assignment]
else:
    __all__.append("apply_zapline_to_raw")
