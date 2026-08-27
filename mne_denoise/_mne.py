"""Internal helpers for optional MNE-Python integration."""

from __future__ import annotations

try:
    import mne
except ImportError:  # pragma: no cover - exercised in no-MNE environments
    mne = None

HAS_MNE = mne is not None


def require_mne(feature: str) -> None:
    """Require MNE-Python for an MNE-specific feature."""
    if mne is None:
        raise ImportError(
            f"{feature} requires MNE-Python. "
            "Install MNE-Python to use this functionality."
        )
