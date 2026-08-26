"""Tests for optional MNE-Python integration helpers."""

import pytest

import mne_denoise._mne as mne_compat


def test_mne_availability_and_require_when_available():
    """The availability flag and requirement helper handle available MNE."""
    assert mne_compat.HAS_MNE is (mne_compat.mne is not None)
    if mne_compat.mne is None:
        pytest.skip("MNE-Python is not installed")

    assert mne_compat.require_mne("test feature") is None


def test_require_mne_when_unavailable(monkeypatch):
    """Requiring MNE reports the requested feature when unavailable."""
    monkeypatch.setattr(mne_compat, "mne", None)

    with pytest.raises(ImportError, match="SSP-SIR.*MNE-Python"):
        mne_compat.require_mne("SSP-SIR")
