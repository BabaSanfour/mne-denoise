"""Tests for the ASR verbose-to-logging wiring (mne_denoise.asr._logging)."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mne_denoise.asr import ASR, AdaptiveASR, JugglerASR

SFREQ = 200.0


def _data(seed: int = 0) -> np.ndarray:
    """Small synthetic multichannel array for fit_transform."""
    return np.random.default_rng(seed).standard_normal((6, 2000))


def test_verbose_true_emits_info(caplog):
    """verbose=True surfaces an INFO calibration message on the asr logger."""
    with caplog.at_level(logging.INFO, logger="mne_denoise.asr"):
        ASR(sfreq=SFREQ, cutoff=20.0, picks=None, verbose=True).fit_transform(_data())
    assert any("ASR calibrated" in rec.message for rec in caplog.records)


def test_verbose_false_is_quiet(caplog):
    """verbose=False raises the asr logger to WARNING, so no INFO is emitted."""
    with caplog.at_level(logging.INFO, logger="mne_denoise.asr"):
        ASR(sfreq=SFREQ, cutoff=20.0, picks=None, verbose=False).fit_transform(_data())
    infos = [
        rec
        for rec in caplog.records
        if rec.levelno >= logging.INFO and "ASR calibrated" in rec.message
    ]
    assert not infos


@pytest.mark.parametrize("estimator_cls", [AdaptiveASR, JugglerASR])
def test_variants_honor_verbose_without_error(estimator_cls, caplog):
    """AdaptiveASR / JugglerASR accept verbose and run cleanly under it."""
    with caplog.at_level(logging.INFO, logger="mne_denoise.asr"):
        est = estimator_cls(sfreq=SFREQ, cutoff=20.0, picks=None, verbose=True)
        cleaned = est.fit_transform(_data())
    assert np.asarray(cleaned).shape == (6, 2000)
