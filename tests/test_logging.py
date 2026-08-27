"""Tests for the verbose-to-logging wiring (mne_denoise._logging)."""

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
    with caplog.at_level(logging.INFO, logger="mne_denoise"):
        ASR(sfreq=SFREQ, cutoff=20.0, picks=None, verbose=True).fit_transform(_data())
    assert any("ASR calibrated" in rec.message for rec in caplog.records)


def test_verbose_false_is_quiet(caplog):
    """verbose=False raises the asr logger to WARNING, so no INFO is emitted."""
    with caplog.at_level(logging.INFO, logger="mne_denoise"):
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
    with caplog.at_level(logging.INFO, logger="mne_denoise"):
        est = estimator_cls(sfreq=SFREQ, cutoff=20.0, picks=None, verbose=True)
        cleaned = est.fit_transform(_data())
    assert np.asarray(cleaned).shape == (6, 2000)


def test_verbose_none():
    from mne_denoise._logging import logger, set_log_level_from_verbose

    prev = logger.level
    set_log_level_from_verbose(None)
    assert logger.level == prev


def test_verbose_str():
    from mne_denoise._logging import logger, set_log_level_from_verbose

    previous = logger.level
    try:
        set_log_level_from_verbose("debug")
        assert logger.level == logging.DEBUG
    finally:
        logger.setLevel(previous)


def test_verbose_int():
    from mne_denoise._logging import logger, set_log_level_from_verbose

    previous = logger.level
    try:
        set_log_level_from_verbose(logging.ERROR)
        assert logger.level == logging.ERROR
    finally:
        logger.setLevel(previous)


def test_verbose_decorator_restores_on_success_and_error():
    """A scoped per-call override must restore the prior level always."""
    from mne_denoise._logging import logger, verbose

    @verbose
    def operation(*, verbose=None, fail=False):
        if fail:
            raise RuntimeError("expected")

    previous = logger.level
    try:
        operation(verbose="DEBUG")
        assert logger.level == previous
        with pytest.raises(RuntimeError, match="expected"):
            operation(verbose=True, fail=True)
        assert logger.level == previous
    finally:
        logger.setLevel(previous)
