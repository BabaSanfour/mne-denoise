"""Statistics-only IIR filter and streaming-edge helpers for ASR.

The light high-pass that shapes which signal drives the ASR variance
statistics (distinct from the AASR Yule-Walker filter in
:mod:`mne_denoise.asr._aasr_filter`), plus the reflected lookahead tail/carry
helpers used by clean_asr / asr_process.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def _design_statistics_filter(
    sfreq: float,
    filter_kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    if filter_kind == "none":
        return np.array([1.0]), np.array([1.0])
    if filter_kind not in ("asr", "highpass"):
        raise ValueError("filter_kind must be 'none', 'asr', or 'highpass'")
    cutoff = min(0.5, sfreq * 0.1)
    if cutoff >= sfreq / 2:
        return np.array([1.0]), np.array([1.0])
    return signal.butter(2, cutoff, btype="highpass", fs=sfreq)


def _apply_statistics_filter(
    X: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
) -> np.ndarray:
    if b.size == 1 and a.size == 1:
        return X.copy()
    padlen = 3 * max(len(a), len(b))
    if X.shape[1] <= padlen:
        return signal.lfilter(b, a, X, axis=1)
    return signal.filtfilt(b, a, X, axis=1)


def _append_clean_rawdata_tail(X: np.ndarray, n_tail: int) -> np.ndarray:
    """Append the reflected lookahead tail used by ``clean_asr.m``."""
    if n_tail == 0:
        return X.copy()
    if X.shape[1] <= n_tail:
        raise ValueError("Data must contain more samples than the lookahead tail")
    tail = 2.0 * X[:, [-1]] - X[:, -2 : -n_tail - 2 : -1]
    return np.concatenate([X, tail], axis=1)


def _prepend_clean_rawdata_carry(X: np.ndarray, n_carry: int) -> np.ndarray:
    """Prepend the reflected initial carry used by ``asr_process.m``."""
    if n_carry == 0:
        return X.copy()
    if X.shape[1] <= n_carry:
        raise ValueError("Data must contain more samples than the lookahead carry")
    carry = 2.0 * X[:, [0]] - X[:, n_carry:0:-1]
    return np.concatenate([carry, X], axis=1)


def _apply_statistics_filter_streaming(
    X: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
) -> np.ndarray:
    """Apply the statistics filter in the causal direction used by ASR."""
    if b.size == 1 and a.size == 1:
        return X.copy()
    return signal.lfilter(b, a, X, axis=1)
