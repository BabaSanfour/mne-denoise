"""Parameter validation and dimension-resolution helpers for ASR.

Low-level, dependency-free checks and resolvers shared across the ASR
calibration, processing, and estimator layers, kept here so higher-level
modules can import them without pulling in the full pipeline.
"""

from __future__ import annotations

import numpy as np


def _validate_common_params(
    *,
    sfreq: float,
    cutoff: float,
    window_length: float,
    window_overlap: float,
    max_dropout_fraction: float,
    min_clean_fraction: float,
    regularization: float,
) -> None:
    if sfreq <= 0:
        raise ValueError("sfreq must be positive")
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    if window_length <= 0:
        raise ValueError("window_length must be positive")
    if not (0 <= window_overlap < 1):
        raise ValueError("window_overlap must be in [0, 1)")
    if not (0 <= max_dropout_fraction < 1):
        raise ValueError("max_dropout_fraction must be in [0, 1)")
    if not (0 < min_clean_fraction <= 1):
        raise ValueError("min_clean_fraction must be in (0, 1]")
    if max_dropout_fraction + min_clean_fraction >= 1:
        raise ValueError(
            "max_dropout_fraction + min_clean_fraction must be less than 1"
        )
    if regularization <= 0:
        raise ValueError("regularization must be positive")


def _validate_array_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"ASR expects a 2D array (n_channels, n_times), got {X.shape}")
    if X.shape[0] < 2:
        raise ValueError("ASR requires at least two channels")
    finite_fraction = np.isfinite(X).mean(axis=1)
    if np.any(finite_fraction < 0.99):
        bad = np.where(finite_fraction < 0.99)[0].tolist()
        raise ValueError(f"Channels contain too many non-finite samples: {bad}")
    X = np.nan_to_num(X, copy=True)
    variances = np.var(X, axis=1)
    max_var = float(np.max(variances))
    # Relative floor so legitimately small-amplitude data (e.g. MEG in Tesla,
    # variance ~1e-26) is not rejected, while genuinely flat/dead channels
    # (variance ~0 relative to the rest) still are.
    if max_var <= 0.0:
        raise ValueError("All channels have zero or near-zero variance")
    bad = np.where(variances <= max_var * 1e-12)[0].tolist()
    if bad:
        raise ValueError(f"Channels with zero or near-zero variance: {bad}")
    return X


def _check_enough_samples(n_times: int, sfreq: float, window_length: float) -> None:
    n_win = int(round(window_length * sfreq))
    if n_win < 2:
        raise ValueError("window_length is too short for the sampling frequency")
    if n_times < n_win:
        raise ValueError(
            f"Window length ({n_win} samples) exceeds data length ({n_times} samples)"
        )


def _round_half_up(value: float) -> int:
    """Round non-negative values like MATLAB ``round``."""
    return int(np.floor(float(value) + 0.5))


def _resolve_max_dims_clean_rawdata(max_dims: float | int, n_channels: int) -> int:
    """Resolve ASR ``maxdims`` using clean_rawdata's convention."""
    if isinstance(max_dims, float) and max_dims < 1:
        if max_dims < 0:
            raise ValueError("max_dims must be non-negative")
        return _round_half_up(n_channels * max_dims)
    max_dims_int = int(max_dims)
    if max_dims_int < 0:
        raise ValueError("max_dims must be non-negative")
    return min(max_dims_int, n_channels)


def _resolve_max_dims(max_dims: float | int, n_channels: int) -> int:
    if isinstance(max_dims, float):
        if not (0 <= max_dims <= 1):
            raise ValueError("float max_dims must be in [0, 1]")
        return int(np.floor(max_dims * n_channels))
    max_dims = int(max_dims)
    if not (0 <= max_dims <= n_channels):
        raise ValueError("integer max_dims must be in [0, n_channels]")
    return max_dims
