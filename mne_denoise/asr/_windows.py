"""Windowing and clean-window selection for ASR calibration.

Window start/weight/RMS helpers, the clean_rawdata-style clean-window selection
and its grid diagnostics, plus the sample-mask / span utilities that translate
retained windows into per-sample masks and annotation spans.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._distribution import fit_eeg_distribution
from ._validation import _round_half_up


def _window_starts(n_times: int, win_len: int, overlap: float) -> np.ndarray:
    if win_len < 2:
        raise ValueError("window length must be at least 2 samples")
    if win_len > n_times:
        raise ValueError(
            f"Window length ({win_len} samples) exceeds data length ({n_times} samples)"
        )
    step = max(1, int(round(win_len * (1 - overlap))))
    starts = list(range(0, n_times - win_len + 1, step))
    last = n_times - win_len
    if starts[-1] != last:
        starts.append(last)
    return np.asarray(starts, dtype=int)


def _clean_rawdata_window_starts(
    n_times: int,
    win_len: int,
    overlap: float,
) -> np.ndarray:
    """Return MATLAB clean_rawdata-style rounded window starts."""
    if win_len < 2:
        raise ValueError("window length must be at least 2 samples")
    if win_len > n_times:
        raise ValueError(
            f"Window length ({win_len} samples) exceeds data length ({n_times} samples)"
        )
    step = win_len * (1.0 - overlap)
    starts_1_based = np.arange(1.0, n_times - win_len + np.finfo(float).eps, step)
    starts = np.asarray(
        [_round_half_up(start) - 1 for start in starts_1_based], dtype=int
    )
    return np.unique(starts)


def _window_weights(win_len: int) -> np.ndarray:
    if win_len <= 2:
        return np.ones(win_len, dtype=np.float64)
    return np.hanning(win_len + 2)[1:-1].astype(np.float64)


def _window_rms(X: np.ndarray, starts: np.ndarray, win_len: int) -> np.ndarray:
    """Compute per-channel RMS values over a set of windows."""
    windows = starts[:, np.newaxis] + np.arange(win_len, dtype=int)[np.newaxis, :]
    squared = X[:, windows] ** 2
    return np.sqrt(np.sum(squared, axis=2) / win_len)


def _resolve_max_bad_channels_count(
    max_bad_channels: float | int,
    n_channels: int,
) -> int:
    """Resolve clean_windows-style tolerated bad-channel count."""
    if isinstance(max_bad_channels, float) and 0 < max_bad_channels < 1:
        resolved = _round_half_up(n_channels * max_bad_channels)
    else:
        resolved = int(max_bad_channels)
    if resolved < 0:
        raise ValueError("max_bad_channels must be non-negative")
    return min(resolved, n_channels)


def _select_clean_windows(
    X: np.ndarray,
    starts: np.ndarray,
    win_len: int,
    *,
    ref_max_bad_channels: float,
    ref_tolerances: tuple[float, float],
    max_dropout_fraction: float,
    min_clean_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    diagnostics = _clean_windows_grid_diagnostics(
        X,
        starts,
        win_len,
        max_bad_channels=ref_max_bad_channels,
        zthresholds=ref_tolerances,
        max_dropout_fraction=max_dropout_fraction,
        min_clean_fraction=min_clean_fraction,
    )
    clean = diagnostics["window_keep_mask"].copy()
    if not np.any(clean):
        zscores = diagnostics["window_rms_zscores"]
        penalty = np.mean(np.maximum(zscores - max(ref_tolerances), 0.0), axis=1)
        clean[np.argmin(penalty)] = True
    return clean, diagnostics["window_rms_zscores"]


def _clean_windows_grid_diagnostics(
    X: np.ndarray,
    starts: np.ndarray,
    win_len: int,
    *,
    max_bad_channels: float | int,
    zthresholds: tuple[float, float],
    max_dropout_fraction: float,
    min_clean_fraction: float,
    fit_quantiles: tuple[float, float] = (0.022, 0.6),
    beta_grid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute clean_windows-style diagnostics on an explicit window grid."""
    n_channels = X.shape[0]
    rms = _window_rms(X, starts, win_len)

    zscores = np.empty_like(rms)
    mu_values = np.empty(n_channels, dtype=np.float64)
    sigma_values = np.empty(n_channels, dtype=np.float64)
    beta_values = np.empty(n_channels, dtype=np.float64)
    fit_errors = np.empty(n_channels, dtype=np.float64)
    fit_intervals = np.empty((n_channels, 2), dtype=np.float64)
    fit_sample_counts = np.empty(n_channels, dtype=int)
    for ch_idx in range(n_channels):
        mu, sigma, info = fit_eeg_distribution(
            rms[ch_idx],
            min_clean_fraction=min_clean_fraction,
            max_dropout_fraction=max_dropout_fraction,
            fit_quantiles=fit_quantiles,
            beta_grid=beta_grid,
            return_info=True,
        )
        mu_values[ch_idx] = mu
        sigma_values[ch_idx] = sigma
        beta_values[ch_idx] = info["beta"]
        fit_errors[ch_idx] = info["fit_error"]
        fit_intervals[ch_idx] = info["fit_interval"]
        fit_sample_counts[ch_idx] = info["n_fit_samples"]
        zscores[ch_idx] = (rms[ch_idx] - mu) / max(sigma, np.finfo(float).eps)

    tolerated_bad_channels = _resolve_max_bad_channels_count(
        max_bad_channels, n_channels
    )
    window_remove_mask = np.zeros(starts.size, dtype=bool)
    if tolerated_bad_channels < n_channels:
        swz = np.sort(zscores, axis=0)
        z_low, z_high = zthresholds
        if z_high > 0:
            window_remove_mask |= swz[-1 - tolerated_bad_channels] > z_high
        if z_low < 0:
            window_remove_mask |= swz[tolerated_bad_channels] < z_low
    window_keep_mask = ~window_remove_mask

    return {
        "window_starts": starts,
        "window_stops": starts + win_len,
        "window_rms": rms.T,
        "window_rms_zscores": zscores.T,
        "window_keep_mask": window_keep_mask,
        "window_remove_mask": window_remove_mask,
        "mu": mu_values,
        "sigma": sigma_values,
        "beta": beta_values,
        "fit_error": fit_errors,
        "fit_interval": fit_intervals,
        "n_fit_samples": fit_sample_counts,
        "n_windows": int(starts.size),
        "n_rejected_windows": int(np.sum(window_remove_mask)),
    }


def _concatenate_windows(
    X: np.ndarray,
    starts: np.ndarray,
    win_len: int,
) -> np.ndarray:
    out = np.empty((X.shape[0], len(starts) * win_len), dtype=np.float64)
    for idx, start in enumerate(starts):
        out[:, idx * win_len : (idx + 1) * win_len] = X[:, start : start + win_len]
    return out


def _sample_mask_from_removed_windows(
    n_times: int,
    starts: np.ndarray,
    win_len: int,
    window_remove_mask: np.ndarray,
) -> np.ndarray:
    """Mirror MATLAB clean_windows sample retention semantics."""
    sample_mask = np.ones(n_times, dtype=bool)
    for start in starts[np.asarray(window_remove_mask, dtype=bool)]:
        sample_mask[start : start + win_len] = False
    return sample_mask


def _good_raw_sample_mask(raw: Any, prefixes: tuple[str, ...]) -> np.ndarray:
    n_times = raw.n_times
    mask = np.ones(n_times, dtype=bool)
    annotations = getattr(raw, "annotations", None)
    if annotations is None or len(annotations) == 0:
        return mask
    sfreq = raw.info["sfreq"]
    first_time = getattr(raw, "_first_time", 0.0)
    for onset, duration, description in zip(
        annotations.onset,
        annotations.duration,
        annotations.description,
    ):
        desc = str(description).lower()
        if not any(desc.startswith(prefix.lower()) for prefix in prefixes):
            continue
        start = max(0, int(np.floor((onset - first_time) * sfreq)))
        stop = min(n_times, int(np.ceil((onset + duration - first_time) * sfreq)))
        mask[start:stop] = False
    return mask


def _merge_sample_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for start, stop in spans[1:]:
        last_start, last_stop = merged[-1]
        if start <= last_stop:
            merged[-1] = (last_start, max(last_stop, stop))
        else:
            merged.append((start, stop))
    return merged


def _mask_to_sample_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    """Convert a 1D boolean mask into inclusive-exclusive sample spans."""
    mask = np.asarray(mask, dtype=bool).ravel()
    if mask.size == 0:
        return []
    edges = np.diff(np.concatenate(([False], mask, [False])).astype(int))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), stops.tolist()))
