"""ASR calibration: fit the reconstruction model from clean reference data.

``calibrate_asr`` selects clean calibration windows, estimates the robust
covariance (standard or Riemannian geometry), and derives the mixing square
root ``M`` and the direction-dependent threshold matrix ``T`` that drive
reconstruction. ``_fit_component_thresholds`` fits the per-component RMS
thresholds from the EEG amplitude distribution.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._covariance import _aggregate_block_covariances
from ._distribution import fit_eeg_distribution
from ._filters import _apply_statistics_filter, _design_statistics_filter
from ._spd import (
    _regularize_spd,
    _riemannian_nonlinear_eigenspace,
    _sqrt_and_eig,
    _sqrtm_spd,
)
from ._types import ASRState
from ._validation import (
    _check_enough_samples,
    _round_half_up,
    _validate_array_2d,
    _validate_common_params,
)
from ._windows import (
    _clean_rawdata_window_starts,
    _sample_mask_from_removed_windows,
    _select_clean_windows,
)


def calibrate_asr(
    X: np.ndarray,
    sfreq: float,
    *,
    cutoff: float = 20.0,
    window_length: float = 0.5,
    window_overlap: float = 0.66,
    calibration: str = "auto",
    calibration_window_length: float = 1.0,
    calibration_window_overlap: float = 0.66,
    ref_max_bad_channels: float = 0.075,
    ref_tolerances: tuple[float, float] = (-np.inf, 5.5),
    blocksize: int = 10,
    max_dropout_fraction: float = 0.1,
    min_clean_fraction: float = 0.25,
    cov_estimator: str = "geometric_median",
    regularization: float = 1e-8,
    filter_kind: str = "none",
    method: str = "standard",
    max_mem_mb: int | None = 512,
) -> tuple[ASRState, dict[str, Any]]:
    """Calibrate a standard ASR model from continuous data.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        Continuous calibration data.
    sfreq : float
        Sampling frequency in Hz.
    cutoff : float
        ASR threshold multiplier. Lower values clean more aggressively.
    window_length : float
        Processing/statistics window length in seconds.
    window_overlap : float
        Overlap fraction for threshold-fitting windows.
    calibration : {'auto', 'manual'}
        Whether to select clean calibration windows automatically or use all
        supplied samples.
    calibration_window_length : float
        Window length in seconds for automatic clean-window selection.
    calibration_window_overlap : float
        Overlap fraction for automatic clean-window selection.
    ref_max_bad_channels : float
        Maximum fraction of channels that may exceed ``ref_tolerances`` for a
        calibration window to be retained.
    ref_tolerances : tuple of float
        Lower and upper robust z-score tolerances for clean-window selection.
    blocksize : int
        Number of successive samples averaged into each covariance block for
        robust calibration covariance estimation.
    max_dropout_fraction : float
        Fraction of the lowest RMS values excluded while fitting thresholds.
    min_clean_fraction : float
        Minimum central fraction used to estimate clean RMS statistics.
    cov_estimator : {'geometric_median', 'mean', 'median'}
        Robust aggregation rule for calibration-window covariance matrices.
    regularization : float
        Relative eigenvalue floor used for SPD regularization.
    filter_kind : {'none', 'asr', 'highpass'}
        Statistics-only filter. ``'none'`` avoids implicit filtering;
        ``'asr'`` and ``'highpass'`` apply a conservative high-pass filter to
        the statistics path only.
    max_mem_mb : int | None
        Reserved memory limit for future chunking. Present for API stability.

    Returns
    -------
    state : ASRState
        Fitted ASR state.
    diagnostics : dict
        Calibration diagnostics.
    """
    _validate_common_params(
        sfreq=sfreq,
        cutoff=cutoff,
        window_length=window_length,
        window_overlap=window_overlap,
        max_dropout_fraction=max_dropout_fraction,
        min_clean_fraction=min_clean_fraction,
        regularization=regularization,
    )
    if calibration not in ("auto", "manual"):
        raise ValueError("calibration must be 'auto' or 'manual'")
    if cov_estimator not in ("geometric_median", "mean", "median"):
        raise ValueError(
            "cov_estimator must be 'geometric_median', 'mean', or 'median'"
        )
    if method not in ("standard", "riemannian", "riemannian_windowed"):
        raise ValueError(
            "method must be 'standard', 'riemannian', or 'riemannian_windowed'"
        )
    if blocksize < 1:
        raise ValueError("blocksize must be at least 1")

    X = _validate_array_2d(X)
    n_channels, n_times = X.shape
    _check_enough_samples(n_times, sfreq, min(window_length, calibration_window_length))

    filter_b, filter_a = _design_statistics_filter(sfreq, filter_kind)
    X_stats = _apply_statistics_filter(X, filter_b, filter_a)

    cal_len = _round_half_up(calibration_window_length * sfreq)
    cal_starts = _clean_rawdata_window_starts(
        n_times,
        cal_len,
        calibration_window_overlap,
    )

    if calibration == "auto":
        clean_window_mask, clean_window_scores = _select_clean_windows(
            X_stats,
            cal_starts,
            cal_len,
            ref_max_bad_channels=ref_max_bad_channels,
            ref_tolerances=ref_tolerances,
            max_dropout_fraction=max_dropout_fraction,
            min_clean_fraction=min_clean_fraction,
        )
        min_clean_windows = max(1, int(np.ceil(min_clean_fraction * len(cal_starts))))
        if clean_window_mask.sum() < min_clean_windows:
            raise ValueError(
                "Not enough clean calibration windows: "
                f"{clean_window_mask.sum()} found, {min_clean_windows} required"
            )
        clean_sample_mask = _sample_mask_from_removed_windows(
            n_times,
            cal_starts,
            cal_len,
            ~clean_window_mask,
        )
        X_clean = X_stats[:, clean_sample_mask]
    else:
        clean_window_mask = np.ones(len(cal_starts), dtype=bool)
        clean_window_scores = np.zeros((len(cal_starts), n_channels), dtype=np.float64)
        clean_sample_mask = np.ones(n_times, dtype=bool)
        X_clean = X_stats
    riemannian_info: dict[str, Any] = {}
    # Both Riemannian variants aggregate block covariances with Riemannian primitives
    # (geometric median + Karcher-style block reduction). The difference is the
    # eigenspace family used for V (and downstream T):
    #   - "riemannian"           : tangent-space V (MATLAB-faithful one-shot processing)
    #   - "riemannian_windowed"  : standard eigh on the Riemannian-aggregated C
    #                              (cutoff-sensitive per-window processing)
    use_riemannian_aggregation = method in ("riemannian", "riemannian_windowed")
    C, memory_info = _aggregate_block_covariances(
        X_clean,
        blocksize,
        cov_estimator,
        covariance_kind="rasr" if use_riemannian_aggregation else "clean_rawdata",
        max_mem_mb=max_mem_mb,
    )
    C = _regularize_spd(C, regularization)
    if method == "riemannian":
        M = _sqrtm_spd(C, regularization)
        eigvals = np.linalg.eigvalsh(C)
        eigvals = np.sort(eigvals)
        _, V = _riemannian_nonlinear_eigenspace(M, regularization)
    else:
        # Both "standard" and "riemannian_windowed" use standard eigh on C.
        # The Riemannian-windowed variant gets robustness from the geometric-
        # median aggregation above; cutoff sensitivity comes from the matching
        # V family at calibration and per-window processing time.
        M, eigvals, V = _sqrt_and_eig(C, regularization)
    rank = int(np.sum(eigvals > regularization * np.max(eigvals)))

    thresholds, threshold_info = _fit_component_thresholds(
        X_clean,
        V,
        sfreq=sfreq,
        window_length=window_length,
        window_overlap=window_overlap,
        cutoff=cutoff,
        min_clean_fraction=min_clean_fraction,
        max_dropout_fraction=max_dropout_fraction,
    )
    T = np.diag(thresholds) @ V.T

    state = ASRState(
        M=M,
        T=T,
        thresholds=thresholds,
        calibration_patterns=V,
        filter_b=filter_b,
        filter_a=filter_a,
        cov=C,
        rank=rank,
        method=method,
        riemannian_solver=(
            "nonlinear_eigenspace"
            if method in ("riemannian", "riemannian_windowed")
            else None
        ),
    )
    diagnostics = {
        "clean_window_mask": clean_window_mask,
        "clean_window_scores": clean_window_scores,
        "clean_sample_mask": clean_sample_mask,
        "calibration_window_starts": cal_starts,
        "calibration_window_length_samples": cal_len,
        "blocksize": int(blocksize),
        "n_clean_windows": int(clean_window_mask.sum()),
        "n_calibration_windows": int(len(cal_starts)),
        "calibration_samples": int(X_clean.shape[1]),
        "rank": rank,
        "thresholds": thresholds.copy(),
        "threshold_mu": threshold_info["mu"].copy(),
        "threshold_sigma": threshold_info["sigma"].copy(),
        "threshold_beta": threshold_info["beta"].copy(),
        "threshold_fit_error": threshold_info["fit_error"].copy(),
        "threshold_fit_interval": threshold_info["fit_interval"].copy(),
        "cov_condition": float(np.linalg.cond(C)),
        "covariance_geometry": method,
        "filter_kind": filter_kind,
    }
    diagnostics.update(memory_info)
    diagnostics.update(riemannian_info)
    return state, diagnostics


def _fit_component_thresholds(
    X: np.ndarray,
    V: np.ndarray,
    *,
    sfreq: float,
    window_length: float,
    window_overlap: float,
    cutoff: float,
    min_clean_fraction: float,
    max_dropout_fraction: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    win_len = _round_half_up(window_length * sfreq)
    starts = _clean_rawdata_window_starts(X.shape[1], win_len, window_overlap)
    projected = V.T @ X
    thresholds = np.empty(projected.shape[0], dtype=np.float64)
    mu_values = np.empty(projected.shape[0], dtype=np.float64)
    sigma_values = np.empty(projected.shape[0], dtype=np.float64)
    beta_values = np.empty(projected.shape[0], dtype=np.float64)
    fit_errors = np.empty(projected.shape[0], dtype=np.float64)
    fit_intervals = np.empty((projected.shape[0], 2), dtype=np.float64)
    for comp_idx, comp in enumerate(projected):
        rms = np.empty(len(starts), dtype=np.float64)
        for idx, start in enumerate(starts):
            segment = comp[start : start + win_len]
            rms[idx] = np.sqrt(np.mean(segment**2))
        mu, sigma, info = fit_eeg_distribution(
            rms,
            min_clean_fraction=min_clean_fraction,
            max_dropout_fraction=max_dropout_fraction,
            return_info=True,
        )
        mu_values[comp_idx] = mu
        sigma_values[comp_idx] = sigma
        beta_values[comp_idx] = info["beta"]
        fit_errors[comp_idx] = info["fit_error"]
        fit_intervals[comp_idx] = info["fit_interval"]
        thresholds[comp_idx] = mu + cutoff * sigma
    info = {
        "mu": mu_values,
        "sigma": sigma_values,
        "beta": beta_values,
        "fit_error": fit_errors,
        "fit_interval": fit_intervals,
    }
    return thresholds, info
