"""Artifact Subspace Reconstruction for MNE and NumPy data.

The implementation in this module provides a clean-room, MNE-compatible
standard ASR estimator plus an experimental Riemannian backend. It follows the
package convention of separating an array-level core from a scikit-learn-style
estimator while storing calibration and processing diagnostics for
auditability.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal, special, stats
from sklearn.base import BaseEstimator, TransformerMixin

from ..utils import extract_data_from_mne, reconstruct_mne_object

try:
    import mne
    from mne.epochs import BaseEpochs
    from mne.evoked import Evoked
    from mne.io import BaseRaw
except ImportError:  # pragma: no cover - MNE is a required project dependency
    mne = None


@dataclass
class ASRState:
    """Fitted ASR calibration state.

    Parameters
    ----------
    M : ndarray, shape (n_channels, n_channels)
        Matrix square root of the robust calibration covariance.
    T : ndarray, shape (n_channels, n_channels)
        Direction-dependent threshold matrix.
    thresholds : ndarray, shape (n_channels,)
        Per-calibration-component RMS thresholds.
    calibration_patterns : ndarray, shape (n_channels, n_channels)
        Calibration covariance eigenvectors.
    filter_b : ndarray
        Numerator coefficients for the statistics-only filter.
    filter_a : ndarray
        Denominator coefficients for the statistics-only filter.
    cov : ndarray, shape (n_channels, n_channels)
        Robust calibration covariance.
    rank : int
        Numerical rank after regularization.
    method : {'standard', 'riemannian'}
        Covariance geometry used by the fitted state.
    riemannian_solver : str | None
        Experimental eigenspace strategy used for Riemannian ASR.
    """

    M: np.ndarray
    T: np.ndarray
    thresholds: np.ndarray
    calibration_patterns: np.ndarray
    filter_b: np.ndarray
    filter_a: np.ndarray
    cov: np.ndarray
    rank: int
    method: str = "standard"
    riemannian_solver: str | None = None


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
    del max_mem_mb  # API placeholder; current calibration is window-streamed.
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
        raise ValueError("cov_estimator must be 'geometric_median', 'mean', or 'median'")
    if method not in ("standard", "riemannian"):
        raise ValueError("method must be 'standard' or 'riemannian'")
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
    if method == "riemannian":
        covariances = _block_covariances_rasr(X_clean, blocksize)
    else:
        covariances = _block_covariances_clean_rawdata(X_clean, blocksize)
    riemannian_info: dict[str, Any] = {}
    C = _aggregate_covariances(covariances, cov_estimator)
    C = _regularize_spd(C, regularization)
    if method == "riemannian":
        M = _sqrtm_spd(C, regularization)
        eigvals = np.linalg.eigvalsh(C)
        eigvals = np.sort(eigvals)
        _, V = _riemannian_nonlinear_eigenspace(M, regularization)
    else:
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
        riemannian_solver="nonlinear_eigenspace" if method == "riemannian" else None,
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
    diagnostics.update(riemannian_info)
    return state, diagnostics


def fit_eeg_distribution(
    values: np.ndarray,
    *,
    min_clean_fraction: float = 0.25,
    max_dropout_fraction: float = 0.1,
    fit_quantiles: tuple[float, float] = (0.022, 0.6),
    beta_grid: np.ndarray | None = None,
    return_info: bool = False,
) -> tuple[float, float] | tuple[float, float, dict[str, Any]]:
    """Fit robust clean EEG RMS statistics.

    This implements the truncated generalized-Gaussian grid search used by
    clean_rawdata's ASR calibration. The fitter sorts finite RMS values,
    searches over plausible low-tail dropout offsets and clean interval
    widths, and selects the generalized-Gaussian shape with minimum
    histogram KL divergence.

    Parameters
    ----------
    values : ndarray, shape (n_windows,)
        RMS or amplitude statistics for one component/channel.
    min_clean_fraction : float
        Minimum fraction of values assumed to be clean.
    max_dropout_fraction : float
        Maximum low-tail fraction that may be ignored as dropouts.
    fit_quantiles : tuple of float
        Lower and upper quantile span used for the clean interval search.
        The upper value also controls the preferred interval width.
    beta_grid : ndarray | None
        Generalized-Gaussian shape grid. If ``None``, use values from 1.7 to
        3.5, matching the range commonly cited for ASR ports.
    return_info : bool
        If True, return an additional diagnostics dictionary.

    Returns
    -------
    mu : float
        Robust location estimate of the clean RMS distribution.
    sigma : float
        Robust standard-deviation estimate of the clean RMS distribution.
    info : dict
        Returned only when ``return_info=True``. Contains ``beta``,
        ``fit_error``, ``fit_interval``, and ``n_fit_samples``.
    """
    if not (0 <= max_dropout_fraction < 1):
        raise ValueError("max_dropout_fraction must be in [0, 1)")
    if not (0 < min_clean_fraction <= 1):
        raise ValueError("min_clean_fraction must be in (0, 1]")
    if max_dropout_fraction + min_clean_fraction >= 1:
        raise ValueError(
            "max_dropout_fraction + min_clean_fraction must be less than 1"
        )
    q_low, q_high = fit_quantiles
    if not (0 <= q_low < q_high <= 1):
        raise ValueError("fit_quantiles must satisfy 0 <= low < high <= 1")

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Cannot fit ASR thresholds from empty RMS distribution")
    finite = np.sort(finite)

    beta_grid = (
        1.7 + 0.15 * np.arange(13, dtype=np.float64)
        if beta_grid is None
        else np.asarray(beta_grid, dtype=np.float64)
    )
    if beta_grid.size == 0:
        raise ValueError("beta_grid must contain positive values")
    if np.any(beta_grid <= 1) or np.any(beta_grid >= 7):
        raise ValueError("beta_grid values must be in the open interval (1, 7)")

    best = _fit_eeg_distribution_clean_rawdata(
        finite,
        min_clean_fraction=min_clean_fraction,
        max_dropout_fraction=max_dropout_fraction,
        fit_quantiles=(q_low, q_high),
        beta_grid=beta_grid,
    )

    if return_info:
        info = {
            "beta": float(best["beta"]),
            "fit_error": float(best["fit_error"]),
            "fit_interval": tuple(best["fit_interval"]),
            "n_fit_samples": int(best["n_fit_samples"]),
            "score": float(best["score"]),
        }
        return float(best["mu"]), float(best["sigma"]), info
    return float(best["mu"]), float(best["sigma"])


def _fit_eeg_distribution_clean_rawdata(
    values: np.ndarray,
    *,
    min_clean_fraction: float,
    max_dropout_fraction: float,
    fit_quantiles: tuple[float, float],
    beta_grid: np.ndarray,
) -> dict[str, Any]:
    """Port of clean_rawdata's ASR EEG distribution fitter."""
    q_low, q_high = fit_quantiles
    step_sizes = (0.01, 0.01)
    n_values = values.size

    bounds_by_beta = []
    rescale = np.empty(beta_grid.size, dtype=np.float64)
    for idx, beta in enumerate(beta_grid):
        sign = np.sign(np.asarray([q_low, q_high]) - 0.5)
        gamma_arg = sign * (2.0 * np.asarray([q_low, q_high]) - 1.0)
        bounds = sign * special.gammaincinv(1.0 / beta, gamma_arg) ** (1.0 / beta)
        bounds_by_beta.append(bounds)
        rescale[idx] = beta / (2.0 * special.gamma(1.0 / beta))

    max_width = q_high - q_low
    min_width = min_clean_fraction * max_width
    n_range = _round_half_up(n_values * max_width)
    offsets = np.asarray(
        [
            _round_half_up(n_values * offset)
            for offset in np.arange(
                q_low,
                q_low + max_dropout_fraction + np.finfo(float).eps,
                step_sizes[0],
            )
        ],
        dtype=int,
    )
    row_idx = np.arange(n_range, dtype=int)[:, np.newaxis]
    sample_idx = row_idx + offsets[np.newaxis, :]
    sample_idx = np.minimum(sample_idx, n_values - 1)
    ranges = values[sample_idx]
    range_start = ranges[0].copy()
    ranges = ranges - range_start[np.newaxis, :]

    opt_val = np.inf
    opt_beta = np.nan
    opt_bounds = None
    opt_lu = None
    opt_m = 0
    widths = np.arange(max_width, min_width - np.finfo(float).eps, -step_sizes[1])
    for width in widths:
        m = _round_half_up(n_values * width)
        if m < 2 or m > ranges.shape[0]:
            continue
        denominators = ranges[m - 1]
        valid = denominators > np.finfo(float).eps
        if not np.any(valid):
            continue
        nbins = max(1, _round_half_up(3.0 * np.log2(1.0 + m / 2.0)))
        scaled = np.empty((m, ranges.shape[1]), dtype=np.float64)
        scaled[:, valid] = ranges[:m, valid] * (nbins / denominators[valid])
        scaled[:, ~valid] = np.nan
        counts = _histc_scaled_bins(scaled, nbins)
        logq = np.log(counts + 0.01)

        for beta_idx, beta in enumerate(beta_grid):
            bounds = bounds_by_beta[beta_idx]
            x = bounds[0] + ((np.arange(nbins) + 0.5) / nbins) * np.diff(bounds)[0]
            p = np.exp(-(np.abs(x) ** beta)) * rescale[beta_idx]
            p = p / np.sum(p)
            kl = np.sum(p[:, np.newaxis] * (np.log(p)[:, np.newaxis] - logq), axis=0)
            kl = kl + np.log(m)
            kl[~valid] = np.inf
            idx = int(np.argmin(kl))
            min_val = float(kl[idx])
            if min_val < opt_val:
                opt_val = min_val
                opt_beta = float(beta)
                opt_bounds = bounds
                opt_lu = (
                    float(range_start[idx]),
                    float(range_start[idx] + ranges[m - 1, idx]),
                )
                opt_m = int(m)

    if opt_lu is None or opt_bounds is None:
        mu, sigma = _robust_location_scale(values)
        return {
            "mu": mu,
            "sigma": sigma,
            "beta": np.nan,
            "fit_error": np.nan,
            "score": np.nan,
            "fit_interval": (0.0, 1.0),
            "n_fit_samples": int(n_values),
        }

    alpha = (opt_lu[1] - opt_lu[0]) / np.diff(opt_bounds)[0]
    mu = opt_lu[0] - opt_bounds[0] * alpha
    sigma = np.sqrt((alpha**2) * special.gamma(3.0 / opt_beta) / special.gamma(1.0 / opt_beta))
    return {
        "mu": float(mu),
        "sigma": float(sigma),
        "beta": float(opt_beta),
        "fit_error": float(opt_val),
        "score": float(opt_val),
        "fit_interval": (float(opt_lu[0]), float(opt_lu[1])),
        "n_fit_samples": int(opt_m),
    }


def _histc_scaled_bins(values: np.ndarray, nbins: int) -> np.ndarray:
    """Histogram columns like MATLAB ``histc(H, [0:nbins-1, Inf])``."""
    counts = np.zeros((nbins, values.shape[1]), dtype=np.float64)
    for col in range(values.shape[1]):
        finite = values[:, col]
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            continue
        bins = np.floor(finite).astype(int)
        bins = np.clip(bins, 0, nbins - 1)
        counts[:, col] = np.bincount(bins, minlength=nbins)
    return counts


def compute_asr_qa_metrics(
    data_before: Any,
    data_after: Any,
    asr: ASR | None = None,
) -> dict[str, Any]:
    """Compute ASR-specific quality-assurance metrics.

    Parameters
    ----------
    data_before, data_after : Raw | Epochs | Evoked | ndarray
        Data before and after ASR. Inputs must have matching shapes.
    asr : ASR | None
        Optional fitted/transformed ASR estimator. When supplied, repair and
        calibration diagnostics are included in the output.

    Returns
    -------
    metrics : dict
        Scalar and per-channel metrics summarizing variance change and ASR
        repair extent.
    """
    before, _, _, _ = extract_data_from_mne(data_before)
    after, _, _, _ = extract_data_from_mne(data_after)
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    if before.shape != after.shape:
        raise ValueError(
            f"data_before and data_after must have matching shapes, got {before.shape} "
            f"and {after.shape}"
        )
    before_2d = before.reshape(-1, before.shape[-1]) if before.ndim > 2 else before
    after_2d = after.reshape(-1, after.shape[-1]) if after.ndim > 2 else after
    delta = before_2d - after_2d
    var_before = np.var(before_2d, axis=1)
    var_after = np.var(after_2d, axis=1)
    per_channel_variance_ratio = var_after / np.maximum(var_before, np.finfo(float).eps)
    metrics: dict[str, Any] = {
        "variance_removed_pct": float(
            100.0 * (1.0 - np.var(after_2d) / max(np.var(before_2d), np.finfo(float).eps))
        ),
        "rms_change": float(np.sqrt(np.mean(delta**2))),
        "max_abs_change": float(np.max(np.abs(delta))),
        "per_channel_variance_ratio": per_channel_variance_ratio,
        "median_channel_variance_ratio": float(np.median(per_channel_variance_ratio)),
    }
    if asr is not None and hasattr(asr, "diagnostics_"):
        counts = np.asarray(asr.n_components_reconstructed_, dtype=float)
        metrics.update(
            {
                "fraction_reconstructed_samples": float(
                    asr.fraction_reconstructed_samples_
                ),
                "fraction_reconstructed_windows": float(
                    asr.fraction_reconstructed_windows_
                ),
                "mean_components_reconstructed": float(np.mean(counts))
                if counts.size
                else 0.0,
                "max_components_reconstructed": int(
                    asr.max_components_reconstructed_
                ),
                "n_windows": int(asr.n_windows_),
            }
        )
        if hasattr(asr, "rejection_sample_mask_"):
            rejection_mask = np.asarray(asr.rejection_sample_mask_, dtype=bool)
            metrics.update(
                {
                    "fraction_retained_after_window_rejection": float(
                        np.mean(rejection_mask)
                    ),
                    "fraction_rejected_after_window_rejection": float(
                        1.0 - np.mean(rejection_mask)
                    ),
                }
            )
    if asr is not None and hasattr(asr, "calibration_info_"):
        cal = asr.calibration_info_
        if "n_clean_windows" in cal and "n_calibration_windows" in cal:
            metrics.update(
                {
                    "calibration_clean_window_fraction": float(
                        cal["n_clean_windows"] / max(cal["n_calibration_windows"], 1)
                    ),
                    "n_clean_calibration_windows": int(cal["n_clean_windows"]),
                    "n_calibration_windows": int(cal["n_calibration_windows"]),
                }
            )
        if "reference_selected_samples" in cal and "reference_candidate_samples" in cal:
            metrics.update(
                {
                    "calibration_clean_sample_fraction": float(
                        cal["reference_selected_samples"]
                        / max(cal["reference_candidate_samples"], 1)
                    ),
                    "n_clean_calibration_samples": int(cal["reference_selected_samples"]),
                    "n_calibration_candidate_samples": int(
                        cal["reference_candidate_samples"]
                    ),
                }
            )
    return metrics


def compute_asr_rejection_mask(
    X: np.ndarray,
    sfreq: float,
    *,
    max_bad_channels: float | int = 0.2,
    zthresholds: tuple[float, float] = (-3.5, 5.0),
    window_length: float = 1.0,
    window_overlap: float = 0.66,
    max_dropout_fraction: float = 0.1,
    min_clean_fraction: float = 0.25,
    fit_quantiles: tuple[float, float] = (0.022, 0.6),
    beta_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute a clean_windows-style retained-sample mask for continuous data.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        Continuous data.
    sfreq : float
        Sampling frequency in Hz.
    max_bad_channels : float | int
        Maximum tolerated number or fraction of bad channels per retained
        window.
    zthresholds : tuple of float
        Lower and upper robust z-score thresholds for channel RMS values.
    window_length : float
        Window length in seconds.
    window_overlap : float
        Overlap fraction between successive windows.
    max_dropout_fraction : float
        Maximum low-tail dropout fraction for robust RMS fitting.
    min_clean_fraction : float
        Minimum clean fraction for robust RMS fitting.
    fit_quantiles : tuple of float
        Lower and upper quantiles for the truncated generalized-Gaussian fit.
    beta_grid : ndarray | None
        Optional generalized-Gaussian beta grid.

    Returns
    -------
    sample_mask : ndarray, shape (n_times,)
        Boolean retained-sample mask. ``False`` entries indicate windows that
        would be removed by clean_windows-style rejection.
    diagnostics : dict
        Window-level RMS, z-score, and retained/removed mask diagnostics.
    """
    _validate_common_params(
        sfreq=sfreq,
        cutoff=1.0,
        window_length=window_length,
        window_overlap=window_overlap,
        max_dropout_fraction=max_dropout_fraction,
        min_clean_fraction=min_clean_fraction,
        regularization=1e-8,
    )
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(
            f"ASR window rejection expects a 2D array (n_channels, n_times), got {X.shape}"
        )
    if X.shape[0] < 1:
        raise ValueError("ASR window rejection requires at least one channel")
    if not np.all(np.isfinite(X)):
        X = np.nan_to_num(X, copy=True)

    n_channels, n_times = X.shape
    _check_enough_samples(n_times, sfreq, window_length)
    window_length_samples = _round_half_up(window_length * sfreq)
    starts = _clean_rawdata_window_starts(n_times, window_length_samples, window_overlap)
    diagnostics = _clean_windows_grid_diagnostics(
        X,
        starts,
        window_length_samples,
        max_bad_channels=max_bad_channels,
        zthresholds=zthresholds,
        max_dropout_fraction=max_dropout_fraction,
        min_clean_fraction=min_clean_fraction,
        fit_quantiles=fit_quantiles,
        beta_grid=beta_grid,
    )
    window_remove_mask = diagnostics["window_remove_mask"]

    sample_mask = np.ones(n_times, dtype=bool)
    for start in starts[window_remove_mask]:
        sample_mask[start : start + window_length_samples] = False

    diagnostics = dict(diagnostics)
    diagnostics["sample_mask"] = sample_mask
    diagnostics["fraction_retained_samples"] = float(np.mean(sample_mask))
    diagnostics["fraction_rejected_samples"] = float(1.0 - np.mean(sample_mask))
    return sample_mask, diagnostics


def process_asr(
    X: np.ndarray,
    sfreq: float,
    state: ASRState,
    *,
    window_length: float = 0.5,
    window_overlap: float = 0.66,
    max_dims: float | int = 0.66,
    regularization: float = 1e-8,
    store_reconstruction_matrices: bool = False,
    max_mem_mb: int | None = 512,
    lookahead: float | None = None,
    stepsize: int | None = None,
    method: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a calibrated ASR model to continuous data.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        Continuous data in the same channel order and units used for
        calibration.
    sfreq : float
        Sampling frequency in Hz.
    state : ASRState
        Fitted calibration state from :func:`calibrate_asr`.
    window_length : float
        Processing window length in seconds.
    window_overlap : float
        Calibration threshold-window overlap. Processing follows the
        ``clean_rawdata`` streaming ASR algorithm and uses ``stepsize`` for
        reconstruction-matrix updates.
    max_dims : float | int
        Maximum number of dimensions reconstructed per window. Floats in
        ``[0, 1]`` are interpreted as a fraction of channels.
    regularization : float
        Relative eigenvalue floor for window covariances.
    store_reconstruction_matrices : bool
        If True, store all window reconstruction matrices in diagnostics.
    max_mem_mb : int | None
        Reserved memory limit for future chunking. Present for API stability.
    lookahead : float | None
        Processing lookahead in seconds. If None, use ``window_length / 2``.
    stepsize : int | None
        Number of samples between reconstruction-matrix updates. If None, use
        ``floor(sfreq * window_length / 2)``, matching ``clean_asr.m``.
    method : {'standard', 'riemannian'} | None
        Covariance geometry for processing. If ``None``, use ``state.method``.

    Returns
    -------
    X_clean : ndarray, shape (n_channels, n_times)
        Cleaned data.
    diagnostics : dict
        Processing diagnostics.
    """
    del max_mem_mb  # API placeholder; current processing uses one offline chunk.
    _validate_common_params(
        sfreq=sfreq,
        cutoff=1.0,
        window_length=window_length,
        window_overlap=window_overlap,
        max_dropout_fraction=0.1,
        min_clean_fraction=0.25,
        regularization=regularization,
    )
    X = _validate_array_2d(X)
    n_channels, n_times = X.shape
    if state.M.shape != (n_channels, n_channels):
        raise ValueError(
            "ASR state channel count does not match data: "
            f"{state.M.shape[0]} vs {n_channels}"
        )
    if method is None:
        method = getattr(state, "method", "standard")
    if method not in ("standard", "riemannian"):
        raise ValueError("method must be 'standard' or 'riemannian'")

    win_len = max(_round_half_up(window_length * sfreq), _round_half_up(1.5 * n_channels))
    if n_times < win_len:
        raise ValueError(
            f"Window length ({win_len} samples) exceeds data length ({n_times} samples)"
        )
    lookahead = (win_len / sfreq) / 2.0 if lookahead is None else float(lookahead)
    if lookahead < 0:
        raise ValueError("lookahead must be non-negative")
    lookahead_samples = _round_half_up(lookahead * sfreq)
    if lookahead_samples >= n_times:
        raise ValueError("lookahead is too long for the data length")
    if stepsize is None:
        stepsize = max(1, int(np.floor(sfreq * (win_len / sfreq) / 2.0)))
    else:
        stepsize = int(stepsize)
    if stepsize < 1:
        raise ValueError("stepsize must be at least 1 sample")
    if stepsize > win_len:
        raise ValueError("stepsize must not exceed window_length in samples")

    max_bad = _resolve_max_dims_clean_rawdata(max_dims, n_channels)
    if max_bad <= 0:
        diagnostics = _empty_process_diagnostics(n_times)
        return X.copy(), diagnostics

    X_proc = _append_clean_rawdata_tail(X, lookahead_samples)
    data_stream = _prepend_clean_rawdata_carry(X_proc, lookahead_samples)
    data_stream[~np.isfinite(data_stream)] = 0.0
    n_stream_input = X_proc.shape[1]
    X_stats = _apply_statistics_filter_streaming(
        data_stream[:, lookahead_samples : lookahead_samples + n_stream_input],
        state.filter_b,
        state.filter_a,
    )
    update_at = np.minimum(
        np.arange(stepsize, n_stream_input + stepsize, stepsize, dtype=int),
        n_stream_input,
    )
    if update_at.size == 0 or update_at[-1] != n_stream_input:
        update_at = np.append(update_at, n_stream_input)
    update_at = np.unique(update_at)
    update_at = np.concatenate(([1], update_at))

    if method == "riemannian":
        return _process_asr_riemannian(
            data_stream,
            X_stats,
            state,
            n_times=n_times,
            n_stream_input=n_stream_input,
            lookahead_samples=lookahead_samples,
            update_at=update_at,
            max_bad=max_bad,
            stepsize=stepsize,
            win_len=win_len,
            regularization=regularization,
            store_reconstruction_matrices=store_reconstruction_matrices,
        )

    outer = np.einsum("it,jt->ijt", X_stats, X_stats, optimize=True)
    Xcov_flat = outer.reshape(n_channels * n_channels, n_stream_input, order="F")
    Xcov_flat, _ = _moving_average_clean_rawdata(win_len, Xcov_flat)

    sample_mask = np.zeros(n_times, dtype=bool)
    n_reconstructed: list[int] = []
    component_variances: list[np.ndarray] = []
    component_thresholds: list[np.ndarray] = []
    reconstruction_matrices: list[np.ndarray] = []
    window_starts: list[int] = []
    window_stops: list[int] = []

    eye = np.eye(n_channels)
    last_R = eye
    last_trivial = True
    last_n = 0
    for n in update_at:
        Cw = Xcov_flat[:, n - 1].reshape(n_channels, n_channels, order="F")
        Cw = (Cw + Cw.T) / 2.0
        D, V = np.linalg.eigh(Cw)
        order = np.argsort(D)
        D = D[order]
        V = V[:, order]

        theta2 = np.sum((state.T @ V) ** 2, axis=0)
        keep = (theta2 > D) | (np.arange(1, n_channels + 1) < (n_channels - max_bad))
        trivial = bool(np.all(keep))

        n_bad = int(n_channels - np.count_nonzero(keep))
        if trivial:
            R = eye
        else:
            basis = keep[:, np.newaxis].astype(np.float64) * (V.T @ state.M)
            R = state.M @ np.linalg.pinv(basis) @ V.T
            R = np.real_if_close(R).astype(np.float64)

        applied = (not trivial) or (not last_trivial)
        if applied and n > last_n:
            subrange = slice(last_n, n)
            width = n - last_n
            blend = (1.0 - np.cos(np.pi * np.arange(1, width + 1) / width)) / 2.0
            segment = data_stream[:, subrange]
            data_stream[:, subrange] = (
                (R @ segment) * blend[np.newaxis, :]
                + (last_R @ segment) * (1.0 - blend[np.newaxis, :])
            )

        start_out = max(last_n, lookahead_samples) - lookahead_samples
        stop_out = min(n, lookahead_samples + n_times) - lookahead_samples
        if stop_out > start_out:
            window_starts.append(int(start_out))
            window_stops.append(int(stop_out))
            n_reconstructed.append(n_bad)
            component_variances.append(D.copy())
            component_thresholds.append(theta2.copy())
            if applied:
                sample_mask[start_out:stop_out] = True
            if store_reconstruction_matrices:
                reconstruction_matrices.append(R.copy())

        last_n = int(n)
        last_R = R
        last_trivial = trivial

    X_clean = data_stream[:, lookahead_samples : lookahead_samples + n_times].copy()

    n_reconstructed_arr = np.asarray(n_reconstructed, dtype=int)
    diagnostics = {
        "window_starts": np.asarray(window_starts, dtype=int),
        "window_stops": np.asarray(window_stops, dtype=int),
        "sample_mask": sample_mask,
        "n_components_reconstructed": n_reconstructed_arr,
        "component_variances": np.asarray(component_variances, dtype=np.float64),
        "component_thresholds": np.asarray(component_thresholds, dtype=np.float64),
        "n_windows": int(len(n_reconstructed_arr)),
        "fraction_reconstructed_windows": float(
            np.mean(n_reconstructed_arr > 0) if n_reconstructed_arr.size else 0.0
        ),
        "fraction_reconstructed_samples": float(np.mean(sample_mask)),
        "max_components_reconstructed": int(n_reconstructed_arr.max(initial=0)),
        "lookahead_samples": int(lookahead_samples),
        "stepsize_samples": int(stepsize),
        "window_length_samples": int(win_len),
        "covariance_geometry": method,
    }
    if store_reconstruction_matrices:
        diagnostics["reconstruction_matrices"] = np.asarray(reconstruction_matrices)
    return X_clean, diagnostics


class ASR(BaseEstimator, TransformerMixin):
    """Artifact Subspace Reconstruction transformer.

    Parameters
    ----------
    sfreq : float | None
        Sampling frequency in Hz. Required for NumPy arrays. For MNE objects,
        this may be ``None`` and is inferred from ``info['sfreq']``.
    cutoff : float
        ASR threshold multiplier. Values around 20 are conservative; lower
        values clean more aggressively.
    window_length : float
        Processing/statistics window length in seconds.
    window_overlap : float
        Overlap fraction for processing and threshold-fitting windows.
    max_dropout_fraction : float
        Fraction of lowest RMS values ignored while estimating thresholds.
    min_clean_fraction : float
        Minimum central fraction used to estimate clean RMS statistics.
    method : {'standard', 'riemannian'}
        ASR backend. ``'riemannian'`` enables an experimental SPD-manifold
        covariance backend.
    experimental : bool
        Explicit opt-in for unstable research backends such as
        ``method='riemannian'``.
    calibration : {'auto', 'manual'}
        Calibration mode. ``'auto'`` selects clean windows before fitting;
        ``'manual'`` uses all supplied calibration samples.
    picks : str | list | None
        Channels to clean for MNE objects. Default ``'eeg'`` excludes channels
        in ``info['bads']``. For NumPy arrays, ``None`` or ``'all'`` uses all
        rows and a list of integers selects rows.
    calibration_window_length : float
        Window length in seconds for automatic clean-window selection.
    calibration_window_overlap : float
        Overlap fraction for automatic clean-window selection.
    ref_max_bad_channels : float
        Maximum fraction of channels exceeding robust tolerances in a clean
        calibration window.
    ref_tolerances : tuple of float
        Lower and upper robust z-score bounds for clean-window selection.
    blocksize : int
        Number of successive samples averaged into each covariance block for
        robust calibration covariance estimation.
    max_dims : float | int
        Maximum number of dimensions reconstructed per processing window.
    reject_by_annotation : bool
        If True, samples under bad annotations are excluded during Raw
        calibration and preserved during Raw transform.
    skip_by_annotation : tuple of str
        Annotation description prefixes treated as bad when
        ``reject_by_annotation=True``.
    cov_estimator : {'geometric_median', 'mean', 'median'}
        Aggregation rule for calibration-window covariance matrices.
    regularization : float
        Relative eigenvalue floor for covariance regularization.
    filter_kind : {'none', 'asr', 'highpass'}
        Statistics-only filter. The cleaned output is reconstructed from the
        original unfiltered data.
    window_criterion : float | int | str | None
        Optional clean_windows-style final rejection criterion. If numeric,
        this is the maximum tolerated number or fraction of bad channels per
        retained window after ASR correction. ``None`` and ``'off'`` disable
        final rejection-mask computation.
    window_criterion_tolerances : tuple of float
        Lower and upper robust z-score thresholds for final clean_windows-style
        retained-sample masking.
    lookahead : float | None
        Processing lookahead in seconds. Defaults to ``window_length / 2``.
    stepsize : int | None
        Number of samples between reconstruction-matrix updates. If ``None``,
        use the clean_rawdata default ``floor(sfreq * window_length / 2)``.
    max_mem_mb : int | None
        Reserved memory limit for future chunking.
    copy : bool
        Reserved API flag. Transform returns a new object/array.
    store_reconstruction_matrices : bool
        Store per-window reconstruction matrices in diagnostics.
    random_state : int | None
        Reserved for future stochastic calibration strategies.
    n_jobs : int | None
        Reserved for future parallel processing.
    verbose : bool | str | int | None
        Verbosity placeholder for API compatibility.

    Attributes
    ----------
    sfreq_ : float
        Sampling frequency used during fitting.
    ch_names_ : list of str | None
        Fitted channel names for MNE inputs.
    picks_ : ndarray
        Row/channel indices cleaned in the fitted data.
    M_ : ndarray
        Calibration covariance square root.
    T_ : ndarray
        Direction-dependent threshold matrix.
    thresholds_ : ndarray
        Per-component RMS thresholds.
    clean_window_mask_ : ndarray
        Calibration windows retained as clean.
    sample_mask_ : ndarray
        Samples reconstructed during the last transform.
    rejection_sample_mask_ : ndarray
        Boolean retained-sample mask from optional clean_windows-style final
        rejection. Present after transforms when ``window_criterion`` is
        enabled.
    n_components_reconstructed_ : ndarray
        Number of reconstructed components per processing window.
    diagnostics_ : dict
        Last-transform diagnostics.
    calibration_info_ : dict
        Calibration diagnostics.
    """

    def __init__(
        self,
        sfreq: float | None = None,
        *,
        cutoff: float = 20.0,
        window_length: float = 0.5,
        window_overlap: float = 0.66,
        max_dropout_fraction: float = 0.1,
        min_clean_fraction: float = 0.25,
        method: str = "standard",
        experimental: bool = False,
        calibration: str = "auto",
        picks: str | list[str] | list[int] | None = "eeg",
        calibration_window_length: float = 1.0,
        calibration_window_overlap: float = 0.66,
        ref_max_bad_channels: float = 0.075,
        ref_tolerances: tuple[float, float] = (-np.inf, 5.5),
        blocksize: int = 10,
        max_dims: float | int = 0.66,
        reject_by_annotation: bool = True,
        skip_by_annotation: tuple[str, ...] = ("bad", "bad_acq_skip"),
        cov_estimator: str = "geometric_median",
        regularization: float = 1e-8,
        filter_kind: str = "none",
        window_criterion: float | int | str | None = None,
        window_criterion_tolerances: tuple[float, float] = (-np.inf, 7.0),
        lookahead: float | None = None,
        stepsize: int | None = None,
        max_mem_mb: int | None = 512,
        copy: bool = True,
        store_reconstruction_matrices: bool = False,
        random_state: int | None = None,
        n_jobs: int | None = None,
        verbose: bool | str | int | None = None,
    ) -> None:
        self.sfreq = sfreq
        self.cutoff = cutoff
        self.window_length = window_length
        self.window_overlap = window_overlap
        self.max_dropout_fraction = max_dropout_fraction
        self.min_clean_fraction = min_clean_fraction
        self.method = method
        self.experimental = experimental
        self.calibration = calibration
        self.picks = picks
        self.calibration_window_length = calibration_window_length
        self.calibration_window_overlap = calibration_window_overlap
        self.ref_max_bad_channels = ref_max_bad_channels
        self.ref_tolerances = ref_tolerances
        self.blocksize = blocksize
        self.max_dims = max_dims
        self.reject_by_annotation = reject_by_annotation
        self.skip_by_annotation = skip_by_annotation
        self.cov_estimator = cov_estimator
        self.regularization = regularization
        self.filter_kind = filter_kind
        self.window_criterion = window_criterion
        self.window_criterion_tolerances = window_criterion_tolerances
        self.lookahead = lookahead
        self.stepsize = stepsize
        self.max_mem_mb = max_mem_mb
        self.copy = copy
        self.store_reconstruction_matrices = store_reconstruction_matrices
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(
        self,
        X: BaseRaw | BaseEpochs | np.ndarray,
        y=None,
        *,
        calibration: BaseRaw | BaseEpochs | np.ndarray | None = None,
        calibration_mask: np.ndarray | None = None,
    ) -> ASR:
        """Fit ASR calibration state.

        Parameters
        ----------
        X : Raw | Epochs | ndarray
            Data used for calibration when ``calibration`` is ``None``.
            NumPy arrays must have shape ``(n_channels, n_times)``.
        y : None
            Ignored.
        calibration : Raw | Epochs | ndarray | None
            Optional separate calibration data with matching channels.
        calibration_mask : ndarray | None
            Optional boolean sample mask for 2D calibration arrays or Raw
            inputs after annotation exclusion.

        Returns
        -------
        self : ASR
            Fitted estimator.
        """
        del y
        self._validate_estimator_params()
        fit_input = X if calibration is None else calibration
        data, sfreq, mne_type, orig_inst = extract_data_from_mne(fit_input)
        if mne_type == "evoked":
            raise ValueError("ASR.fit() does not support Evoked calibration data")
        sfreq = self._resolve_sfreq(sfreq)
        picks, ch_names = self._resolve_picks(fit_input, data, mne_type)
        data_2d = self._select_fit_data(data, mne_type, picks)
        if calibration_mask is not None:
            calibration_mask = np.asarray(calibration_mask, dtype=bool)
            if calibration_mask.shape != (data_2d.shape[1],):
                raise ValueError(
                    "calibration_mask must have shape (n_times,), got "
                    f"{calibration_mask.shape}"
                )
            data_2d = data_2d[:, calibration_mask]

        if mne_type == "raw" and self.reject_by_annotation:
            good_mask = _good_raw_sample_mask(orig_inst, self.skip_by_annotation)
            data_2d = data_2d[:, good_mask]

        self._warn_preprocessing_state(orig_inst, mne_type)
        state, cal_info = calibrate_asr(
            data_2d,
            sfreq,
            cutoff=self.cutoff,
            window_length=self.window_length,
            window_overlap=self.window_overlap,
            calibration=self.calibration,
            calibration_window_length=self.calibration_window_length,
            calibration_window_overlap=self.calibration_window_overlap,
            ref_max_bad_channels=self.ref_max_bad_channels,
            ref_tolerances=self.ref_tolerances,
            blocksize=self.blocksize,
            max_dropout_fraction=self.max_dropout_fraction,
            min_clean_fraction=self.min_clean_fraction,
            cov_estimator=self.cov_estimator,
            regularization=self.regularization,
            filter_kind=self.filter_kind,
            method=self.method,
            max_mem_mb=self.max_mem_mb,
        )

        self.state_ = state
        self.sfreq_ = float(sfreq)
        self.picks_ = picks
        self.ch_names_ = ch_names
        self.n_channels_ = int(len(picks))
        self.M_ = state.M
        self.mixing_ = state.M
        self.T_ = state.T
        self.threshold_matrix_ = state.T
        self.thresholds_ = state.thresholds
        self.calibration_patterns_ = state.calibration_patterns
        self.patterns_ = state.calibration_patterns
        self.rank_ = state.rank
        self.clean_window_mask_ = cal_info["clean_window_mask"]
        self.clean_window_scores_ = cal_info["clean_window_scores"]
        self.calibration_info_ = cal_info
        self.history_ = {
            "method": self.method,
            "calibration": self.calibration,
            "source_type": mne_type,
            "n_channels": self.n_channels_,
            "sfreq": self.sfreq_,
        }
        return self

    def transform(
        self,
        X: BaseRaw | BaseEpochs | Evoked | np.ndarray,
        y=None,
        *,
        copy: bool | None = None,
        return_diagnostics: bool = False,
    ) -> Any:
        """Apply the fitted ASR model.

        Parameters
        ----------
        X : Raw | Epochs | Evoked | ndarray
            Data to clean.
        y : None
            Ignored.
        copy : bool | None
            Reserved API flag. Transform returns a new object/array.
        return_diagnostics : bool
            If True, return ``(cleaned, diagnostics)``.

        Returns
        -------
        cleaned : Raw | Epochs | Evoked | ndarray
            Cleaned data with the same type/shape as ``X``.
        diagnostics : dict
            Returned only when ``return_diagnostics=True``.
        """
        del y, copy
        self._check_is_fitted()
        data, sfreq, mne_type, orig_inst = extract_data_from_mne(X)
        sfreq = self._resolve_sfreq(sfreq, fitted=True)
        if not np.isclose(sfreq, self.sfreq_):
            raise ValueError(
                f"Input sfreq {sfreq} does not match fitted sfreq {self.sfreq_}"
            )
        picks, ch_names = self._resolve_picks(X, data, mne_type)
        self._check_transform_channels(picks, ch_names)
        self._warn_preprocessing_state(orig_inst, mne_type)

        if mne_type == "epochs":
            cleaned_data, diagnostics = self._transform_epochs(data, picks, sfreq)
        else:
            data_out = np.asarray(data, dtype=np.float64).copy()
            selected = data_out[picks, :]
            selected_clean, diagnostics = process_asr(
                selected,
                sfreq,
                self.state_,
                window_length=self.window_length,
                window_overlap=self.window_overlap,
                max_dims=self.max_dims,
                regularization=self.regularization,
                store_reconstruction_matrices=self.store_reconstruction_matrices,
                max_mem_mb=self.max_mem_mb,
                lookahead=self.lookahead,
                stepsize=self.stepsize,
                method=self.method,
            )
            if mne_type == "raw" and self.reject_by_annotation:
                good_mask = _good_raw_sample_mask(orig_inst, self.skip_by_annotation)
                selected_clean[:, ~good_mask] = selected[:, ~good_mask]
                diagnostics["sample_mask"] = diagnostics["sample_mask"] & good_mask
            if self._window_criterion_enabled():
                rejection_mask, rejection_diag = self._compute_window_rejection(
                    selected_clean,
                    sfreq,
                )
                if mne_type == "raw" and self.reject_by_annotation:
                    rejection_mask = rejection_mask & good_mask
                diagnostics.update(
                    {
                        "rejection_sample_mask": rejection_mask,
                        "rejection_window_starts": rejection_diag["window_starts"],
                        "rejection_window_stops": rejection_diag["window_stops"],
                        "rejection_window_keep_mask": rejection_diag["window_keep_mask"],
                        "rejection_window_remove_mask": rejection_diag[
                            "window_remove_mask"
                        ],
                        "fraction_retained_after_window_rejection": float(
                            np.mean(rejection_mask)
                        ),
                        "fraction_rejected_after_window_rejection": float(
                            1.0 - np.mean(rejection_mask)
                        ),
                    }
                )
            data_out[picks, :] = selected_clean
            cleaned_data = data_out

        self._store_transform_diagnostics(diagnostics)
        cleaned = reconstruct_mne_object(cleaned_data, orig_inst, mne_type, verbose=False)
        if return_diagnostics:
            return cleaned, diagnostics
        return cleaned

    def fit_transform(
        self,
        X: BaseRaw | BaseEpochs | np.ndarray,
        y=None,
        *,
        calibration: BaseRaw | BaseEpochs | np.ndarray | None = None,
        return_diagnostics: bool = False,
    ) -> Any:
        """Fit ASR and apply it to ``X``."""
        self.fit(X, y=y, calibration=calibration)
        return self.transform(X, return_diagnostics=return_diagnostics)

    def get_diagnostics(self) -> dict[str, Any]:
        """Return diagnostics from the last transform."""
        self._check_is_fitted()
        if not hasattr(self, "diagnostics_"):
            return {}
        return dict(self.diagnostics_)

    def get_clean_window_mask(self) -> np.ndarray:
        """Return the calibration clean-window mask."""
        self._check_is_fitted()
        return self.clean_window_mask_.copy()

    def get_rejection_mask(self) -> np.ndarray:
        """Return the retained-sample mask from final clean_windows-style rejection."""
        self._check_is_fitted()
        if not hasattr(self, "rejection_sample_mask_"):
            raise RuntimeError(
                "No final rejection mask is available. Enable window_criterion and "
                "run transform first."
            )
        return np.asarray(self.rejection_sample_mask_, dtype=bool).copy()

    def to_annotations(
        self,
        *,
        min_components: int = 1,
        description: str = "ASR_REPAIR",
    ) -> Any:
        """Convert last-transform repaired windows into MNE annotations.

        Parameters
        ----------
        min_components : int
            Minimum reconstructed component count required for annotation.
        description : str
            Annotation description.

        Returns
        -------
        annotations : mne.Annotations
            Annotation spans for the last transform.
        """
        self._check_is_fitted()
        if mne is None:
            raise RuntimeError("MNE is required to create annotations")
        if not hasattr(self, "diagnostics_"):
            raise RuntimeError("No transform diagnostics available")
        starts = self.diagnostics_["window_starts"]
        stops = self.diagnostics_["window_stops"]
        counts = self.diagnostics_["n_components_reconstructed"]
        spans = [(int(s), int(e)) for s, e, c in zip(starts, stops, counts) if c >= min_components]
        spans = _merge_sample_spans(spans)
        onsets = [s / self.sfreq_ for s, _ in spans]
        durations = [(e - s) / self.sfreq_ for s, e in spans]
        return mne.Annotations(onsets, durations, [description] * len(spans))

    def to_rejection_annotations(
        self,
        *,
        description: str = "ASR_REJECT",
    ) -> Any:
        """Convert final retained-sample mask into rejection annotations."""
        self._check_is_fitted()
        if mne is None:
            raise RuntimeError("MNE is required to create annotations")
        if not hasattr(self, "rejection_sample_mask_"):
            raise RuntimeError(
                "No final rejection mask is available. Enable window_criterion and "
                "run transform first."
            )
        mask = np.asarray(self.rejection_sample_mask_, dtype=bool)
        if mask.ndim != 1:
            raise RuntimeError(
                "Rejection annotations require continuous transform diagnostics."
            )
        rejected = ~mask
        if not np.any(rejected):
            return mne.Annotations([], [], [])
        spans = _mask_to_sample_spans(rejected)
        onsets = [s / self.sfreq_ for s, _ in spans]
        durations = [(e - s) / self.sfreq_ for s, e in spans]
        return mne.Annotations(onsets, durations, [description] * len(spans))

    def _validate_estimator_params(self) -> None:
        if self.method not in ("standard", "riemannian"):
            raise NotImplementedError(
                "Supported methods are 'standard' and experimental 'riemannian'."
            )
        if self.method == "riemannian" and not self.experimental:
            raise ValueError(
                "Riemannian ASR is experimental. Set experimental=True to enable it."
            )
        if self.lookahead is not None and self.lookahead < 0:
            raise ValueError("lookahead must be non-negative")
        if self.stepsize is not None and self.stepsize < 1:
            raise ValueError("stepsize must be at least 1 sample")
        if (
            self.window_criterion is not None
            and self.window_criterion != "off"
            and isinstance(self.window_criterion, str)
        ):
            raise ValueError("window_criterion must be numeric, None, or 'off'")
        _validate_common_params(
            sfreq=self.sfreq if self.sfreq is not None else 1.0,
            cutoff=self.cutoff,
            window_length=self.window_length,
            window_overlap=self.window_overlap,
            max_dropout_fraction=self.max_dropout_fraction,
            min_clean_fraction=self.min_clean_fraction,
            regularization=self.regularization,
        )

    def _resolve_sfreq(self, sfreq: float | None, fitted: bool = False) -> float:
        if sfreq is None:
            sfreq = self.sfreq_ if fitted and hasattr(self, "sfreq_") else self.sfreq
        if sfreq is None:
            raise ValueError("sfreq must be provided for NumPy array inputs")
        if sfreq <= 0:
            raise ValueError("sfreq must be positive")
        return float(sfreq)

    def _resolve_picks(
        self,
        inst: Any,
        data: np.ndarray,
        mne_type: str,
    ) -> tuple[np.ndarray, list[str] | None]:
        if mne_type == "array":
            n_channels = data.shape[0]
            if self.picks is None or self.picks in ("all", "data", "eeg"):
                picks = np.arange(n_channels, dtype=int)
            else:
                picks = np.asarray(self.picks, dtype=int)
            if picks.size == 0:
                raise ValueError("No channels selected for ASR")
            return picks, None

        if mne is None:
            raise RuntimeError("MNE is required for MNE object inputs")
        info = inst.info
        if self.picks is None or self.picks == "all":
            picks = np.arange(len(info["ch_names"]), dtype=int)
        elif self.picks == "eeg":
            picks = mne.pick_types(
                info,
                meg=False,
                eeg=True,
                eog=False,
                ecg=False,
                stim=False,
                misc=False,
                exclude="bads",
            )
        elif isinstance(self.picks, str):
            picks = mne.pick_types(info, meg=False, eeg=self.picks == "eeg", exclude="bads")
        elif all(isinstance(pick, str) for pick in self.picks):
            picks = mne.pick_channels(info["ch_names"], include=list(self.picks), ordered=True)
        else:
            picks = np.asarray(self.picks, dtype=int)
        if picks.size == 0:
            raise ValueError("No channels selected for ASR")
        ch_names = [info["ch_names"][idx] for idx in picks]
        return np.asarray(picks, dtype=int), ch_names

    def _select_fit_data(
        self,
        data: np.ndarray,
        mne_type: str,
        picks: np.ndarray,
    ) -> np.ndarray:
        if mne_type == "epochs":
            # MNE Epochs are (n_epochs, n_channels, n_times). Concatenate epochs
            # for calibration while preserving channel order.
            return np.transpose(data[:, picks, :], (1, 0, 2)).reshape(len(picks), -1)
        return np.asarray(data[picks, :], dtype=np.float64)

    def _check_transform_channels(
        self,
        picks: np.ndarray,
        ch_names: list[str] | None,
    ) -> None:
        if len(picks) != self.n_channels_:
            raise ValueError(
                "Input channel count does not match fitted ASR state: "
                f"{len(picks)} vs {self.n_channels_}"
            )
        if self.ch_names_ is not None and ch_names != self.ch_names_:
            raise ValueError("Input channel names/order do not match fitted ASR state")

    def _transform_epochs(
        self,
        data: np.ndarray,
        picks: np.ndarray,
        sfreq: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        cleaned = np.asarray(data, dtype=np.float64).copy()
        epoch_diags = []
        starts_all: list[np.ndarray] = []
        stops_all: list[np.ndarray] = []
        sample_masks: list[np.ndarray] = []
        rejection_masks: list[np.ndarray] = []
        rejection_starts_all: list[np.ndarray] = []
        rejection_stops_all: list[np.ndarray] = []
        rejection_keep_masks: list[np.ndarray] = []
        rejection_remove_masks: list[np.ndarray] = []
        counts: list[np.ndarray] = []
        for epoch_idx in range(cleaned.shape[0]):
            selected = cleaned[epoch_idx, picks, :]
            selected_clean, diag = process_asr(
                selected,
                sfreq,
                self.state_,
                window_length=self.window_length,
                window_overlap=self.window_overlap,
                max_dims=self.max_dims,
                regularization=self.regularization,
                store_reconstruction_matrices=self.store_reconstruction_matrices,
                max_mem_mb=self.max_mem_mb,
                lookahead=self.lookahead,
                stepsize=self.stepsize,
                method=self.method,
            )
            cleaned[epoch_idx, picks, :] = selected_clean
            if self._window_criterion_enabled():
                rejection_mask, rejection_diag = self._compute_window_rejection(
                    selected_clean,
                    sfreq,
                )
                diag["rejection_sample_mask"] = rejection_mask
                diag["rejection_window_starts"] = rejection_diag["window_starts"]
                diag["rejection_window_stops"] = rejection_diag["window_stops"]
                diag["rejection_window_keep_mask"] = rejection_diag["window_keep_mask"]
                diag["rejection_window_remove_mask"] = rejection_diag["window_remove_mask"]
                diag["fraction_retained_after_window_rejection"] = float(
                    np.mean(rejection_mask)
                )
                diag["fraction_rejected_after_window_rejection"] = float(
                    1.0 - np.mean(rejection_mask)
                )
            epoch_diags.append(diag)
            starts_all.append(diag["window_starts"])
            stops_all.append(diag["window_stops"])
            sample_masks.append(diag["sample_mask"])
            if "rejection_sample_mask" in diag:
                rejection_masks.append(diag["rejection_sample_mask"])
                rejection_starts_all.append(diag["rejection_window_starts"])
                rejection_stops_all.append(diag["rejection_window_stops"])
                rejection_keep_masks.append(diag["rejection_window_keep_mask"])
                rejection_remove_masks.append(diag["rejection_window_remove_mask"])
            counts.append(diag["n_components_reconstructed"])
        diagnostics = {
            "epoch_diagnostics": epoch_diags,
            "window_starts": np.concatenate(starts_all) if starts_all else np.array([], dtype=int),
            "window_stops": np.concatenate(stops_all) if stops_all else np.array([], dtype=int),
            "sample_mask": np.vstack(sample_masks) if sample_masks else np.empty((0, 0), dtype=bool),
            "n_components_reconstructed": np.concatenate(counts) if counts else np.array([], dtype=int),
            "n_windows": int(sum(diag["n_windows"] for diag in epoch_diags)),
        }
        diagnostics["fraction_reconstructed_windows"] = (
            float(np.mean(diagnostics["n_components_reconstructed"] > 0))
            if diagnostics["n_components_reconstructed"].size
            else 0.0
        )
        diagnostics["fraction_reconstructed_samples"] = (
            float(np.mean(diagnostics["sample_mask"]))
            if diagnostics["sample_mask"].size
            else 0.0
        )
        diagnostics["max_components_reconstructed"] = int(
            diagnostics["n_components_reconstructed"].max(initial=0)
        )
        if rejection_masks:
            diagnostics["rejection_sample_mask"] = np.vstack(rejection_masks)
            diagnostics["rejection_window_starts"] = (
                np.concatenate(rejection_starts_all)
                if rejection_starts_all
                else np.array([], dtype=int)
            )
            diagnostics["rejection_window_stops"] = (
                np.concatenate(rejection_stops_all)
                if rejection_stops_all
                else np.array([], dtype=int)
            )
            diagnostics["rejection_window_keep_mask"] = (
                np.concatenate(rejection_keep_masks)
                if rejection_keep_masks
                else np.array([], dtype=bool)
            )
            diagnostics["rejection_window_remove_mask"] = (
                np.concatenate(rejection_remove_masks)
                if rejection_remove_masks
                else np.array([], dtype=bool)
            )
            diagnostics["fraction_retained_after_window_rejection"] = float(
                np.mean(diagnostics["rejection_sample_mask"])
            )
            diagnostics["fraction_rejected_after_window_rejection"] = float(
                1.0 - np.mean(diagnostics["rejection_sample_mask"])
            )
        return cleaned, diagnostics

    def _store_transform_diagnostics(self, diagnostics: dict[str, Any]) -> None:
        self.diagnostics_ = diagnostics
        self.sample_mask_ = diagnostics["sample_mask"]
        self.window_starts_ = diagnostics["window_starts"]
        self.window_stops_ = diagnostics["window_stops"]
        self.n_components_reconstructed_ = diagnostics["n_components_reconstructed"]
        self.n_windows_ = diagnostics["n_windows"]
        self.fraction_reconstructed_windows_ = diagnostics[
            "fraction_reconstructed_windows"
        ]
        self.fraction_reconstructed_samples_ = diagnostics[
            "fraction_reconstructed_samples"
        ]
        self.max_components_reconstructed_ = diagnostics[
            "max_components_reconstructed"
        ]
        if "rejection_sample_mask" in diagnostics:
            self.rejection_sample_mask_ = diagnostics["rejection_sample_mask"]
            self.rejection_window_starts_ = diagnostics["rejection_window_starts"]
            self.rejection_window_stops_ = diagnostics["rejection_window_stops"]
            self.rejection_window_keep_mask_ = diagnostics[
                "rejection_window_keep_mask"
            ]
            self.rejection_window_remove_mask_ = diagnostics[
                "rejection_window_remove_mask"
            ]
            self.fraction_retained_after_window_rejection_ = diagnostics[
                "fraction_retained_after_window_rejection"
            ]
            self.fraction_rejected_after_window_rejection_ = diagnostics[
                "fraction_rejected_after_window_rejection"
            ]
        elif hasattr(self, "rejection_sample_mask_"):
            del self.rejection_sample_mask_
            del self.rejection_window_starts_
            del self.rejection_window_stops_
            del self.rejection_window_keep_mask_
            del self.rejection_window_remove_mask_
            del self.fraction_retained_after_window_rejection_
            del self.fraction_rejected_after_window_rejection_

    def _warn_preprocessing_state(self, inst: Any, mne_type: str) -> None:
        if mne_type == "array" or inst is None:
            return
        highpass = inst.info.get("highpass", None)
        if highpass is not None and highpass < 0.25:
            warnings.warn(
                "ASR assumes high-pass filtered data; input info reports "
                f"highpass={highpass} Hz.",
                UserWarning,
                stacklevel=3,
            )
        if len(inst.info.get("projs", [])) > 0:
            warnings.warn(
                "ASR is sensitive to data rank; active or unapplied projectors "
                "may affect covariance estimates.",
                UserWarning,
                stacklevel=3,
            )

    def _window_criterion_enabled(self) -> bool:
        return self.window_criterion not in (None, "off")

    def _compute_window_rejection(
        self,
        X: np.ndarray,
        sfreq: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        return compute_asr_rejection_mask(
            X,
            sfreq,
            max_bad_channels=self.window_criterion,
            zthresholds=self.window_criterion_tolerances,
            window_length=self.calibration_window_length,
            window_overlap=self.calibration_window_overlap,
            max_dropout_fraction=self.max_dropout_fraction,
            min_clean_fraction=self.min_clean_fraction,
        )

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "state_"):
            raise RuntimeError("ASR is not fitted. Call fit() first.")


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
    if np.any(variances <= np.finfo(float).eps):
        bad = np.where(variances <= np.finfo(float).eps)[0].tolist()
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
    starts = np.asarray([_round_half_up(start) - 1 for start in starts_1_based], dtype=int)
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


def _round_half_up(value: float) -> int:
    """Round non-negative values like MATLAB ``round``."""
    return int(np.floor(float(value) + 0.5))


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


def _moving_average_clean_rawdata(
    window_length: int,
    X: np.ndarray,
    zi: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Moving-average filter equivalent to clean_rawdata's helper."""
    if zi is None:
        zi = np.zeros((X.shape[0], window_length), dtype=np.float64)
    Y = np.concatenate([zi, X], axis=1)
    diffs = (Y[:, window_length:] - Y[:, : -window_length]) / window_length
    out = np.cumsum(diffs, axis=1)
    zf_first = Y[:, [-window_length]] - out[:, [-1]] * window_length
    zf_tail = Y[:, -window_length + 1 :] if window_length > 1 else np.empty((X.shape[0], 0))
    zf = np.concatenate([zf_first, zf_tail], axis=1)
    return out, zf


def _empty_process_diagnostics(n_times: int) -> dict[str, Any]:
    """Return identity-processing diagnostics."""
    return {
        "window_starts": np.array([0], dtype=int),
        "window_stops": np.array([n_times], dtype=int),
        "sample_mask": np.zeros(n_times, dtype=bool),
        "n_components_reconstructed": np.array([0], dtype=int),
        "component_variances": np.empty((1, 0), dtype=np.float64),
        "component_thresholds": np.empty((1, 0), dtype=np.float64),
        "n_windows": 1,
        "fraction_reconstructed_windows": 0.0,
        "fraction_reconstructed_samples": 0.0,
        "max_components_reconstructed": 0,
        "lookahead_samples": 0,
        "stepsize_samples": 0,
        "window_length_samples": 0,
    }


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


def _window_covariances(
    X: np.ndarray,
    starts: np.ndarray,
    win_len: int,
) -> np.ndarray:
    covariances = np.empty((len(starts), X.shape[0], X.shape[0]), dtype=np.float64)
    for idx, start in enumerate(starts):
        window = X[:, start : start + win_len]
        window = window - window.mean(axis=1, keepdims=True)
        covariances[idx] = (window @ window.T) / max(win_len - 1, 1)
    return covariances


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


def _process_asr_riemannian(
    data_stream: np.ndarray,
    X_stats: np.ndarray,
    state: ASRState,
    *,
    n_times: int,
    n_stream_input: int,
    lookahead_samples: int,
    update_at: np.ndarray,
    max_bad: int,
    stepsize: int,
    win_len: int,
    regularization: float,
    store_reconstruction_matrices: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the experimental rASRMatlab-style chunk covariance backend."""
    n_channels = data_stream.shape[0]
    eye = np.eye(n_channels)
    Cw = (X_stats @ X_stats.T) / n_stream_input
    Cw = _regularize_spd(Cw, regularization)

    D, V = _riemannian_nonlinear_eigenspace(Cw, regularization)
    theta2 = np.sum((state.T @ V) ** 2, axis=0)
    keep = (theta2 > D) | (np.arange(1, n_channels + 1) < (n_channels - max_bad))
    trivial = bool(np.all(keep))
    n_bad = int(n_channels - np.count_nonzero(keep))

    if trivial:
        R = eye
    else:
        basis = keep[:, np.newaxis].astype(np.float64) * (V.T @ state.M)
        R = state.M @ np.linalg.pinv(basis) @ V.T
        R = np.real_if_close(R).astype(np.float64)

    sample_mask = np.zeros(n_times, dtype=bool)
    n_reconstructed: list[int] = []
    component_variances: list[np.ndarray] = []
    component_thresholds: list[np.ndarray] = []
    reconstruction_matrices: list[np.ndarray] = []
    window_starts: list[int] = []
    window_stops: list[int] = []

    last_R = eye
    last_trivial = True
    last_n = 0
    for n in update_at:
        applied = (not trivial) or (not last_trivial)
        if applied and n > last_n:
            subrange = slice(last_n, n)
            width = n - last_n
            blend = (1.0 - np.cos(np.pi * np.arange(1, width + 1) / width)) / 2.0
            segment = data_stream[:, subrange]
            data_stream[:, subrange] = (
                (R @ segment) * blend[np.newaxis, :]
                + (last_R @ segment) * (1.0 - blend[np.newaxis, :])
            )

        start_out = max(last_n, lookahead_samples) - lookahead_samples
        stop_out = min(n, lookahead_samples + n_times) - lookahead_samples
        if stop_out > start_out:
            window_starts.append(int(start_out))
            window_stops.append(int(stop_out))
            n_reconstructed.append(n_bad)
            component_variances.append(D.copy())
            component_thresholds.append(theta2.copy())
            if applied:
                sample_mask[start_out:stop_out] = True
            if store_reconstruction_matrices:
                reconstruction_matrices.append(R.copy())

        last_n = int(n)
        last_R = R
        last_trivial = trivial

    X_clean = data_stream[:, lookahead_samples : lookahead_samples + n_times].copy()
    n_reconstructed_arr = np.asarray(n_reconstructed, dtype=int)
    diagnostics = {
        "window_starts": np.asarray(window_starts, dtype=int),
        "window_stops": np.asarray(window_stops, dtype=int),
        "sample_mask": sample_mask,
        "n_components_reconstructed": n_reconstructed_arr,
        "component_variances": np.asarray(component_variances, dtype=np.float64),
        "component_thresholds": np.asarray(component_thresholds, dtype=np.float64),
        "n_windows": int(len(n_reconstructed_arr)),
        "fraction_reconstructed_windows": float(
            np.mean(n_reconstructed_arr > 0) if n_reconstructed_arr.size else 0.0
        ),
        "fraction_reconstructed_samples": float(np.mean(sample_mask)),
        "max_components_reconstructed": int(n_reconstructed_arr.max(initial=0)),
        "lookahead_samples": int(lookahead_samples),
        "stepsize_samples": int(stepsize),
        "window_length_samples": int(win_len),
        "covariance_geometry": "riemannian",
        "riemannian_solver": "nonlinear_eigenspace",
        "riemannian_mean_iterations": np.zeros(
            len(n_reconstructed_arr),
            dtype=int,
        ),
        "riemannian_mean_converged": np.ones(
            len(n_reconstructed_arr),
            dtype=bool,
        ),
        "riemannian_mean_update_norm": np.zeros(
            len(n_reconstructed_arr),
            dtype=np.float64,
        ),
    }
    if store_reconstruction_matrices:
        diagnostics["reconstruction_matrices"] = np.asarray(reconstruction_matrices)
    return X_clean, diagnostics


def _block_covariances_clean_rawdata(X: np.ndarray, blocksize: int) -> np.ndarray:
    """Match clean_rawdata's padded trailing-block covariance accumulation."""
    n_channels, n_times = X.shape
    n_blocks = max(1, int(np.ceil(n_times / blocksize)))
    covariances = np.zeros((n_blocks, n_channels, n_channels), dtype=np.float64)
    X_t = X.T
    for offset in range(blocksize):
        sample_idx = np.minimum(
            n_times - 1,
            np.arange(offset, n_times + offset, blocksize, dtype=int),
        )
        samples = X_t[sample_idx]
        covariances += np.einsum("bi,bj->bij", samples, samples)
    return covariances / blocksize


def _block_covariances_rasr(X: np.ndarray, blocksize: int) -> np.ndarray:
    """Match rASRMatlab's contiguous blocks with a full blocksize divisor."""
    n_channels, n_times = X.shape
    starts = np.arange(0, n_times, blocksize, dtype=int)
    covariances = np.empty((len(starts), n_channels, n_channels), dtype=np.float64)
    for idx, start in enumerate(starts):
        stop = min(start + blocksize, n_times)
        block = X[:, start:stop]
        covariances[idx] = (block @ block.T) / blocksize
    return covariances


def _aggregate_covariances(covariances: np.ndarray, method: str) -> np.ndarray:
    if method == "mean":
        return np.mean(covariances, axis=0)
    if method == "median":
        return np.median(covariances, axis=0)
    return _geometric_median(covariances)


def _geometric_median(
    covariances: np.ndarray,
    *,
    max_iter: int = 128,
    tol: float = 1e-7,
) -> np.ndarray:
    current = np.mean(covariances, axis=0)
    for _ in range(max_iter):
        distances = np.linalg.norm(
            (covariances - current).reshape(covariances.shape[0], -1),
            axis=1,
        )
        distances = np.maximum(distances, np.finfo(float).eps)
        weights = 1.0 / distances
        update = np.tensordot(weights, covariances, axes=(0, 0)) / np.sum(weights)
        if np.linalg.norm(update - current) <= tol * max(np.linalg.norm(current), 1.0):
            current = update
            break
        current = update
    return current


def _regularize_spd(C: np.ndarray, regularization: float) -> np.ndarray:
    C = (C + C.T) / 2
    w, V = np.linalg.eigh(C)
    floor = regularization * max(float(np.trace(C)) / C.shape[0], np.max(w), 1.0)
    w = np.maximum(w, floor)
    return (V * w) @ V.T


def _sqrtm_spd(C: np.ndarray, regularization: float) -> np.ndarray:
    C = _regularize_spd(C, regularization)
    w, V = np.linalg.eigh(C)
    return (V * np.sqrt(np.maximum(w, np.finfo(float).eps))) @ V.T


def _invsqrtm_spd(C: np.ndarray, regularization: float) -> np.ndarray:
    C = _regularize_spd(C, regularization)
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, np.finfo(float).eps)
    return (V * (1.0 / np.sqrt(w))) @ V.T


def _logm_spd(C: np.ndarray, regularization: float) -> np.ndarray:
    C = _regularize_spd(C, regularization)
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, np.finfo(float).eps)
    return (V * np.log(w)) @ V.T


def _expm_sym(S: np.ndarray) -> np.ndarray:
    S = (S + S.T) / 2.0
    w, V = np.linalg.eigh(S)
    return (V * np.exp(w)) @ V.T


def _karcher_mean_spd(
    covariances: np.ndarray,
    *,
    regularization: float,
    init: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    max_iter: int = 32,
    tol: float = 1e-7,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute the affine-invariant Karcher mean of SPD matrices."""
    covariances = np.asarray(covariances, dtype=np.float64)
    if covariances.ndim != 3:
        raise ValueError(
            "covariances must have shape (n_matrices, n_channels, n_channels)"
        )
    n_matrices = covariances.shape[0]
    if n_matrices < 1:
        raise ValueError("At least one covariance matrix is required")

    if sample_weight is None:
        weights = np.full(n_matrices, 1.0 / n_matrices, dtype=np.float64)
    else:
        weights = np.asarray(sample_weight, dtype=np.float64)
        if weights.shape != (n_matrices,):
            raise ValueError(
                "sample_weight must have shape "
                f"({n_matrices},), got {weights.shape}"
            )
        if np.any(weights < 0):
            raise ValueError("sample_weight must be non-negative")
        weight_sum = np.sum(weights)
        if weight_sum <= 0:
            raise ValueError("sample_weight must sum to a positive value")
        weights = weights / weight_sum

    current = (
        _regularize_spd(init, regularization)
        if init is not None
        else _regularize_spd(np.mean(covariances, axis=0), regularization)
    )
    converged = False
    update_norm = np.inf
    iteration_count = 0
    for _ in range(max_iter):
        iteration_count += 1
        sqrt_current = _sqrtm_spd(current, regularization)
        invsqrt_current = _invsqrtm_spd(current, regularization)
        tangent = np.zeros_like(current)
        for weight, cov in zip(weights, covariances):
            centered = (
                invsqrt_current
                @ _regularize_spd(cov, regularization)
                @ invsqrt_current
            )
            tangent += weight * _logm_spd(centered, regularization)
        update_norm = float(np.linalg.norm(tangent, ord="fro"))
        if update_norm <= tol * max(float(np.linalg.norm(current, ord="fro")), 1.0):
            converged = True
            break
        current = sqrt_current @ _expm_sym(tangent) @ sqrt_current
        current = _regularize_spd(current, regularization)

    info = {
        "riemannian_mean_iterations": int(iteration_count),
        "riemannian_mean_converged": bool(converged),
        "riemannian_mean_update_norm": float(update_norm),
    }
    return current, info


def _sqrt_and_eig(
    C: np.ndarray,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)
    w = w[order]
    V = V[:, order]
    floor = regularization * max(np.max(w), 1.0)
    w = np.maximum(w, floor)
    M = (V * np.sqrt(w)) @ V.T
    return M, w, V


def _riemannian_nonlinear_eigenspace(
    L: np.ndarray,
    regularization: float,
    *,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Approximate rASR's nonlinear eigenspace for the full-basis case.

    The MATLAB ``rasr_nonlinear_eigenspace`` helper is called with ``k=C`` in
    both calibration and processing. In that regime, the initialization step
    dominates and yields a deterministic modified eigenproblem based on
    ``L + alpha * diag(L \\ 1)``.
    """
    L = _regularize_spd(L, regularization)
    ones = np.ones(L.shape[0], dtype=np.float64)
    correction = np.linalg.solve(L, ones)
    modified = L + alpha * np.diag(correction)
    modified = (modified + modified.T) / 2.0
    D, V = np.linalg.eigh(modified)
    order = np.argsort(D)
    return D[order], V[:, order]


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


def _fit_gennorm_quantiles(
    values: np.ndarray,
    beta_grid: np.ndarray,
) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    mu = float(np.median(values))
    empirical = np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9])
    best_beta = np.nan
    best_scale = np.nan
    best_error = np.inf
    for beta in beta_grid:
        theoretical = stats.gennorm.ppf([0.1, 0.25, 0.5, 0.75, 0.9], beta)
        keep = np.abs(theoretical) > np.finfo(float).eps
        if not np.any(keep):
            continue
        scale_estimates = np.abs(empirical[keep] - mu) / np.abs(theoretical[keep])
        scale = float(np.median(scale_estimates[scale_estimates > 0]))
        if not np.isfinite(scale) or scale <= 0:
            continue
        predicted = mu + scale * theoretical
        error = float(np.mean(((empirical - predicted) / scale) ** 2))
        if error < best_error:
            best_error = error
            best_beta = float(beta)
            best_scale = scale

    if not np.isfinite(best_scale) or best_scale <= 0:
        mu, sigma = _robust_location_scale(values)
        return mu, sigma, np.nan, np.nan

    variance_factor = np.exp(
        special.gammaln(3.0 / best_beta) - special.gammaln(1.0 / best_beta)
    )
    sigma = best_scale * float(np.sqrt(variance_factor))
    if not np.isfinite(sigma) or sigma <= np.finfo(float).eps:
        _, sigma = _robust_location_scale(values)
    return mu, float(sigma), best_beta, best_error


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    mu = float(np.median(values))
    mad = float(np.median(np.abs(values - mu)))
    sigma = 1.4826 * mad
    if sigma <= np.finfo(float).eps:
        sigma = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    if sigma <= np.finfo(float).eps:
        sigma = max(abs(mu) * 1e-6, np.finfo(float).eps)
    return mu, sigma


def _resolve_max_dims(max_dims: float | int, n_channels: int) -> int:
    if isinstance(max_dims, float):
        if not (0 <= max_dims <= 1):
            raise ValueError("float max_dims must be in [0, 1]")
        return int(np.floor(max_dims * n_channels))
    max_dims = int(max_dims)
    if not (0 <= max_dims <= n_channels):
        raise ValueError("integer max_dims must be in [0, n_channels]")
    return max_dims


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
