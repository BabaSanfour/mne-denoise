"""Quality-assurance metrics and rejection-mask summaries for ASR.

``compute_asr_qa_metrics`` summarises variance change and repair statistics for
a before/after pair; ``compute_asr_rejection_mask`` derives the clean_rawdata
final window-rejection mask. Both are reporting helpers, independent of the
estimator class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..utils import extract_data_from_mne
from ._validation import _check_enough_samples, _round_half_up, _validate_common_params
from ._windows import _clean_rawdata_window_starts, _clean_windows_grid_diagnostics

if TYPE_CHECKING:
    from .core import ASR


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
    before, _, _, _, _, _ = extract_data_from_mne(data_before, auto_pick=False)
    after, _, _, _, _, _ = extract_data_from_mne(data_after, auto_pick=False)
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
            100.0
            * (1.0 - np.var(after_2d) / max(np.var(before_2d), np.finfo(float).eps))
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
                "max_components_reconstructed": int(asr.max_components_reconstructed_),
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
                    "n_clean_calibration_samples": int(
                        cal["reference_selected_samples"]
                    ),
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
    starts = _clean_rawdata_window_starts(
        n_times, window_length_samples, window_overlap
    )
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
