"""ZapLine linear functions."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from ..dss.linear import compute_dss
from ..dss.utils import compute_covariance, iterative_outlier_removal
from ..dss.denoisers.spectral import HarmonicFFTBias
from ..dss.denoisers.temporal import PeriodSmoothingBias


@dataclass
class ZapLineResult:
    """Results from DSS-ZapLine processing.

    Attributes
    ----------
    cleaned : ndarray
        Cleaned data with line noise removed.
    removed : ndarray
        Removed noise components.
    n_removed : int
        Number of components removed.
    dss_filters : ndarray
        DSS spatial filters for line noise.
    dss_patterns : ndarray
        DSS spatial patterns (mixing matrix).
    dss_eigenvalues : ndarray
        DSS eigenvalues (noise power per component).
    line_freq : float
        Target line frequency.
    n_harmonics : int
        Number of harmonics processed.
    """

    cleaned: np.ndarray
    removed: np.ndarray
    n_removed: int
    dss_filters: np.ndarray
    dss_patterns: np.ndarray
    dss_eigenvalues: np.ndarray
    line_freq: float
    n_harmonics: int = 1


def dss_zapline(
    data: np.ndarray,
    line_freq: float,
    sfreq: float,
    *,
    n_remove: Union[int, str] = "auto",
    n_harmonics: Optional[int] = None,
    nfft: int = 1024,
    nkeep: Optional[int] = None,
    rank: Optional[int] = None,
    reg: float = 1e-9,
    threshold: float = 3.0,
) -> ZapLineResult:
    """Remove line noise using DSS-based spatial filtering.

    Implements the ZapLine algorithm (de Cheveigné, 2020):
    1. Smooth data to isolate low-frequency content
    2. Compute residual (line noise + high-freq)
    3. Use FFT-based bias for DSS at line frequency harmonics
    4. Find spatial components that maximize line noise variance
    5. Project out top components from original data

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Input data with line noise.
    line_freq : float
        Line noise frequency in Hz (e.g., 50 or 60). Must be positive.
    sfreq : float
        Sampling frequency in Hz.
    n_remove : int or 'auto'
        Number of components to remove. If 'auto', determine using
        outlier detection. Default 'auto'.
    n_harmonics : int, optional
        Number of harmonics to include. If None, use all harmonics
        up to Nyquist.
    nfft : int
        FFT size for bias computation. Default 1024.
    nkeep : int, optional
        Number of PCA components to keep before DSS. Helps avoid
        overfitting with many channels.
    rank : int, optional
        Rank for DSS whitening.
    reg : float
        Regularization for DSS. Default 1e-9.
    threshold : float
        Z-score threshold for auto component selection. Default 3.0.

    Returns
    -------
    result : ZapLineResult
        Container with cleaned data and metadata.

    Examples
    --------
    >>> # Remove 50 Hz line noise
    >>> result = dss_zapline(eeg_data, line_freq=50, sfreq=1000)
    >>> cleaned = result.cleaned

    >>> # Remove 60 Hz with all harmonics
    >>> result = dss_zapline(eeg_data, line_freq=60, sfreq=1000)
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(
            f"Data must be 2D (n_channels, n_times), got shape {data.shape}"
        )

    # Issue 3: Validate line_freq
    if line_freq <= 0:
        raise ValueError(f"line_freq must be positive, got {line_freq}")

    n_channels, n_times = data.shape
    nyquist = sfreq / 2

    # Issue 3: Check period validity
    period_float = sfreq / line_freq
    period = int(round(period_float))
    if abs(period_float - period) > 0.1:
        warnings.warn(
            f"sfreq/line_freq = {period_float:.2f} is not close to an integer. "
            f"Smoothing will use period={period} samples, which may not perfectly "
            f"cancel line frequency {line_freq} Hz. Consider resampling data.",
            UserWarning,
        )

    # Compute number of harmonics if not specified
    if n_harmonics is None:
        n_harmonics = int(np.floor(nyquist / line_freq))
    else:
        max_harmonics = int(np.floor(nyquist / line_freq))
        n_harmonics = min(n_harmonics, max_harmonics)

    # Step 1: Smooth data to get low-frequency component using PeriodSmoothingBias
    smoother = PeriodSmoothingBias(period=period, n_iterations=1)
    data_smooth = smoother.apply(data)

    # Step 2: Residual contains line noise and high frequencies
    data_residual = data - data_smooth

    # Step 3: Reduce dimensionality if requested (avoid overfitting)
    if nkeep is not None and nkeep < n_channels:
        # PCA reduction
        cov = compute_covariance(data_residual)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1][:nkeep]
        pca_filters = eigvecs[:, idx].T  # (nkeep, n_channels)
        data_reduced = pca_filters @ data_residual  # (nkeep, n_times)
    else:
        data_reduced = data_residual
        pca_filters = None

    # Step 4: Compute FFT-based bias covariance matrices using HarmonicFFTBias
    bias = HarmonicFFTBias(
        line_freq=line_freq,
        sfreq=sfreq,
        n_harmonics=n_harmonics,
        nfft=nfft,
        overlap=0.5,
    )
    c0, c1 = bias.compute_covariances(data_reduced)

    # Step 5: Apply DSS with covariance matrices
    dss_filters, dss_patterns, eigvals = compute_dss(
        covariance_baseline=c0,
        covariance_biased=c1,
        n_components=None,  # Return all for selection
        rank=rank,
        reg=reg,
    )

    # Eigenvalues are the scores (ratio of biased to baseline variance)
    scores = np.abs(eigvals)

    # Step 6: Determine number of components to remove
    if n_remove == "auto":
        n_remove_count = iterative_outlier_removal(scores, threshold)
    else:
        n_remove_count = min(int(n_remove), len(scores))

    # Map DSS filters/patterns to original space
    if pca_filters is not None:
        dss_filters_original = dss_filters @ pca_filters
        dss_patterns_original = pca_filters.T @ dss_patterns
    else:
        dss_filters_original = dss_filters
        dss_patterns_original = dss_patterns

    # Issue 2 & 4: Build result directly instead of calling apply_zapline
    # This preserves the scores and avoids redundant pinv computation
    if n_remove_count <= 0:
        return ZapLineResult(
            cleaned=data.copy(),
            removed=np.zeros_like(data),
            n_removed=0,
            dss_filters=dss_filters_original,
            dss_patterns=dss_patterns_original,
            dss_eigenvalues=scores,
            line_freq=line_freq,
            n_harmonics=n_harmonics,
        )

    # Apply projection using pinv(filters) as the mixing matrix
    # The proper mixing matrix A for projection is pinv(W), not the "patterns" from DSS
    # which are C0 @ W (useful for visualization, not projection)
    A = np.linalg.pinv(dss_filters_original)
    
    # P = I - A(:, :d) @ W(:d, :)
    P = np.eye(n_channels) - A[:, :n_remove_count] @ dss_filters_original[:n_remove_count, :]

    residual_clean = P @ data_residual
    cleaned = data_smooth + residual_clean
    removed = data - cleaned

    return ZapLineResult(
        cleaned=cleaned,
        removed=removed,
        n_removed=n_remove_count,
        dss_filters=dss_filters_original,
        dss_patterns=dss_patterns_original,
        dss_eigenvalues=scores,
        line_freq=line_freq,
        n_harmonics=n_harmonics,
    )


def apply_zapline(
    data: np.ndarray,
    filters: np.ndarray,
    n_remove: int,
    line_freq: float,
    sfreq: float,
    patterns: Optional[np.ndarray] = None,
) -> ZapLineResult:
    """Apply learned ZapLine filters to new data.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Data to clean.
    filters : ndarray, shape (n_components, n_channels)
        Learned DSS filters.
    n_remove : int
        Number of components to remove.
    line_freq : float
        Line frequency.
    sfreq : float
        Sampling frequency.
    patterns : ndarray, optional
        Pre-computed patterns (mixing matrix). If None, computed via pinv(filters).

    Returns
    -------
    result : ZapLineResult
        Cleaned data.
    """
    data = np.asarray(data)
    n_channels, n_times = data.shape

    # Validate line_freq
    if line_freq <= 0:
        raise ValueError(f"line_freq must be positive, got {line_freq}")

    # Compute or use provided patterns
    if patterns is None:
        A = np.linalg.pinv(filters)
    else:
        A = patterns

    if n_remove <= 0:
        return ZapLineResult(
            cleaned=data.copy(),
            removed=np.zeros_like(data),
            n_removed=0,
            dss_filters=filters,
            dss_patterns=A,
            dss_eigenvalues=np.array([]),  # Not available in apply only
            line_freq=line_freq,
        )

    # Step 1: Smooth using PeriodSmoothingBias
    period = int(round(sfreq / line_freq))
    smoother = PeriodSmoothingBias(period=period, n_iterations=1)
    data_smooth = smoother.apply(data)
    data_residual = data - data_smooth

    # Step 2: Project out components
    P = np.eye(n_channels) - A[:, :n_remove] @ filters[:n_remove, :]

    residual_clean = P @ data_residual

    # Step 3: Reconstruct
    cleaned = data_smooth + residual_clean
    removed = data - cleaned

    return ZapLineResult(
        cleaned=cleaned,
        removed=removed,
        n_removed=n_remove,
        dss_filters=filters,
        dss_patterns=A,
        dss_eigenvalues=np.array([]),  # Not available here
        line_freq=line_freq,
    )
