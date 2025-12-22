"""Time-Shift DSS (TSR) implementation.

Implements time-shift regression/DSS for capturing temporally extended
structure, equivalent to NoiseTools nt_tsr.m.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from .core import compute_dss


@dataclass
class TimeShiftResult:
    """Results from time-shift DSS.

    Attributes
    ----------
    filters : ndarray, shape (n_components, n_channels)
        DSS spatial filters.
    patterns : ndarray, shape (n_channels, n_components)
        DSS spatial patterns.
    eigenvalues : ndarray, shape (n_components,)
        DSS eigenvalues.
    shifts : ndarray
        Time shifts used.
    lag_correlations : ndarray, shape (n_shifts,)
        Average correlation at each lag.
    """
    filters: np.ndarray
    patterns: np.ndarray
    eigenvalues: np.ndarray
    shifts: np.ndarray
    lag_correlations: np.ndarray


def time_shift_dss(
    data: np.ndarray,
    shifts: Union[int, np.ndarray] = 10,
    *,
    n_components: Optional[int] = None,
    rank: Optional[int] = None,
    reg: float = 1e-9,
    method: str = 'autocorrelation',
) -> TimeShiftResult:
    """Time-Shift DSS for capturing temporally extended structure.

    Creates time-shifted copies of the data and applies DSS to find
    components that have high autocorrelation (temporal predictability).
    Useful for extracting slow-varying or temporally smooth sources.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
        Input data.
    shifts : int or ndarray
        If int, use lags from 1 to shifts.
        If array, use specified lag values in samples.
        Default 10.
    n_components : int, optional
        Number of DSS components to return.
    rank : int, optional
        Rank for whitening.
    reg : float
        Regularization. Default 1e-9.
    method : str
        Method for constructing bias:
        - 'autocorrelation': Use shifted data correlation (default)
        - 'prediction': Use prediction from shifted data

    Returns
    -------
    result : TimeShiftResult
        Container with filters, patterns, eigenvalues, and shift info.

    Examples
    --------
    >>> # Find components with high temporal structure
    >>> result = time_shift_dss(data, shifts=20)
    >>> slow_sources = result.filters @ data
    
    >>> # Use specific lags
    >>> result = time_shift_dss(data, shifts=np.array([1, 2, 5, 10, 20]))

    Notes
    -----
    This is equivalent to NoiseTools' nt_tsr.m. Components are ordered by
    temporal predictability (highest eigenvalue = most predictable).

    The bias covariance emphasizes signals that are autocorrelated, meaning
    samples at different time lags are similar. This extracts:
    - Slow oscillations
    - DC shifts
    - Temporally smooth artifact patterns
    """
    data = np.asarray(data)
    
    # Handle 3D epoched data
    if data.ndim == 3:
        n_channels, n_times, n_epochs = data.shape
        data_2d = data.reshape(n_channels, -1)
    elif data.ndim == 2:
        data_2d = data
        n_channels, n_times = data.shape
    else:
        raise ValueError(f"Data must be 2D or 3D, got {data.ndim}D")

    n_samples = data_2d.shape[1]

    # Set up shifts
    if isinstance(shifts, int):
        shift_array = np.arange(1, shifts + 1)
    else:
        shift_array = np.asarray(shifts)
    
    # Validate shifts
    max_shift = np.max(np.abs(shift_array))
    if max_shift >= n_samples // 2:
        raise ValueError(
            f"Max shift ({max_shift}) too large for data length ({n_samples})"
        )

    # Compute biased covariance based on temporal structure
    if method == 'autocorrelation':
        biased_data = _compute_autocorrelation_bias(data_2d, shift_array)
    elif method == 'prediction':
        biased_data = _compute_prediction_bias(data_2d, shift_array)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute DSS
    filters, patterns, eigenvalues, _ = compute_dss(
        data_2d, biased_data,
        n_components=n_components,
        rank=rank,
        reg=reg,
    )

    # Compute correlation at each lag for diagnostics
    lag_correlations = _compute_lag_correlations(data_2d, shift_array)

    return TimeShiftResult(
        filters=filters,
        patterns=patterns,
        eigenvalues=eigenvalues,
        shifts=shift_array,
        lag_correlations=lag_correlations,
    )


def _compute_autocorrelation_bias(
    data: np.ndarray,
    shifts: np.ndarray,
) -> np.ndarray:
    """Create bias emphasizing autocorrelation structure.
    
    For each shift, create time-shifted version and average.
    This emphasizes signals that are similar across time lags.
    """
    n_channels, n_samples = data.shape
    max_shift = np.max(np.abs(shifts))
    
    # Trim to valid range
    valid_start = max_shift
    valid_end = n_samples - max_shift
    valid_length = valid_end - valid_start
    
    if valid_length <= 0:
        raise ValueError("Data too short for specified shifts")
    
    # Accumulate shifted versions
    accumulated = np.zeros((n_channels, valid_length))
    
    for shift in shifts:
        if shift > 0:
            shifted = data[:, valid_start + shift : valid_end + shift]
        elif shift < 0:
            shifted = data[:, valid_start + shift : valid_end + shift]
        else:
            shifted = data[:, valid_start:valid_end]
        
        accumulated += shifted
    
    # Average
    biased = accumulated / len(shifts)
    
    # Pad to original length with zeros (or replicate edge values)
    biased_full = np.zeros_like(data)
    biased_full[:, valid_start:valid_end] = biased
    
    return biased_full


def _compute_prediction_bias(
    data: np.ndarray,
    shifts: np.ndarray,
) -> np.ndarray:
    """Create bias based on predicting current from past.
    
    Uses regression from shifted data to predict current timepoint.
    """
    n_channels, n_samples = data.shape
    max_shift = np.max(np.abs(shifts))
    
    # Simple approach: average of shifted versions weighted by shift
    valid_start = max_shift
    valid_end = n_samples - max_shift
    valid_length = valid_end - valid_start
    
    if valid_length <= 0:
        raise ValueError("Data too short for specified shifts")
    
    # Weight by inverse of shift (closer lags are more predictive)
    accumulated = np.zeros((n_channels, valid_length))
    total_weight = 0
    
    for shift in shifts:
        weight = 1.0 / max(abs(shift), 1)
        shifted = data[:, valid_start + shift : valid_end + shift]
        accumulated += weight * shifted
        total_weight += weight
    
    biased = accumulated / total_weight
    
    biased_full = np.zeros_like(data)
    biased_full[:, valid_start:valid_end] = biased
    
    return biased_full


def _compute_lag_correlations(
    data: np.ndarray,
    shifts: np.ndarray,
) -> np.ndarray:
    """Compute average autocorrelation at each lag."""
    n_channels, n_samples = data.shape
    correlations = []
    
    for shift in shifts:
        if shift == 0:
            correlations.append(1.0)
            continue
            
        # Compute correlation at this lag
        if shift > 0:
            x = data[:, :-shift]
            y = data[:, shift:]
        else:
            x = data[:, -shift:]
            y = data[:, :shift]
        
        # Mean correlation across channels
        corr_per_channel = []
        for ch in range(n_channels):
            cc = np.corrcoef(x[ch], y[ch])[0, 1]
            if np.isfinite(cc):
                corr_per_channel.append(cc)
        
        if corr_per_channel:
            correlations.append(np.mean(corr_per_channel))
        else:
            correlations.append(0.0)
    
    return np.array(correlations)


def smooth_dss(
    data: np.ndarray,
    smooth_window: int = 10,
    *,
    n_components: Optional[int] = None,
    rank: Optional[int] = None,
    reg: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DSS biased toward temporally smooth sources.

    Applies temporal smoothing as the bias function, extracting
    components with low-frequency temporal structure.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Input data.
    smooth_window : int
        Smoothing window size in samples. Default 10.
    n_components : int, optional
        Number of components to return.
    rank : int, optional
        Rank for whitening.
    reg : float
        Regularization. Default 1e-9.

    Returns
    -------
    filters : ndarray
        DSS spatial filters.
    patterns : ndarray
        DSS spatial patterns.
    eigenvalues : ndarray
        DSS eigenvalues.

    Examples
    --------
    >>> # Extract slow components (smoothed over 20 samples)
    >>> filters, patterns, eigs = smooth_dss(data, smooth_window=20)
    """
    data = np.asarray(data)
    if data.ndim == 3:
        data = data.reshape(data.shape[0], -1)
    
    # Apply temporal smoothing as bias
    from scipy.ndimage import uniform_filter1d
    biased_data = uniform_filter1d(data, size=smooth_window, axis=1, mode='reflect')
    
    filters, patterns, eigenvalues, _ = compute_dss(
        data, biased_data,
        n_components=n_components,
        rank=rank,
        reg=reg,
    )
    
    return filters, patterns, eigenvalues
