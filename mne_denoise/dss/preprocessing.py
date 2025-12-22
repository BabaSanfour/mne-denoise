"""Preprocessing and robustness utilities for DSS.

Provides functions for handling bad channels, outlier segments,
and data quality issues that can affect DSS performance.
"""

from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np
from scipy import stats


def detect_bad_channels(
    data: np.ndarray,
    *,
    z_threshold: float = 3.5,
    correlation_threshold: float = 0.4,
    variance_threshold: float = 0.01,
) -> Tuple[np.ndarray, dict]:
    """Detect bad channels using multiple criteria.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Input data.
    z_threshold : float
        Z-score threshold for amplitude outliers. Default 3.5.
    correlation_threshold : float
        Minimum correlation with neighbors. Default 0.4.
    variance_threshold : float
        Minimum relative variance (ratio to median). Default 0.01.

    Returns
    -------
    bad_mask : ndarray, shape (n_channels,)
        Boolean mask of bad channels.
    details : dict
        Dictionary with detection details per criterion.

    Examples
    --------
    >>> bad_mask, details = detect_bad_channels(data)
    >>> good_data = data[~bad_mask]
    """
    n_channels, n_times = data.shape
    
    # Initialize mask
    bad_amplitude = np.zeros(n_channels, dtype=bool)
    bad_correlation = np.zeros(n_channels, dtype=bool)
    bad_variance = np.zeros(n_channels, dtype=bool)
    bad_flat = np.zeros(n_channels, dtype=bool)
    
    # 1. Check for flat channels
    channel_vars = np.var(data, axis=1)
    median_var = np.median(channel_vars[channel_vars > 0])
    if median_var > 0:
        bad_flat = channel_vars < variance_threshold * median_var
    
    # 2. Check for amplitude outliers (z-score of RMS)
    rms = np.sqrt(np.mean(data**2, axis=1))
    rms_z = np.abs(stats.zscore(rms))
    bad_amplitude = rms_z > z_threshold
    
    # 3. Check for low correlation with other channels
    if n_channels > 1:
        # Compute correlation matrix
        corr_matrix = np.corrcoef(data)
        np.fill_diagonal(corr_matrix, np.nan)
        
        # Mean correlation with other channels
        mean_corr = np.nanmean(np.abs(corr_matrix), axis=1)
        bad_correlation = mean_corr < correlation_threshold
    
    # Combine
    bad_mask = bad_amplitude | bad_correlation | bad_variance | bad_flat
    
    details = {
        'bad_amplitude': bad_amplitude,
        'bad_correlation': bad_correlation,
        'bad_variance': bad_variance,
        'bad_flat': bad_flat,
        'rms': rms,
        'mean_correlation': mean_corr if n_channels > 1 else np.ones(n_channels),
        'channel_variance': channel_vars,
    }
    
    return bad_mask, details


def detect_bad_segments(
    data: np.ndarray,
    sfreq: float,
    *,
    segment_length: float = 1.0,
    z_threshold: float = 4.0,
    overlap: float = 0.5,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Detect bad temporal segments using amplitude criteria.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Input data.
    sfreq : float
        Sampling frequency.
    segment_length : float
        Length of segments to check in seconds. Default 1.0.
    z_threshold : float
        Z-score threshold for segment rejection. Default 4.0.
    overlap : float
        Overlap between segments (0-1). Default 0.5.

    Returns
    -------
    bad_mask : ndarray, shape (n_times,)
        Boolean mask of bad time points.
    bad_segments : list of tuple
        List of (start, end) sample indices of bad segments.
    """
    n_channels, n_times = data.shape
    segment_samples = int(segment_length * sfreq)
    step = int(segment_samples * (1 - overlap))
    
    if segment_samples >= n_times:
        # Data too short for segmentation
        return np.zeros(n_times, dtype=bool), []
    
    # Compute segment statistics
    segment_starts = list(range(0, n_times - segment_samples + 1, step))
    segment_rms = []
    
    for start in segment_starts:
        end = start + segment_samples
        segment = data[:, start:end]
        rms = np.sqrt(np.mean(segment**2))
        segment_rms.append(rms)
    
    segment_rms = np.array(segment_rms)
    
    # Z-score based detection
    z_scores = np.abs(stats.zscore(segment_rms))
    bad_segments_idx = np.where(z_scores > z_threshold)[0]
    
    # Create sample-level mask
    bad_mask = np.zeros(n_times, dtype=bool)
    bad_segments = []
    
    for idx in bad_segments_idx:
        start = segment_starts[idx]
        end = min(start + segment_samples, n_times)
        bad_mask[start:end] = True
        bad_segments.append((start, end))
    
    return bad_mask, bad_segments


def interpolate_bad_channels(
    data: np.ndarray,
    bad_mask: np.ndarray,
    *,
    method: str = "spline",
) -> np.ndarray:
    """Interpolate bad channels using good channels.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Input data.
    bad_mask : ndarray, shape (n_channels,)
        Boolean mask of bad channels.
    method : str
        Interpolation method: 'spline', 'average', or 'zero'.
        Default 'spline'.

    Returns
    -------
    data_interp : ndarray
        Data with bad channels interpolated.
    """
    data_interp = data.copy()
    good_mask = ~bad_mask
    n_good = np.sum(good_mask)
    
    if n_good == 0:
        raise ValueError("No good channels available for interpolation")
    
    good_data = data[good_mask]
    
    if method == "average":
        # Simple average of good channels
        mean_good = np.mean(good_data, axis=0)
        data_interp[bad_mask] = mean_good
        
    elif method == "zero":
        # Zero out bad channels
        data_interp[bad_mask] = 0
        
    elif method == "spline":
        # Spline interpolation based on correlation
        # Use weighted average based on correlation with good channels
        good_indices = np.where(good_mask)[0]
        bad_indices = np.where(bad_mask)[0]
        
        for bad_idx in bad_indices:
            # Compute correlation with each good channel
            correlations = []
            for good_idx in good_indices:
                corr = np.corrcoef(data[bad_idx], data[good_idx])[0, 1]
                if np.isnan(corr):
                    corr = 0
                correlations.append(np.abs(corr))
            
            correlations = np.array(correlations)
            
            if correlations.sum() > 0:
                weights = correlations / correlations.sum()
            else:
                weights = np.ones(n_good) / n_good
            
            data_interp[bad_idx] = np.sum(good_data * weights[:, np.newaxis], axis=0)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    return data_interp


def robust_covariance(
    data: np.ndarray,
    *,
    method: str = "empirical",
    shrinkage: Optional[float] = None,
) -> np.ndarray:
    """Compute robust covariance matrix.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Input data.
    method : str
        Method: 'empirical', 'shrinkage', 'oas', or 'mcd'.
        Default 'empirical'.
    shrinkage : float, optional
        Shrinkage parameter for 'shrinkage' method.

    Returns
    -------
    cov : ndarray, shape (n_channels, n_channels)
        Covariance matrix.
    """
    n_channels, n_times = data.shape
    
    # Center data
    data_centered = data - data.mean(axis=1, keepdims=True)
    
    if method == "empirical":
        cov = data_centered @ data_centered.T / n_times
        
    elif method == "shrinkage":
        # Ledoit-Wolf-like shrinkage
        emp_cov = data_centered @ data_centered.T / n_times
        
        if shrinkage is None:
            # Estimate optimal shrinkage
            shrinkage = _ledoit_wolf_shrinkage(data_centered)
        
        target = np.eye(n_channels) * np.trace(emp_cov) / n_channels
        cov = (1 - shrinkage) * emp_cov + shrinkage * target
        
    elif method == "oas":
        # Oracle Approximating Shrinkage
        try:
            from sklearn.covariance import OAS
            oas = OAS().fit(data_centered.T)
            cov = oas.covariance_
        except ImportError:
            # Fallback to shrinkage
            return robust_covariance(data, method="shrinkage")
            
    elif method == "mcd":
        # Minimum Covariance Determinant (robust)
        try:
            from sklearn.covariance import MinCovDet
            mcd = MinCovDet().fit(data_centered.T)
            cov = mcd.covariance_
        except ImportError:
            return robust_covariance(data, method="shrinkage")
    else:
        raise ValueError(f"Unknown covariance method: {method}")
    
    # Ensure symmetry
    cov = (cov + cov.T) / 2
    
    return cov


def _ledoit_wolf_shrinkage(data: np.ndarray) -> float:
    """Estimate optimal Ledoit-Wolf shrinkage parameter."""
    n_channels, n_times = data.shape
    
    # Sample covariance
    S = data @ data.T / n_times
    
    # Target: scaled identity
    mu = np.trace(S) / n_channels
    
    # Compute shrinkage intensity
    delta = ((S - mu * np.eye(n_channels))**2).sum() / n_channels
    
    # Estimate beta
    X2 = data**2
    beta = np.sum(X2 @ X2.T / n_times - S**2) / (n_channels * n_times)
    
    # Shrinkage
    shrinkage = min(1.0, beta / max(delta, 1e-10))
    
    return max(0.0, shrinkage)


def reject_epochs_by_amplitude(
    data: np.ndarray,
    *,
    threshold: Optional[float] = None,
    z_threshold: float = 4.0,
) -> np.ndarray:
    """Reject epochs with extreme amplitudes.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times, n_epochs)
        Epoched data.
    threshold : float, optional
        Absolute amplitude threshold. If None, use z-score method.
    z_threshold : float
        Z-score threshold if threshold is None. Default 4.0.

    Returns
    -------
    good_epochs : ndarray, shape (n_epochs,)
        Boolean mask of good epochs.
    """
    n_channels, n_times, n_epochs = data.shape
    
    # Compute max amplitude per epoch
    max_amp = np.max(np.abs(data), axis=(0, 1))  # (n_epochs,)
    
    if threshold is not None:
        good_epochs = max_amp < threshold
    else:
        # Z-score based
        z_scores = np.abs(stats.zscore(max_amp))
        good_epochs = z_scores < z_threshold
    
    return good_epochs


class RobustDSS:
    """DSS with automatic bad channel/segment handling.

    Parameters
    ----------
    bias : callable
        Bias function.
    n_components : int, optional
        Number of components.
    detect_bad_channels : bool
        Auto-detect and interpolate bad channels. Default True.
    detect_bad_segments : bool
        Auto-detect and exclude bad segments. Default True.
    channel_z_threshold : float
        Z-score for bad channel detection. Default 3.5.
    segment_z_threshold : float
        Z-score for bad segment detection. Default 4.0.
    covariance_method : str
        Robust covariance method. Default 'shrinkage'.
    """

    def __init__(
        self,
        bias,
        *,
        n_components: Optional[int] = None,
        detect_bad_channels: bool = True,
        detect_bad_segments: bool = True,
        channel_z_threshold: float = 3.5,
        segment_z_threshold: float = 4.0,
        covariance_method: str = "shrinkage",
    ):
        self.bias = bias
        self.n_components = n_components
        self.detect_bad_channels = detect_bad_channels
        self.detect_bad_segments = detect_bad_segments
        self.channel_z_threshold = channel_z_threshold
        self.segment_z_threshold = segment_z_threshold
        self.covariance_method = covariance_method
        
        # Fitted attributes
        self.filters_ = None
        self.patterns_ = None
        self.eigenvalues_ = None
        self.bad_channels_ = None
        self.bad_segments_ = None

    def fit(self, data: np.ndarray, sfreq: float = 1.0):
        """Fit robust DSS."""
        data_clean = data.copy()
        
        # Handle bad channels
        if self.detect_bad_channels:
            bad_ch, _ = detect_bad_channels(data, z_threshold=self.channel_z_threshold)
            self.bad_channels_ = bad_ch
            if np.any(bad_ch):
                data_clean = interpolate_bad_channels(data_clean, bad_ch)
        
        # Handle bad segments
        if self.detect_bad_segments:
            bad_seg, segments = detect_bad_segments(
                data_clean, sfreq, z_threshold=self.segment_z_threshold
            )
            self.bad_segments_ = segments
            # Use only good segments for covariance
            if np.any(bad_seg):
                good_data = data_clean[:, ~bad_seg]
            else:
                good_data = data_clean
        else:
            good_data = data_clean
        
        # Apply bias
        biased_data = self.bias(good_data)
        
        # Compute DSS with robust covariance
        from .core import compute_dss
        self.filters_, self.patterns_, self.eigenvalues_, _ = compute_dss(
            good_data, biased_data, n_components=self.n_components
        )
        
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted filters."""
        if self.filters_ is None:
            raise RuntimeError("RobustDSS not fitted")
        
        data_centered = data - data.mean(axis=1, keepdims=True)
        return self.filters_ @ data_centered

    def inverse_transform(self, sources: np.ndarray) -> np.ndarray:
        """Reconstruct from sources."""
        if self.patterns_ is None:
            raise RuntimeError("RobustDSS not fitted")
        
        n_sources = sources.shape[0]
        return self.patterns_[:, :n_sources] @ sources
