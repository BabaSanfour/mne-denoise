"""
Robust DSS with Data Cleaning
=============================

This example demonstrates how to use DSS with automatic bad channel
and segment detection for noisy real-world data.

The key insight is that DSS is sensitive to artifacts in the covariance
estimation. By cleaning the data before fitting DSS, we get more reliable
spatial filters.

This is a recipe, not a core DSS variant - you should adapt this workflow
to your specific preprocessing needs.

Note: The helper functions below are simplified numpy-based implementations.
For production use with MNE data, consider using MNE's built-in methods:
- `mne.preprocessing.find_bad_channels_maxwell()`
- `raw.interpolate_bads()`
- `mne.preprocessing.annotate_*`
"""

import numpy as np
from scipy import stats

# Import DSS
from mne_denoise.dss import DSS


# =============================================================================
# Helper Functions (Simplified numpy-based implementations)
# =============================================================================

def detect_bad_channels(
    data: np.ndarray,
    *,
    z_threshold: float = 3.5,
    correlation_threshold: float = 0.4,
    variance_threshold: float = 0.01,
):
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
    """
    n_channels, n_times = data.shape

    bad_amplitude = np.zeros(n_channels, dtype=bool)
    bad_correlation = np.zeros(n_channels, dtype=bool)
    bad_flat = np.zeros(n_channels, dtype=bool)

    # Check for flat channels
    channel_vars = np.var(data, axis=1)
    median_var = np.median(channel_vars[channel_vars > 0])
    if median_var > 0:
        bad_flat = channel_vars < variance_threshold * median_var

    # Check for amplitude outliers (z-score of RMS)
    rms = np.sqrt(np.mean(data**2, axis=1))
    rms_z = np.abs(stats.zscore(rms))
    bad_amplitude = rms_z > z_threshold

    # Check for low correlation with other channels
    mean_corr = np.ones(n_channels)
    if n_channels > 1:
        corr_matrix = np.corrcoef(data)
        np.fill_diagonal(corr_matrix, np.nan)
        mean_corr = np.nanmean(np.abs(corr_matrix), axis=1)
        bad_correlation = mean_corr < correlation_threshold

    bad_mask = bad_amplitude | bad_correlation | bad_flat

    details = {
        "bad_amplitude": bad_amplitude,
        "bad_correlation": bad_correlation,
        "bad_flat": bad_flat,
        "rms": rms,
        "mean_correlation": mean_corr,
    }

    return bad_mask, details


def interpolate_bad_channels(
    data: np.ndarray,
    bad_mask: np.ndarray,
    *,
    method: str = "average",
) -> np.ndarray:
    """Interpolate bad channels using good channels.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Input data.
    bad_mask : ndarray, shape (n_channels,)
        Boolean mask of bad channels.
    method : str
        Interpolation method: 'average' or 'zero'. Default 'average'.

    Returns
    -------
    data_interp : ndarray
        Data with bad channels interpolated.
    """
    data_interp = data.copy()
    good_mask = ~bad_mask

    if np.sum(good_mask) == 0:
        raise ValueError("No good channels available for interpolation")

    if method == "average":
        mean_good = np.mean(data[good_mask], axis=0)
        data_interp[bad_mask] = mean_good
    elif method == "zero":
        data_interp[bad_mask] = 0
    else:
        raise ValueError(f"Unknown method: {method}")

    return data_interp


def detect_bad_segments(
    data: np.ndarray,
    sfreq: float,
    *,
    segment_length: float = 1.0,
    z_threshold: float = 4.0,
):
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

    Returns
    -------
    bad_mask : ndarray, shape (n_times,)
        Boolean mask of bad time points.
    """
    n_channels, n_times = data.shape
    segment_samples = int(segment_length * sfreq)
    step = segment_samples // 2  # 50% overlap

    if segment_samples >= n_times:
        return np.zeros(n_times, dtype=bool)

    segment_starts = list(range(0, n_times - segment_samples + 1, step))
    segment_rms = []

    for start in segment_starts:
        end = start + segment_samples
        segment = data[:, start:end]
        rms = np.sqrt(np.mean(segment**2))
        segment_rms.append(rms)

    segment_rms = np.array(segment_rms)
    z_scores = np.abs(stats.zscore(segment_rms))
    bad_segments_idx = np.where(z_scores > z_threshold)[0]

    bad_mask = np.zeros(n_times, dtype=bool)
    for idx in bad_segments_idx:
        start = segment_starts[idx]
        end = min(start + segment_samples, n_times)
        bad_mask[start:end] = True

    return bad_mask


# =============================================================================
# Main Workflow
# =============================================================================

def robust_dss_workflow(
    data: np.ndarray,
    sfreq: float,
    bias,
    *,
    channel_z_threshold: float = 3.5,
    segment_z_threshold: float = 4.0,
    n_components: int = None,
    **dss_kws,
):
    """
    Example workflow for robust DSS with data cleaning.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Raw data that may contain artifacts.
    sfreq : float
        Sampling frequency.
    bias : LinearDenoiser or callable
        Bias function for DSS.
    channel_z_threshold : float
        Z-score threshold for bad channel detection.
    segment_z_threshold : float
        Z-score threshold for bad segment detection.
    n_components : int, optional
        Number of DSS components to keep.
    **dss_kws
        Additional keyword arguments for DSS.

    Returns
    -------
    dss : DSS
        Fitted DSS object.
    info : dict
        Dictionary with cleaning information.
    """
    data_clean = data.copy()
    info = {}

    # Step 1: Detect and interpolate bad channels
    bad_channels, _ = detect_bad_channels(data, z_threshold=channel_z_threshold)
    info["n_bad_channels"] = np.sum(bad_channels)

    if np.any(bad_channels):
        print(f"Detected {info['n_bad_channels']} bad channels, interpolating...")
        data_clean = interpolate_bad_channels(data_clean, bad_channels)

    # Step 2: Detect and exclude bad segments
    bad_segments = detect_bad_segments(
        data_clean, sfreq, z_threshold=segment_z_threshold
    )
    info["n_bad_samples"] = np.sum(bad_segments)

    if np.any(bad_segments):
        print(f"Detected {info['n_bad_samples']} bad samples, excluding from fit...")
        data_for_fit = data_clean[:, ~bad_segments]
    else:
        data_for_fit = data_clean

    # Step 3: Fit DSS on clean data
    dss = DSS(bias=bias, n_components=n_components, **dss_kws)
    dss.fit(data_for_fit)

    return dss, info


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Generate synthetic data with artifacts
    np.random.seed(42)
    n_channels, n_times = 32, 1000
    sfreq = 250

    # Clean signal
    t = np.arange(n_times) / sfreq
    signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz oscillation
    data = np.random.randn(n_channels, n_times) * 0.5
    data += np.outer(np.random.randn(n_channels), signal)

    # Add artifact to channel 5
    data[5, :] += 10 * np.random.randn(n_times)

    # Add artifact segment
    data[:, 400:450] += 5 * np.random.randn(n_channels, 50)

    # Define a simple bias (smoothing)
    def simple_bias(x):
        from scipy.ndimage import uniform_filter1d

        return uniform_filter1d(x, size=25, axis=1)

    # Run robust DSS workflow
    dss, info = robust_dss_workflow(
        data, sfreq=sfreq, bias=simple_bias, n_components=5
    )

    print(f"Bad channels: {info['n_bad_channels']}")
    print(f"Bad samples: {info['n_bad_samples']}")
    print(f"DSS eigenvalues: {dss.eigenvalues_}")

    # Transform data
    sources = dss.transform(data)
    print(f"Sources shape: {sources.shape}")
