"""CleanLine-style fallback for residual noise.

Provides a simple notch filter as a fallback when ZapLine cannot fully remove
noise without causing spectral notches.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import signal


def apply_cleanline_notch(
    data: np.ndarray,
    sfreq: float,
    freq: float,
    bandwidth: float = 0.5,
    order: int = 4,
) -> np.ndarray:
    """Apply a narrow notch filter to remove residual line noise.
    
    This is a fallback for cases where ZapLine cannot fully remove noise
    without over-cleaning. Uses a Butterworth notch filter.
    
    Parameters
    ----------
    data : ndarray
        Input data (n_channels, n_times).
    sfreq : float
        Sampling frequency.
    freq : float
        Center frequency to notch.
    bandwidth : float
        Width of the notch in Hz (default 0.5 Hz).
    order : int
        Filter order (default 4).
        
    Returns
    -------
    filtered : ndarray
        Notch-filtered data.
    """
    # Design notch filter
    nyquist = sfreq / 2
    low = (freq - bandwidth / 2) / nyquist
    high = (freq + bandwidth / 2) / nyquist
    
    # Ensure within valid range
    low = max(0.001, min(low, 0.999))
    high = max(0.001, min(high, 0.999))
    
    if low >= high:
        return data  # Cannot filter, return unchanged
    
    # Use bandstop (notch) filter
    sos = signal.butter(order, [low, high], btype='bandstop', output='sos')
    filtered = signal.sosfiltfilt(sos, data, axis=1)
    
    return filtered


def apply_hybrid_cleanup(
    data: np.ndarray,
    sfreq: float,
    freq: float,
    bandwidth: float = 0.5,
    max_power_reduction_db: float = 3.0,
) -> np.ndarray:
    """Apply hybrid cleanup: notch filter with spectral impact check.
    
    Only applies the notch if it doesn't cause excessive power reduction
    below the noise frequency (which would indicate over-cleaning).
    
    Parameters
    ----------
    data : ndarray
        Input data.
    sfreq : float
        Sampling frequency.
    freq : float
        Frequency to clean. 
    bandwidth : float
        Notch bandwidth (default 0.5 Hz).
    max_power_reduction_db : float
        Maximum allowed power reduction in surrounding frequencies (dB).
        If exceeded, cleanup is not applied.
        
    Returns
    -------
    cleaned : ndarray
        Cleaned data (or original if cleanup would cause issues).
    """
    from scipy.signal import welch
    
    # Apply notch
    filtered = apply_cleanline_notch(data, sfreq, freq, bandwidth)
    
    # Check spectral impact
    n_times = data.shape[1]
    n_fft = min(n_times, int(sfreq * 4))
    
    freqs, psd_orig = welch(data, fs=sfreq, nperseg=n_fft, axis=-1)
    freqs, psd_filt = welch(filtered, fs=sfreq, nperseg=n_fft, axis=-1)
    
    # Check power in surrounding frequencies (excluding notch region)
    surr_low = (freqs > freq - 3) & (freqs < freq - bandwidth)
    surr_high = (freqs > freq + bandwidth) & (freqs < freq + 3)
    surr_mask = surr_low | surr_high
    
    if not np.any(surr_mask):
        return filtered  # Can't check, apply anyway
    
    mean_orig = np.mean(psd_orig[:, surr_mask])
    mean_filt = np.mean(psd_filt[:, surr_mask])
    
    # Power reduction in dB
    reduction_db = 10 * np.log10(mean_orig / max(mean_filt, 1e-20))
    
    if reduction_db > max_power_reduction_db:
        # Cleanup would cause too much collateral damage
        return data
    
    return filtered
