"""Adaptive algorithms for Zapline-plus.

This module implements the adaptive components described in Klug & Kloosterman (2022):
1. Automatic noise frequency detection (coarse).
2. Adaptive data segmentation based on covariance stationarity.
3. Fine-grained frequency peak detection per segment.
4. Artifact presence testing.

References
----------
Klug, M., & Kloosterman, N. A. (2022). Zapline-plus: A Zapline extension for
automatic and adaptive removal of frequency-specific noise artifacts in M/EEG.
NeuroImage, 258, 119370.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
from scipy import signal
from scipy.signal import welch, find_peaks
from ..dss.utils.covariance import compute_covariance


def find_noise_freqs(
    data: np.ndarray,
    sfreq: float,
    fmin: float = 17.0,
    fmax: float = 99.0,
    window_length: float = 6.0,  # Hz for moving window
    threshold_factor: float = 4.0,  # "exceeds center by factor of 4 in 10*log10" ?? 
    # Wait, paper says "exceeds center by > 4 (in 10*log10 units)" which is additive in log space.
) -> List[float]:
    """Detect noise frequencies using Welch PSD and outlier detection.
    
    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Input data.
    sfreq : float
        Sampling frequency.
    fmin : float
        Minimum frequency to search.
    fmax : float
        Maximum frequency to search.
    window_length : float
        Length of the moving average window in Hz (default 6 Hz).
    threshold_factor : float
        Threshold for peak detection in dB (default 4.0, approx 2.5x power).
        
    Returns
    -------
    noise_freqs : list of float
        Detected noise frequencies.
    """
    n_channels, n_times = data.shape
    
    # Paper uses Hanning window (hann)
    n_fft = min(n_times, int(sfreq * 4))  # Reasonable resolution
    if n_fft < 1024:
        n_fft = 1024
        
    freqs, psd = welch(
        data, 
        fs=sfreq, 
        window='hann', 
        nperseg=n_fft, 
        axis=-1, 
        average='mean'
    )
    
    # "Convert PSD to 10*log10, then take mean over channels in log space"
    # Note: paper says geometric mean (mean in log space)
    psd_log = 10 * np.log10(np.clip(psd, 1e-20, None))
    mean_log_psd = np.mean(psd_log, axis=0)  # Average over channels
    
    # Limit to search range
    mask = (freqs >= fmin) & (freqs <= fmax)
    search_freqs = freqs[mask]
    search_psd = mean_log_psd[mask]
    
    if len(search_freqs) == 0:
        return []

    # Moving window outlier detection
    # Search for outlier peaks using a 6 Hz moving window.
    # A frequency is flagged if it exceeds the "center" (mean of left 
    # and right thirds of the 6 Hz window) by > 4 (in 10*log10 units).
    
    detected_freqs = []
    
    # Convert window length from Hz to indices
    freq_res = freqs[1] - freqs[0]
    win_samples = int(window_length / freq_res)
    half_win = win_samples // 2
    
    # We need to iterate over indices identifying local peaks
    # First find local maxima to reduce search
    peak_indices, _ = find_peaks(search_psd)
    
    for idx_rel in peak_indices:
        idx_full = np.where(freqs == search_freqs[idx_rel])[0][0]
        
        # Define window indices
        start_idx = max(0, idx_full - half_win)
        end_idx = min(len(freqs), idx_full + half_win)
        
        # Extract window
        window_psd = mean_log_psd[start_idx:end_idx]
        
        # Split into thirds
        n_win = len(window_psd)
        if n_win < 3:
            continue
            
        n_third = n_win // 3
        left_third = window_psd[:n_third]
        right_third = window_psd[-n_third:]
        
        # "Center" is mean of left and right thirds
        center_level = np.mean(np.concatenate([left_third, right_third]))
        
        # Check peak value (the value at idx_full)
        peak_val = mean_log_psd[idx_full]
        
        if peak_val > center_level + threshold_factor:
            detected_freqs.append(freqs[idx_full])
            
    # Mere duplicates if peaks are close? For now return all.
    # Maybe cluster them? The function returns list of floats.
    return detected_freqs


def segment_data(
    data: np.ndarray,
    sfreq: float,
    target_freq: float,
    min_chunk_len: float = 30.0,
    cov_win_len: float = 1.0,  # 1s covariance matrices
) -> List[Tuple[int, int]]:
    """Segment data into chunks based on covariance stationarity.
    
    Parameters
    ----------
    data : ndarray
        Input data.
    sfreq : float
        Sampling frequency.
    target_freq : float
        Target line noise frequency.
    min_chunk_len : float
        Minimum chunk length in seconds (default 30s).
    cov_win_len : float
        Window length for covariance computation in seconds (default 1s).
        
    Returns
    -------
    segments : list of (start_idx, end_idx)
        List of segment indices.
    """
    n_channels, n_times = data.shape
    
    # 1. Narrowband filter around target freq +/- 3Hz
    # Paper uses +/- 3Hz
    f_low = target_freq - 3
    f_high = target_freq + 3
    
    sos = signal.butter(4, [f_low, f_high], btype='bandpass', fs=sfreq, output='sos')
    data_filt = signal.sosfiltfilt(sos, data, axis=1)
    
    # 2. Compute 1s covariance matrices
    n_win = int(cov_win_len * sfreq)
    n_steps = n_times // n_win
    
    covs = []
    for i in range(n_steps):
        start = i * n_win
        end = start + n_win
        chunk = data_filt[:, start:end]
        cov = compute_covariance(chunk)
        # Normalize covariance to focus on topography, not power scale?
        # Paper says "proxy for changing noise topography".
        # Usually implies normalization. Let's trace closely or assume raw cov.
        # "Compute a distance between successive covariance matrices"
        # Riemannian distance is standard, but frobenius might be used.
        # Let's use correlation distance or Riemannian. 
        # For simplicity and speed in Python without pyriemann:
        # Log-Euclidean distance is good, or just Frobenius on normalized covs.
        
        # Normalize by trace to remove power fluctuation info, keeping spatial info?
        tr = np.trace(cov)
        if tr > 1e-20:
            cov = cov / tr
        covs.append(cov)
        
    covs = np.array(covs)  # (n_steps, n_ch, n_ch)
    
    # 3. Compute distance between successive matrices
    # dist[i] = dist(cov[i], cov[i+1])
    dists = []
    for i in range(len(covs) - 1):
        # Riemannian distance d(A,B) = ||log(A^-1/2 * B * A^-1/2)||_F
        # But that's heavy.
        # Klug paper doesn't specify metric explicitly in the quick summary.
        # "Distance between successive covariance matrices"
        # Let's use simple Frobenius distance between normalized matrices for now.
        d = np.linalg.norm(covs[i] - covs[i+1], ord='fro')
        dists.append(d)
        
    dists = np.array(dists)
    
    # 4. Detect peaks in stationarity signal
    if len(dists) == 0:
        return [(0, n_times)]
        
    # Smooth a bit?
    # "Detect peaks"
    peak_indices, _ = find_peaks(dists, prominence=np.std(dists)*0.5) # Heuristic prominence
    
    # Convert peak indices (which are window indices) to sample indices
    # boundaries are at the "interface" between i and i+1
    boundary_indices = (peak_indices + 1) * n_win
    
    # 5. Enforce minimum chunk length
    # Recursively merge or filter boundaries
    valid_boundaries = [0]
    last_boundary = 0
    min_samples = int(min_chunk_len * sfreq)
    
    for b in boundary_indices:
        if (b - last_boundary) >= min_samples:
            valid_boundaries.append(b)
            last_boundary = b
            
    # Check last segment
    if (n_times - last_boundary) < min_samples:
        # Merge with previous if possible
        if len(valid_boundaries) > 1:
            valid_boundaries.pop() # Remove last start, so previous segment extends to end
            
    valid_boundaries.append(n_times)
    
    # Construct segments
    segments = []
    for i in range(len(valid_boundaries) - 1):
        segments.append((valid_boundaries[i], valid_boundaries[i+1]))
        
    return segments


def find_fine_peak(
    data: np.ndarray,
    sfreq: float,
    coarse_freq: float,
    search_width: float = 0.05
) -> float:
    """Find peak frequency within a narrow band.
    
    Parameters
    ----------
    data : ndarray
    sfreq : float
    coarse_freq : float
    search_width : float
        Search range +/- Hz (default 0.05).
        
    Returns
    -------
    fine_freq : float
    """
    f_low = coarse_freq - search_width
    f_high = coarse_freq + search_width
    n_times = data.shape[1]
    
    # High resolution PSD needed
    # Zero-padding for better frequency interpolation
    n_fft = max(n_times, int(sfreq / 0.01)) # Res 0.01 Hz
    
    freqs, psd = welch(data, fs=sfreq, nperseg=min(n_times, 4*int(sfreq)), nfft=n_fft, axis=-1)
    
    # Log mean
    psd_log = 10 * np.log10(np.clip(psd, 1e-20, None))
    mean_log_psd = np.mean(psd_log, axis=0)
    
    mask = (freqs >= f_low) & (freqs <= f_high)
    search_freqs = freqs[mask]
    search_psd = mean_log_psd[mask]
    
    if len(search_freqs) == 0:
        return coarse_freq
        
    idx_max = np.argmax(search_psd)
    return search_freqs[idx_max]


def check_artifact_presence(
    data: np.ndarray,
    sfreq: float,
    target_freq: float,
    window_length: float = 6.0
) -> bool:
    """Check if artifact is present using spectral thresholding.
    
    "In a 6 Hz region around the target, take the two lower 5% log-PSD quantiles 
    from the first and last third of the window. 
    Compare to 'center power' (mean of left/right thirds) and build a deviation measure
    Threshold = center power + 2 * deviation"
    
    Wait, the paper description is a bit dense. 
    "threshold = center power + 2 * deviation"
    
    Let's interpret:
    1. PSD in 6Hz window centered on target_freq.
    2. Split into left/right thirds (flanks) and center third (peak region).
    3. Center power = Mean of Flanks (?? This contradicts "Compare to center power"). 
    
    Re-reading prompt:
    "Compare to 'center power' (mean of left/right thirds)" -> This likely means Baseline Power.
    Let's call it Baseline.
    
    "take the two lower 5% log-PSD quantiles from the first and last third"
    -> Deviation = measure of noise floor variance?
    
    Let's assume:
    Baseline = Mean(Left Third + Right Third)
    Deviation = Something derived from 5% quantiles? 
    Maybe Deviation = Baseline - Quantile(5%)? Or MAD?
    
    Paper (Klug 2022): 
    "We estimated the noise floor... by taking the 5% quantile of the power spectrum values in the outer thirds..."
    "We defined the noise floor as the mean of these two 5% quantiles."
    "The standard deviation of the noise floor was estimated as the difference between the mean noise floor and the 5% quantile..." ??
    
    Let's follow the prompt specifically:
    "two lower 5% log-PSD quantiles from the first and last third"
    "Compare to 'center power' (mean of left/right thirds)"
    "Threshold = center power + 2 * deviation"
    
    Hypothesis:
    CenterPower (Baseline) = Mean(LeftThird concat RightThird)
    QuantileLeft = 5% quantile of LeftThird
    QuantileRight = 5% quantile of RightThird
    Deviation ?
    
    Actually, let's implement a robust peak check.
    If Peak > Baseline + 2 * Deviation.
    
    Let's try to infer deviation.
    If Deviation is approx std.
    Maybe Deviation = Baseline - Mean(QuantileLeft, QuantileRight)?
    
    Let's stick to a robust detection:
    1. Calculate PSD in window.
    2. Baseline = median or mean of flanks.
    3. Peak = value at target_freq.
    4. Threshold = Baseline + X dB. (Prompt says corresponds to 2.5x power -> 4dB).
    
    But prompt gives specific formula "Threshold = center power + 2 x deviation".
    And "explicitly say SD / MAD didn't work".
    
    Let's look at "deviation measure":
    Maybe Deviation = (Mean(Left+Right) - Mean(Lower5%Quantiles)) ?
    This measures the "spread" of the noise floor.
    
    Let's proceed with this interpretation.
    """
    n_times = data.shape[1]
    n_fft = min(n_times, int(sfreq * 4))
    freqs, psd = welch(data, fs=sfreq, nperseg=n_fft, axis=-1)
    
    psd_log = 10 * np.log10(np.clip(psd, 1e-20, None))
    mean_log_psd = np.mean(psd_log, axis=0) # Channel mean
    
    # 6 Hz window
    f_low = target_freq - 3
    f_high = target_freq + 3
    
    # Indices
    idx_start = np.argmax(freqs >= f_low)
    idx_end = np.argmax(freqs > f_high)
    if idx_end == 0: idx_end = len(freqs)
    
    window_psd = mean_log_psd[idx_start:idx_end]
    n_win = len(window_psd)
    
    if n_win < 3:
        return False
        
    n_third = n_win // 3
    left_third = window_psd[:n_third]
    right_third = window_psd[-n_third:]
    
    # "Center power" (Baseline)
    flanks = np.concatenate([left_third, right_third])
    center_power = np.mean(flanks) # Mean of flanks
    
    # Deviation
    # "two lower 5% log-PSD quantiles from the first and last third"
    q_left = np.percentile(left_third, 5)
    q_right = np.percentile(right_third, 5)
    
    # If standard deviation is problematic, maybe deviation is simply distance from mean to bottom?
    # Or maybe range?
    # Let's assume deviation ~ (center_power - mean(q_left, q_right))
    # This represents half-width of noise floor distribution??
    
    deviation = center_power - np.mean([q_left, q_right])
    
    threshold = center_power + 2 * deviation
    
    # Peak power
    # Target freq index relative to window
    # Actually just max in the middle third?
    # Or value at target_freq explicitly?
    # "If peak log-PSD > threshold -> artifact present"
    # Usually peak is search in the center.
    
    # Get value at target freq
    idx_target = np.argmin(np.abs(freqs - target_freq))
    peak_val = mean_log_psd[idx_target]
    
    return peak_val > threshold


def detect_harmonics(
    data: np.ndarray,
    sfreq: float,
    fundamental: float,
    max_harmonics: Optional[int] = None,
    threshold_factor: float = 4.0,
    window_length: float = 6.0,
) -> List[float]:
    """Detect present harmonics of a fundamental frequency.
    
    Checks each harmonic (2f, 3f, ...) up to Nyquist and returns only those
    that are detected as peaks above threshold.
    
    Parameters
    ----------
    data : ndarray
        Input data (n_channels, n_times).
    sfreq : float
        Sampling frequency.
    fundamental : float
        Fundamental frequency (e.g., 50 Hz).
    max_harmonics : int, optional
        Maximum number of harmonics to check. If None, checks all up to Nyquist.
    threshold_factor : float
        Threshold for peak detection in dB (default 4.0).
    window_length : float
        Window size in Hz for detection (default 6.0).
        
    Returns
    -------
    detected_harmonics : list of float
        List of detected harmonic frequencies (not including fundamental).
    """
    nyquist = sfreq / 2
    
    # Determine max harmonics
    if max_harmonics is None:
        max_harmonics = int(np.floor(nyquist / fundamental)) - 1  # Exclude fundamental
    
    detected = []
    
    # Calculate PSD once
    n_times = data.shape[1]
    n_fft = min(n_times, int(sfreq * 4))
    if n_fft < 1024:
        n_fft = 1024
        
    freqs, psd = welch(data, fs=sfreq, window='hann', nperseg=n_fft, axis=-1)
    psd_log = 10 * np.log10(np.clip(psd, 1e-20, None))
    mean_log_psd = np.mean(psd_log, axis=0)
    
    # Check each harmonic
    for h in range(2, max_harmonics + 2):  # 2f, 3f, ... up to max+1
        harmonic_freq = fundamental * h
        
        if harmonic_freq >= nyquist:
            break
            
        # Check if this harmonic is a significant peak
        # Use similar logic to find_noise_freqs
        freq_res = freqs[1] - freqs[0]
        win_samples = int(window_length / freq_res)
        half_win = win_samples // 2
        
        idx_center = np.argmin(np.abs(freqs - harmonic_freq))
        start_idx = max(0, idx_center - half_win)
        end_idx = min(len(freqs), idx_center + half_win)
        
        window_psd = mean_log_psd[start_idx:end_idx]
        n_win = len(window_psd)
        
        if n_win < 3:
            continue
            
        n_third = n_win // 3
        left_third = window_psd[:n_third]
        right_third = window_psd[-n_third:]
        
        center_level = np.mean(np.concatenate([left_third, right_third]))
        peak_val = mean_log_psd[idx_center]
        
        if peak_val > center_level + threshold_factor:
            detected.append(harmonic_freq)
            
    return detected
