"""Narrowband frequency scanning for DSS.

Implements nt_narrowband_scan.m equivalent for finding optimal
narrowband DSS components across a frequency range.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import signal

from .core import compute_dss


@dataclass
class NarrowbandScanResult:
    """Results from narrowband frequency scanning.

    Attributes
    ----------
    frequencies : ndarray, shape (n_freqs,)
        Frequencies that were scanned.
    eigenvalues : ndarray, shape (n_freqs, n_components)
        DSS eigenvalues at each frequency.
    peak_freqs : ndarray
        Frequencies with highest eigenvalues (top peaks).
    peak_eigenvalues : ndarray
        Eigenvalues at peak frequencies.
    best_freq : float
        Single best frequency (highest eigenvalue).
    best_filters : ndarray
        DSS filters at the best frequency.
    best_patterns : ndarray
        DSS patterns at the best frequency.
    """
    frequencies: np.ndarray
    eigenvalues: np.ndarray
    peak_freqs: np.ndarray
    peak_eigenvalues: np.ndarray
    best_freq: float
    best_filters: np.ndarray
    best_patterns: np.ndarray


def narrowband_scan(
    data: np.ndarray,
    sfreq: float,
    *,
    freq_range: Tuple[float, float] = (1, 40),
    freq_step: float = 1.0,
    bandwidth: float = 2.0,
    n_components: int = 1,
    n_peaks: int = 3,
    rank: Optional[int] = None,
    reg: float = 1e-9,
) -> NarrowbandScanResult:
    """Scan frequencies to find optimal narrowband DSS components.

    Sweeps through a frequency range, applying bandpass filtering at each
    frequency and computing DSS. Returns eigenvalue spectrum showing which
    frequencies have the strongest spatially coherent activity.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
        Input data.
    sfreq : float
        Sampling frequency in Hz.
    freq_range : tuple of float
        (min_freq, max_freq) range to scan. Default (1, 40).
    freq_step : float
        Frequency step size in Hz. Default 1.0.
    bandwidth : float
        Bandwidth of bandpass filter at each frequency. Default 2.0.
    n_components : int
        Number of DSS components to compute at each frequency. Default 1.
    n_peaks : int
        Number of peak frequencies to return. Default 3.
    rank : int, optional
        Rank for DSS whitening.
    reg : float
        Regularization threshold. Default 1e-9.

    Returns
    -------
    result : NarrowbandScanResult
        Container with scanning results including peak frequencies
        and eigenvalue spectrum.

    Examples
    --------
    >>> # Find dominant alpha frequency
    >>> result = narrowband_scan(eeg_data, sfreq=250, freq_range=(7, 14))
    >>> print(f"Peak alpha at {result.best_freq:.1f} Hz")
    
    >>> # Scan broad range for oscillatory peaks
    >>> result = narrowband_scan(data, sfreq=500, freq_range=(1, 45), n_peaks=5)
    >>> for f, ev in zip(result.peak_freqs, result.peak_eigenvalues):
    ...     print(f"Peak at {f:.1f} Hz, eigenvalue={ev:.3f}")

    Notes
    -----
    This is equivalent to NoiseTools' nt_narrowband_scan.m. The eigenvalue
    at each frequency indicates how much spatially coherent power exists
    in that band. Peaks indicate dominant oscillatory rhythms.
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

    nyquist = sfreq / 2
    min_freq, max_freq = freq_range
    
    # Validate frequency range
    if min_freq < 0.5:
        min_freq = 0.5
    if max_freq >= nyquist * 0.95:
        max_freq = nyquist * 0.9
    
    # Generate frequencies to scan
    frequencies = np.arange(min_freq, max_freq + freq_step, freq_step)
    n_freqs = len(frequencies)
    
    # Storage for results
    all_eigenvalues = np.zeros((n_freqs, n_components))
    best_idx = 0
    best_eigenvalue = -np.inf
    best_filters = None
    best_patterns = None
    
    for i, freq in enumerate(frequencies):
        # Design bandpass filter around this frequency
        low = max((freq - bandwidth / 2) / nyquist, 0.01)
        high = min((freq + bandwidth / 2) / nyquist, 0.99)
        
        if low >= high:
            continue
            
        try:
            sos = signal.butter(4, [low, high], btype='band', output='sos')
            biased_data = signal.sosfiltfilt(sos, data_2d, axis=1)
        except Exception:
            # Skip problematic frequencies
            continue
        
        # Compute DSS with this bandpass as bias
        try:
            filters, patterns, eigenvalues, _ = compute_dss(
                data_2d, biased_data,
                n_components=n_components,
                rank=rank,
                reg=reg,
            )
            
            all_eigenvalues[i, :len(eigenvalues)] = eigenvalues[:n_components]
            
            # Track best frequency
            if eigenvalues[0] > best_eigenvalue:
                best_eigenvalue = eigenvalues[0]
                best_idx = i
                best_filters = filters
                best_patterns = patterns
                
        except Exception:
            # Skip if DSS fails at this frequency
            continue
    
    # Find peak frequencies
    first_eigenvalues = all_eigenvalues[:, 0]
    peak_indices = _find_spectral_peaks(first_eigenvalues, n_peaks=n_peaks)
    
    peak_freqs = frequencies[peak_indices]
    peak_eigenvalues = first_eigenvalues[peak_indices]
    
    # Sort by eigenvalue (descending)
    sort_order = np.argsort(peak_eigenvalues)[::-1]
    peak_freqs = peak_freqs[sort_order]
    peak_eigenvalues = peak_eigenvalues[sort_order]
    
    return NarrowbandScanResult(
        frequencies=frequencies,
        eigenvalues=all_eigenvalues,
        peak_freqs=peak_freqs,
        peak_eigenvalues=peak_eigenvalues,
        best_freq=frequencies[best_idx],
        best_filters=best_filters if best_filters is not None else np.array([]),
        best_patterns=best_patterns if best_patterns is not None else np.array([]),
    )


def _find_spectral_peaks(
    values: np.ndarray,
    n_peaks: int = 3,
    min_distance: int = 2,
) -> np.ndarray:
    """Find peaks in eigenvalue spectrum."""
    from scipy.signal import find_peaks as scipy_find_peaks
    
    if len(values) < 3:
        return np.array([np.argmax(values)])
    
    # Find local maxima
    peaks, properties = scipy_find_peaks(values, distance=min_distance)
    
    if len(peaks) == 0:
        # No peaks found, return top n indices by value
        return np.argsort(values)[-n_peaks:][::-1]
    
    # Sort peaks by value and take top n
    peak_values = values[peaks]
    sorted_idx = np.argsort(peak_values)[::-1][:n_peaks]
    
    return peaks[sorted_idx]


def narrowband_dss(
    data: np.ndarray,
    sfreq: float,
    freq: float,
    *,
    bandwidth: float = 2.0,
    n_components: Optional[int] = None,
    rank: Optional[int] = None,
    reg: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply DSS with narrowband bias at a specific frequency.

    Convenience function for extracting oscillatory components at a
    known frequency of interest.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Input data.
    sfreq : float
        Sampling frequency in Hz.
    freq : float
        Target frequency in Hz.
    bandwidth : float
        Bandwidth of bandpass filter. Default 2.0 Hz.
    n_components : int, optional
        Number of components to return.
    rank : int, optional
        Rank for whitening.
    reg : float
        Regularization. Default 1e-9.

    Returns
    -------
    filters : ndarray, shape (n_components, n_channels)
        DSS spatial filters.
    patterns : ndarray, shape (n_channels, n_components)
        DSS spatial patterns.
    eigenvalues : ndarray, shape (n_components,)
        DSS eigenvalues.

    Examples
    --------
    >>> # Extract 10 Hz (alpha) components
    >>> filters, patterns, eigenvalues = narrowband_dss(data, sfreq=250, freq=10)
    >>> alpha_sources = filters @ data
    """
    data = np.asarray(data)
    if data.ndim == 3:
        data = data.reshape(data.shape[0], -1)
    
    nyquist = sfreq / 2
    low = max((freq - bandwidth / 2) / nyquist, 0.01)
    high = min((freq + bandwidth / 2) / nyquist, 0.99)
    
    if low >= high:
        raise ValueError(f"Invalid frequency {freq} Hz for sampling rate {sfreq} Hz")
    
    sos = signal.butter(4, [low, high], btype='band', output='sos')
    biased_data = signal.sosfiltfilt(sos, data, axis=1)
    
    filters, patterns, eigenvalues, _ = compute_dss(
        data, biased_data,
        n_components=n_components,
        rank=rank,
        reg=reg,
    )
    
    return filters, patterns, eigenvalues
