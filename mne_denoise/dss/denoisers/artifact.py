"""Artifact-based bias functions for DSS.

Implements cycle averaging for quasi-periodic artifacts like ECG and blinks.
This emphasizes reproducible artifact morphology while canceling neural activity.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy import signal

from .base import LinearDenoiser


class CycleAverageBias(LinearDenoiser):
    """Bias function for quasi-periodic artifact extraction.

    Applies cycle averaging synchronized to artifact events (e.g., R-peaks
    for ECG, blink onsets for EOG). This emphasizes the stereotyped
    artifact waveform while canceling non-phase-locked neural activity.

    Parameters
    ----------
    event_samples : array-like
        Sample indices of artifact events (e.g., R-peak locations).
    window : tuple of int
        (pre, post) samples around each event to include.
        Default (-100, 200) for ~300ms window at 1kHz.
    sfreq : float, optional
        Sampling frequency for window specification in seconds.
        If provided, window can be in seconds instead of samples.

    Examples
    --------
    >>> # ECG artifact removal
    >>> r_peaks = find_r_peaks(ecg_channel)  # Get R-peak locations
    >>> bias = CycleAverageBias(event_samples=r_peaks, window=(-0.1, 0.3), sfreq=1000)
    >>> biased = bias.apply(eeg_data)  # Shape (n_channels, n_times)
    """

    def __init__(
        self,
        event_samples: Sequence[int],
        window: tuple[int, int] = (-100, 200),
        *,
        sfreq: Optional[float] = None,
    ) -> None:
        self.event_samples = np.asarray(event_samples, dtype=int)
        
        # Convert window to samples if sfreq provided
        if sfreq is not None:
            self.window = (int(window[0] * sfreq), int(window[1] * sfreq))
        else:
            self.window = (int(window[0]), int(window[1]))
        
        self.sfreq = sfreq
        self._window_length = self.window[1] - self.window[0]

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply cycle averaging bias.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
            Input data.

        Returns
        -------
        biased : ndarray, same shape as input
            Data where artifact-locked segments are replaced by cycle average.
        """
        original_shape = data.shape
        
        # Handle 3D epoched data by concatenating
        if data.ndim == 3:
            n_channels, n_times, n_epochs = data.shape
            # Adjust events for concatenated epochs
            data_2d = data.reshape(n_channels, -1)
            total_samples = n_times * n_epochs
        elif data.ndim == 2:
            data_2d = data
            n_channels, total_samples = data.shape
            n_times = total_samples
            n_epochs = 1
        else:
            raise ValueError(f"Data must be 2D or 3D, got {data.ndim}D")

        # Filter valid events (within data bounds)
        pre, post = self.window
        valid_mask = (
            (self.event_samples + pre >= 0) &
            (self.event_samples + post <= total_samples)
        )
        valid_events = self.event_samples[valid_mask]

        if len(valid_events) == 0:
            # No valid events, return zeros (no artifact signal)
            if data.ndim == 3:
                return np.zeros_like(data)
            return np.zeros_like(data_2d)

        # Compute cycle average
        window_len = post - pre
        epochs_matrix = np.zeros((len(valid_events), n_channels, window_len))
        
        for i, event in enumerate(valid_events):
            start = event + pre
            end = event + post
            epochs_matrix[i] = data_2d[:, start:end]

        # Average across artifact cycles
        cycle_average = np.mean(epochs_matrix, axis=0)  # (n_channels, window_len)

        # Create biased output: each artifact window gets the average
        biased_2d = np.zeros_like(data_2d)
        
        for event in valid_events:
            start = event + pre
            end = event + post
            biased_2d[:, start:end] = cycle_average

        # Reshape back if needed
        if len(original_shape) == 3:
            biased = biased_2d.reshape(original_shape)
        else:
            biased = biased_2d

        return biased


def find_ecg_events(
    ecg_signal: np.ndarray,
    sfreq: float,
    *,
    threshold_factor: float = 0.6,
    min_distance_seconds: float = 0.4,
) -> np.ndarray:
    """Detect R-peaks in an ECG signal.

    Simple peak detection for ECG artifact extraction. For more robust
    detection, consider using MNE's find_ecg_events or dedicated
    ECG processing libraries.

    Parameters
    ----------
    ecg_signal : ndarray, shape (n_times,)
        Single-channel ECG signal.
    sfreq : float
        Sampling frequency in Hz.
    threshold_factor : float
        Fraction of max amplitude for peak threshold. Default 0.6.
    min_distance_seconds : float
        Minimum distance between peaks in seconds. Default 0.4.

    Returns
    -------
    r_peaks : ndarray
        Sample indices of detected R-peaks.
    """
    # Bandpass filter to isolate QRS complex (5-15 Hz)
    nyq = sfreq / 2
    low = 5 / nyq
    high = min(15 / nyq, 0.99)
    
    if low >= high:
        # Can't filter, use raw signal
        filtered = np.abs(ecg_signal)
    else:
        sos = signal.butter(4, [low, high], btype='band', output='sos')
        filtered = np.abs(signal.sosfiltfilt(sos, ecg_signal))

    # Find peaks
    min_distance = int(min_distance_seconds * sfreq)
    threshold = threshold_factor * np.max(filtered)
    
    peaks, _ = signal.find_peaks(
        filtered,
        height=threshold,
        distance=max(1, min_distance),
    )
    
    return peaks


def find_eog_events(
    eog_signal: np.ndarray,
    sfreq: float,
    *,
    threshold_factor: float = 0.5,
    min_distance_seconds: float = 0.5,
) -> np.ndarray:
    """Detect blink events in an EOG signal.

    Simple peak detection for EOG artifact extraction. For more robust
    detection, consider using MNE's find_eog_events.

    Parameters
    ----------
    eog_signal : ndarray, shape (n_times,)
        Single-channel vertical EOG signal.
    sfreq : float
        Sampling frequency in Hz.
    threshold_factor : float
        Fraction of max amplitude for peak threshold. Default 0.5.
    min_distance_seconds : float
        Minimum distance between peaks in seconds. Default 0.5.

    Returns
    -------
    blink_events : ndarray
        Sample indices of detected blink peaks.
    """
    # Low-pass filter to isolate blink component (< 5 Hz)
    nyq = sfreq / 2
    cutoff = min(5 / nyq, 0.99)
    
    sos = signal.butter(4, cutoff, btype='low', output='sos')
    filtered = signal.sosfiltfilt(sos, eog_signal)

    # Find positive peaks (upward deflection for blinks)
    min_distance = int(min_distance_seconds * sfreq)
    threshold = threshold_factor * np.max(filtered)
    
    peaks, _ = signal.find_peaks(
        filtered,
        height=threshold,
        distance=max(1, min_distance),
    )
    
    return peaks
