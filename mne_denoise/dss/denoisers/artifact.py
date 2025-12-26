"""Artifact-based bias functions for DSS.

Implements cycle averaging for quasi-periodic artifacts like ECG and blinks.
This emphasizes reproducible artifact morphology while canceling neural activity.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

References
----------
.. [1] Särelä & Valpola (2005). Denoising Source Separation. J. Mach. Learn. Res., 6, 233-272.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


from .base import LinearDenoiser


class CycleAverageBias(LinearDenoiser):
    """Bias for removing quasi-periodic artifacts (e.g., ECG, EOG).

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
    >>> from mne.preprocessing import find_ecg_events
    >>> from mne_denoise.dss.denoisers import CycleAverageBias
    >>> r_peaks, _ = find_ecg_events(raw) # MNE returns events array
    >>> # Extract sample indices (column 0)
    >>> r_peak_samples = r_peaks[:, 0]
    >>> bias = CycleAverageBias(event_samples=r_peak_samples, window=(-100, 200))
    >>> biased_data = bias.apply(raw.get_data())

    >>> # EOG (blink) artifact removal
    >>> from mne.preprocessing import find_eog_events
    >>> blinks = find_eog_events(raw)
    >>> blink_samples = blinks[:, 0]
    >>> bias_eog = CycleAverageBias(event_samples=blink_samples, window=(-200, 200))
    >>> biased_eog = bias_eog.apply(raw.get_data())

    References
    ----------
    Särelä & Valpola (2005). Section 4.1.4 "DENOISING OF QUASIPERIODIC SIGNALS"
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
