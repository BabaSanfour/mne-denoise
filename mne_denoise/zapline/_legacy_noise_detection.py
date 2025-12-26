"""
Frequency search utilities used by the Zapline-plus pipeline.

The logic closely mirrors the behaviour of the original MATLAB helper:
we slide a fixed-width window across the averaged spectrum, compare the
centre bin against its neighbourhood, and stop on the first location
that exceeds the configured thresholds.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _validate_inputs(pxx: np.ndarray, freqs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    data = np.asarray(pxx)
    freq = np.asarray(freqs, dtype=float)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    if data.ndim != 2:
        raise ValueError("pxx must be 1D or 2D array-like.")
    if freq.ndim != 1 or freq.size != data.shape[0]:
        raise ValueError("Frequency axis mismatch between pxx and freqs.")
    return data, freq


def find_next_noisefreq(
    pxx: np.ndarray,
    freqs: np.ndarray,
    minfreq: float = 0.0,
    threshdiff: float = 5.0,
    winsizeHz: float = 3.0,
    maxfreq: Optional[float] = None,
    lower_threshdiff: float = 1.76091259055681,
    verbose: bool = False,
) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """
    Identify the next prominent narrow-band peak in the averaged PSD.

    Parameters
    ----------
    pxx : array_like, shape (n_frequencies, n_channels)
        Power values (linear or dB). Channels are averaged internally.
    freqs : array_like, shape (n_frequencies,)
        Frequency grid corresponding to ``pxx``.
    minfreq : float
        Lower bound for the search window.
    threshdiff : float
        Primary threshold: the centre bin must sit this many units above the
        reference level to count as a detection.
    winsizeHz : float
        Width of the sliding window expressed in Hertz.
    maxfreq : float | None
        Upper bound for the search window. Defaults to 85% of max(freqs).
    lower_threshdiff : float
        Relaxed threshold applied once a candidate region has been entered.
    verbose : bool
        If True, prints simple progress information (useful for debugging).

    Returns
    -------
    noisefreq : float or None
        The detected frequency, or ``None`` if no peak exceeds the thresholds.
    window_freqs : ndarray or None
        Frequencies covered by the final evaluation window.
    window_power : ndarray or None
        Averaged power values inside the evaluation window.
    threshold_used : float or None
        Threshold that caused the detection.
    """
    spectrum, freq = _validate_inputs(pxx, freqs)
    mean_power = np.mean(spectrum, axis=1)

    if maxfreq is None:
        maxfreq = float(freq.max()) * 0.85

    mask = (freq >= minfreq) & (freq <= maxfreq)
    if not np.any(mask):
        return None, None, None, None

    freq = freq[mask]
    mean_power = mean_power[mask]

    step_hz = np.median(np.diff(freq))
    if step_hz <= 0:
        return None, None, None, None

    half_bins = int(round((winsizeHz / step_hz) / 2))
    window_size = max(3, 2 * half_bins + 1)

    start_idx = half_bins
    end_idx = len(freq) - half_bins
    detection_started = False
    detection_start_idx = start_idx
    detection_end_idx = start_idx
    last_print = -np.inf

    for centre in range(start_idx, end_idx):
        slice_start = centre - half_bins
        slice_end = centre + half_bins + 1
        window_power = mean_power[slice_start:slice_end]
        window_freq = freq[slice_start:slice_end]

        leading = window_power[:half_bins]
        trailing = window_power[-half_bins:]
        reference_level = float(np.mean(np.concatenate([leading, trailing])))
        primary_threshold = reference_level + threshdiff
        relaxed_threshold = reference_level + lower_threshdiff

        centre_value = float(window_power[half_bins])
        current_freq = float(window_freq[half_bins])

        if verbose and current_freq - last_print >= 1.0:
            print(f"{current_freq:.0f} Hz", end=" ")
            last_print = current_freq

        if not detection_started:
            if centre_value > primary_threshold:
                detection_started = True
                detection_start_idx = centre
                detection_end_idx = centre
                threshold_used = primary_threshold
            continue

        # Detection already started; track the end index as long as we stay over the relaxed threshold
        if centre_value > relaxed_threshold:
            detection_end_idx = centre
            continue

        # The streak ended -> pick the largest peak inside the detected window
        slice_power = mean_power[detection_start_idx:detection_end_idx + 1]
        slice_freq = freq[detection_start_idx:detection_end_idx + 1]
        if slice_power.size == 0:
            detection_started = False
            continue

        max_idx = int(np.argmax(slice_power))
        detected_freq = float(slice_freq[max_idx])
        if verbose:
            print(f"\nDetected candidate at {detected_freq:.2f} Hz")
        return detected_freq, window_freq, window_power, threshold_used

    if detection_started:
        slice_power = mean_power[detection_start_idx:detection_end_idx + 1]
        slice_freq = freq[detection_start_idx:detection_end_idx + 1]
        if slice_power.size:
            max_idx = int(np.argmax(slice_power))
            detected_freq = float(slice_freq[max_idx])
            if verbose:
                print(f"\nDetected candidate at {detected_freq:.2f} Hz (window ended)")
            return detected_freq, freq[detection_start_idx:detection_end_idx + 1], slice_power, threshold_used

    if verbose and not detection_started:
        print("\nNo noise peaks found in the requested band.")
    return None, freq, mean_power, None
