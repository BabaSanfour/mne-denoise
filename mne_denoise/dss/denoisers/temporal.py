"""Temporal bias functions for DSS.

Implements time-shift and smoothing biases for extracting temporally
extended structure (slow waves, autocorrelated signals).

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

References
----------
.. [1] de Cheveigné, A. (2010). Time-shift denoising source separation.
       Journal of Neuroscience Methods, 189(1), 113-120.
.. [2] de Cheveigné, A. & Simon, J.Z. (2008). Denoising based on spatial filtering.
       Journal of Neuroscience Methods, 171(2), 331-339.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from .base import LinearDenoiser


class TimeShiftBias(LinearDenoiser):
    """Time-shift bias for extracting autocorrelated signals.

    Creates a bias by averaging time-shifted versions of the data,
    emphasizing signals that are predictable across time lags.

    Parameters
    ----------
    shifts : int or array-like
        If int, use lags from 1 to shifts.
        If array, use specified lag values in samples.
        Default 10.
    method : str
        Method for constructing bias:
        - 'autocorrelation': Average of shifted versions (default)
        - 'prediction': Weighted average (closer lags weighted more)

    Examples
    --------
    >>> bias = TimeShiftBias(shifts=10)
    >>> dss = DSS(bias=bias)
    >>> dss.fit(data)
    """

    def __init__(
        self,
        shifts: Union[int, np.ndarray] = 10,
        method: str = "autocorrelation",
    ) -> None:
        self.shifts = shifts
        self.method = method

        # Resolve shifts to array
        if isinstance(shifts, int):
            self._shift_array = np.arange(1, shifts + 1)
        else:
            self._shift_array = np.asarray(shifts)

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply time-shift bias.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
            Input data.

        Returns
        -------
        biased : ndarray, same shape as input
            Time-shifted averaged data.
        """
        # Handle 3D data
        orig_shape = data.shape
        if data.ndim == 3:
            n_ch, n_times, n_epochs = data.shape
            data_2d = data.reshape(n_ch, -1)
        else:
            data_2d = data

        if self.method == "autocorrelation":
            biased_2d = self._autocorrelation_bias(data_2d)
        elif self.method == "prediction":
            biased_2d = self._prediction_bias(data_2d)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Restore shape
        if data.ndim == 3:
            return biased_2d.reshape(orig_shape)
        return biased_2d

    def _autocorrelation_bias(self, data: np.ndarray) -> np.ndarray:
        """Average of time-shifted versions."""
        n_channels, n_samples = data.shape
        shifts = self._shift_array
        max_shift = np.max(np.abs(shifts))

        if max_shift >= n_samples // 2:
            raise ValueError(f"Max shift ({max_shift}) too large for data length ({n_samples})")

        valid_start = max_shift
        valid_end = n_samples - max_shift
        valid_length = valid_end - valid_start

        accumulated = np.zeros((n_channels, valid_length))
        for shift in shifts:
            shifted = data[:, valid_start + shift : valid_end + shift]
            accumulated += shifted

        biased = accumulated / len(shifts)

        # Pad to original length
        biased_full = np.zeros_like(data)
        biased_full[:, valid_start:valid_end] = biased
        return biased_full

    def _prediction_bias(self, data: np.ndarray) -> np.ndarray:
        """Weighted average (closer lags weighted more)."""
        n_channels, n_samples = data.shape
        shifts = self._shift_array
        max_shift = np.max(np.abs(shifts))

        valid_start = max_shift
        valid_end = n_samples - max_shift
        valid_length = valid_end - valid_start

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


class SmoothingBias(LinearDenoiser):
    """Temporal smoothing bias for extracting slow signals.

    Applies a moving average filter to create a bias that emphasizes
    low-frequency temporal structure.

    Parameters
    ----------
    window : int
        Smoothing window size in samples. Default 10.

    Examples
    --------
    >>> bias = SmoothingBias(window=20)
    >>> dss = DSS(bias=bias)
    >>> dss.fit(data)
    """

    def __init__(self, window: int = 10) -> None:
        self.window = window

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing bias.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
            Input data.

        Returns
        -------
        biased : ndarray, same shape as input
            Smoothed data.
        """
        from scipy.ndimage import uniform_filter1d

        orig_shape = data.shape
        if data.ndim == 3:
            data_2d = data.reshape(data.shape[0], -1)
        else:
            data_2d = data

        smoothed = uniform_filter1d(data_2d, size=self.window, axis=1, mode="reflect")

        if data.ndim == 3:
            return smoothed.reshape(orig_shape)
        return smoothed
