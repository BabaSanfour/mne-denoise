"""Spectral bias functions for DSS.

Implements bandpass and notch filters for narrow-band rhythm extraction
and line noise isolation.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

References
----------
.. [1] Särelä & Valpola (2005). Denoising Source Separation. J. Mach. Learn. Res., 6, 233-272.
.. [2] de Cheveigné, A. (2020). ZapLine: A simple and effective method to remove
       power line artifacts. NeuroImage, 207, 116356.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import signal

from .base import LinearDenoiser


class BandpassBias(LinearDenoiser):
    """Bandpass filter bias for narrow-band rhythm extraction.

    Applies a bandpass filter to emphasize a specific frequency band,
    useful for extracting oscillatory sources (alpha, beta, etc.).


    Parameters
    ----------
    freq_band : tuple of float
        (low_freq, high_freq) defining the passband in Hz.
    sfreq : float
        Sampling frequency in Hz.
    order : int
        Filter order. Default 4.
    method : str
        Filter design method: 'butter' or 'fir'. Default 'butter'.

    Examples
    --------
    >>> from mne_denoise.dss.denoisers import BandpassBias
    >>> bias = BandpassBias(freq_band=(8, 12), sfreq=250)  # Alpha band
    >>> dss.fit(data)

    See Also
    --------
    mne_denoise.dss.denoisers.PeakFilterBias : For strictly periodic signals.

    References
    ----------
    Särelä & Valpola (2005). Section 4.1.2 "DENOISING BASED ON FREQUENCY CONTENT"
    """

    def __init__(
        self,
        freq_band: Tuple[float, float],
        sfreq: float,
        *,
        order: int = 4,
        method: str = "butter",
    ) -> None:
        self.freq_band = freq_band
        self.sfreq = sfreq
        self.order = order
        self.method = method

        # Pre-compute filter coefficients
        self._b: Optional[np.ndarray] = None
        self._a: Optional[np.ndarray] = None
        self._sos: Optional[np.ndarray] = None
        self._design_filter()

    def _design_filter(self) -> None:
        """Design the bandpass filter."""
        low, high = self.freq_band
        nyq = self.sfreq / 2

        if low <= 0:
            raise ValueError(f"Low frequency must be > 0, got {low}")
        if high >= nyq:
            raise ValueError(f"High frequency ({high}) must be < Nyquist ({nyq})")

        if self.method == "butter":
            # Use second-order sections for stability
            self._sos = signal.butter(
                self.order,
                [low / nyq, high / nyq],
                btype="band",
                output="sos",
            )
        else:
            raise ValueError(f"Unknown filter method: {self.method}")

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply bandpass filter bias.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
            Input data.

        Returns
        -------
        biased : ndarray, same shape as input
            Bandpass filtered data.
        """
        # Handle 3D epoched data
        if data.ndim == 3:
            n_channels, n_times, n_epochs = data.shape
            # Process each epoch separately to avoid edge effects between epochs
            biased = np.zeros_like(data)
            for ep in range(n_epochs):
                biased[:, :, ep] = signal.sosfiltfilt(self._sos, data[:, :, ep], axis=1)
        elif data.ndim == 2:
            biased = signal.sosfiltfilt(self._sos, data, axis=1)
        else:
            raise ValueError(f"Data must be 2D or 3D, got {data.ndim}D")

        return biased


class NotchBias(LinearDenoiser):
    """Notch filter bias for isolating a specific frequency.

    Applies a narrow notch (bandpass) filter to isolate power at a
    specific frequency. This is the core bias operation used in the
    ZapLine algorithm (de Cheveigné, 2020) to find and remove line noise.

    Parameters
    ----------
    freq : float
        Center frequency to isolate in Hz.
    sfreq : float
        Sampling frequency in Hz.
    bandwidth : float
        Filter bandwidth in Hz. Default 1.0.

    Examples
    --------
    >>> from mne_denoise.dss.denoisers import NotchBias
    >>> bias = NotchBias(freq=60, sfreq=250, bandwidth=2)
    >>> biased_data = bias.apply(data)

    See Also
    --------
    BandpassBias : Band-pass filter bias.

    References
    ----------
    de Cheveigné (2020). ZapLine
    Särelä & Valpola (2005). Section 4.1.2 "DENOISING BASED ON FREQUENCY CONTENT"
    """

    def __init__(
        self,
        freq: float,
        sfreq: float,
        *,
        bandwidth: float = 1.0,
    ) -> None:
        self.freq = freq
        self.sfreq = sfreq
        self.bandwidth = bandwidth

        # Create a narrow bandpass around the target frequency
        low = freq - bandwidth / 2
        high = freq + bandwidth / 2
        self._bandpass = BandpassBias(
            freq_band=(low, high),
            sfreq=sfreq,
            order=4,
        )

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply notch isolation bias.

        Parameters
        ----------
        data : ndarray
            Input data.

        Returns
        -------
        biased : ndarray
            Data filtered to isolate the target frequency.
        """
        return self._bandpass.apply(data)
