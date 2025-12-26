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

from .base import LinearDenoiser, NonlinearDenoiser


class BandpassBias(LinearDenoiser):
    """Bandpass filter bias for narrow-band rhythm extraction.

    Applies a bandpass filter to emphasize a specific frequency band,
    useful for extracting oscillatory sources (alpha, beta, etc.).

    References
    ----------
    Särelä & Valpola (2005). Section 4.1.2 "DENOISING BASED ON FREQUENCY CONTENT"

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
    >>> bias = BandpassBias(freq_band=(8, 12), sfreq=250)  # Alpha band
    >>> biased_data = bias.apply(raw_data)
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
            raise ValueError(
                f"High frequency ({high}) must be < Nyquist ({nyq})"
            )

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
        if self._sos is None:
            raise RuntimeError("Filter not designed")

        # Handle 3D epoched data
        if data.ndim == 3:
            n_channels, n_times, n_epochs = data.shape
            # Process each epoch separately to avoid edge effects between epochs
            biased = np.zeros_like(data)
            for ep in range(n_epochs):
                biased[:, :, ep] = signal.sosfiltfilt(
                    self._sos, data[:, :, ep], axis=1
                )
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

    References
    ----------
    de Cheveigné (2020). ZapLine
    Särelä & Valpola (2005). Section 4.1.2 "DENOISING BASED ON FREQUENCY CONTENT"

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
    >>> bias = NotchBias(freq=50, sfreq=250, bandwidth=2)  # Line noise
    >>> biased = bias.apply(data)  # Contains mostly 50 Hz component
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


class DCTDenoiser(NonlinearDenoiser):
    """DCT domain denoiser (MATLAB denoise_dct.m).

    Applies a mask in the DCT (Discrete Cosine Transform) domain.
    Useful for frequency-selective denoising without explicit bandpass.

    References
    ----------
    Särelä & Valpola (2005). Section 4.1.2 "DENOISING BASED ON FREQUENCY CONTENT"

    Parameters
    ----------
    mask : ndarray or None
        DCT domain mask. Must have same length as signal, or will be
        expanded/truncated. If None, creates lowpass mask.
        If mask is None, this fraction of DCT coefficients are kept.
        Default 0.5 (lowpass, keep first 50% of coefficients).
    """

    def __init__(
        self, 
        mask: Optional[np.ndarray] = None, 
        cutoff_fraction: float = 0.5
    ) -> None:
        self.mask = mask
        self.cutoff_fraction = cutoff_fraction
        self._cached_mask = None
        self._cached_len = None

    def denoise(self, source: np.ndarray) -> np.ndarray:
        """Apply DCT filtering."""
        from scipy.fftpack import dct, idct
        
        n = len(source)
        
        # Create or retrieve mask
        if self.mask is not None:
            if len(self.mask) == n:
                mask = self.mask
            else:
                # Resample mask to match signal length
                mask = np.interp(
                    np.linspace(0, 1, n),
                    np.linspace(0, 1, len(self.mask)),
                    self.mask
                )
        else:
            # Create lowpass mask if not cached or length changed
            if self._cached_mask is None or self._cached_len != n:
                cutoff = int(n * self.cutoff_fraction)
                mask = np.zeros(n)
                mask[:cutoff] = 1.0
                self._cached_mask = mask
                self._cached_len = n
            else:
                mask = self._cached_mask
        
        if source.ndim == 1:
            dct_coeffs = dct(source, type=2, norm='ortho')
            dct_filtered = dct_coeffs * mask
            return idct(dct_filtered, type=2, norm='ortho')
        elif source.ndim == 2:            
            n_times, n_epochs = source.shape
            denoised = np.zeros_like(source)
            # Recompute mask if needed for n_times
            if self._cached_mask is None or self._cached_len != n_times:
                 pass 
                 
            for ep in range(n_epochs):
                 denoised[:, ep] = self._denoise_1d(source[:, ep], mask)
            return denoised

        return source

    def _denoise_1d(self, source, mask):
        from scipy.fftpack import dct, idct
        dct_coeffs = dct(source, type=2, norm='ortho')
        dct_filtered = dct_coeffs * mask
        return idct(dct_filtered, type=2, norm='ortho')


class TemporalSmoothnessDenoiser(NonlinearDenoiser):
    """Nonlinear denoiser emphasizing temporally smooth sources.

    Promotes sources with high autocorrelation by penalizing
    rapid fluctuations. Useful for slow-wave or DC-shift artifacts.

    References
    ----------
    Särelä & Valpola (2005). Section 4.1.2 "DENOISING BASED ON FREQUENCY CONTENT"

    Parameters
    ----------
    smoothing_factor : float
        Weight for temporal smoothness penalty. Default 0.1.
    order : int
        Derivative order for smoothness measure. Default 1.

    Examples
    --------
    >>> denoiser = TemporalSmoothnessDenoiser(smoothing_factor=0.2)
    >>> smooth_source = denoiser.denoise(source)
    """

    def __init__(
        self,
        smoothing_factor: float = 0.1,
        order: int = 1,
    ) -> None:
        self.smoothing_factor = smoothing_factor
        self.order = max(1, order)

    def denoise(self, source: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing denoiser."""
        if source.ndim == 1:
            window = max(3, int(len(source) * self.smoothing_factor))
            weights = np.ones(window) / window
            smoothed = np.convolve(source, weights, mode='same')
            return (1 - self.smoothing_factor) * source + self.smoothing_factor * smoothed
        elif source.ndim == 2:
            n_times, n_trials = source.shape
            window = max(3, int(n_times * self.smoothing_factor))
            weights = np.ones(window) / window
            denoised = np.zeros_like(source)
            for t in range(n_trials):
                smoothed = np.convolve(source[:, t], weights, mode='same')
                denoised[:, t] = (1 - self.smoothing_factor) * source[:, t] + self.smoothing_factor * smoothed
            return denoised
        else:
            raise ValueError(f"Source must be 1D or 2D, got {source.ndim}D")
