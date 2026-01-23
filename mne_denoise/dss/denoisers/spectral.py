"""Spectral bias functions for DSS.

Implements bandpass filters and unified line noise removal (Notch/FFT).

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

References
----------
.. [1] Särelä & Valpola (2005). Denoising Source Separation. J. Mach. Learn. Res., 6, 233-272.
.. [2] de Cheveigné, A. (2020). ZapLine: A simple and effective method to remove
       power line artifacts. NeuroImage, 207, 116356.
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft

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
        freq_band: tuple[float, float],
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
        self._b: np.ndarray | None = None
        self._a: np.ndarray | None = None
        self._sos: np.ndarray | None = None
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


class LineNoiseBias(LinearDenoiser):
    """A bias LinearDenoiser for line noise isolation (Notch/IIR or FFT/Harmonic).

    Isolates power at a specific frequency (e.g., 50/60 Hz) and potentially
    its harmonics. Supports two methods:
    1. 'fft': Use FFT masking to isolate exact frequency bins (ZapLine style).
       Best for sharp line noise with harmonics.
    2. 'iir': Use a narrow bandpass (notch) filter.
       Simpler, but affects broader band.

    Parameters
    ----------
    freq : float
        Line frequency in Hz.
    sfreq : float
        Sampling frequency in Hz.
    method : {'fft', 'iir'}
        Method to use. Default 'fft'.
    n_harmonics : int, optional
        Number of harmonics (for 'fft' method). If None, all up to Nyquist.
    bandwidth : float, optional
        Bandwidth in Hz (for 'iir' method). Default 1.0.
    order : int, optional
        Filter order (for 'iir' method). Default 4.
    nfft : int, optional
        FFT window size (for 'fft' method). Default 1024.
    overlap : float, optional
        Overlap fraction (for 'fft' method). Default 0.5.

    Examples
    --------
    >>> bias = LineNoiseBias(freq=50, sfreq=1000, method="fft")
    >>> biased = bias.apply(data)
    """

    def __init__(
        self,
        freq: float,
        sfreq: float,
        *,
        method: str = "fft",
        n_harmonics: int | None = None,
        bandwidth: float = 1.0,
        order: int = 4,
        nfft: int = 1024,
        overlap: float = 0.5,
    ) -> None:
        self.freq = freq
        self.sfreq = sfreq
        self.method = method
        self.n_harmonics = n_harmonics
        self.bandwidth = bandwidth
        self.order = order
        self.nfft = nfft
        self.overlap = overlap

        if method == "iir":
            low = freq - bandwidth / 2
            high = freq + bandwidth / 2
            self._bandpass = BandpassBias(
                freq_band=(low, high), sfreq=sfreq, order=order
            )
        elif method == "fft":
            # FFT setup logic
            nyquist = sfreq / 2
            if n_harmonics is None:
                self.n_harmonics = int(np.floor(nyquist / freq))
            else:
                max_harmonics = int(np.floor(nyquist / freq))
                self.n_harmonics = min(n_harmonics, max_harmonics)

            self._harmonic_freqs = np.array(
                [freq * (h + 1) for h in range(self.n_harmonics)]
            )
            self._harmonic_freqs = self._harmonic_freqs[self._harmonic_freqs < nyquist]
        else:
            raise ValueError(f"Unknown method '{method}', must be 'fft' or 'iir'.")

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply the selected bias."""
        if self.method == "iir":
            return self._bandpass.apply(data)
        elif self.method == "fft":
            return self._apply_fft(data)
        return data

    def _apply_fft(self, data: np.ndarray) -> np.ndarray:
        """Apply FFT-based harmonic bias."""
        if data.ndim == 3:
            n_channels, n_times, n_epochs = data.shape
            biased = np.zeros_like(data)
            for ep in range(n_epochs):
                biased[:, :, ep] = self._apply_fft_2d(data[:, :, ep])
            return biased
        elif data.ndim == 2:
            return self._apply_fft_2d(data)
        else:
            raise ValueError(f"Data must be 2D or 3D, got {data.ndim}D")

    def _get_target_indices(self, nfft: int) -> list:
        """Get FFT bin indices for target frequencies."""
        freq_bins = np.fft.fftfreq(nfft, 1 / self.sfreq)
        target_indices = []

        for f in self._harmonic_freqs:
            idx = np.argmin(np.abs(freq_bins - f))
            if idx not in target_indices:
                target_indices.append(idx)
            # Negative frequency
            idx_neg = np.argmin(np.abs(freq_bins + f))
            if idx_neg not in target_indices:
                target_indices.append(idx_neg)

        return target_indices

    def _apply_fft_2d(self, data: np.ndarray) -> np.ndarray:
        """Apply bias to 2D data using FFT."""
        n_channels, n_times = data.shape

        # Use data length or nfft, whichever is smaller
        actual_nfft = min(self.nfft, n_times)
        target_indices = self._get_target_indices(actual_nfft)

        # If data is shorter than nfft, process as single block
        if n_times <= actual_nfft:
            X = fft(data, n=actual_nfft, axis=1)
            X_bias = np.zeros_like(X)
            for idx in target_indices:
                X_bias[:, idx] = X[:, idx]
            biased = np.real(ifft(X_bias, axis=1))[:, :n_times]
            return biased

        # Welch-style block processing
        step = int(actual_nfft * (1 - self.overlap))
        step = max(step, 1)

        biased = np.zeros_like(data)
        counts = np.zeros(n_times)

        for start in range(0, n_times - actual_nfft + 1, step):
            end = start + actual_nfft
            segment = data[:, start:end]

            X = fft(segment, axis=1)
            X_bias = np.zeros_like(X)
            for idx in target_indices:
                X_bias[:, idx] = X[:, idx]

            segment_biased = np.real(ifft(X_bias, axis=1))
            biased[:, start:end] += segment_biased
            counts[start:end] += 1

        # Normalize by overlap counts
        counts = np.maximum(counts, 1)
        biased /= counts

        return biased
