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

from typing import Optional, Tuple, Union

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
    HarmonicFFTBias : FFT-based harmonic bias (preferred for line noise).

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


class HarmonicFFTBias(LinearDenoiser):
    """FFT-based harmonic bias for precise line noise isolation.

    Uses FFT to isolate exact frequency bins at a fundamental frequency
    and its harmonics. This is the preferred bias for line noise removal
    as used in the ZapLine algorithm (de Cheveigné, 2020).

    Compared to filter-based approaches (NotchBias), this method:
    - Uses exact frequency bins (no filter roll-off)
    - Handles multiple harmonics naturally
    - Supports Welch-style averaging over blocks

    Parameters
    ----------
    line_freq : float
        Fundamental frequency in Hz (e.g., 50 or 60 for line noise).
    sfreq : float
        Sampling frequency in Hz.
    n_harmonics : int or None
        Number of harmonics to include. If None, include all up to Nyquist.
    nfft : int
        FFT size (window length). Default 1024.
    overlap : float
        Overlap fraction between blocks (0 to 1). Default 0.5.

    Examples
    --------
    >>> from mne_denoise.dss.denoisers import HarmonicFFTBias
    >>> bias = HarmonicFFTBias(line_freq=50, sfreq=1000, n_harmonics=5)
    >>> biased_data = bias.apply(data)

    >>> # For ZapLine-style direct covariance computation:
    >>> c0, c1 = bias.compute_covariances(data)

    See Also
    --------
    NotchBias : Filter-based single frequency bias.
    BandpassBias : General bandpass filter bias.

    References
    ----------
    de Cheveigné (2020). ZapLine: A simple and effective method to remove
        power line artifacts. NeuroImage, 207, 116356.
    """

    def __init__(
        self,
        line_freq: float,
        sfreq: float,
        *,
        n_harmonics: Optional[int] = None,
        nfft: int = 1024,
        overlap: float = 0.5,
    ) -> None:
        if line_freq <= 0:
            raise ValueError(f"line_freq must be positive, got {line_freq}")
        
        self.line_freq = line_freq
        self.sfreq = sfreq
        self.nfft = nfft
        self.overlap = overlap
        
        # Compute number of harmonics
        nyquist = sfreq / 2
        if n_harmonics is None:
            self.n_harmonics = int(np.floor(nyquist / line_freq))
        else:
            max_harmonics = int(np.floor(nyquist / line_freq))
            self.n_harmonics = min(n_harmonics, max_harmonics)
        
        # Pre-compute harmonic frequencies
        self._harmonic_freqs = np.array(
            [line_freq * (h + 1) for h in range(self.n_harmonics)]
        )
        self._harmonic_freqs = self._harmonic_freqs[self._harmonic_freqs < nyquist]

    def _get_target_indices(self, nfft: int) -> list:
        """Get FFT bin indices for target frequencies."""
        freq_bins = np.fft.fftfreq(nfft, 1 / self.sfreq)
        target_indices = []
        
        for f in self._harmonic_freqs:
            # Positive frequency
            idx = np.argmin(np.abs(freq_bins - f))
            if idx not in target_indices:
                target_indices.append(idx)
            # Negative frequency (for real signal symmetry)
            idx_neg = np.argmin(np.abs(freq_bins + f))
            if idx_neg not in target_indices:
                target_indices.append(idx_neg)
        
        return target_indices

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply FFT-based harmonic bias.

        Isolates power at harmonic frequencies by zeroing out all
        non-target FFT bins.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
            Input data.

        Returns
        -------
        biased : ndarray, same shape as input
            Data with only harmonic frequencies retained.
        """
        if data.ndim == 3:
            # Handle 3D epoched data
            n_channels, n_times, n_epochs = data.shape
            biased = np.zeros_like(data)
            for ep in range(n_epochs):
                biased[:, :, ep] = self._apply_2d(data[:, :, ep])
            return biased
        elif data.ndim == 2:
            return self._apply_2d(data)
        else:
            raise ValueError(f"Data must be 2D or 3D, got {data.ndim}D")

    def _apply_2d(self, data: np.ndarray) -> np.ndarray:
        """Apply bias to 2D data."""
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
        
        # Welch-style block processing with averaging
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

    def compute_covariances(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute baseline and biased covariance matrices.

        This is the ZapLine-style direct covariance computation that
        avoids computing biased data explicitly. More efficient for
        the ZapLine algorithm.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times)
            Input data (2D only).

        Returns
        -------
        c0 : ndarray, shape (n_channels, n_channels)
            Baseline covariance (all frequencies).
        c1 : ndarray, shape (n_channels, n_channels)
            Biased covariance (target frequencies only).
        """
        if data.ndim != 2:
            raise ValueError(
                f"compute_covariances requires 2D data, got {data.ndim}D. "
                "For 3D data, concatenate epochs first."
            )
        
        n_channels, n_times = data.shape
        actual_nfft = min(self.nfft, n_times)
        
        # Handle short data
        if n_times < actual_nfft:
            actual_nfft = n_times
            step = actual_nfft
        else:
            step = int(actual_nfft * (1 - self.overlap))
            step = max(step, 1)
        
        target_indices = self._get_target_indices(actual_nfft)
        
        c0 = np.zeros((n_channels, n_channels))
        c1 = np.zeros((n_channels, n_channels))
        n_blocks = 0
        
        # Accumulate over blocks
        for start in range(0, n_times - actual_nfft + 1, step):
            end = start + actual_nfft
            segment = data[:, start:end]
            
            X = fft(segment, axis=1)
            
            # Baseline: covariance of all frequencies
            c0 += np.real(X @ X.conj().T)
            
            # Biased: covariance of target frequencies only
            X_bias = np.zeros_like(X)
            for idx in target_indices:
                X_bias[:, idx] = X[:, idx]
            
            c1 += np.real(X_bias @ X_bias.conj().T)
            n_blocks += 1
        
        if n_blocks > 0:
            c0 /= n_blocks
            c1 /= n_blocks
            # Normalize by nfft
            c0 /= actual_nfft
            c1 /= actual_nfft
        
        return c0, c1
