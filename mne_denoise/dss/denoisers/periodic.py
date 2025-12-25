"""Periodic filter biases for DSS.

Implements peak and comb filter biases for extracting periodic/oscillatory
signals, particularly useful for SSVEP analysis.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np
from scipy import signal

from .base import LinearDenoiser


class PeakFilterBias(LinearDenoiser):
    """Peak filter bias for single-frequency extraction.

    Applies a narrow bandpass (peak) filter to emphasize activity at a
    specific frequency. More selective than BandpassBias, using a
    resonant filter design.

    References
    ----------
    Särelä & Valpola (2005). Denoising Source Separation. J. Mach. Learn. Res., 6, 233-272.
    Section 4.1.2 "DENOISING BASED ON FREQUENCY CONTENT":

    Parameters
    ----------
    freq : float
        Target frequency in Hz.
    sfreq : float
        Sampling frequency in Hz.
    q_factor : float
        Quality factor controlling bandwidth. Higher Q = narrower band.
        Default 30 (roughly 1 Hz bandwidth at 30 Hz).
    order : int
        Filter order. Default 2.

    Examples
    --------
    >>> # Extract 10 Hz alpha with tight filter
    >>> bias = PeakFilterBias(freq=10, sfreq=250, q_factor=20)
    >>> biased_data = bias.apply(data)

    Notes
    -----
    Uses a second-order IIR peak filter design. Q factor determines
    the bandwidth: bandwidth ≈ freq / Q.
    """

    def __init__(
        self,
        freq: float,
        sfreq: float,
        *,
        q_factor: float = 30.0,
        order: int = 2,
    ) -> None:
        self.freq = freq
        self.sfreq = sfreq
        self.q_factor = q_factor
        self.order = order

        # Design peak filter
        self._sos = self._design_peak_filter()

    def _design_peak_filter(self) -> np.ndarray:
        """Design IIR peak filter using second-order sections."""
        nyq = self.sfreq / 2

        if self.freq >= nyq:
            raise ValueError(
                f"Target frequency ({self.freq} Hz) must be < Nyquist ({nyq} Hz)"
            )

        # Normalized frequency
        w0 = self.freq / nyq

        # Bandwidth from Q factor
        bw = w0 / self.q_factor

        # Use iirpeak for resonant filter
        try:
            b, a = signal.iirpeak(w0, self.q_factor)
            sos = signal.tf2sos(b, a)
        except Exception:
            # Fallback to narrow bandpass
            low = max(w0 - bw / 2, 0.01)
            high = min(w0 + bw / 2, 0.99)
            sos = signal.butter(self.order, [low, high], btype='band', output='sos')

        return sos

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply peak filter bias.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
            Input data.

        Returns
        -------
        biased : ndarray, same shape as input
            Peak-filtered data.
        """
        if data.ndim == 3:
            n_channels, n_times, n_epochs = data.shape
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


class CombFilterBias(LinearDenoiser):
    """Comb filter bias for harmonic frequency extraction.

    Applies a comb filter that passes the fundamental frequency and its
    harmonics. Ideal for SSVEP analysis where stimulus frequency creates
    responses at multiple harmonic frequencies.

    Parameters
    ----------
    fundamental_freq : float
        Fundamental frequency in Hz (e.g., 15 Hz for SSVEP).
    sfreq : float
        Sampling frequency in Hz.
    n_harmonics : int
        Number of harmonics to include (1 = fundamental only).
        Default 3.
    q_factor : float
        Quality factor for each peak. Default 30.
    weights : array-like, optional
        Weights for each harmonic. If None, uses 1/harmonic_number
        weighting (decreasing importance of higher harmonics).

    Examples
    --------
    >>> # SSVEP at 15 Hz with 3 harmonics (15, 30, 45 Hz)
    >>> bias = CombFilterBias(fundamental_freq=15, sfreq=250, n_harmonics=3)
    >>> biased_data = bias.apply(data)

    >>> # Custom weighting (equal weight for all harmonics)
    >>> bias = CombFilterBias(
    ...     fundamental_freq=12, sfreq=500, n_harmonics=4,
    ...     weights=[1.0, 1.0, 1.0, 1.0]
    ... )

    Notes
    -----
    Implements a sum of peak filters at each harmonic. Harmonics above
    Nyquist frequency are automatically excluded.
    """

    def __init__(
        self,
        fundamental_freq: float,
        sfreq: float,
        *,
        n_harmonics: int = 3,
        q_factor: float = 30.0,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        self.fundamental_freq = fundamental_freq
        self.sfreq = sfreq
        self.n_harmonics = n_harmonics
        self.q_factor = q_factor
        
        # Set up weights
        if weights is None:
            self.weights = np.array([1.0 / h for h in range(1, n_harmonics + 1)])
        else:
            self.weights = np.asarray(weights)
            if len(self.weights) != n_harmonics:
                raise ValueError(
                    f"weights length ({len(self.weights)}) must match "
                    f"n_harmonics ({n_harmonics})"
                )

        # Create peak filters for each valid harmonic
        self._peak_filters: List[Tuple[np.ndarray, float]] = []
        self._create_harmonic_filters()

    def _create_harmonic_filters(self) -> None:
        """Create peak filter for each harmonic within Nyquist."""
        nyq = self.sfreq / 2

        for h in range(1, self.n_harmonics + 1):
            freq = self.fundamental_freq * h
            
            if freq >= nyq * 0.95:
                continue  # Skip harmonics too close to Nyquist

            w0 = freq / nyq
            weight = self.weights[h - 1]

            try:
                b, a = signal.iirpeak(w0, self.q_factor)
                sos = signal.tf2sos(b, a)
                self._peak_filters.append((sos, weight))
            except Exception:
                # Fallback to bandpass
                bw = w0 / self.q_factor
                low = max(w0 - bw / 2, 0.01)
                high = min(w0 + bw / 2, 0.99)
                try:
                    sos = signal.butter(2, [low, high], btype='band', output='sos')
                    self._peak_filters.append((sos, weight))
                except Exception:
                    continue

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply comb filter bias.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
            Input data.

        Returns
        -------
        biased : ndarray, same shape as input
            Comb-filtered data (sum of harmonics).
        """
        if len(self._peak_filters) == 0:
            raise ValueError("No valid harmonics within Nyquist frequency")

        biased = np.zeros_like(data)

        for sos, weight in self._peak_filters:
            if data.ndim == 3:
                n_channels, n_times, n_epochs = data.shape
                for ep in range(n_epochs):
                    biased[:, :, ep] += weight * signal.sosfiltfilt(
                        sos, data[:, :, ep], axis=1
                    )
            elif data.ndim == 2:
                biased += weight * signal.sosfiltfilt(sos, data, axis=1)
            else:
                raise ValueError(f"Data must be 2D or 3D, got {data.ndim}D")

        return biased

    @property
    def harmonic_frequencies(self) -> List[float]:
        """Return list of harmonic frequencies being filtered."""
        nyq = self.sfreq / 2
        return [
            self.fundamental_freq * h
            for h in range(1, self.n_harmonics + 1)
            if self.fundamental_freq * h < nyq * 0.95
        ]

