"""Narrowband DSS variant.

Frequency-targeted DSS for extracting oscillatory components.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..denoisers.spectral import BandpassBias
from ..linear import DSS


def narrowband_dss(
    sfreq: float,
    freq: float,
    *,
    bandwidth: float = 2.0,
    n_components: Optional[int] = None,
    **dss_kws,
) -> DSS:
    """Create a DSS configured for a specific frequency band.

    Returns a pre-configured DSS object that extracts components with
    maximum power in the specified frequency band.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz.
    freq : float
        Target center frequency in Hz.
    bandwidth : float
        Bandwidth of the bandpass filter in Hz. Default 2.0.
    n_components : int, optional
        Number of DSS components to keep. If None, keep all.
    **dss_kws
        Additional keyword arguments passed to `DSS`.

    Returns
    -------
    dss : DSS
        A DSS object configured with a BandpassBias.

    Examples
    --------
    >>> # Extract 10 Hz (alpha) components
    >>> dss = narrowband_dss(sfreq=250, freq=10)
    >>> dss.fit(data)
    >>> alpha_sources = dss.transform(data)
    """
    low = freq - bandwidth / 2
    high = freq + bandwidth / 2
    bias = BandpassBias(freq_band=(low, high), sfreq=sfreq)
    return DSS(bias=bias, n_components=n_components, **dss_kws)


def narrowband_scan(
    data: np.ndarray,
    sfreq: float,
    *,
    freq_range: Tuple[float, float] = (1, 40),
    freq_step: float = 1.0,
    bandwidth: float = 2.0,
    n_components: int = 1,
    **dss_kws,
) -> Tuple[DSS, np.ndarray, np.ndarray]:
    """Scan frequencies to find optimal narrowband DSS components.

    Sweeps through a frequency range, computing DSS at each frequency.
    Returns the fitted DSS at the best frequency, along with the full
    eigenvalue spectrum for visualization.

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
    **dss_kws
        Additional keyword arguments passed to `DSS`.

    Returns
    -------
    best_dss : DSS
        Fitted DSS at the frequency with highest eigenvalue.
    frequencies : ndarray, shape (n_freqs,)
        Frequencies that were scanned.
    eigenvalues : ndarray, shape (n_freqs,)
        First eigenvalue at each frequency.

    Examples
    --------
    >>> # Find dominant alpha frequency
    >>> best_dss, freqs, eigs = narrowband_scan(data, sfreq=250, freq_range=(7, 14))
    >>> print(f"Peak alpha at {freqs[np.argmax(eigs)]:.1f} Hz")
    >>> alpha_sources = best_dss.transform(data)

    >>> # Plot eigenvalue spectrum
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(freqs, eigs)
    >>> plt.xlabel('Frequency (Hz)')
    >>> plt.ylabel('DSS Eigenvalue')
    """
    data = np.asarray(data)

    nyquist = sfreq / 2
    min_freq, max_freq = freq_range

    # Validate frequency range
    min_freq = max(min_freq, 0.5)
    max_freq = min(max_freq, nyquist * 0.9)

    frequencies = np.arange(min_freq, max_freq + freq_step, freq_step)
    n_freqs = len(frequencies)
    eigenvalues = np.zeros(n_freqs)

    best_eigenvalue = -np.inf
    best_dss = None

    for i, freq in enumerate(frequencies):
        try:
            dss = narrowband_dss(
                sfreq=sfreq,
                freq=freq,
                bandwidth=bandwidth,
                n_components=n_components,
                **dss_kws,
            )
            dss.fit(data)
            if dss.eigenvalues_ is not None:
                eigenvalues[i] = dss.eigenvalues_[0]

            if eigenvalues[i] > best_eigenvalue:
                best_eigenvalue = eigenvalues[i]
                best_dss = dss

        except Exception:
            # Skip problematic frequencies
            continue

    if best_dss is None:
        raise RuntimeError("Failed to fit DSS at any frequency")

    return best_dss, frequencies, eigenvalues
