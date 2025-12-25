"""SSVEP denoising utilities.

Convenience functions for SSVEP analysis using DSS.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ._core import compute_dss
from .denoisers.periodic import CombFilterBias


def ssvep_dss(
    data: np.ndarray,
    sfreq: float,
    stim_freq: float,
    *,
    n_harmonics: int = 3,
    n_components: Optional[int] = None,
    rank: Optional[int] = None,
    reg: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract SSVEP components using comb-filter DSS.

    Convenience function for extracting steady-state visual evoked
    potential components locked to a stimulus frequency.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Input data.
    sfreq : float
        Sampling frequency in Hz.
    stim_freq : float
        SSVEP stimulus frequency in Hz.
    n_harmonics : int
        Number of harmonics to include. Default 3.
    n_components : int, optional
        Number of DSS components to return.
    rank : int, optional
        Rank for whitening.
    reg : float
        Regularization. Default 1e-9.

    Returns
    -------
    filters : ndarray, shape (n_components, n_channels)
        DSS spatial filters for SSVEP.
    patterns : ndarray, shape (n_channels, n_components)
        DSS spatial patterns.
    eigenvalues : ndarray, shape (n_components,)
        DSS eigenvalues.

    Examples
    --------
    >>> # Extract 12 Hz SSVEP components
    >>> filters, patterns, eigenvalues = ssvep_dss(data, sfreq=250, stim_freq=12)
    >>> ssvep_sources = filters @ data
    """
    data = np.asarray(data)
    if data.ndim == 3:
        data = data.reshape(data.shape[0], -1)

    # Create comb filter bias
    comb_bias = CombFilterBias(
        fundamental_freq=stim_freq,
        sfreq=sfreq,
        n_harmonics=n_harmonics,
    )
    
    biased_data = comb_bias.apply(data)
    
    filters, patterns, eigenvalues, _ = compute_dss(
        data, biased_data,
        n_components=n_components,
        rank=rank,
        reg=reg,
    )
    
    return filters, patterns, eigenvalues
