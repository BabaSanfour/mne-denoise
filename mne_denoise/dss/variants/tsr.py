"""Time-Shift DSS (TSR) variant.

Convenience wrapper for extracting temporally extended structure
(autocorrelated signals, slow waves, DC shifts) [1]_.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

References
----------
.. [1] de Cheveigné, A. (2010). Time-shift denoising source separation.
       Journal of Neuroscience Methods, 189(1), 113-120.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from ..denoisers.temporal import SmoothingBias, TimeShiftBias
from ..linear import DSS


def time_shift_dss(
    shifts: Union[int, np.ndarray] = 10,
    *,
    method: str = "autocorrelation",
    n_components: Optional[int] = None,
    **dss_kws,
) -> DSS:
    """Create a DSS configured for temporal predictability.

    Returns a pre-configured DSS object that extracts components with
    high autocorrelation (temporally smooth or predictable signals).

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
    n_components : int, optional
        Number of DSS components to keep. If None, keep all.
    **dss_kws
        Additional keyword arguments passed to `DSS`.

    Returns
    -------
    dss : DSS
        A DSS object configured with a TimeShiftBias.

    Examples
    --------
    >>> # Extract temporally predictable components
    >>> dss = time_shift_dss(shifts=20)
    >>> dss.fit(data)
    >>> slow_sources = dss.transform(data)

    >>> # Use specific lags
    >>> dss = time_shift_dss(shifts=np.array([1, 2, 5, 10, 20]))
    >>> dss.fit(data)
    """
    bias = TimeShiftBias(shifts=shifts, method=method)
    return DSS(bias=bias, n_components=n_components, **dss_kws)


def smooth_dss(
    window: int = 10,
    *,
    n_components: Optional[int] = None,
    **dss_kws,
) -> DSS:
    """Create a DSS configured for temporally smooth sources.

    Returns a pre-configured DSS that extracts components with
    low-frequency temporal structure.

    Parameters
    ----------
    window : int
        Smoothing window size in samples. Default 10.
    n_components : int, optional
        Number of DSS components to keep. If None, keep all.
    **dss_kws
        Additional keyword arguments passed to `DSS`.

    Returns
    -------
    dss : DSS
        A DSS object configured with a SmoothingBias.

    Examples
    --------
    >>> # Extract slow components
    >>> dss = smooth_dss(window=20)
    >>> dss.fit(data)
    >>> slow_sources = dss.transform(data)
    """
    bias = SmoothingBias(window=window)
    return DSS(bias=bias, n_components=n_components, **dss_kws)
