"""MNE-Python integration utilities for DSS.

Provides functions to apply DSS to MNE Raw, Epochs, and Evoked objects,
preserving metadata and measurement info.
"""

from __future__ import annotations

from typing import Optional, Union, TYPE_CHECKING

import numpy as np

from .core import DSS, compute_dss
from .iterative import IterativeDSS
from .zapline import dss_zapline, ZapLineResult
from .denoisers import LinearDenoiser, BandpassBias, TrialAverageBias

if TYPE_CHECKING:
    import mne


def apply_dss_to_raw(
    raw: "mne.io.BaseRaw",
    bias: Union[str, LinearDenoiser],
    *,
    n_components: Optional[int] = None,
    picks: Optional[str] = "data",
    copy: bool = True,
) -> "mne.io.BaseRaw":
    """Apply DSS to MNE Raw object.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw data object.
    bias : str or LinearDenoiser
        Bias function. Can be:
        - 'alpha': Bandpass 8-12 Hz
        - 'beta': Bandpass 13-30 Hz
        - 'theta': Bandpass 4-8 Hz
        - 'delta': Bandpass 1-4 Hz
        - LinearDenoiser instance
    n_components : int, optional
        Number of DSS components to keep. If None, keep all.
    picks : str or array-like
        Channels to include. Default 'data' (all data channels).
    copy : bool
        Whether to copy the data. Default True.

    Returns
    -------
    raw_dss : mne.io.BaseRaw
        Raw object with DSS applied.

    Examples
    --------
    >>> raw_alpha = apply_dss_to_raw(raw, bias='alpha', n_components=10)
    """
    import mne

    if copy:
        raw = raw.copy()

    # Get data and info
    picks_idx = mne.pick_types(raw.info, meg=True, eeg=True, exclude='bads') if picks == "data" else mne.pick_channels(raw.ch_names, picks)
    data = raw.get_data(picks=picks_idx)
    sfreq = raw.info['sfreq']

    # Create bias function
    if isinstance(bias, str):
        bias = _create_bias_from_string(bias, sfreq)
    
    # Apply DSS
    biased_data = bias.apply(data)
    filters, patterns, eigenvalues, _ = compute_dss(
        data, biased_data, n_components=n_components
    )

    # Transform data
    sources = filters @ (data - data.mean(axis=1, keepdims=True))
    
    # Reconstruct with selected components
    if n_components is not None:
        n_keep = min(n_components, filters.shape[0])
        reconstructed = patterns[:, :n_keep] @ sources[:n_keep]
    else:
        reconstructed = patterns @ sources

    # Update raw data
    raw._data[picks_idx] = reconstructed

    return raw


def apply_dss_to_epochs(
    epochs: "mne.Epochs",
    bias: Union[str, LinearDenoiser] = "evoked",
    *,
    n_components: Optional[int] = None,
    picks: Optional[str] = "data",
    copy: bool = True,
) -> "mne.Epochs":
    """Apply DSS to MNE Epochs object.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs data object.
    bias : str or LinearDenoiser
        Bias function. Special values:
        - 'evoked': Trial averaging (classic DSS for evoked responses)
        - Band names: 'alpha', 'beta', 'theta', 'delta'
        - LinearDenoiser instance
    n_components : int, optional
        Number of DSS components to keep.
    picks : str or array-like
        Channels to include.
    copy : bool
        Whether to copy. Default True.

    Returns
    -------
    epochs_dss : mne.Epochs
        Epochs with DSS applied.

    Examples
    --------
    >>> # Enhance evoked response
    >>> epochs_clean = apply_dss_to_epochs(epochs, bias='evoked', n_components=5)
    """
    import mne

    if copy:
        epochs = epochs.copy()

    # Get data: (n_epochs, n_channels, n_times) -> (n_channels, n_times, n_epochs)
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    data = np.transpose(data, (1, 2, 0))  # (n_channels, n_times, n_epochs)
    
    sfreq = epochs.info['sfreq']
    n_channels, n_times, n_epochs = data.shape

    # Create bias function
    if isinstance(bias, str):
        if bias == "evoked":
            bias = TrialAverageBias()
        else:
            bias = _create_bias_from_string(bias, sfreq)

    # Apply DSS
    biased_data = bias.apply(data)
    filters, patterns, eigenvalues, _ = compute_dss(
        data, biased_data, n_components=n_components
    )

    # Transform data
    data_2d = data.reshape(n_channels, -1)
    data_centered = data_2d - data_2d.mean(axis=1, keepdims=True)
    sources = filters @ data_centered  # (n_components, n_times*n_epochs)

    # Reconstruct
    if n_components is not None:
        n_keep = min(n_components, filters.shape[0])
        reconstructed = patterns[:, :n_keep] @ sources[:n_keep]
    else:
        reconstructed = patterns @ sources

    # Reshape back to epochs format
    reconstructed = reconstructed.reshape(n_channels, n_times, n_epochs)
    reconstructed = np.transpose(reconstructed, (2, 0, 1))  # (n_epochs, n_channels, n_times)

    # Update epochs data
    epochs._data = reconstructed

    return epochs


def apply_zapline_to_raw(
    raw: "mne.io.BaseRaw",
    line_freq: float,
    *,
    n_remove: Union[int, str] = "auto",
    n_harmonics: int = 1,
    picks: Optional[str] = "data",
    copy: bool = True,
) -> "mne.io.BaseRaw":
    """Apply DSS-ZapLine to remove line noise from Raw object.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw data object.
    line_freq : float
        Line noise frequency (e.g., 50 or 60 Hz).
    n_remove : int or 'auto'
        Number of components to remove.
    n_harmonics : int
        Number of harmonics to include.
    picks : str or array-like
        Channels to include.
    copy : bool
        Whether to copy.

    Returns
    -------
    raw_clean : mne.io.BaseRaw
        Raw with line noise removed.

    Examples
    --------
    >>> raw_clean = apply_zapline_to_raw(raw, line_freq=50)
    """
    import mne

    if copy:
        raw = raw.copy()

    picks_idx = mne.pick_types(raw.info, meg=True, eeg=True, exclude='bads') if picks == "data" else mne.pick_channels(raw.ch_names, picks)
    data = raw.get_data(picks=picks_idx)
    sfreq = raw.info['sfreq']

    result = dss_zapline(
        data, line_freq, sfreq,
        n_remove=n_remove,
        n_harmonics=n_harmonics,
    )

    raw._data[picks_idx] = result.cleaned

    return raw


def apply_zapline_to_epochs(
    epochs: "mne.Epochs",
    line_freq: float,
    *,
    n_remove: Union[int, str] = "auto",
    n_harmonics: int = 1,
    copy: bool = True,
) -> "mne.Epochs":
    """Apply DSS-ZapLine to remove line noise from Epochs.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs data object.
    line_freq : float
        Line noise frequency.
    n_remove : int or 'auto'
        Number of components to remove.
    n_harmonics : int
        Number of harmonics.
    copy : bool
        Whether to copy.

    Returns
    -------
    epochs_clean : mne.Epochs
        Epochs with line noise removed.
    """
    import mne

    if copy:
        epochs = epochs.copy()

    # Concatenate epochs for DSS
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    n_epochs, n_channels, n_times = data.shape
    data_concat = data.transpose(1, 0, 2).reshape(n_channels, -1)

    sfreq = epochs.info['sfreq']

    result = dss_zapline(
        data_concat, line_freq, sfreq,
        n_remove=n_remove,
        n_harmonics=n_harmonics,
    )

    # Reshape back
    cleaned = result.cleaned.reshape(n_channels, n_epochs, n_times).transpose(1, 0, 2)
    epochs._data = cleaned

    return epochs


def _create_bias_from_string(name: str, sfreq: float) -> LinearDenoiser:
    """Create a bias function from a string name."""
    bands = {
        'delta': (1.0, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 12.0),
        'beta': (13.0, 30.0),
        'gamma': (30.0, 100.0),
    }
    
    if name.lower() not in bands:
        raise ValueError(f"Unknown bias name: {name}. Use one of {list(bands.keys())}")
    
    low, high = bands[name.lower()]
    return BandpassBias(freq_band=(low, high), sfreq=sfreq)


def get_dss_components(
    epochs: "mne.Epochs",
    bias: Union[str, LinearDenoiser] = "evoked",
    *,
    n_components: Optional[int] = None,
) -> dict:
    """Extract DSS components and patterns from epochs for visualization.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs data.
    bias : str or LinearDenoiser
        Bias function.
    n_components : int, optional
        Number of components.

    Returns
    -------
    results : dict
        Dictionary with:
        - 'filters': Spatial filters (n_components, n_channels)
        - 'patterns': Spatial patterns (n_channels, n_components)
        - 'eigenvalues': DSS eigenvalues
        - 'sources': Source time series (n_components, n_times, n_epochs)
        - 'evoked': Evoked response of top component
    """
    # Get data
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    data = np.transpose(data, (1, 2, 0))  # (n_channels, n_times, n_epochs)
    
    sfreq = epochs.info['sfreq']
    n_channels, n_times, n_epochs = data.shape

    # Create bias
    if isinstance(bias, str):
        if bias == "evoked":
            bias = TrialAverageBias()
        else:
            bias = _create_bias_from_string(bias, sfreq)

    # Compute DSS
    biased_data = bias.apply(data)
    filters, patterns, eigenvalues, _ = compute_dss(
        data, biased_data, n_components=n_components
    )

    # Extract sources
    data_2d = data.reshape(n_channels, -1)
    data_centered = data_2d - data_2d.mean(axis=1, keepdims=True)
    sources = filters @ data_centered
    sources = sources.reshape(filters.shape[0], n_times, n_epochs)

    # Compute evoked for top component
    evoked_sources = sources.mean(axis=2)  # (n_components, n_times)

    return {
        'filters': filters,
        'patterns': patterns,
        'eigenvalues': eigenvalues,
        'sources': sources,
        'evoked_sources': evoked_sources,
        'times': epochs.times,
        'info': epochs.info,
    }
