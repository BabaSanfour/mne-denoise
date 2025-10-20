"""
Integration helpers for working with MNE-Python.

The public helper :func:`apply_zapline_to_raw` applies the zapline_plus
algorithm to a subset of channels in an ``mne.io.BaseRaw`` object while
preserving the rest of the signal.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

from .core import zapline_plus


def apply_zapline_to_raw(
    raw,
    *,
    picks: Optional[Sequence[str]] = None,
    copy: bool = True,
    line_freqs: Optional[Sequence[float] | str] = "line",
    verbose: Optional[bool] = None,
    **zap_kwargs: Any,
):
    """
    Run zapline_plus on an MNE Raw object and return the modified Raw.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Preloaded raw instance.
    picks : sequence of str | None
        Channel names that should be cleaned. Defaults to EEG channels.
    copy : bool
        If True (default) operate on a copy of ``raw``.
    line_freqs : sequence|\"line\"|None
        Noise frequencies to target. ``\"line\"`` triggers automatic selection.
    verbose : bool | None
        Optional verbosity override passed to MNE's logging context.
    zap_kwargs : dict
        Extra keyword arguments forwarded to :func:`zapline_plus`.

    Returns
    -------
    raw_out : mne.io.BaseRaw
        Cleaned Raw object.
    config, analytics, figures : tuple
        Additional information returned by :func:`zapline_plus`.
    """
    try:
        import numpy as np
        import mne  # type: ignore
        from mne.utils import use_log_level  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "apply_zapline_to_raw requires the 'mne' extra. "
            "Install with 'pip install pyzaplineplus[mne]' "
            "or 'pip install mne'."
        ) from exc

    if not getattr(raw, "preload", False):
        raise ValueError("The Raw object must be preloaded (raw.preload == True).")

    if picks is None:
        pick_idx = mne.pick_types(raw.info, eeg=True, meg=False, ref_meg=False, eog=False)
        pick_names = [raw.ch_names[i] for i in pick_idx]
    else:
        pick_names = list(picks)
        missing = [name for name in pick_names if name not in raw.ch_names]
        if missing:
            raise ValueError(f"Unknown channel names in picks: {missing}")
        pick_idx = [raw.ch_names.index(name) for name in pick_names]

    raw_out = raw.copy() if copy else raw
    data = raw_out.get_data(picks=pick_idx)
    fs = float(raw_out.info["sfreq"])

    extra = dict(zap_kwargs)
    if line_freqs is not None:
        extra.setdefault("noisefreqs", line_freqs)
    extra.setdefault("plotResults", False)

    ctx = None
    if verbose is not None:
        ctx = use_log_level("INFO" if verbose else "WARNING")
        ctx.__enter__()
    try:
        clean, config, analytics, figs = zapline_plus(data.T, fs, **extra)
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)

    # zapline_plus returns samples x channels; Raw expects channels x samples
    clean = clean.T
    if clean.dtype != data.dtype:
        clean = clean.astype(data.dtype, copy=False)
    raw_out._data[pick_idx, :] = clean

    return raw_out, config, analytics, figs

