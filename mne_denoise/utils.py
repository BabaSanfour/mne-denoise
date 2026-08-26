"""Utilities for reconstructing processed MNE objects."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._mne import HAS_MNE


def reconstruct_mne_object(
    data: np.ndarray,
    orig_inst: Any,
    mne_type: str,
    picks: np.ndarray | None = None,
    verbose: bool = False,
) -> Any:
    """Insert processed data into a copy of an MNE object.

    Parameters
    ----------
    data : array
        The cleaned/processed data.
    orig_inst : object
        The original MNE instance (template).
    mne_type : str
        Type string returned by extract_data_from_mne ('raw', 'epochs', 'evoked', 'array').
    picks : array of int | None
        If provided, `data` is re-inserted into a copy of `orig_inst` only at these channel indices.
    verbose : bool
        Retained for API compatibility. Copy-based reconstruction does not
        create a new MNE object.

    Returns
    -------
    out : Raw | Epochs | Evoked | array
        Reconstructed object or the data array.
    """
    if mne_type == "array" or orig_inst is None:
        return data

    if not HAS_MNE:
        return data

    if mne_type in ("raw", "epochs"):
        out = orig_inst.copy().load_data()
        target = out._data
        if picks is None:
            if target.shape != data.shape:
                raise ValueError(
                    f"Processed data shape {data.shape} does not match {mne_type} "
                    f"shape {target.shape}"
                )
            target[...] = data
        elif mne_type == "epochs":
            target[:, picks, :] = data
        else:
            target[picks, :] = data
        return out

    if mne_type == "evoked":
        out = orig_inst.copy()
        if picks is None:
            if out.data.shape != data.shape:
                raise ValueError(
                    f"Processed data shape {data.shape} does not match evoked "
                    f"shape {out.data.shape}"
                )
            out.data[...] = data
        else:
            out.data[picks, :] = data
        return out

    return data
