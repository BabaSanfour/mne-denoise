"""General utilities for mne-denoise."""

from typing import Any

import numpy as np

try:
    import mne
    from mne.epochs import BaseEpochs
    from mne.evoked import Evoked
    from mne.io import BaseRaw

    _HAS_MNE = True
except ImportError:
    mne = None
    # Mock classes for type hinting/checking when MNE not available

    class BaseRaw:  # noqa: D101
        """Mock BaseRaw class."""

    class BaseEpochs:  # noqa: D101
        """Mock BaseEpochs class."""

    class Evoked:  # noqa: D101
        """Mock Evoked class."""

    _HAS_MNE = False


def extract_data_from_mne(X: Any) -> tuple[np.ndarray, float | None, str, Any]:
    """Extract data and metadata from input (MNE object or array).

    Parameters
    ----------
    X : Raw | Epochs | Evoked | array
        Input data.

    Returns
    -------
    data : array
        Extracted data. If MNE Epochs, data is (n_epochs, n_channels, n_times).
    sfreq : float | None
        Sampling frequency.
    mne_type : str
        'raw', 'epochs', 'evoked', or 'array'.
    orig_inst : object
        Original MNE instance (or None).
    """
    sfreq = None
    mne_type = "array"
    orig_inst = None

    if _HAS_MNE and isinstance(X, (BaseRaw, BaseEpochs, Evoked)):
        orig_inst = X
        sfreq = X.info["sfreq"]
        data = X.get_data()

        if isinstance(X, BaseEpochs):
            mne_type = "epochs"
        elif isinstance(X, Evoked):
            mne_type = "evoked"
        else:
            mne_type = "raw"
    else:
        # Assume array
        data = np.asarray(X)

    return data, sfreq, mne_type, orig_inst


def reconstruct_mne_object(
    data: np.ndarray, orig_inst: Any, mne_type: str, verbose: bool = False
) -> Any:
    """Reconstruct MNE object from data and template instance.

    Parameters
    ----------
    data : array
        The cleaned/processed data.
    orig_inst : object
        The original MNE instance (template).
    mne_type : str
        Type string returned by extract_data_from_mne ('raw', 'epochs', 'evoked', 'array').
    verbose : bool
        Verbosity flag for MNE object creation.

    Returns
    -------
    out : Raw | Epochs | Evoked | array
        Reconstructed object or the data array.
    """
    if mne_type == "array" or orig_inst is None:
        return data

    if not _HAS_MNE:
        return data

    if mne_type == "raw":
        out = mne.io.RawArray(data, orig_inst.info, verbose=verbose)
        if hasattr(orig_inst, "annotations") and orig_inst.annotations:
            out.set_annotations(orig_inst.annotations)
        return out

    elif mne_type == "epochs":
        events = getattr(orig_inst, "events", None)
        event_id = getattr(orig_inst, "event_id", None)
        tmin = getattr(orig_inst, "tmin", 0)
        metadata = getattr(orig_inst, "metadata", None)

        out = mne.EpochsArray(
            data,
            orig_inst.info,
            events=events,
            event_id=event_id,
            tmin=tmin,
            verbose=verbose,
        )
        if metadata is not None:
            out.metadata = metadata.copy()
        return out

    elif mne_type == "evoked":
        nave = getattr(orig_inst, "nave", 1)
        tmin = getattr(orig_inst, "tmin", 0)
        comment = getattr(orig_inst, "comment", "")

        out = mne.EvokedArray(
            data, orig_inst.info, tmin=tmin, nave=nave, comment=comment, verbose=verbose
        )
        return out

    return data
