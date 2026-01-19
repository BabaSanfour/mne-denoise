"""ZapLine Transformer API."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .core import apply_zapline, dss_zapline


def _extract_data_and_info(X):
    """Extract data array and metadata from input.
    
    Returns
    -------
    data : ndarray
        Data array.
    sfreq : float or None
        Sampling frequency if available.
    input_type : str
        One of 'raw', 'epochs', 'evoked', 'array'.
    """
    # Check for MNE types by class name to avoid import
    type_name = type(X).__name__
    
    if type_name == "Evoked":
        raise TypeError(
            "Evoked objects are not supported by ZapLine. "
            "ZapLine requires continuous or epoched data (Raw or Epochs)."
        )
    
    if hasattr(X, "get_data") and hasattr(X, "info"):
        # MNE object (Raw or Epochs)
        data = X.get_data()
        sfreq = X.info.get("sfreq", None)
        if type_name == "Epochs" or data.ndim == 3:
            return data, sfreq, "epochs"
        else:
            return data, sfreq, "raw"
    else:
        # NumPy array
        return np.asarray(X), None, "array"


def _apply_to_mne_object(X, cleaned_data, input_type):
    """Apply cleaned data back to MNE object properly.
    
    Parameters
    ----------
    X : MNE object
        Original MNE object.
    cleaned_data : ndarray
        Cleaned data array.
    input_type : str
        Type of input ('raw', 'epochs').
    
    Returns
    -------
    out : MNE object
        New MNE object with cleaned data.
    """
    if input_type == "raw":
        # For Raw, use apply_function or create new RawArray
        try:
            from mne.io import RawArray
            out = RawArray(cleaned_data, X.info.copy(), verbose=False)
            # Preserve annotations if present
            if hasattr(X, "annotations") and X.annotations is not None:
                out.set_annotations(X.annotations)
            return out
        except ImportError:
            # Fallback: copy and set data (less ideal)
            out = X.copy()
            out._data = cleaned_data
            return out
    elif input_type == "epochs":
        # For Epochs, create new EpochsArray
        try:
            from mne import EpochsArray
            out = EpochsArray(
                cleaned_data,
                X.info.copy(),
                events=getattr(X, "events", None),
                tmin=X.tmin if hasattr(X, "tmin") else 0,
                event_id=getattr(X, "event_id", None),
                verbose=False,
            )
            # Preserve metadata if present
            if hasattr(X, "metadata") and X.metadata is not None:
                out.metadata = X.metadata.copy()
            return out
        except ImportError:
            out = X.copy()
            out._data = cleaned_data
            return out
    else:
        return cleaned_data


class ZapLine(BaseEstimator, TransformerMixin):
    """ZapLine Transformer for line noise removal.

    This estimator wraps the `dss_zapline` function into a scikit-learn
    compatible Transformer.

    Parameters
    ----------
    line_freq : float
        Line noise frequency in Hz (e.g., 50 or 60).
    sfreq : float
        Sampling frequency in Hz.
    n_remove : int or 'auto', default='auto'
        Number of components to remove.
    n_harmonics : int, optional
        Number of harmonics to include. If None, use all harmonics
        up to Nyquist.
    nfft : int, default=1024
        FFT size for bias computation.
    nkeep : int, optional
        Number of PCA components to keep before DSS.
    rank : int, optional
        Rank for DSS whitening.
    reg : float, default=1e-9
        Regularization for DSS.
    threshold : float, default=3.0
        Z-score threshold for auto component selection.

    Attributes
    ----------
    filters_ : ndarray
        DSS spatial filters.
    patterns_ : ndarray
        DSS spatial patterns (mixing matrix).
    scores_ : ndarray
        DSS eigenvalues (component scores).
    n_removed_ : int
        Number of components actually removed.
    """

    def __init__(
        self,
        line_freq: float = 60.0,
        sfreq: Optional[float] = None,
        n_remove: Union[int, str] = "auto",
        n_harmonics: Optional[int] = None,
        nfft: int = 1024,
        nkeep: Optional[int] = None,
        rank: Optional[int] = None,
        reg: float = 1e-9,
        threshold: float = 3.0,
    ):
        self.line_freq = line_freq
        self.sfreq = sfreq
        self.n_remove = n_remove
        self.n_harmonics = n_harmonics
        self.nfft = nfft
        self.nkeep = nkeep
        self.rank = rank
        self.reg = reg
        self.threshold = threshold

        # Attributes (set during fit)
        self.filters_ = None
        self.patterns_ = None
        self.scores_ = None
        self.n_removed_ = None

    def fit(self, X, y=None):
        """Fit ZapLine to data (calculate filters).

        Parameters
        ----------
        X : ndarray or MNE Raw/Epochs
            Data to fit. Evoked objects are not supported.
        y : None
            Ignored.

        Returns
        -------
        self : ZapLine
            Fitted transformer.
        """
        # Extract data and validate type
        data, mne_sfreq, input_type = _extract_data_and_info(X)

        # Determine sfreq
        sfreq = self.sfreq
        if sfreq is None:
            if mne_sfreq is not None:
                sfreq = mne_sfreq
            else:
                raise ValueError("sfreq must be provided if X is not an MNE object.")

        # Handle 3D data (epochs) by concatenation for fit
        if data.ndim == 3:
            n_epochs, n_ch, n_times = data.shape
            data_cont = np.transpose(data, (1, 0, 2)).reshape(n_ch, -1)
        else:
            data_cont = data

        # Run dss_zapline to estimate parameters
        res = dss_zapline(
            data_cont,
            line_freq=self.line_freq,
            sfreq=sfreq,
            n_remove=self.n_remove,
            n_harmonics=self.n_harmonics,
            nfft=self.nfft,
            nkeep=self.nkeep,
            rank=self.rank,
            reg=self.reg,
            threshold=self.threshold,
        )

        self.filters_ = res.dss_filters
        self.patterns_ = res.dss_patterns
        self.scores_ = res.dss_eigenvalues
        self.n_removed_ = res.n_removed

        return self

    def transform(self, X):
        """Clean data using ZapLine.

        Parameters
        ----------
        X : ndarray or MNE Raw/Epochs
            Data to clean. Evoked objects are not supported.

        Returns
        -------
        X_clean : ndarray or MNE object
            Cleaned data (same type as input).
        """
        # Check if fitted
        if self.filters_ is None:
            raise RuntimeError("ZapLine must be fitted before transform.")

        # Extract data and validate type
        data, mne_sfreq, input_type = _extract_data_and_info(X)

        # Determine sfreq
        sfreq = self.sfreq
        if sfreq is None:
            if mne_sfreq is not None:
                sfreq = mne_sfreq
            else:
                raise ValueError("sfreq must be provided if X is not an MNE object.")

        is_mne = input_type in ("raw", "epochs")

        # Clean using apply_zapline
        if data.ndim == 3:
            # 3D epochs: (n_epochs, n_ch, n_times)
            n_epochs, n_ch, n_times = data.shape
            data_cont = np.transpose(data, (1, 0, 2)).reshape(n_ch, -1)

            res = apply_zapline(
                data_cont,
                filters=self.filters_,
                n_remove=self.n_removed_,
                line_freq=self.line_freq,
                sfreq=sfreq,
                patterns=self.patterns_,
            )
            cleaned_cont = res.cleaned
            cleaned_epochs = cleaned_cont.reshape(n_ch, n_epochs, n_times).transpose(
                1, 0, 2
            )

            if is_mne:
                return _apply_to_mne_object(X, cleaned_epochs, input_type)
            return cleaned_epochs

        # 2D case
        res = apply_zapline(
            data,
            filters=self.filters_,
            n_remove=self.n_removed_,
            line_freq=self.line_freq,
            sfreq=sfreq,
            patterns=self.patterns_,
        )

        if is_mne:
            return _apply_to_mne_object(X, res.cleaned, input_type)

        return res.cleaned
