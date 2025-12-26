"""ZapLine Transformer API (Placeholder)."""

from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin

from ..dss.linear import DSS
from ..dss.denoisers.spectral import BandpassBias

class ZapLine(DSS):
    """ZapLine Transformer (Placeholder).
    
    This class is currently a placeholder for the future ZapLine implementation
    that will inherit directly from DSS and provide a clean TransformerMixin API.
    
    For now, use `mne_denoise.dss.dss_zapline` for functional usage.
    """
    def __init__(
        self,
        line_freq: float = 60.0,
        sfreq: float = None,
        n_remove: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.line_freq = line_freq
        self.sfreq = sfreq
        self.n_remove = n_remove
