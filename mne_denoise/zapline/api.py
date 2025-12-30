"""ZapLine Transformer API (Placeholder)."""

from __future__ import annotations

from typing import Optional

from sklearn.base import BaseEstimator, TransformerMixin


class ZapLine(BaseEstimator, TransformerMixin):
    """ZapLine line noise removal."""

    def __init__(
        self,
        line_freq: float = 60.0,
        sfreq: Optional[float] = None,
        n_remove: Optional[int] = None,
        n_harmonics: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.line_freq = line_freq
        self.sfreq = sfreq
        self.n_remove = n_remove
