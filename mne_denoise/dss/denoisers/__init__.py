"""Pluggable denoiser functions for DSS.
"""

from __future__ import annotations

from .base import LinearDenoiser, NonlinearDenoiser
from .evoked import TrialAverageBias

from .artifact import CycleAverageBias
from .spectrogram import SpectrogramBias, SpectrogramDenoiser
from .ica import (
    TanhMaskDenoiser,
    RobustTanhDenoiser,
    GaussDenoiser,
    SkewDenoiser,
    KurtosisDenoiser,
    SmoothTanhDenoiser,
    beta_tanh,
    beta_pow3,
    beta_gauss,
)
from .masking import (
    WienerMaskDenoiser,
    VarianceMaskDenoiser,
)
from .periodic import PeakFilterBias, CombFilterBias, QuasiPeriodicDenoiser
from .spectral import (
    BandpassBias,
    NotchBias,
)
from .temporal import (
    TimeShiftBias,
    SmoothingBias,
    DCTDenoiser,
    TemporalSmoothnessDenoiser,
)

__all__ = [
    # Base classes
    "LinearDenoiser",
    "NonlinearDenoiser",
    # Linear biases
    "TrialAverageBias",
    "BandpassBias",
    "NotchBias",
    "PeakFilterBias",
    "CombFilterBias",
    "CycleAverageBias",
    # Nonlinear denoisers (paper-faithful)
    "WienerMaskDenoiser",
    "TanhMaskDenoiser",
    "RobustTanhDenoiser",
    "GaussDenoiser",
    "SkewDenoiser",
    "DCTDenoiser",
    "SpectrogramBias",
    "SpectrogramDenoiser",
    "QuasiPeriodicDenoiser",
    "KurtosisDenoiser",
    "SmoothTanhDenoiser",
    # Beta helpers (FastICA Newton step)
    "beta_tanh",
    "beta_pow3",
    "beta_gauss",
    # Deprecated
    "VarianceMaskDenoiser",
    "TemporalSmoothnessDenoiser",
    # Temporal biases
    "TimeShiftBias",
    "SmoothingBias",
]
