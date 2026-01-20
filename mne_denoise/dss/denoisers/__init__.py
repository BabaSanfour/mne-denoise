"""Pluggable denoiser functions for DSS."""

from __future__ import annotations

from .artifact import CycleAverageBias
from .base import LinearDenoiser, NonlinearDenoiser
from .evoked import TrialAverageBias
from .ica import (
    GaussDenoiser,
    KurtosisDenoiser,
    RobustTanhDenoiser,
    SkewDenoiser,
    SmoothTanhDenoiser,
    TanhMaskDenoiser,
    beta_gauss,
    beta_pow3,
    beta_tanh,
)
from .masking import (
    VarianceMaskDenoiser,
    WienerMaskDenoiser,
)
from .periodic import CombFilterBias, PeakFilterBias, QuasiPeriodicDenoiser
from .spectral import (
    BandpassBias,
    HarmonicFFTBias,
    NotchBias,
)
from .spectrogram import SpectrogramBias, SpectrogramDenoiser
from .temporal import (
    DCTDenoiser,
    PeriodSmoothingBias,
    SmoothingBias,
    TemporalSmoothnessDenoiser,
    TimeShiftBias,
)

__all__ = [
    # Base classes
    "LinearDenoiser",
    "NonlinearDenoiser",
    # Linear biases
    "TrialAverageBias",
    "BandpassBias",
    "NotchBias",
    "HarmonicFFTBias",
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
    "PeriodSmoothingBias",
]
