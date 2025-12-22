"""Pluggable denoiser functions for DSS.

This module provides bias functions that emphasize different signal types:
- TrialAverageBias: Evoked responses via epoch averaging
- BandpassBias: Narrow-band rhythms
- PeakFilterBias, CombFilterBias: Periodic/SSVEP signals
- CycleAverageBias: Periodic artifacts (ECG, blinks)
- WienerMaskDenoiser, TanhMaskDenoiser: Paper-faithful nonlinear DSS (Särelä & Valpola 2005)
- GaussDenoiser, SkewDenoiser: FastICA nonlinearities
- QuasiPeriodicDenoiser: Cycle-averaging for quasi-periodic signals
"""

from __future__ import annotations

from .base import LinearDenoiser, NonlinearDenoiser
from .evoked import TrialAverageBias
from .spectral import BandpassBias, NotchBias
from .artifact import CycleAverageBias, find_ecg_events, find_eog_events
from .variance import (
    # Paper-faithful nonlinear denoisers
    WienerMaskDenoiser,
    TanhMaskDenoiser,
    RobustTanhDenoiser,
    GaussDenoiser,
    SkewDenoiser,
    DCTDenoiser,
    Spectrogram2DDenoiser,
    QuasiPeriodicDenoiser,
    KurtosisDenoiser,
    # Beta helpers for Newton updates
    beta_tanh,
    beta_pow3,
    beta_gauss,
    # Gamma helpers for adaptive learning rate
    Gamma179,
    GammaPredictive,
    # Deprecated (kept for backwards compatibility)
    VarianceMaskDenoiser,
    TemporalSmoothnessDenoiser,
)
from .periodic import PeakFilterBias, CombFilterBias, ssvep_dss

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
    "ssvep_dss",
    "CycleAverageBias",
    "find_ecg_events",
    "find_eog_events",
    # Nonlinear denoisers (paper-faithful)
    "WienerMaskDenoiser",
    "TanhMaskDenoiser",
    "RobustTanhDenoiser",
    "GaussDenoiser",
    "SkewDenoiser",
    "DCTDenoiser",
    "Spectrogram2DDenoiser",
    "QuasiPeriodicDenoiser",
    "KurtosisDenoiser",
    # Beta helpers (FastICA Newton step)
    "beta_tanh",
    "beta_pow3",
    "beta_gauss",
    # Gamma helpers (adaptive learning rate)
    "Gamma179",
    "GammaPredictive",
    # Deprecated
    "VarianceMaskDenoiser",
    "TemporalSmoothnessDenoiser",
]
