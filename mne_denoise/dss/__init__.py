"""Denoising Source Separation (DSS).

This module contains:
- Core DSS algorithms (linear and nonlinear)
- Variants and applications (TSR, SSVEP, Narrowband)

For ZapLine, see `mne_denoise.zapline`.
"""

# Core
# Denoisers & Biases (Flat API)
from .denoisers import (
    BandpassBias,
    CombFilterBias,
    CycleAverageBias,
    DCTDenoiser,
    GaussDenoiser,
    KurtosisDenoiser,
    LinearDenoiser,
    NonlinearDenoiser,
    NotchBias,
    PeakFilterBias,
    QuasiPeriodicDenoiser,
    RobustTanhDenoiser,
    SkewDenoiser,
    SmoothingBias,
    SmoothTanhDenoiser,
    SpectrogramBias,
    SpectrogramDenoiser,
    TanhMaskDenoiser,
    TimeShiftBias,
    TrialAverageBias,
    WienerMaskDenoiser,
    beta_gauss,
    beta_pow3,
    beta_tanh,
)
from .linear import DSS, compute_dss
from .jdss import JDSS, compute_jdss
from .nonlinear import IterativeDSS, iterative_dss, iterative_dss_one

# Utils (exposed for convenience if needed)
from .utils import convergence, whitening

# Variants (Modules)
from .variants import narrowband, ssvep, tsr
from .variants.narrowband import narrowband_dss, narrowband_scan
from .variants.ssvep import ssvep_dss

# Variants (Direct Access)
from .variants.tsr import smooth_dss, time_shift_dss

__all__ = [
    # Core
    "compute_dss",
    "DSS",
    "compute_jdss",
    "JDSS",
    "iterative_dss",
    "iterative_dss_one",
    "IterativeDSS",
    # Variants modules
    "tsr",
    "ssvep",
    "narrowband",
    # Variants functions
    "time_shift_dss",
    "smooth_dss",
    "ssvep_dss",
    "narrowband_scan",
    "narrowband_dss",
    # Utils
    "whitening",
    "convergence",
    # Denoisers (from .denoisers)
    "LinearDenoiser",
    "NonlinearDenoiser",
    "TrialAverageBias",
    "BandpassBias",
    "NotchBias",
    "PeakFilterBias",
    "CombFilterBias",
    "CycleAverageBias",
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
    "beta_tanh",
    "beta_pow3",
    "beta_gauss",
    "TimeShiftBias",
    "SmoothingBias",
]
