"""Denoising Source Separation (DSS).

This module contains:
- Core DSS algorithms (linear and nonlinear)
- Variants and applications (TSR, SSVEP, Narrowband)

For ZapLine, see `mne_denoise.zapline`.
"""

# Core
from .linear import compute_dss, DSS
from .nonlinear import iterative_dss, iterative_dss_one, IterativeDSS

# Variants (Modules)
from .variants import tsr, ssvep, narrowband

# Variants (Direct Access)
from .variants.tsr import time_shift_dss, smooth_dss
from .variants.ssvep import ssvep_dss
from .variants.narrowband import narrowband_scan, narrowband_dss

# Denoisers & Biases (Flat API)
from .denoisers import *

# Utils (exposed for convenience if needed)
from .utils import whitening, convergence

__all__ = [
    # Core
    "compute_dss",
    "DSS",
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
