"""Narrow-band artifact suppression utilities integrated with MNE-Python.

mne-denoise provides DSS (Denoising Source Separation) algorithms for:
- Evoked response enhancement
- Line noise removal (ZapLine)
- Artifact extraction (ECG, EOG)
- Rhythm extraction (alpha, SSVEP)
"""

from __future__ import annotations

__version__ = "0.0.1"

# =============================================================================
# DSS Module - Main API
# =============================================================================
from .dss import (
    # Linear DSS (de Cheveigné & Simon 2008)
    DSS,
    DSSConfig,
    compute_dss,
    whiten_data,
    compute_whitener,
    # Iterative/Nonlinear DSS (Särelä & Valpola 2005)
    IterativeDSS,
    iterative_dss,
    iterative_dss_one,
    # ZapLine DSS (de Cheveigné 2020, Klug & Kloosterman 2022)
    dss_zapline,
    dss_zapline_adaptive,
    zapline_plus,
    ZapLineResult,
    ZapLinePlusResult,
    compute_psd_reduction,
    # Narrowband scan
    narrowband_scan,
    narrowband_dss,
    NarrowbandScanResult,
    # Time-shift DSS
    time_shift_dss,
    smooth_dss,
    TimeShiftResult,
    # Preprocessing
    detect_bad_channels,
    detect_bad_segments,
    robust_covariance,
    RobustDSS,
    # Denoisers - Base
    LinearDenoiser,
    NonlinearDenoiser,
    # Denoisers - Linear biases
    TrialAverageBias,
    BandpassBias,
    NotchBias,
    PeakFilterBias,
    CombFilterBias,
    ssvep_dss,
    CycleAverageBias,
    find_ecg_events,
    find_eog_events,
    # Denoisers - Nonlinear (Särelä & Valpola 2005)
    WienerMaskDenoiser,
    TanhMaskDenoiser,
    RobustTanhDenoiser,
    GaussDenoiser,
    SkewDenoiser,
    DCTDenoiser,
    Spectrogram2DDenoiser,
    QuasiPeriodicDenoiser,
    KurtosisDenoiser,
    # Beta helpers (Newton step)
    beta_tanh,
    beta_pow3,
    beta_gauss,
    # Gamma helpers (adaptive LR)
    Gamma179,
    GammaPredictive,
)

# Noise detection (shared utility)
from .noise_detection import find_next_noisefreq

__all__ = [
    "__version__",
    # Linear DSS
    "DSS",
    "DSSConfig",
    "compute_dss",
    "whiten_data",
    "compute_whitener",
    # Iterative/Nonlinear DSS
    "IterativeDSS",
    "iterative_dss",
    "iterative_dss_one",
    # ZapLine DSS
    "dss_zapline",
    "dss_zapline_adaptive",
    "zapline_plus",
    "ZapLineResult",
    "ZapLinePlusResult",
    "compute_psd_reduction",
    # Narrowband scan
    "narrowband_scan",
    "narrowband_dss",
    "NarrowbandScanResult",
    # Time-shift DSS
    "time_shift_dss",
    "smooth_dss",
    "TimeShiftResult",
    # Preprocessing
    "detect_bad_channels",
    "detect_bad_segments",
    "robust_covariance",
    "RobustDSS",
    # Denoisers - Base
    "LinearDenoiser",
    "NonlinearDenoiser",
    # Denoisers - Linear
    "TrialAverageBias",
    "BandpassBias",
    "NotchBias",
    "PeakFilterBias",
    "CombFilterBias",
    "ssvep_dss",
    "CycleAverageBias",
    "find_ecg_events",
    "find_eog_events",
    # Denoisers - Nonlinear
    "WienerMaskDenoiser",
    "TanhMaskDenoiser",
    "RobustTanhDenoiser",
    "GaussDenoiser",
    "SkewDenoiser",
    "DCTDenoiser",
    "Spectrogram2DDenoiser",
    "QuasiPeriodicDenoiser",
    "KurtosisDenoiser",
    # Beta helpers
    "beta_tanh",
    "beta_pow3",
    "beta_gauss",
    # Gamma helpers
    "Gamma179",
    "GammaPredictive",
    # Utils
    "find_next_noisefreq",
]

# MNE integration (optional)
try:
    from ._mne import apply_zapline_to_raw
    __all__.append("apply_zapline_to_raw")
except ImportError:
    pass

# =============================================================================
# Legacy imports (deprecated, will be removed in future versions)
# =============================================================================
try:
    from ._legacy_zapline import PyZaplinePlus
    __all__.append("PyZaplinePlus")
except ImportError:
    pass
