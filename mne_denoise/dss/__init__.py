"""Denoising Source Separation (DSS) module for mne-denoise.

This module provides implementations of linear and nonlinear DSS algorithms
for evoked response enhancement, artifact removal, and rhythm extraction.

References
----------
.. [1] Särelä & Valpola (2005). Denoising Source Separation. JMLR.
.. [2] de Cheveigné & Simon (2008). DSS for evoked responses. NeuroImage.
.. [3] de Cheveigné (2020). ZapLine for line noise removal. NeuroImage.
"""

from __future__ import annotations

from .whitening import whiten_data, compute_whitener
from .core import compute_dss, DSS, DSSConfig
from .iterative import iterative_dss, iterative_dss_one, IterativeDSS
from .zapline import (
    dss_zapline, dss_zapline_adaptive, ZapLineResult, ZapLinePlusResult,
    zapline_plus, compute_psd_reduction,
)
from .narrowband_scan import narrowband_scan, narrowband_dss, NarrowbandScanResult
from .tsr import time_shift_dss, smooth_dss, TimeShiftResult
from .preprocessing import (
    detect_bad_channels,
    detect_bad_segments,
    interpolate_bad_channels,
    robust_covariance,
    reject_epochs_by_amplitude,
    RobustDSS,
)
from .denoisers import (
    LinearDenoiser,
    NonlinearDenoiser,
    TrialAverageBias,
    BandpassBias,
    NotchBias,
    PeakFilterBias,
    CombFilterBias,
    ssvep_dss,
    CycleAverageBias,
    find_ecg_events,
    find_eog_events,
    # Nonlinear denoisers (paper-faithful)
    WienerMaskDenoiser,
    TanhMaskDenoiser,
    RobustTanhDenoiser,
    GaussDenoiser,
    SkewDenoiser,
    DCTDenoiser,
    Spectrogram2DDenoiser,
    QuasiPeriodicDenoiser,
    KurtosisDenoiser,
    # Beta helpers
    beta_tanh,
    beta_pow3,
    beta_gauss,
    # Gamma helpers
    Gamma179,
    GammaPredictive,
    # Deprecated
    VarianceMaskDenoiser,
    TemporalSmoothnessDenoiser,
)

# MNE integration (optional, only if MNE is installed)
try:
    from .mne_integration import (
        apply_dss_to_raw,
        apply_dss_to_epochs,
        apply_zapline_to_raw,
        apply_zapline_to_epochs,
        get_dss_components,
    )
    _HAS_MNE_INTEGRATION = True
except ImportError:
    _HAS_MNE_INTEGRATION = False

__all__ = [
    # Whitening
    "whiten_data",
    "compute_whitener",
    # Linear DSS
    "compute_dss",
    "DSS",
    "DSSConfig",
    # Iterative/Nonlinear DSS
    "iterative_dss",
    "iterative_dss_one",
    "IterativeDSS",
    # ZapLine
    "dss_zapline",
    "dss_zapline_adaptive",
    "zapline_plus",
    "ZapLineResult",
    "ZapLinePlusResult",
    "compute_psd_reduction",
    # Narrowband Scan
    "narrowband_scan",
    "narrowband_dss",
    "NarrowbandScanResult",
    # Time-Shift DSS
    "time_shift_dss",
    "smooth_dss",
    "TimeShiftResult",
    # Preprocessing
    "detect_bad_channels",
    "detect_bad_segments",
    "interpolate_bad_channels",
    "robust_covariance",
    "reject_epochs_by_amplitude",
    "RobustDSS",
    # Denoisers base
    "LinearDenoiser",
    "NonlinearDenoiser",
    # Linear denoisers
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
    "QuasiPeriodicDenoiser",
    "KurtosisDenoiser",
    # Beta helpers (FastICA Newton step)
    "beta_tanh",
    "beta_pow3",
    "beta_gauss",
    # Deprecated
    "VarianceMaskDenoiser",
    "TemporalSmoothnessDenoiser",
]

# Add MNE integration to exports if available
if _HAS_MNE_INTEGRATION:
    __all__.extend([
        "apply_dss_to_raw",
        "apply_dss_to_epochs",
        "apply_zapline_to_raw",
        "apply_zapline_to_epochs",
        "get_dss_components",
    ])
