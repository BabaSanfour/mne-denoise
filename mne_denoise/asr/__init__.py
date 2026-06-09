"""Artifact Subspace Reconstruction.

This module provides standard, adaptive, Juggler-style, and experimental
Riemannian ASR implementations for continuous EEG denoising workflows.
"""

from ._calibration import calibrate_asr
from ._distribution import fit_eeg_distribution
from ._estimator import ASR
from ._qa import compute_asr_qa_metrics, compute_asr_rejection_mask
from ._reconstruction import process_asr
from ._types import ASRState
from .adaptive import AdaptiveASR
from .juggler import JugglerASR, select_juggler_reference_samples

__all__ = [
    "ASR",
    "AdaptiveASR",
    "JugglerASR",
    "ASRState",
    "calibrate_asr",
    "compute_asr_rejection_mask",
    "compute_asr_qa_metrics",
    "fit_eeg_distribution",
    "process_asr",
    "select_juggler_reference_samples",
]
