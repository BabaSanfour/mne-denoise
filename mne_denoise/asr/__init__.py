"""Artifact Subspace Reconstruction.

This module provides standard, adaptive, Juggler-style, and experimental
Riemannian ASR implementations for continuous EEG denoising workflows.
"""

from .adaptive import AdaptiveASR
from .core import (
    ASR,
    ASRState,
    calibrate_asr,
    compute_asr_qa_metrics,
    compute_asr_rejection_mask,
    fit_eeg_distribution,
    process_asr,
)
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
