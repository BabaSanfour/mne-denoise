"""DSS Internal Utilities."""

from .convergence import Gamma179, GammaPredictive
from .covariance import compute_covariance
from .selection import (
    auto_select_components,
    auto_select_components_robust,
    detect_eigenvalue_knee,
    iterative_outlier_removal,
)
from .whitening import compute_whitener, whiten_data

__all__ = [
    "whiten_data",
    "compute_whitener",
    "compute_covariance",
    "iterative_outlier_removal",
    "auto_select_components",
    "auto_select_components_robust",
    "detect_eigenvalue_knee",
    "Gamma179",
    "GammaPredictive",
]
