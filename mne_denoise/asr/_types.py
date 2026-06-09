"""Shared dataclasses and types for the ASR package.

This module holds the fitted-state container shared by the standard, adaptive,
Juggler, and Riemannian ASR variants. It is kept separate from the calibration
and processing logic so every variant module can depend on the type without
importing the whole pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ASRState:
    """Fitted ASR calibration state.

    Parameters
    ----------
    M : ndarray, shape (n_channels, n_channels)
        Matrix square root of the robust calibration covariance.
    T : ndarray, shape (n_channels, n_channels)
        Direction-dependent threshold matrix.
    thresholds : ndarray, shape (n_channels,)
        Per-calibration-component RMS thresholds.
    calibration_patterns : ndarray, shape (n_channels, n_channels)
        Calibration covariance eigenvectors.
    filter_b : ndarray
        Numerator coefficients for the statistics-only filter.
    filter_a : ndarray
        Denominator coefficients for the statistics-only filter.
    cov : ndarray, shape (n_channels, n_channels)
        Robust calibration covariance.
    rank : int
        Numerical rank after regularization.
    method : {'standard', 'riemannian'}
        Covariance geometry used by the fitted state.
    riemannian_solver : str | None
        Experimental eigenspace strategy used for Riemannian ASR.
    """

    M: np.ndarray
    T: np.ndarray
    thresholds: np.ndarray
    calibration_patterns: np.ndarray
    filter_b: np.ndarray
    filter_a: np.ndarray
    cov: np.ndarray
    rank: int
    method: str = "standard"
    riemannian_solver: str | None = None
