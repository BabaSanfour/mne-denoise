"""MATLAB Parity Tests for DSS Implementation.

This module provides infrastructure to compare Python DSS implementations
against MATLAB NoiseTools (nt_dss0, nt_dss1, nt_zapline) for publication-quality
validation.

Requirements:
- MATLAB Engine for Python (installed in amica310 environment)
- NoiseTools MATLAB toolbox

Usage:
    conda activate amica310
    python -m pytest tests/parity/ -v
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# MATLAB Engine will be imported dynamically
_matlab_engine = None


def get_matlab_engine():
    """Get or start MATLAB Engine."""
    global _matlab_engine
    if _matlab_engine is None:
        import matlab.engine

        _matlab_engine = matlab.engine.start_matlab()
        # Add NoiseTools to path
        noisetools_path = os.environ.get(
            "NOISETOOLS_PATH", r"C:\Users\s\Documents\MATLAB\NoiseTools"
        )
        if os.path.exists(noisetools_path):
            _matlab_engine.addpath(_matlab_engine.genpath(noisetools_path))
    return _matlab_engine


def close_matlab_engine():
    """Close MATLAB Engine."""
    global _matlab_engine
    if _matlab_engine is not None:
        _matlab_engine.quit()
        _matlab_engine = None


def to_matlab(arr: np.ndarray):
    """Convert numpy array to MATLAB array."""
    import matlab

    # Ensure float64 type for MATLAB
    arr = np.asarray(arr, dtype=np.float64)
    # MATLAB expects row-major order
    if arr.ndim == 1:
        return matlab.double(arr.tolist())
    elif arr.ndim == 2:
        return matlab.double(arr.tolist())
    else:
        # For 3D, flatten to list of lists of lists
        return matlab.double(arr.tolist())


def from_matlab(mat_arr) -> np.ndarray:
    """Convert MATLAB array to numpy."""
    return np.array(mat_arr)


class ParityMetrics:
    """Metrics for comparing MATLAB and Python implementations."""

    def __init__(self, name: str):
        self.name = name
        self.correlations = []
        self.rmse_values = []
        self.max_abs_diffs = []

    def add_comparison(
        self,
        python_result: np.ndarray,
        matlab_result: np.ndarray,
        component_name: str = "component",
    ):
        """Add a comparison between Python and MATLAB results."""
        # Handle sign ambiguity in eigenvectors
        corr = np.abs(np.corrcoef(python_result.ravel(), matlab_result.ravel())[0, 1])

        # Normalize for RMSE (eigenvectors can have different signs)
        p_norm = python_result / (np.linalg.norm(python_result) + 1e-12)
        m_norm = matlab_result / (np.linalg.norm(matlab_result) + 1e-12)

        # Check both signs
        rmse_pos = np.sqrt(np.mean((p_norm - m_norm) ** 2))
        rmse_neg = np.sqrt(np.mean((p_norm + m_norm) ** 2))
        rmse = min(rmse_pos, rmse_neg)

        max_diff = min(np.max(np.abs(p_norm - m_norm)), np.max(np.abs(p_norm + m_norm)))

        self.correlations.append(corr)
        self.rmse_values.append(rmse)
        self.max_abs_diffs.append(max_diff)

        return {
            "component": component_name,
            "correlation": corr,
            "rmse": rmse,
            "max_abs_diff": max_diff,
        }

    def summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        return {
            "mean_correlation": np.mean(self.correlations),
            "min_correlation": np.min(self.correlations),
            "mean_rmse": np.mean(self.rmse_values),
            "max_rmse": np.max(self.rmse_values),
            "mean_max_diff": np.mean(self.max_abs_diffs),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"ParityMetrics({self.name}): "
            f"corr={s['mean_correlation']:.4f} (min={s['min_correlation']:.4f}), "
            f"RMSE={s['mean_rmse']:.6f}"
        )


def generate_test_data(
    n_channels: int = 16,
    n_samples: int = 5000,
    n_epochs: int = 50,
    sfreq: float = 250.0,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate synthetic test data for parity testing.

    Creates data with:
    - Background noise
    - Evoked response (phase-locked across epochs)
    - Alpha oscillation (10 Hz)
    - Line noise (50 Hz)

    Returns dict with 'continuous', 'epoched', 'evoked_template', etc.
    """
    rng = np.random.default_rng(seed)

    n_times = int(0.5 * sfreq)  # 500ms epochs
    t = np.arange(n_times) / sfreq

    # Evoked response template (hanning-windowed sinusoid)
    evoked_template = np.hanning(n_times) * np.sin(2 * np.pi * 5 * t)
    evoked_mixing = rng.standard_normal(n_channels)
    evoked_mixing /= np.linalg.norm(evoked_mixing)

    # Alpha oscillation (10 Hz)
    alpha_template = np.sin(2 * np.pi * 10 * t)
    alpha_mixing = rng.standard_normal(n_channels)
    alpha_mixing /= np.linalg.norm(alpha_mixing)

    # Create epoched data
    epoched = np.zeros((n_channels, n_times, n_epochs))
    for epoch in range(n_epochs):
        noise = rng.standard_normal((n_channels, n_times)) * 0.5
        evoked = np.outer(evoked_mixing, evoked_template) * 2
        alpha = np.outer(alpha_mixing, alpha_template) * rng.uniform(0.5, 1.5)
        epoched[:, :, epoch] = noise + evoked + alpha

    # Create continuous data
    n_channels * n_samples
    continuous = rng.standard_normal((n_channels, n_samples)) * 0.5

    # Add 50 Hz line noise to continuous
    t_cont = np.arange(n_samples) / sfreq
    line_noise = 1.5 * np.sin(2 * np.pi * 50 * t_cont)
    line_mixing = rng.standard_normal(n_channels)
    line_mixing /= np.linalg.norm(line_mixing)
    continuous += np.outer(line_mixing, line_noise)

    # Add 10 Hz alpha
    alpha_cont = np.sin(2 * np.pi * 10 * t_cont)
    continuous += np.outer(alpha_mixing, alpha_cont)

    return {
        "continuous": continuous,
        "epoched": epoched,
        "sfreq": sfreq,
        "evoked_template": evoked_template,
        "evoked_mixing": evoked_mixing,
        "alpha_mixing": alpha_mixing,
        "line_mixing": line_mixing,
    }


def save_test_data_for_matlab(
    data: Dict[str, np.ndarray],
    output_dir: Path,
):
    """Save test data in format readable by MATLAB."""
    import scipy.io as sio

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as .mat file
    mat_data = {
        "continuous": data["continuous"],
        "epoched": data["epoched"],
        "sfreq": data["sfreq"],
    }
    sio.savemat(output_dir / "test_data.mat", mat_data)

    return output_dir / "test_data.mat"
