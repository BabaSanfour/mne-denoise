"""Robust covariance estimation.

Provides methods for computing covariance matrices robust to outliers
or low sample counts (shrinkage).
"""

from __future__ import annotations
from typing import Optional
import numpy as np


def robust_covariance(
    data: np.ndarray,
    *,
    method: str = "empirical",
    shrinkage: Optional[float] = None,
) -> np.ndarray:
    """Compute robust covariance matrix.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Input data.
    method : str
        Method: 'empirical', 'shrinkage', 'oas', or 'mcd'.
        Default 'empirical'.
    shrinkage : float, optional
        Shrinkage parameter for 'shrinkage' method.

    Returns
    -------
    cov : ndarray, shape (n_channels, n_channels)
        Covariance matrix.
    """
    n_channels, n_times = data.shape
    
    # Center data
    data_centered = data - data.mean(axis=1, keepdims=True)
    
    if method == "empirical":
        cov = data_centered @ data_centered.T / n_times
        
    elif method == "shrinkage":
        # Ledoit-Wolf-like shrinkage
        emp_cov = data_centered @ data_centered.T / n_times
        
        if shrinkage is None:
            # Estimate optimal shrinkage
            shrinkage = _ledoit_wolf_shrinkage(data_centered)
        
        target = np.eye(n_channels) * np.trace(emp_cov) / n_channels
        cov = (1 - shrinkage) * emp_cov + shrinkage * target
        
    elif method == "oas":
        # Oracle Approximating Shrinkage
        try:
            from sklearn.covariance import OAS
            oas = OAS().fit(data_centered.T)
            cov = oas.covariance_
        except ImportError:
            # Fallback to shrinkage
            return robust_covariance(data, method="shrinkage")
            
    elif method == "mcd":
        # Minimum Covariance Determinant (robust)
        try:
            from sklearn.covariance import MinCovDet
            mcd = MinCovDet().fit(data_centered.T)
            cov = mcd.covariance_
        except ImportError:
            return robust_covariance(data, method="shrinkage")
    else:
        raise ValueError(f"Unknown covariance method: {method}")
    
    # Ensure symmetry
    cov = (cov + cov.T) / 2
    
    return cov


def _ledoit_wolf_shrinkage(data: np.ndarray) -> float:
    """Estimate optimal Ledoit-Wolf shrinkage parameter."""
    n_channels, n_times = data.shape
    
    # Sample covariance
    S = data @ data.T / n_times
    
    # Target: scaled identity
    mu = np.trace(S) / n_channels
    
    # Compute shrinkage intensity
    delta = ((S - mu * np.eye(n_channels))**2).sum() / n_channels
    
    # Estimate beta
    X2 = data**2
    beta = np.sum(X2 @ X2.T / n_times - S**2) / (n_channels * n_times)
    
    # Shrinkage
    shrinkage = min(1.0, beta / max(delta, 1e-10))
    
    return max(0.0, shrinkage)
