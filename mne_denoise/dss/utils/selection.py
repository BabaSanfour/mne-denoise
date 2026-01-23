"""Component selection utilities for DSS.

Provides automatic component selection using outlier detection.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

import numpy as np


def iterative_outlier_removal(scores: np.ndarray, sigma: float = 3.0) -> int:
    """Detect outliers iteratively using mean + sigma threshold.

    This algorithm iteratively identifies values that exceed `mean + sigma * std`,
    removes them from consideration, and repeats until no more outliers are found.
    This is equivalent to MATLAB's `iterative_outlier_removal` from NoiseTools.

    Useful for automatic component selection in DSS applications, such as:
    - ZapLine: Selecting how many line-noise components to remove
    - Narrowband scan: Identifying significant frequency peaks
    - Any DSS where components need automatic thresholding

    Parameters
    ----------
    scores : ndarray
        Component scores (e.g., eigenvalues, power ratios).
        Higher values are considered more significant.
    sigma : float
        Sigma threshold for outlier detection. Default 3.0.
        Components with `score > mean + sigma * std` are outliers.

    Returns
    -------
    n_outliers : int
        Number of outliers (significant components) detected.

    Examples
    --------
    >>> from mne_denoise.dss.utils import iterative_outlier_removal
    >>> scores = np.array([0.9, 0.8, 0.2, 0.15, 0.1, 0.08])
    >>> n_significant = iterative_outlier_removal(scores, sigma=2.0)
    >>> print(f"Found {n_significant} significant components")

    References
    ----------
    NoiseTools: http://audition.ens.fr/adc/NoiseTools/
    """
    scores = np.asarray(scores)
    n_outliers = 0
    remaining = scores.copy()

    while len(remaining) > 2:
        mean_val = np.mean(remaining)
        std_val = np.std(remaining)

        if std_val < 1e-12:
            break

        threshold = mean_val + sigma * std_val
        outliers = remaining > threshold

        if not np.any(outliers):
            break

        n_outliers += np.sum(outliers)
        remaining = remaining[~outliers]

    return n_outliers


def auto_select_components(eigenvalues: np.ndarray, threshold: float = 3.0) -> int:
    """Select number of components using outlier detection.

    Convenience wrapper around `iterative_outlier_removal` for DSS eigenvalues.

    Parameters
    ----------
    eigenvalues : ndarray
        DSS eigenvalues (component scores).
    threshold : float
        Sigma threshold for outlier detection. Default 3.0.

    Returns
    -------
    n_components : int
        Number of significant components to keep/remove.
    """
    return iterative_outlier_removal(eigenvalues, threshold)
