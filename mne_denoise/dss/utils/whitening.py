"""PCA whitening utilities for DSS.

This module provides functions for computing whitening transformations
with robust handling of rank deficiency and numerical stability.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import LinAlgError


def compute_whitener(
    cov: np.ndarray,
    *,
    rank: int | None = None,
    reg: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute whitening and de-whitening matrices from covariance.

    Parameters
    ----------
    cov : ndarray, shape (n_channels, n_channels)
        Covariance matrix of the data.
    rank : int, optional
        Number of principal components to retain. If None, determined
        automatically based on eigenvalue threshold.
    reg : float
        Regularization threshold. Eigenvalues smaller than
        reg * max(eigenvalue) are discarded. Default 1e-9.

    Returns
    -------
    whitener : ndarray, shape (rank, n_channels)
        Matrix to whiten data: X_white = whitener @ X
    dewhitener : ndarray, shape (n_channels, rank)
        Matrix to de-whiten: X = dewhitener @ X_white
    eigenvalues : ndarray, shape (rank,)
        Retained eigenvalues (descending order).


    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.dss import compute_whitener
    >>> cov = np.random.randn(64, 64)
    >>> whitener, dewhitener, eigenvalues = compute_whitener(cov)
    >>> whitened = whitener @ np.random.randn(64, 1000)
    >>> de_whitened = dewhitener @ whitened
    >>> np.allclose(de_whitened, np.random.randn(64, 1000))
    True
    """
    # Ensure symmetry
    cov = (cov + cov.T) / 2

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except LinAlgError as e:
        raise ValueError(f"Eigendecomposition failed: {e}") from e

    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Use MNE-like logic for rank
    max_ev = np.max(np.abs(eigenvalues))
    if max_ev < 1e-30:
        raise ValueError("Covariance matrix has no significant variance")

    # Regularization
    eigenvalues = np.clip(eigenvalues, a_min=0, a_max=None)  # Ensure positive
    threshold = reg * max_ev
    keep_mask = eigenvalues > threshold

    if rank is not None:
        keep_mask[rank:] = False

    # Truncate
    eigenvalues = eigenvalues[keep_mask]
    eigenvectors = eigenvectors[:, keep_mask]

    # Whitening scaling
    # X_white = diag(1/sqrt(lambda)) * V.T * X
    scales = 1.0 / np.sqrt(eigenvalues)
    whitener = scales[:, np.newaxis] * eigenvectors.T
    dewhitener = eigenvectors * np.sqrt(eigenvalues)

    return whitener, dewhitener, eigenvalues


def whiten_data(
    data: np.ndarray,
    *,
    rank: int | None = None,
    reg: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Whiten multichannel data using PCA.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
        Input data to whiten.
    rank : int, optional
        Number of principal components to retain. If None, auto-determined.
    reg : float
        Regularization threshold for eigenvalue cutoff. Default 1e-9.

    Returns
    -------
    whitened : ndarray, shape (rank, n_times) or (rank, n_times, n_epochs)
        Whitened data with unit covariance.
    whitener : ndarray, shape (rank, n_channels)
        Whitening matrix.
    dewhitener : ndarray, shape (n_channels, rank)
        De-whitening matrix for reconstruction.

    Examples
    --------
    >>> data = np.random.randn(64, 1000)  # 64 channels, 1000 samples
    >>> whitened, W, D = whiten_data(data)
    >>> # Verify whitened covariance is approximately identity
    >>> np.allclose(whitened @ whitened.T / 1000, np.eye(whitened.shape[0]), atol=0.1)
    True
    """
    # Handle 2D and 3D data
    input_shape = data.shape
    if data.ndim == 3:
        # Epochs: (n_channels, n_times, n_epochs) -> (n_channels, n_times*n_epochs)
        n_channels, n_times, n_epochs = data.shape
        data_2d = data.reshape(n_channels, -1)
    elif data.ndim == 2:
        data_2d = data
    else:
        raise ValueError(f"Data must be 2D or 3D, got {data.ndim}D")

    # Remove mean for covariance computation
    data_centered = data_2d - data_2d.mean(axis=1, keepdims=True)

    # Compute covariance: C = X @ X.T / n_samples
    n_samples = data_centered.shape[1]
    cov = data_centered @ data_centered.T / n_samples

    # Get whitening matrices
    whitener, dewhitener, eigenvalues = compute_whitener(cov, rank=rank, reg=reg)

    # Apply whitening
    whitened_2d = whitener @ data_centered

    # Reshape back if input was 3D
    if len(input_shape) == 3:
        whitened = whitened_2d.reshape(whitener.shape[0], n_times, n_epochs)
    else:
        whitened = whitened_2d

    return whitened, whitener, dewhitener
