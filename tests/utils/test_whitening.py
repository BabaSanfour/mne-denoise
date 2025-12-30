"""Unit tests for whitening utilities."""

from __future__ import annotations

import numpy as np
import pytest

from mne_denoise.dss.utils.whitening import (
    compute_whitener,
    whiten_data,
)


def test_whiten_identity_covariance():
    """Whitened data should have approximately identity covariance."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 16, 5000

    # Create correlated data
    mixing = rng.standard_normal((n_channels, n_channels))
    sources = rng.standard_normal((n_channels, n_samples))
    data = mixing @ sources

    whitened, W, D = whiten_data(data)

    # Check covariance is approximately identity
    cov = whitened @ whitened.T / n_samples
    np.testing.assert_allclose(cov, np.eye(whitened.shape[0]), atol=0.1)


def test_whiten_rank_deficient():
    """Whitening should handle rank-deficient data."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 16, 1000
    true_rank = 8

    # Create rank-deficient data
    sources = rng.standard_normal((true_rank, n_samples))
    mixing = rng.standard_normal((n_channels, true_rank))
    data = mixing @ sources

    whitened, W, D = whiten_data(data)

    # Should auto-detect reduced rank
    assert whitened.shape[0] <= true_rank + 1


def test_whiten_3d_data():
    """Whitening should work on 3D epoched data."""
    rng = np.random.default_rng(42)
    n_channels, n_times, n_epochs = 8, 100, 20

    data = rng.standard_normal((n_channels, n_times, n_epochs))
    whitened, W, D = whiten_data(data)

    assert whitened.ndim == 3
    assert whitened.shape[1:] == (n_times, n_epochs)


def test_compute_whitener_matrices():
    """Whitener and dewhitener should be inverses."""
    rng = np.random.default_rng(42)
    n_channels = 8

    # Create covariance
    A = rng.standard_normal((n_channels, n_channels))
    cov = A @ A.T

    W, D, eigenvalues = compute_whitener(cov)

    # W @ D should be approximately identity (up to truncation)
    product = W @ D
    np.testing.assert_allclose(product, np.eye(W.shape[0]), atol=1e-10)


def test_compute_whitener_with_rank():
    """Test compute_whitener with explicit rank truncation."""
    rng = np.random.default_rng(42)
    n_channels = 10

    # Create full rank covariance
    A = rng.standard_normal((n_channels, n_channels))
    cov = A @ A.T

    W, D, eigenvalues = compute_whitener(cov, rank=5)

    # Should truncate to specified rank
    assert W.shape[0] == 5
    assert len(eigenvalues) == 5


def test_compute_whitener_no_variance_error():
    """Test compute_whitener raises error for zero covariance."""
    # All zeros = no variance
    cov = np.zeros((5, 5))

    with pytest.raises(ValueError, match="no significant variance"):
        compute_whitener(cov)


def test_compute_whitener_no_components_error():
    """Test compute_whitener raises error when all eigenvalues below threshold."""
    np.random.default_rng(42)
    n_channels = 5

    # Create very small covariance (all eigenvalues tiny)
    cov = np.eye(n_channels) * 1e-35

    with pytest.raises(ValueError, match="no significant variance|No components"):
        compute_whitener(cov)


def test_whiten_data_invalid_ndim():
    """Test whiten_data raises error for invalid dimensions."""
    data = np.array([1, 2, 3, 4, 5])  # 1D

    with pytest.raises(ValueError, match="must be 2D or 3D"):
        whiten_data(data)
