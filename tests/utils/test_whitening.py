"""Unit tests for whitening utilities."""

from __future__ import annotations

import numpy as np
import pytest

from mne_denoise.dss.utils.whitening import (
    whiten_data,
    compute_whitener,
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
    np.testing.assert_allclose(
        cov, np.eye(whitened.shape[0]), atol=0.1
    )

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
    np.testing.assert_allclose(
        product, np.eye(W.shape[0]), atol=1e-10
    )
