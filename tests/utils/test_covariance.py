"""Unit tests for covariance utilities."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne_denoise.dss.utils.covariance import compute_covariance, _ledoit_wolf_shrinkage


def test_empirical_covariance_shape():
    """Empirical covariance should return correct shape."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 5, 100
    data = rng.standard_normal((n_channels, n_samples))
    
    cov = compute_covariance(data)
    
    assert cov.shape == (n_channels, n_channels)
    assert_allclose(cov, cov.T)  # Symmetry


def test_empirical_covariance_value():
    """Empirical covariance should match numpy calculation."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((3, 1000))
    
    # Center manually for comparison
    data_centered = data - data.mean(axis=1, keepdims=True)
    expected = data_centered @ data_centered.T / 1000
    
    cov = compute_covariance(data, method="empirical")
    
    assert_allclose(cov, expected)


def test_shrinkage_covariance_identity():
    """Shrinkage should return identity for identity input (mostly)."""
    # Ideally if data is uncorrelated, shrinkage target (diagonal) matches empirical (diagonal)
    rng = np.random.default_rng(42)
    data = rng.standard_normal((5, 200)) # Uncorrelated
    
    # With enough samples, empirical is close to identity
    # Shrinkage target is also identity-like
    cov_shrink = compute_covariance(data, method="shrinkage")
    
    # Check diagonal dominance
    diag = np.diag(cov_shrink)
    off_diag = cov_shrink - np.diag(diag)
    
    assert np.all(diag > 0.8)
    assert np.all(np.abs(off_diag) < 0.3)


def test_ledoit_wolf_shrinkage_calculation():
    """Test internal LW shrinkage calculation."""
    rng = np.random.default_rng(42)
    mixing = rng.standard_normal((10, 10))
    source = rng.standard_normal((10, 5000))
    data = mixing @ source
    data = data - data.mean(axis=1, keepdims=True)
    
    shrinkage = _ledoit_wolf_shrinkage(data)
    
    assert shrinkage < 0.2 
    
    data_small = data[:, :15] # 15 samples
    data_small = data_small - data_small.mean(axis=1, keepdims=True)
    shrinkage_small = _ledoit_wolf_shrinkage(data_small)
    assert shrinkage_small > shrinkage


def test_covariance_methods():
    """Test that all method strings are accepted."""
    data = np.random.randn(3, 50)
    
    compute_covariance(data, method="empirical")
    compute_covariance(data, method="shrinkage")
    
    compute_covariance(data, method="oas")
    
    # Invalid method
    with pytest.raises(ValueError, match="Unknown covariance method"):
        compute_covariance(data, method="invalid_method")


def test_covariance_mcd_method():
    """Test MCD (Minimum Covariance Determinant) method."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((3, 100))
    
    cov = compute_covariance(data, method="mcd")
    
    assert cov.shape == (3, 3)
    assert_allclose(cov, cov.T)  # Symmetric


def test_covariance_3d_data():
    """Test covariance with 3D epoched data."""
    rng = np.random.default_rng(42)
    n_ch, n_times, n_epochs = 3, 100, 5
    data = rng.standard_normal((n_ch, n_times, n_epochs))
    
    cov = compute_covariance(data, method="empirical")
    
    assert cov.shape == (n_ch, n_ch)


def test_covariance_3d_with_weights():
    """Test covariance with 3D data and per-time-point weights."""
    rng = np.random.default_rng(42)
    n_ch, n_times, n_epochs = 3, 100, 5
    data = rng.standard_normal((n_ch, n_times, n_epochs))
    
    # Weights matching n_times (will be tiled)
    weights = rng.uniform(0.5, 1.5, n_times)
    
    cov = compute_covariance(data, weights=weights)
    
    assert cov.shape == (n_ch, n_ch)


def test_covariance_3d_with_full_weights():
    """Test covariance with 3D data and full-length weights."""
    rng = np.random.default_rng(42)
    n_ch, n_times, n_epochs = 3, 100, 5
    data = rng.standard_normal((n_ch, n_times, n_epochs))
    
    # Weights matching total samples (n_times * n_epochs)
    weights = rng.uniform(0.5, 1.5, n_times * n_epochs)
    
    cov = compute_covariance(data, weights=weights)
    
    assert cov.shape == (n_ch, n_ch)


def test_covariance_weights_mismatch_error():
    """Test covariance raises error for weights length mismatch."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((3, 100))
    weights = rng.uniform(0, 1, 50)  # Wrong length
    
    with pytest.raises(ValueError, match="does not match"):
        compute_covariance(data, weights=weights)


def test_covariance_zero_weights_error():
    """Test covariance raises error when sum of weights is zero."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((3, 100))
    weights = np.zeros(100)  # All zero weights
    
    with pytest.raises(ValueError, match="Sum of weights is zero"):
        compute_covariance(data, weights=weights)


def test_covariance_weighted_non_empirical_error():
    """Test that weighted covariance only works with empirical method."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((3, 100))
    weights = rng.uniform(0.5, 1.5, 100)
    
    with pytest.raises(ValueError, match="not implemented"):
        compute_covariance(data, weights=weights, method="shrinkage")


def test_covariance_explicit_shrinkage():
    """Test covariance with explicit shrinkage parameter."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((3, 100))
    
    cov = compute_covariance(data, method="shrinkage", shrinkage=0.5)
    
    assert cov.shape == (3, 3)

