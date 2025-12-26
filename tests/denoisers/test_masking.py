"""Unit tests for adaptive masking denoisers."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne_denoise.dss.denoisers.masking import WienerMaskDenoiser, VarianceMaskDenoiser

def test_wiener_mask_denoiser():
    """Test WienerMaskDenoiser on bursty data."""
    sfreq = 200
    times = np.arange(1000) / sfreq
    
    # Create bursty signal (high variance in middle)
    burst = np.sin(2 * np.pi * 10 * times) * np.exp(-0.5 * (times - 2.5)**2 / 0.1)
    
    # Create background noise (constant variance)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.2, len(times))
    
    data = burst + noise
    
    # Denoise
    # noise_percentile=25 means we estimate noise from the quietest 25% of data
    denoiser = WienerMaskDenoiser(window_samples=50, noise_percentile=25)
    denoised = denoiser.denoise(data)
    
    # Verification:
    # 1. Burst should be preserved (mask ~ 1)
    # Burst peak is around t=2.5s -> index 500
    burst_idx = 500
    # Expected: burst amplitude >> noise -> mask -> 1
    # Check ratio input/output at peak
    ratio_peak = np.abs(denoised[burst_idx]) / (np.abs(data[burst_idx]) + 1e-15)
    assert ratio_peak > 0.9, "Burst peak should be preserved"
    
    # 2. Quiet regions should be suppressed (mask < 1)
    # Quiet region: start of file
    quiet_idx = 0
    ratio_quiet = np.abs(denoised[quiet_idx]) / (np.abs(data[quiet_idx]) + 1e-15)
    # It won't be perfectly 0 due to min_gain, but should be small
    assert ratio_quiet < 0.6, "Quiet noise should be suppressed"

def test_wiener_mask_fixed_noise():
    """Test WienerMaskDenoiser with fixed noise variance."""
    data = np.ones(100)  # Signal variance ~ 0 (DC)
    # If we say noise_variance is huge, output should be close to 0
    denoiser = WienerMaskDenoiser(noise_variance=100.0)
    denoised = denoiser.denoise(data)
    
    # mask = sig_var / (sig_var + noise_var) ~ 0 / (0 + 100) ~ 0
    # min_gain default 0.01
    assert_allclose(denoised, data * 0.01, atol=1e-5)

def test_variance_mask_denoiser():
    """Test VarianceMaskDenoiser (binary and soft)."""
    # Simply test that high variance pass, low variance stop
    data = np.array([0.1, 0.1, 10.0, 10.0, 0.1, 0.1])
    # window=1 ensures local variance is calculated immediately
    
    # Binary
    denoiser = VarianceMaskDenoiser(window_samples=1, percentile=50, soft=False)
    denoised = denoiser.denoise(data)
    
    # High values (10) should stay, low values (0.1) should be zeroed
    expected = np.array([0.0, 0.0, 10.0, 10.0, 0.0, 0.0])
    assert_allclose(denoised, expected)
    
    # We need *fluctuating* signal to have variance.
    data = np.array([1, -1, 1, -1, 10, -10, 10, -10], dtype=float)
    # High amplitude region has high variance.
    
    denoiser = VarianceMaskDenoiser(window_samples=2, percentile=50, soft=False)
    denoised = denoiser.denoise(data)
    
    # Last part should be preserved
    assert np.all(denoised[4:] != 0) 
    
    # Soft
    denoiser_soft = VarianceMaskDenoiser(window_samples=2, percentile=50, soft=True)
    denoised_soft = denoiser_soft.denoise(data)
    
    # Just check valid output
    assert denoised_soft.shape == data.shape
