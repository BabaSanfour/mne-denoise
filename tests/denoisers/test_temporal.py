"""Unit tests for temporal denoisers (TimeShiftBias, SmoothingBias)."""

import numpy as np
import pytest

from mne_denoise.dss.denoisers.temporal import TimeShiftBias, SmoothingBias


def test_timeshift_basic_2d():
    """Test TimeShiftBias with 2D data."""
    rng = np.random.default_rng(42)
    n_ch, n_times = 3, 200
    data = rng.normal(0, 1, (n_ch, n_times))
    
    bias = TimeShiftBias(shifts=5)
    biased = bias.apply(data)
    
    assert biased.shape == data.shape

def test_timeshift_basic_3d():
    """Test TimeShiftBias with 3D epoched data."""
    rng = np.random.default_rng(42)
    n_ch, n_times, n_epochs = 3, 200, 4
    data = rng.normal(0, 1, (n_ch, n_times, n_epochs))
    
    bias = TimeShiftBias(shifts=5)
    biased = bias.apply(data)
    
    assert biased.shape == data.shape
    assert biased.ndim == 3

def test_shifts_as_array():
    """Test TimeShiftBias with shifts specified as array."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (2, 100))
    
    shifts = np.array([1, 2, 5, 10])
    bias = TimeShiftBias(shifts=shifts)
    biased = bias.apply(data)
    
    assert biased.shape == data.shape

def test_autocorrelation_method():
    """Test TimeShiftBias with autocorrelation method."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (2, 100))
    
    bias = TimeShiftBias(shifts=5, method="autocorrelation")
    biased = bias.apply(data)
    
    assert biased.shape == data.shape

def test_prediction_method():
    """Test TimeShiftBias with prediction method."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (2, 100))
    
    bias = TimeShiftBias(shifts=5, method="prediction")
    biased = bias.apply(data)
    
    assert biased.shape == data.shape

def test_unknown_method_error():
    """Test TimeShiftBias raises error for unknown method."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (2, 100))
    
    bias = TimeShiftBias(shifts=5, method="unknown")
    with pytest.raises(ValueError, match="Unknown method"):
        bias.apply(data)

def test_shift_too_large_error():
    """Test TimeShiftBias raises error when shift too large."""
    data = np.ones((2, 20))
    
    bias = TimeShiftBias(shifts=15)  # Too large for data length 20
    with pytest.raises(ValueError, match="too large"):
        bias.apply(data)

def test_data_too_short_error():
    """Test TimeShiftBias raises error when data too short."""
    data = np.ones((2, 10))
    
    bias = TimeShiftBias(shifts=5)
    with pytest.raises(ValueError, match="too short|too large"):
        bias.apply(data)

def test_autocorrelated_signal_preserved():
    """Test TimeShiftBias preserves autocorrelated signals."""
    n_times = 500
    times = np.arange(n_times) / 100
    
    # Slow signal (high autocorrelation)
    slow_signal = np.sin(2 * np.pi * 0.5 * times)  # 0.5 Hz
    
    # Fast noise (low autocorrelation)
    rng = np.random.default_rng(42)
    fast_noise = rng.normal(0, 0.1, n_times)
    
    data = (slow_signal + fast_noise)[np.newaxis, :]
    
    bias = TimeShiftBias(shifts=10)
    biased = bias.apply(data)
    
    # Slow signal should be better correlated with biased output
    # (ignoring edges where padding is zero)
    center = slice(50, -50)
    corr = np.corrcoef(biased[0, center], slow_signal[center])[0, 1]
    assert corr > 0.9, f"Autocorrelated signal should be preserved (corr={corr:.3f})"


def test_smoothing_basic_2d():
    """Test SmoothingBias with 2D data."""
    rng = np.random.default_rng(42)
    n_ch, n_times = 3, 200
    data = rng.normal(0, 1, (n_ch, n_times))
    
    bias = SmoothingBias(window=10)
    biased = bias.apply(data)
    
    assert biased.shape == data.shape

def test_smoothing_basic_3d():
    """Test SmoothingBias with 3D epoched data."""
    rng = np.random.default_rng(42)
    n_ch, n_times, n_epochs = 3, 200, 4
    data = rng.normal(0, 1, (n_ch, n_times, n_epochs))
    
    bias = SmoothingBias(window=10)
    biased = bias.apply(data)
    
    assert biased.shape == data.shape
    assert biased.ndim == 3

def test_smoothing_reduces_variance():
    """Test that smoothing reduces signal variance."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (2, 500))
    
    bias = SmoothingBias(window=20)
    biased = bias.apply(data)
    
    # Smoothed data should have lower variance
    orig_var = np.var(data)
    smooth_var = np.var(biased)
    assert smooth_var < orig_var, "Smoothing should reduce variance"

def test_slow_signal_preserved():
    """Test that slow signals are preserved by smoothing."""
    n_times = 500
    times = np.arange(n_times) / 100
    
    # Slow signal
    slow_signal = np.sin(2 * np.pi * 0.5 * times)
    
    # Fast noise
    rng = np.random.default_rng(42)
    fast_noise = rng.normal(0, 0.3, n_times)
    
    data = (slow_signal + fast_noise)[np.newaxis, :]
    
    bias = SmoothingBias(window=20)
    biased = bias.apply(data)
    
    # Slow signal should correlate highly with smoothed output
    corr = np.corrcoef(biased[0], slow_signal)[0, 1]
    assert corr > 0.95, f"Slow signal should be preserved (corr={corr:.3f})"

def test_different_window_sizes():
    """Test SmoothingBias with different window sizes."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (2, 200))
    
    for window in [5, 10, 20, 50]:
        bias = SmoothingBias(window=window)
        biased = bias.apply(data)
        assert biased.shape == data.shape
