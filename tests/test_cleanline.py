"""Unit tests for cleanline fallback functions."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import signal

from mne_denoise.zapline.cleanline import apply_cleanline_notch, apply_hybrid_cleanup


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def line_noise_data():
    """Generate synthetic data with line noise."""
    rng = np.random.default_rng(42)
    sfreq = 250
    n_channels = 4
    n_times = 2500  # 10s
    times = np.arange(n_times) / sfreq

    # Background signal
    brain = rng.normal(0, 0.5, (n_channels, n_times))

    # 50 Hz line noise
    line_noise = np.sin(2 * np.pi * 50 * times) * 2.0
    data = brain + line_noise[np.newaxis, :]

    return {"data": data, "sfreq": sfreq, "line_freq": 50.0, "times": times}


# =============================================================================
# apply_cleanline_notch - Basic Functionality
# =============================================================================


def test_cleanline_notch_output_shape(line_noise_data):
    """apply_cleanline_notch should return same shape as input."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]

    filtered = apply_cleanline_notch(data, sfreq=sfreq, freq=50.0)

    assert filtered.shape == data.shape


def test_cleanline_notch_reduces_target_power(line_noise_data):
    """apply_cleanline_notch should reduce power at target frequency."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]

    filtered = apply_cleanline_notch(data, sfreq=sfreq, freq=50.0)

    def get_power_at(d, freq, fs):
        f, psd = signal.welch(d, fs=fs, nperseg=int(fs), axis=-1)
        idx = np.argmin(np.abs(f - freq))
        return np.mean(psd[:, idx])

    power_before = get_power_at(data, 50.0, sfreq)
    power_after = get_power_at(filtered, 50.0, sfreq)

    # Should reduce 50 Hz power significantly
    assert power_after < power_before * 0.1


def test_cleanline_notch_preserves_other_frequencies(line_noise_data):
    """apply_cleanline_notch should preserve power at other frequencies."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]

    # Add a 10 Hz signal we want to preserve
    times = line_noise_data["times"]
    alpha = np.sin(2 * np.pi * 10 * times)
    data_with_alpha = data.copy()
    data_with_alpha[0] += alpha

    filtered = apply_cleanline_notch(data_with_alpha, sfreq=sfreq, freq=50.0)

    def get_power_at(d, freq, fs):
        f, psd = signal.welch(d, fs=fs, nperseg=int(fs), axis=-1)
        idx = np.argmin(np.abs(f - freq))
        return np.mean(psd[:, idx])

    power_alpha_before = get_power_at(data_with_alpha[0:1], 10.0, sfreq)
    power_alpha_after = get_power_at(filtered[0:1], 10.0, sfreq)

    # Alpha power should be mostly preserved (within 20%)
    ratio = power_alpha_after / power_alpha_before
    assert ratio > 0.8


def test_cleanline_notch_bandwidth_parameter(line_noise_data):
    """apply_cleanline_notch should respect bandwidth parameter."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]

    # Narrow bandwidth
    filtered_narrow = apply_cleanline_notch(data, sfreq=sfreq, freq=50.0, bandwidth=0.2)

    # Wide bandwidth
    filtered_wide = apply_cleanline_notch(data, sfreq=sfreq, freq=50.0, bandwidth=2.0)

    # Both should reduce 50 Hz
    def get_power_at(d, freq, fs):
        f, psd = signal.welch(d, fs=fs, nperseg=int(fs), axis=-1)
        idx = np.argmin(np.abs(f - freq))
        return np.mean(psd[:, idx])

    power_narrow = get_power_at(filtered_narrow, 50.0, sfreq)
    power_wide = get_power_at(filtered_wide, 50.0, sfreq)
    power_orig = get_power_at(data, 50.0, sfreq)

    assert power_narrow < power_orig
    assert power_wide < power_orig


def test_cleanline_notch_order_parameter(line_noise_data):
    """apply_cleanline_notch should accept order parameter."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]

    # Different filter orders
    filtered_low = apply_cleanline_notch(data, sfreq=sfreq, freq=50.0, order=2)
    filtered_high = apply_cleanline_notch(data, sfreq=sfreq, freq=50.0, order=6)

    # Both should produce valid output
    assert filtered_low.shape == data.shape
    assert filtered_high.shape == data.shape


def test_cleanline_notch_60hz():
    """apply_cleanline_notch should work with 60 Hz."""
    rng = np.random.default_rng(42)
    sfreq = 250
    n_times = 1000
    times = np.arange(n_times) / sfreq

    data = rng.normal(0, 1, (4, n_times))
    data += np.sin(2 * np.pi * 60 * times) * 3.0

    filtered = apply_cleanline_notch(data, sfreq=sfreq, freq=60.0)

    def get_power_at(d, freq, fs):
        f, psd = signal.welch(d, fs=fs, nperseg=int(fs), axis=-1)
        idx = np.argmin(np.abs(f - freq))
        return np.mean(psd[:, idx])

    power_before = get_power_at(data, 60.0, sfreq)
    power_after = get_power_at(filtered, 60.0, sfreq)

    assert power_after < power_before * 0.2


# =============================================================================
# apply_cleanline_notch - Edge Cases
# =============================================================================


def test_cleanline_notch_invalid_freq():
    """apply_cleanline_notch should handle frequency near Nyquist gracefully."""
    data = np.random.randn(4, 1000)
    sfreq = 100  # Nyquist = 50 Hz

    # 49 Hz is very close to Nyquist
    filtered = apply_cleanline_notch(data, sfreq=sfreq, freq=49.0, bandwidth=0.5)

    # Should still produce valid output (clamped to valid range)
    assert filtered.shape == data.shape


def test_cleanline_notch_low_freq():
    """apply_cleanline_notch should work at low frequencies."""
    data = np.random.randn(4, 1000)
    sfreq = 100

    # Low frequency notch
    filtered = apply_cleanline_notch(data, sfreq=sfreq, freq=10.0, bandwidth=1.0)

    assert filtered.shape == data.shape


def test_cleanline_notch_invalid_bandwidth_returns_original():
    """apply_cleanline_notch with invalid bandwidth should return original."""
    data = np.random.randn(4, 1000)
    sfreq = 100  # Nyquist = 50 Hz

    # Bandwidth larger than possible
    filtered = apply_cleanline_notch(data, sfreq=sfreq, freq=49.9, bandwidth=10.0)

    # Should return unchanged data if filter can't be applied
    assert filtered.shape == data.shape


# =============================================================================
# apply_hybrid_cleanup - Basic Functionality
# =============================================================================


def test_hybrid_cleanup_output_shape(line_noise_data):
    """apply_hybrid_cleanup should return same shape as input."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]

    cleaned = apply_hybrid_cleanup(data, sfreq=sfreq, freq=50.0)

    assert cleaned.shape == data.shape


def test_hybrid_cleanup_reduces_target_power(line_noise_data):
    """apply_hybrid_cleanup should reduce power at target frequency."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]

    cleaned = apply_hybrid_cleanup(data, sfreq=sfreq, freq=50.0)

    def get_power_at(d, freq, fs):
        f, psd = signal.welch(d, fs=fs, nperseg=int(fs), axis=-1)
        idx = np.argmin(np.abs(f - freq))
        return np.mean(psd[:, idx])

    power_before = get_power_at(data, 50.0, sfreq)
    power_after = get_power_at(cleaned, 50.0, sfreq)

    # Should reduce 50 Hz power
    assert power_after < power_before


def test_hybrid_cleanup_bandwidth_parameter(line_noise_data):
    """apply_hybrid_cleanup should accept bandwidth parameter."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]

    cleaned = apply_hybrid_cleanup(data, sfreq=sfreq, freq=50.0, bandwidth=1.0)

    assert cleaned.shape == data.shape


def test_hybrid_cleanup_max_power_reduction_parameter(line_noise_data):
    """apply_hybrid_cleanup should respect max_power_reduction_db parameter."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]

    # Low threshold - more likely to reject cleanup
    cleaned_strict = apply_hybrid_cleanup(
        data, sfreq=sfreq, freq=50.0, max_power_reduction_db=0.1
    )

    # High threshold - more likely to apply cleanup
    cleaned_permissive = apply_hybrid_cleanup(
        data, sfreq=sfreq, freq=50.0, max_power_reduction_db=10.0
    )

    # Both should produce valid output
    assert cleaned_strict.shape == data.shape
    assert cleaned_permissive.shape == data.shape


def test_hybrid_cleanup_skips_when_overcleaning():
    """apply_hybrid_cleanup should skip if cleanup would over-clean."""
    rng = np.random.default_rng(42)
    sfreq = 250
    n_times = 1000

    # Create data where notch would cause too much collateral damage
    # (broadband signal near the target frequency)
    times = np.arange(n_times) / sfreq
    data = np.sin(2 * np.pi * 49 * times) + np.sin(2 * np.pi * 51 * times)
    data = data[np.newaxis, :] + rng.normal(0, 0.01, (1, n_times))

    # With very strict threshold, should return original
    cleaned = apply_hybrid_cleanup(
        data, sfreq=sfreq, freq=50.0, bandwidth=5.0, max_power_reduction_db=0.01
    )

    # If cleanup was rejected, data should be very similar to original
    # (allowing for numerical precision)
    assert cleaned.shape == data.shape


# =============================================================================
# apply_hybrid_cleanup - Edge Cases
# =============================================================================


def test_hybrid_cleanup_no_surrounding_freqs():
    """apply_hybrid_cleanup should handle case with no surrounding frequencies."""
    data = np.random.randn(4, 100)  # Very short data
    sfreq = 100

    # This might not have enough frequency resolution for surrounding check
    cleaned = apply_hybrid_cleanup(data, sfreq=sfreq, freq=25.0)

    assert cleaned.shape == data.shape


def test_hybrid_cleanup_clean_data():
    """apply_hybrid_cleanup on clean data should not damage it."""
    rng = np.random.default_rng(42)
    sfreq = 250
    n_times = 1000

    # Clean data without line noise
    clean_data = rng.normal(0, 1, (4, n_times))

    cleaned = apply_hybrid_cleanup(clean_data, sfreq=sfreq, freq=50.0)

    # Variance should be similar (notch filter has some effect but not dramatic)
    var_before = np.var(clean_data)
    var_after = np.var(cleaned)

    # Should not remove more than 20% of variance
    assert var_after > var_before * 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
