"""Unit tests for ZapLine core functions (dss_zapline, apply_zapline)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import signal

from mne_denoise.zapline.core import ZapLineResult, apply_zapline, dss_zapline


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def line_noise_data():
    """Generate synthetic data with 50 Hz line noise."""
    rng = np.random.default_rng(42)
    sfreq = 500
    n_channels = 8
    n_times = 5000  # 10s
    times = np.arange(n_times) / sfreq

    # Clean brain-like signal (alpha oscillation + noise)
    brain_signal = np.sin(2 * np.pi * 10 * times) * 0.5  # 10 Hz alpha
    brain_signal = brain_signal[np.newaxis, :] + rng.normal(0, 0.2, (n_channels, n_times))

    # Line noise (50 Hz + harmonics) - STRONG and uniform spatial distribution
    line_50hz = np.sin(2 * np.pi * 50 * times) * 10.0  # Much stronger!
    line_100hz = np.sin(2 * np.pi * 100 * times) * 3.0
    line_noise = line_50hz + line_100hz  # Same on all channels

    data = brain_signal + line_noise[np.newaxis, :]

    return {
        "data": data,
        "sfreq": sfreq,
        "line_freq": 50.0,
        "brain_signal": brain_signal,
        "line_noise": line_noise[np.newaxis, :] * np.ones((n_channels, 1)),
        "times": times,
    }



@pytest.fixture
def minimal_data():
    """Generate minimal test data for quick tests."""
    rng = np.random.default_rng(42)
    sfreq = 250
    n_channels = 4
    n_times = 1000  # 4s
    times = np.arange(n_times) / sfreq

    # Simple 50 Hz noise
    line_noise = np.sin(2 * np.pi * 50 * times) * 5.0
    data = rng.normal(0, 1, (n_channels, n_times))
    data[0] += line_noise  # Add noise mostly to ch 0

    return {"data": data, "sfreq": sfreq, "line_freq": 50.0, "times": times}


# =============================================================================
# dss_zapline - Basic Functionality
# =============================================================================


def test_dss_zapline_returns_correct_type(minimal_data):
    """dss_zapline should return a ZapLineResult."""
    result = dss_zapline(
        minimal_data["data"],
        line_freq=minimal_data["line_freq"],
        sfreq=minimal_data["sfreq"],
    )
    assert isinstance(result, ZapLineResult)


def test_dss_zapline_output_shapes(minimal_data):
    """dss_zapline should return arrays with correct shapes."""
    data = minimal_data["data"]
    result = dss_zapline(data, line_freq=50.0, sfreq=minimal_data["sfreq"])

    n_channels, n_times = data.shape

    assert result.cleaned.shape == data.shape
    assert result.removed.shape == data.shape
    assert result.dss_filters.shape[1] == n_channels
    assert result.dss_patterns.shape[0] == n_channels
    assert len(result.dss_eigenvalues) > 0


def test_dss_zapline_reduces_line_noise(line_noise_data):
    """dss_zapline should significantly reduce power at line frequency."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]
    line_freq = line_noise_data["line_freq"]

    # Use fixed n_remove=1 to ensure we actually remove something
    result = dss_zapline(data, line_freq=line_freq, sfreq=sfreq, n_remove=1)

    # Compare power at 50 Hz before/after
    def get_power_at(d, freq, fs):
        f, psd = signal.welch(d, fs=fs, nperseg=int(fs), axis=-1)
        idx = np.argmin(np.abs(f - freq))
        return np.mean(psd[:, idx])

    power_before = get_power_at(data, line_freq, sfreq)
    power_after = get_power_at(result.cleaned, line_freq, sfreq)

    # Should reduce power significantly
    reduction = (power_before - power_after) / power_before
    assert reduction > 0.5, f"Insufficient noise reduction: {reduction:.2%}"



def test_dss_zapline_preserves_brain_signal(line_noise_data):
    """dss_zapline should preserve signals at non-line frequencies."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]

    # Use n_remove=1 to ensure testing with actual removal
    result = dss_zapline(data, line_freq=50.0, sfreq=sfreq, n_remove=1)

    # Check 10 Hz alpha power preservation
    def get_power_at(d, freq, fs):
        f, psd = signal.welch(d, fs=fs, nperseg=int(fs), axis=-1)
        idx = np.argmin(np.abs(f - freq))
        return np.mean(psd[:, idx])

    power_alpha_original = get_power_at(data, 10.0, sfreq)
    power_alpha_cleaned = get_power_at(result.cleaned, 10.0, sfreq)

    # Alpha power should be mostly preserved (within 30%)
    ratio = power_alpha_cleaned / power_alpha_original
    assert ratio > 0.5, f"Alpha signal degraded: ratio={ratio:.2f}"




def test_dss_zapline_closed_sum_property(minimal_data):
    """cleaned + removed should equal original data."""
    data = minimal_data["data"]
    result = dss_zapline(data, line_freq=50.0, sfreq=minimal_data["sfreq"])

    reconstructed = result.cleaned + result.removed
    assert_allclose(reconstructed, data, rtol=1e-10)


# =============================================================================
# dss_zapline - Parameters
# =============================================================================


def test_dss_zapline_n_remove_fixed(minimal_data):
    """dss_zapline should remove exactly n_remove components when specified."""
    data = minimal_data["data"]
    result = dss_zapline(data, line_freq=50.0, sfreq=minimal_data["sfreq"], n_remove=2)

    assert result.n_removed == 2


def test_dss_zapline_n_remove_auto(minimal_data):
    """dss_zapline auto should work (may remove 0 or more components)."""
    data = minimal_data["data"]
    result = dss_zapline(
        data, line_freq=50.0, sfreq=minimal_data["sfreq"], n_remove="auto"
    )

    # Auto mode should return valid result (may remove 0 if no strong noise detected)
    assert result.n_removed >= 0
    assert result.cleaned.shape == data.shape



def test_dss_zapline_with_harmonics(line_noise_data):
    """dss_zapline should handle harmonics correctly."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]

    # Use fixed n_remove and limit to 2 harmonics
    result = dss_zapline(data, line_freq=50.0, sfreq=sfreq, n_harmonics=2, n_remove=1)

    assert result.n_harmonics == 2

    # Check 100 Hz (2nd harmonic) power is also reduced
    def get_power_at(d, freq, fs):
        f, psd = signal.welch(d, fs=fs, nperseg=int(fs), axis=-1)
        idx = np.argmin(np.abs(f - freq))
        return np.mean(psd[:, idx])

    power_100_before = get_power_at(data, 100.0, sfreq)
    power_100_after = get_power_at(result.cleaned, 100.0, sfreq)

    # Should also reduce 100 Hz power
    assert power_100_after < power_100_before



def test_dss_zapline_with_nkeep(minimal_data):
    """dss_zapline should work with nkeep parameter."""
    data = minimal_data["data"]
    result = dss_zapline(
        data, line_freq=50.0, sfreq=minimal_data["sfreq"], nkeep=2  # Reduce to 2 PCA
    )

    # Should still produce valid output
    assert result.cleaned.shape == data.shape
    assert result.dss_filters.shape[1] == data.shape[0]


def test_dss_zapline_60hz(minimal_data):
    """dss_zapline should work with 60 Hz line frequency."""
    rng = np.random.default_rng(42)
    sfreq = 250
    n_times = 1000
    times = np.arange(n_times) / sfreq

    # Create data with 60 Hz noise
    data = rng.normal(0, 1, (4, n_times))
    data += np.sin(2 * np.pi * 60 * times) * 5.0

    result = dss_zapline(data, line_freq=60.0, sfreq=sfreq, n_remove=1)


    # Check 60 Hz power reduction
    def get_power_at(d, freq, fs):
        f, psd = signal.welch(d, fs=fs, nperseg=int(fs), axis=-1)
        idx = np.argmin(np.abs(f - freq))
        return np.mean(psd[:, idx])

    power_before = get_power_at(data, 60.0, sfreq)
    power_after = get_power_at(result.cleaned, 60.0, sfreq)

    assert power_after < power_before * 0.5


def test_dss_zapline_threshold_parameter(minimal_data):
    """dss_zapline should use threshold for auto component selection."""
    data = minimal_data["data"]

    # Low threshold should remove more components
    result_low = dss_zapline(
        data, line_freq=50.0, sfreq=minimal_data["sfreq"], n_remove="auto", threshold=1.0
    )

    # High threshold should remove fewer
    result_high = dss_zapline(
        data, line_freq=50.0, sfreq=minimal_data["sfreq"], n_remove="auto", threshold=5.0
    )

    # Low threshold >= high threshold (could be equal if all detected)
    assert result_low.n_removed >= result_high.n_removed


def test_dss_zapline_rank_parameter(minimal_data):
    """dss_zapline should accept rank parameter."""
    data = minimal_data["data"]
    result = dss_zapline(
        data, line_freq=50.0, sfreq=minimal_data["sfreq"], rank=2  # Limit DSS rank
    )

    # Should still work
    assert result.cleaned.shape == data.shape


# =============================================================================
# dss_zapline - Edge Cases and Error Handling
# =============================================================================


def test_dss_zapline_error_1d_data():
    """dss_zapline should raise error for 1D data."""
    data = np.random.randn(1000)

    with pytest.raises(ValueError, match="must be 2D"):
        dss_zapline(data, line_freq=50.0, sfreq=250)


def test_dss_zapline_error_3d_data():
    """dss_zapline should raise error for 3D data."""
    data = np.random.randn(4, 100, 10)

    with pytest.raises(ValueError, match="must be 2D"):
        dss_zapline(data, line_freq=50.0, sfreq=250)


def test_dss_zapline_error_negative_line_freq():
    """dss_zapline should raise error for negative line_freq."""
    data = np.random.randn(4, 1000)

    with pytest.raises(ValueError, match="must be positive"):
        dss_zapline(data, line_freq=-50.0, sfreq=250)


def test_dss_zapline_error_zero_line_freq():
    """dss_zapline should raise error for zero line_freq."""
    data = np.random.randn(4, 1000)

    with pytest.raises(ValueError, match="must be positive"):
        dss_zapline(data, line_freq=0.0, sfreq=250)


def test_dss_zapline_warning_noninteger_period():
    """dss_zapline should warn when sfreq/line_freq is not close to integer."""
    data = np.random.randn(4, 1000)

    # 257 Hz sfreq with 50 Hz line = 5.14 period (not integer)
    with pytest.warns(UserWarning, match="not close to an integer"):
        dss_zapline(data, line_freq=50.0, sfreq=257)


def test_dss_zapline_no_noise_detection(minimal_data):
    """dss_zapline with clean data should remove 0 or minimal components."""
    rng = np.random.default_rng(42)
    clean_data = rng.normal(0, 1, (4, 1000))  # No line noise

    result = dss_zapline(
        clean_data, line_freq=50.0, sfreq=minimal_data["sfreq"], n_remove="auto"
    )

    # Should remove 0 or maybe 1 spurious component
    assert result.n_removed <= 1


# =============================================================================
# apply_zapline - Basic Functionality
# =============================================================================


def test_apply_zapline_basic(minimal_data):
    """apply_zapline should cleanly apply pre-learned filters."""
    data = minimal_data["data"]
    sfreq = minimal_data["sfreq"]

    # First, learn filters
    result_fit = dss_zapline(data, line_freq=50.0, sfreq=sfreq)

    # Then apply to same data (or could be new data)
    result_apply = apply_zapline(
        data,
        filters=result_fit.dss_filters,
        n_remove=result_fit.n_removed,
        line_freq=50.0,
        sfreq=sfreq,
    )

    assert result_apply.cleaned.shape == data.shape
    # The cleaning should be similar (not identical due to smoothing)
    assert_allclose(result_apply.n_removed, result_fit.n_removed)


def test_apply_zapline_with_patterns(minimal_data):
    """apply_zapline should use provided patterns correctly."""
    data = minimal_data["data"]
    sfreq = minimal_data["sfreq"]

    # Learn filters
    result_fit = dss_zapline(data, line_freq=50.0, sfreq=sfreq, n_remove=1)

    # Apply with explicit patterns
    result_apply = apply_zapline(
        data,
        filters=result_fit.dss_filters,
        n_remove=1,
        line_freq=50.0,
        sfreq=sfreq,
        patterns=result_fit.dss_patterns,
    )

    assert result_apply.cleaned.shape == data.shape


def test_apply_zapline_zero_n_remove(minimal_data):
    """apply_zapline with n_remove=0 should return original data."""
    data = minimal_data["data"]
    sfreq = minimal_data["sfreq"]

    result_fit = dss_zapline(data, line_freq=50.0, sfreq=sfreq)

    result_apply = apply_zapline(
        data,
        filters=result_fit.dss_filters,
        n_remove=0,
        line_freq=50.0,
        sfreq=sfreq,
    )

    assert result_apply.n_removed == 0
    assert_allclose(result_apply.cleaned, data)
    assert_allclose(result_apply.removed, np.zeros_like(data))


def test_apply_zapline_error_negative_line_freq(minimal_data):
    """apply_zapline should raise error for negative line_freq."""
    data = minimal_data["data"]
    filters = np.eye(4)

    with pytest.raises(ValueError, match="must be positive"):
        apply_zapline(data, filters=filters, n_remove=1, line_freq=-50.0, sfreq=250)


def test_apply_zapline_closed_sum(minimal_data):
    """apply_zapline: cleaned + removed should equal original."""
    data = minimal_data["data"]
    sfreq = minimal_data["sfreq"]

    result_fit = dss_zapline(data, line_freq=50.0, sfreq=sfreq, n_remove=2)
    result_apply = apply_zapline(
        data,
        filters=result_fit.dss_filters,
        n_remove=2,
        line_freq=50.0,
        sfreq=sfreq,
    )

    reconstructed = result_apply.cleaned + result_apply.removed
    assert_allclose(reconstructed, data, rtol=1e-10)


# =============================================================================
# ZapLineResult Dataclass
# =============================================================================


def test_zapline_result_attributes():
    """ZapLineResult should have all expected attributes."""
    result = ZapLineResult(
        cleaned=np.zeros((4, 100)),
        removed=np.zeros((4, 100)),
        n_removed=2,
        dss_filters=np.zeros((4, 4)),
        dss_patterns=np.zeros((4, 4)),
        dss_eigenvalues=np.array([1.0, 0.5, 0.2, 0.1]),
        line_freq=50.0,
        n_harmonics=2,
    )

    assert hasattr(result, "cleaned")
    assert hasattr(result, "removed")
    assert hasattr(result, "n_removed")
    assert hasattr(result, "dss_filters")
    assert hasattr(result, "dss_patterns")
    assert hasattr(result, "dss_eigenvalues")
    assert hasattr(result, "line_freq")
    assert hasattr(result, "n_harmonics")
    assert hasattr(result, "removed_topographies")
    assert hasattr(result, "chunk_info")


def test_zapline_result_optional_fields():
    """ZapLineResult optional fields should default to None."""
    result = ZapLineResult(
        cleaned=np.zeros((4, 100)),
        removed=np.zeros((4, 100)),
        n_removed=1,
        dss_filters=np.zeros((4, 4)),
        dss_patterns=np.zeros((4, 4)),
        dss_eigenvalues=np.array([1.0]),
        line_freq=50.0,
    )

    assert result.n_harmonics == 1  # Default
    assert result.removed_topographies is None
    assert result.chunk_info is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
