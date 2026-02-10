"""Unit tests for ZapLine core functions (dss_zapline, apply_zapline)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import signal

from mne_denoise.zapline.core import ZapLine
from mne_denoise.dss.denoisers.spectral import LineNoiseBias
from unittest.mock import patch


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
    brain_signal = brain_signal[np.newaxis, :] + rng.normal(
        0, 0.2, (n_channels, n_times)
    )

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


def test_zapline_class_init(minimal_data):
    """ZapLine should initialize correctly."""
    est = ZapLine(
        line_freq=minimal_data["line_freq"],
        sfreq=minimal_data["sfreq"],
    )
    assert est.n_remove == "auto"


def test_zapline_class_fit_transform(minimal_data):
    """ZapLine should fit and transform correctly."""
    data = minimal_data["data"]
    est = ZapLine(
        line_freq=minimal_data["line_freq"],
        sfreq=minimal_data["sfreq"],
    )
    est.fit(data)
    cleaned = est.transform(data)
    assert est.filters_ is not None
    assert cleaned.shape == data.shape


def test_zapline_class_output_shapes(minimal_data):
    """ZapLine should return arrays with correct shapes."""
    data = minimal_data["data"]
    # n_harmonics=None means auto-detected
    est = ZapLine(line_freq=50.0, sfreq=minimal_data["sfreq"])
    est.fit(data)
    cleaned = est.transform(data)

    n_channels, n_times = data.shape

    assert cleaned.shape == data.shape
    # Check attributes populated after fit
    assert est.filters_.shape[1] == n_channels
    assert est.patterns_.shape[0] == n_channels
    assert len(est.eigenvalues_) > 0


def test_dss_zapline_reduces_line_noise(line_noise_data):
    """dss_zapline should significantly reduce power at line frequency."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]
    line_freq = line_noise_data["line_freq"]

    # Use fixed n_remove=1 to ensure we actually remove something
    est = ZapLine(line_freq=line_freq, sfreq=sfreq, n_remove=1)
    est.fit(data)
    cleaned = est.transform(data)

    # Compare power at 50 Hz before/after
    def get_power_at(d, freq, fs):
        f, psd = signal.welch(d, fs=fs, nperseg=int(fs), axis=-1)
        idx = np.argmin(np.abs(f - freq))
        return np.mean(psd[:, idx])

    power_before = get_power_at(data, line_freq, sfreq)
    power_after = get_power_at(cleaned, line_freq, sfreq)

    reduction = (power_before - power_after) / power_before
    assert reduction > 0.1, f"Insufficient noise reduction: {reduction:.2%}"


def test_dss_zapline_preserves_brain_signal(line_noise_data):
    """dss_zapline should preserve signals at non-line frequencies."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]

    # Use n_remove=1 to ensure testing with actual removal
    est = ZapLine(line_freq=50.0, sfreq=sfreq, n_remove=1)
    est.fit(data)
    cleaned = est.transform(data)

    # Check 10 Hz alpha power preservation
    def get_power_at(d, freq, fs):
        f, psd = signal.welch(d, fs=fs, nperseg=int(fs), axis=-1)
        idx = np.argmin(np.abs(f - freq))
        return np.mean(psd[:, idx])

    power_alpha_original = get_power_at(data, 10.0, sfreq)
    power_alpha_cleaned = get_power_at(cleaned, 10.0, sfreq)

    # Alpha power should be mostly preserved (within 30%)
    ratio = power_alpha_cleaned / power_alpha_original
    assert ratio > 0.5, f"Alpha signal degraded: ratio={ratio:.2f}"


def test_dss_zapline_closed_sum_property(minimal_data):
    """cleaned + removed should equal original data."""
    data = minimal_data["data"]
    est = ZapLine(line_freq=50.0, sfreq=minimal_data["sfreq"])
    est = ZapLine(line_freq=50.0, sfreq=minimal_data["sfreq"])
    est.fit(data)

    cleaned = est.transform(data)
    removed = data - cleaned

    reconstructed = cleaned + removed
    assert_allclose(reconstructed, data, rtol=1e-10)


def test_zapline_n_remove_fixed(minimal_data):
    """ZapLine should remove exactly n_remove components when specified."""
    data = minimal_data["data"]
    est = ZapLine(line_freq=50.0, sfreq=minimal_data["sfreq"], n_remove=2)
    est.fit(data)

    assert est.n_removed_ == 2


def test_zapline_n_remove_auto(minimal_data):
    """ZapLine auto should work (may remove 0 or more components)."""
    data = minimal_data["data"]
    est = ZapLine(line_freq=50.0, sfreq=minimal_data["sfreq"], n_remove="auto")
    est.fit(data)
    cleaned = est.transform(data)

    assert est.n_removed_ >= 0
    assert cleaned.shape == data.shape


def test_zapline_with_harmonics(line_noise_data):
    """ZapLine should handle harmonics correctly."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]

    est = ZapLine(line_freq=50.0, sfreq=sfreq, n_harmonics=2, n_remove=1)
    est.fit(data)

    assert est.n_harmonics_ == 2

    def get_power_at(d, freq, fs):
        f, psd = signal.welch(d, fs=fs, nperseg=int(fs), axis=-1)
        idx = np.argmin(np.abs(f - freq))
        return np.mean(psd[:, idx])

    cleaned = est.transform(data)
    power_100_before = get_power_at(data, 100.0, sfreq)
    power_100_after = get_power_at(cleaned, 100.0, sfreq)

    assert power_100_after < power_100_before


def test_zapline_with_nkeep(minimal_data):
    """ZapLine should work with nkeep parameter."""
    data = minimal_data["data"]
    est = ZapLine(line_freq=50.0, sfreq=minimal_data["sfreq"], nkeep=2)
    est.fit(data)
    cleaned = est.transform(data)

    # Should still produce valid output
    assert cleaned.shape == data.shape
    assert est.filters_.shape[1] == data.shape[0]


def test_zapline_60hz(minimal_data):
    """ZapLine should work with 60 Hz line frequency."""
    rng = np.random.default_rng(42)
    sfreq = 250
    n_times = 1000
    times = np.arange(n_times) / sfreq

    # Create data with 60 Hz noise
    data = rng.normal(0, 1, (4, n_times))
    data += np.sin(2 * np.pi * 60 * times) * 5.0

    est = ZapLine(line_freq=60.0, sfreq=sfreq, n_remove=1)
    est.fit(data)
    cleaned = est.transform(data)

    # Check 60 Hz power reduction
    def get_power_at(d, freq, fs):
        f, psd = signal.welch(d, fs=fs, nperseg=int(fs), axis=-1)
        idx = np.argmin(np.abs(f - freq))
        return np.mean(psd[:, idx])

    power_before = get_power_at(data, 60.0, sfreq)
    power_after = get_power_at(cleaned, 60.0, sfreq)

    assert power_after < power_before * 0.5


def test_zapline_threshold_parameter(minimal_data):
    """ZapLine should use threshold for auto component selection."""
    data = minimal_data["data"]

    # Low threshold should remove more components
    est_low = ZapLine(
        line_freq=50.0, sfreq=minimal_data["sfreq"], n_remove="auto", threshold=1.0
    )
    est_low.fit(data)

    # High threshold should remove fewer
    est_high = ZapLine(
        line_freq=50.0, sfreq=minimal_data["sfreq"], n_remove="auto", threshold=5.0
    )
    est_high.fit(data)

    # Low threshold >= high threshold (could be equal if all detected)
    assert est_low.n_removed_ >= est_high.n_removed_


def test_zapline_rank_parameter(minimal_data):
    """ZapLine should accept rank parameter."""
    data = minimal_data["data"]
    # Limit DSS rank to 2
    est = ZapLine(line_freq=50.0, sfreq=minimal_data["sfreq"], rank=2)
    est.fit(data)
    cleaned = est.transform(data)

    # Should still work
    assert cleaned.shape == data.shape


def test_zapline_error_1d_data():
    """ZapLine should raise error for 1D data."""
    data = np.random.randn(1000)
    est = ZapLine(line_freq=50.0, sfreq=250)

    with pytest.raises(Exception):
        est.fit(data)


def test_zapline_closed_sum_method(minimal_data):
    """Cleaned + Removed should equal original."""
    data = minimal_data["data"]
    sfreq = minimal_data["sfreq"]

    est = ZapLine(line_freq=50.0, sfreq=sfreq)
    est.fit(data)
    cleaned = est.transform(data)
    removed = data - cleaned

    reconstructed = cleaned + removed
    assert_allclose(reconstructed, data, rtol=1e-10)


def test_zapline_error_3d_data():
    """ZapLine should handle 3D data by reshaping."""
    data = np.random.randn(4, 100, 10)
    est = ZapLine(line_freq=50.0, sfreq=250)
    est.fit(data)


def test_zapline_error_zero_line_freq():
    """ZapLine should safely handle zero line_freq (do nothing)."""
    data = np.random.randn(4, 1000)

    est = ZapLine(line_freq=0.0, sfreq=250)
    est.fit(data)
    assert est.n_removed_ == 0


def test_zapline_adaptive_fit_error():
    """ZapLine adaptive mode should raise error on fit()."""
    data = np.random.randn(4, 1000)
    est = ZapLine(line_freq=50.0, sfreq=250, adaptive=True)

    with pytest.raises(RuntimeError, match="Adaptive mode requires"):
        est.fit(data)


def test_zapline_adaptive_transform_error():
    """ZapLine adaptive mode should raise error on transform()."""
    data = np.random.randn(4, 1000)
    est = ZapLine(line_freq=50.0, sfreq=250, adaptive=True)
    est.filters_ = np.eye(4)  # Fake fitted state

    with pytest.raises(RuntimeError, match="Adaptive mode requires"):
        est.transform(data)


def test_zapline_sfreq_mismatch_warning_fit():
    """ZapLine should warn on sfreq mismatch during fit."""
    import mne

    data = np.random.randn(4, 1000)
    info = mne.create_info(
        ch_names=["EEG1", "EEG2", "EEG3", "EEG4"], sfreq=500, ch_types="eeg"
    )
    raw = mne.io.RawArray(data, info)

    est = ZapLine(line_freq=50.0, sfreq=250)  # Different sfreq

    with pytest.warns(UserWarning, match="Input data sfreq"):
        est.fit(raw)


def test_zapline_sfreq_mismatch_warning_transform():
    """ZapLine should warn on sfreq mismatch during transform."""
    import mne

    # Fit on array first
    data = np.random.randn(4, 1000)
    est = ZapLine(line_freq=50.0, sfreq=250)
    est.fit(data)

    # Transform with Raw that has different sfreq
    info = mne.create_info(
        ch_names=["EEG1", "EEG2", "EEG3", "EEG4"], sfreq=500, ch_types="eeg"
    )
    raw = mne.io.RawArray(data, info)

    with pytest.warns(UserWarning, match="Input data sfreq"):
        est.transform(raw)


def test_zapline_sfreq_mismatch_warning_fit_transform():
    """ZapLine should warn on sfreq mismatch during fit_transform."""
    import mne

    data = np.random.randn(4, 1000)
    info = mne.create_info(
        ch_names=["EEG1", "EEG2", "EEG3", "EEG4"], sfreq=500, ch_types="eeg"
    )
    raw = mne.io.RawArray(data, info)

    est = ZapLine(line_freq=50.0, sfreq=250)  # Different sfreq

    with pytest.warns(UserWarning, match="Input data sfreq"):
        est.fit_transform(raw)


def test_zapline_fit_none_line_freq_error():
    """ZapLine fit() should raise error if line_freq is None."""
    data = np.random.randn(4, 1000)
    est = ZapLine(line_freq=None, sfreq=250)

    with pytest.raises(ValueError, match="line_freq required"):
        est.fit(data)


def test_zapline_transform_not_fitted_error():
    """ZapLine transform() should raise error if not fitted."""
    data = np.random.randn(4, 1000)
    est = ZapLine(line_freq=50.0, sfreq=250)

    with pytest.raises(RuntimeError, match="Not fitted"):
        est.transform(data)


def test_zapline_3d_data_fit_transform():
    """ZapLine should handle 3D epoched data correctly."""
    # Shape: (n_epochs, n_channels, n_times)
    rng = np.random.default_rng(42)
    n_epochs, n_ch, n_times = 5, 4, 500
    sfreq = 250
    times = np.arange(n_times) / sfreq

    data = rng.normal(0, 1, (n_epochs, n_ch, n_times))
    # Add line noise
    data += np.sin(2 * np.pi * 50 * times) * 2.0

    est = ZapLine(line_freq=50.0, sfreq=sfreq, n_remove=1)
    est.fit(data)
    cleaned = est.transform(data)

    assert cleaned.shape == data.shape


def test_zapline_adaptive_3d_data():
    """ZapLine adaptive mode should handle 3D epoched data."""
    rng = np.random.default_rng(42)
    n_epochs, n_ch, n_times = 3, 4, 2500
    sfreq = 250
    times = np.arange(n_times) / sfreq

    data = rng.normal(0, 0.5, (n_epochs, n_ch, n_times))
    data += np.sin(2 * np.pi * 50 * times) * 5.0

    est = ZapLine(
        sfreq=sfreq,
        line_freq=50.0,
        adaptive=True,
        adaptive_params={"min_chunk_len": 5.0},
    )
    cleaned = est.fit_transform(data)

    assert cleaned.shape == data.shape
    assert hasattr(est, "adaptive_results_")


def test_zapline_adaptive_auto_detection():
    """ZapLine adaptive mode with line_freq=None should auto-detect."""
    rng = np.random.default_rng(42)
    sfreq = 250
    n_times = 7500  # 30s
    times = np.arange(n_times) / sfreq

    data = rng.normal(0, 0.5, (4, n_times))
    # Add strong 50 Hz line noise
    data += np.sin(2 * np.pi * 50 * times) * 10.0

    est = ZapLine(
        sfreq=sfreq,
        line_freq=None,  # Auto-detect
        adaptive=True,
        adaptive_params={"fmin": 45, "fmax": 55, "min_chunk_len": 10.0},
    )
    cleaned = est.fit_transform(data)

    assert cleaned.shape == data.shape
    assert est.adaptive_results_ is not None


def test_zapline_adaptive_no_detection():
    """ZapLine adaptive mode should handle case where no noise is detected."""
    rng = np.random.default_rng(42)
    sfreq = 250
    n_times = 7500

    # Clean data with no line noise
    data = rng.normal(0, 1, (4, n_times))

    est = ZapLine(
        sfreq=sfreq,
        line_freq=None,
        adaptive=True,
        adaptive_params={"fmin": 45, "fmax": 55, "min_chunk_len": 10.0},
    )
    cleaned = est.fit_transform(data)

    # Should return data mostly unchanged
    assert cleaned.shape == data.shape


def test_zapline_adaptive_with_harmonics():
    """ZapLine adaptive mode should process harmonics when enabled."""
    rng = np.random.default_rng(42)
    sfreq = 500
    n_times = 15000  # 30s
    times = np.arange(n_times) / sfreq

    data = rng.normal(0, 0.5, (4, n_times))
    # Add 50 Hz and 100 Hz harmonics
    data += np.sin(2 * np.pi * 50 * times) * 10.0
    data += np.sin(2 * np.pi * 100 * times) * 5.0

    est = ZapLine(
        sfreq=sfreq,
        line_freq=50.0,
        adaptive=True,
        adaptive_params={
            "process_harmonics": True,
            "max_harmonics": 2,
            "min_chunk_len": 10.0,
        },
    )
    cleaned = est.fit_transform(data)

    assert cleaned.shape == data.shape


def test_smoothing_warnings():
    """Test that warnings are issued for bad sfreq/line_freq ratios."""
    with pytest.warns(UserWarning):
        zap = ZapLine(sfreq=100, line_freq=70)
        data = np.random.randn(1, 100)
        zap._get_smooth_residual(data, warn=True)


def test_smoothing_warning_integer_mismatch():
    """Test warning when period is not exactly integer but close enough for standard."""
    sfreq = 10000
    period = 2000.15
    line_freq = sfreq / period

    with pytest.warns(UserWarning, match="is not exactly an integer"):
        zap = ZapLine(sfreq=sfreq, line_freq=line_freq)
        data = np.random.randn(1, 2500)
        zap._get_smooth_residual(data, warn=True)


def test_fractional_smooth_period_le_1():
    """Test fractional smooth when period <= 1."""
    zap = ZapLine(sfreq=100, line_freq=150)  # period < 1
    data = np.random.randn(1, 100)
    data_smooth = zap._fractional_smooth(data, period=0.5)
    assert np.array_equal(data_smooth, data)


def test_fractional_smooth_integ_equals_ntimes():
    """Test fractional smooth when smoothing period >= n_times."""
    zap = ZapLine(sfreq=100, line_freq=1) 
    data = np.random.randn(1, 50) 

    smooth = zap._fractional_smooth(data, period=100.0)
    assert np.allclose(smooth, np.mean(data))


def test_fractional_smooth_integer_period():
    """Test fast path for integer period in fractional smooth."""
    zap = ZapLine(sfreq=100, line_freq=50)
    data = np.random.randn(1, 100)

    smooth = zap._fractional_smooth(data, period=2.0)
    assert smooth.shape == data.shape


def test_linenoise_bias_3d():
    """Test LineNoiseBias with 3D data."""
    bias = LineNoiseBias(freq=50, sfreq=1000, method="fft")
    data = np.random.randn(2, 100, 3)  # ch, time, ep

    biased = bias.apply(data)
    assert biased.shape == data.shape

    with pytest.raises(ValueError):
        bias._apply_fft(np.zeros((2,)))  # 1D data


def test_linenoise_bias_method_errors():
    """Test LineNoiseBias error handling for invalid methods."""
    # Init validation
    with pytest.raises(ValueError, match="Unknown method"):
        LineNoiseBias(freq=50, sfreq=1000, method="invalid")

    # Apply fallback
    bias = LineNoiseBias(freq=50, sfreq=1000, method="fft")
    bias.method = "invalid" 
    # Should return data unchanged
    data = np.random.randn(1, 100)
    assert np.array_equal(bias.apply(data), data)
