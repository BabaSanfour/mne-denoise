"""Unit tests for artifact denoisers (CycleAverageBias)."""

import numpy as np
from numpy.testing import assert_allclose

from mne_denoise.dss.denoisers.artifact import CycleAverageBias


def test_cycle_average_bias():
    """Test CycleAverageBias on synthetic data."""
    rng = np.random.default_rng(42)
    n_channels = 3
    n_times = 1000

    # 1. Create synthetic periodic artifact (e.g., heartbeat)
    events = np.arange(100, n_times - 100, 200)
    artifact_signal = np.zeros(n_times)

    template_len = 50
    template = np.hanning(template_len)

    for event in events:
        start = event - template_len // 2
        end = start + template_len
        if start >= 0 and end <= n_times:
            artifact_signal[start:end] += template

    artifact_data = np.outer([1.0, 0.5, -0.5], artifact_signal)

    # 2. Add asynchronous noise (simulated brain activity)
    noise = rng.normal(0, 0.1, (n_channels, n_times))
    data = artifact_data + noise

    # 3. Apply CycleAverageBias
    window = (-25, 25)
    bias = CycleAverageBias(event_samples=events, window=window)
    biased_data = bias.apply(data)

    # 4. Verification
    mask = np.ones(n_times, dtype=bool)
    for event in events:
        mask[event + window[0] : event + window[1]] = False

    assert_allclose(
        biased_data[:, mask],
        0,
        atol=1e-10,
        err_msg="Biased data should be zero outside windows",
    )

    # Extract one window from biased data
    event = events[0]
    biased_epoch = biased_data[:, event + window[0] : event + window[1]]

    # Theoretical clean epoch
    clean_epoch = artifact_data[:, event + window[0] : event + window[1]]

    # Correlation should be high
    corr = np.corrcoef(biased_epoch.ravel(), clean_epoch.ravel())[0, 1]
    assert (
        corr > 0.95
    ), f"Biased data should correlate with clean artifact (got {corr:.3f})"

    # Check shape preservation
    assert biased_data.shape == data.shape


def test_cycle_average_bias_sfreq():
    """Test CycleAverageBias with second-based window."""
    events = [100, 200]
    sfreq = 100.0

    # Window: -0.1s to +0.2s -> -10 to +20 samples
    bias = CycleAverageBias(event_samples=events, window=(-0.1, 0.2), sfreq=sfreq)

    assert bias.window == (-10, 20)


def test_cycle_average_bias_3d_data():
    """Test CycleAverageBias with 3D epoched data."""
    rng = np.random.default_rng(42)
    n_channels, n_times, n_epochs = 3, 100, 5

    # Create 3D data
    data = rng.normal(0, 1, (n_channels, n_times, n_epochs))

    # Add a periodic artifact at known locations
    events = np.array([20, 50, 80])  # Within first epoch

    bias = CycleAverageBias(event_samples=events, window=(-5, 5))
    biased = bias.apply(data)

    # Output should be 3D with same shape
    assert biased.shape == data.shape
    assert biased.ndim == 3


def test_cycle_average_bias_empty_events():
    """Test CycleAverageBias with no valid events returns zeros."""
    rng = np.random.default_rng(42)
    n_channels, n_times = 3, 100
    data = rng.normal(0, 1, (n_channels, n_times))

    # Events outside data bounds
    events = np.array([1000, 2000])  # Way outside

    bias = CycleAverageBias(event_samples=events, window=(-10, 10))
    biased = bias.apply(data)

    # Should return zeros when no valid events
    assert_allclose(biased, 0)
    assert biased.shape == data.shape


def test_cycle_average_bias_invalid_ndim():
    """Test CycleAverageBias raises error for invalid dimensions."""
    import pytest

    events = [50]
    bias = CycleAverageBias(event_samples=events, window=(-5, 5))

    # 1D data should raise ValueError
    data_1d = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="Data must be 2D or 3D"):
        bias.apply(data_1d)


def test_cycle_average_bias_partial_valid_events():
    """Test CycleAverageBias handles mix of valid and invalid events."""
    rng = np.random.default_rng(42)
    n_channels, n_times = 2, 200
    data = rng.normal(0, 1, (n_channels, n_times))

    # Mix of valid and invalid events (some at edges)
    events = np.array([5, 50, 100, 195])  # 5 and 195 may be clipped by window

    bias = CycleAverageBias(event_samples=events, window=(-10, 10))
    biased = bias.apply(data)

    # Should still work with the valid events in the middle
    assert biased.shape == data.shape
    # Middle event (50) window should be non-zero
    assert np.any(biased[:, 40:60] != 0)
