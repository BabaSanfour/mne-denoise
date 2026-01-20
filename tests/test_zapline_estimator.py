"""Unit tests for ZapLine Transformer (estimator class)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import signal

import mne
from mne_denoise.zapline.estimator import ZapLine


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

    # Clean signal
    brain = rng.normal(0, 0.5, (n_channels, n_times))
    brain[0] += np.sin(2 * np.pi * 10 * times)  # Add alpha

    # 50 Hz noise
    line_noise = np.sin(2 * np.pi * 50 * times) * 3.0
    data = brain + line_noise[np.newaxis, :]

    return {"data": data, "sfreq": sfreq, "line_freq": 50.0, "times": times}


@pytest.fixture
def mne_raw(line_noise_data):
    """Create MNE Raw object."""
    info = mne.create_info(
        ch_names=["EEG1", "EEG2", "EEG3", "EEG4"],
        sfreq=line_noise_data["sfreq"],
        ch_types="eeg",
    )
    raw = mne.io.RawArray(line_noise_data["data"], info, verbose=False)
    return raw


@pytest.fixture
def mne_epochs(line_noise_data):
    """Create MNE Epochs object."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]

    # Reshape to epochs (5 epochs x 4 channels x 500 samples)
    n_epochs = 5
    epoch_len = 500
    data_epochs = data[:, : n_epochs * epoch_len].reshape(
        data.shape[0], n_epochs, epoch_len
    )
    data_epochs = data_epochs.transpose(1, 0, 2)  # (n_epochs, n_ch, n_times)

    info = mne.create_info(
        ch_names=["EEG1", "EEG2", "EEG3", "EEG4"], sfreq=sfreq, ch_types="eeg"
    )
    epochs = mne.EpochsArray(data_epochs, info, verbose=False)
    return epochs


# =============================================================================
# ZapLine Class - Initialization
# =============================================================================


def test_zapline_init_defaults():
    """ZapLine should initialize with sensible defaults."""
    zl = ZapLine()
    
    assert zl.line_freq == 60.0  # US default
    assert zl.n_remove == "auto"
    assert zl.adaptive is False


def test_zapline_init_custom():
    """ZapLine should accept custom parameters."""
    zl = ZapLine(
        line_freq=50.0,
        sfreq=250,
        n_remove=3,
        n_harmonics=2,
        nfft=512,
        threshold=2.5,
        adaptive=True,
    )

    assert zl.line_freq == 50.0
    assert zl.sfreq == 250
    assert zl.n_remove == 3
    assert zl.n_harmonics == 2
    assert zl.nfft == 512
    assert zl.threshold == 2.5
    assert zl.adaptive is True


# =============================================================================
# ZapLine Class - fit()
# =============================================================================


def test_zapline_fit_array(line_noise_data):
    """ZapLine.fit should work with numpy arrays."""
    data = line_noise_data["data"]
    zl = ZapLine(line_freq=50.0, sfreq=line_noise_data["sfreq"])
    
    result = zl.fit(data)
    
    assert result is zl  # Returns self
    assert hasattr(zl, "filters_")
    assert hasattr(zl, "patterns_")
    assert hasattr(zl, "n_removed_")


def test_zapline_fit_mne_raw(mne_raw):
    """ZapLine.fit should work with MNE Raw objects."""
    zl = ZapLine(line_freq=50.0)
    
    result = zl.fit(mne_raw)
    
    assert result is zl
    assert hasattr(zl, "filters_")
    assert zl.sfreq_ == mne_raw.info["sfreq"]


def test_zapline_fit_mne_epochs(mne_epochs):
    """ZapLine.fit should work with MNE Epochs objects."""
    zl = ZapLine(line_freq=50.0)
    
    result = zl.fit(mne_epochs)
    
    assert result is zl
    assert hasattr(zl, "filters_")


def test_zapline_fit_extracts_sfreq_from_mne(mne_raw):
    """ZapLine should extract sfreq from MNE object if not provided."""
    zl = ZapLine(line_freq=50.0, sfreq=None)
    zl.fit(mne_raw)
    
    assert zl.sfreq_ == mne_raw.info["sfreq"]


def test_zapline_fit_fixed_n_remove(line_noise_data):
    """ZapLine.fit should respect fixed n_remove."""
    data = line_noise_data["data"]
    zl = ZapLine(line_freq=50.0, sfreq=line_noise_data["sfreq"], n_remove=2)
    
    zl.fit(data)
    
    assert zl.n_removed_ == 2


# =============================================================================
# ZapLine Class - transform()
# =============================================================================


def test_zapline_transform_array(line_noise_data):
    """ZapLine.transform should clean numpy arrays."""
    data = line_noise_data["data"]
    zl = ZapLine(line_freq=50.0, sfreq=line_noise_data["sfreq"])
    
    zl.fit(data)
    cleaned = zl.transform(data)
    
    assert cleaned.shape == data.shape
    assert isinstance(cleaned, np.ndarray)


def test_zapline_transform_mne_raw(mne_raw):
    """ZapLine.transform should return MNE Raw object when given Raw."""
    zl = ZapLine(line_freq=50.0)
    
    zl.fit(mne_raw)
    cleaned = zl.transform(mne_raw)
    
    assert isinstance(cleaned, mne.io.BaseRaw)
    assert cleaned.info["nchan"] == mne_raw.info["nchan"]


def test_zapline_transform_mne_epochs(mne_epochs):
    """ZapLine.transform should return MNE Epochs when given Epochs."""
    zl = ZapLine(line_freq=50.0)
    
    zl.fit(mne_epochs)
    cleaned = zl.transform(mne_epochs)
    
    assert isinstance(cleaned, mne.BaseEpochs)


def test_zapline_transform_reduces_line_power(line_noise_data):
    """ZapLine.transform should reduce line noise power."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]
    
    zl = ZapLine(line_freq=50.0, sfreq=sfreq, n_remove=1)
    zl.fit(data)

    cleaned = zl.transform(data)
    
    def get_power_at(d, freq, fs):
        f, psd = signal.welch(d, fs=fs, nperseg=int(fs), axis=-1)
        idx = np.argmin(np.abs(f - freq))
        return np.mean(psd[:, idx])
    
    power_before = get_power_at(data, 50.0, sfreq)
    power_after = get_power_at(cleaned, 50.0, sfreq)
    
    assert power_after < power_before * 0.3


def test_zapline_transform_before_fit_error(line_noise_data):
    """ZapLine.transform should raise error if called before fit."""
    data = line_noise_data["data"]
    zl = ZapLine(line_freq=50.0, sfreq=line_noise_data["sfreq"])
    
    with pytest.raises((AttributeError, ValueError, RuntimeError)):
        zl.transform(data)



# =============================================================================
# ZapLine Class - fit_transform()
# =============================================================================


def test_zapline_fit_transform_array(line_noise_data):
    """ZapLine.fit_transform should fit and clean in one step."""
    data = line_noise_data["data"]
    zl = ZapLine(line_freq=50.0, sfreq=line_noise_data["sfreq"])
    
    cleaned = zl.fit_transform(data)
    
    assert cleaned.shape == data.shape
    assert hasattr(zl, "filters_")


def test_zapline_fit_transform_mne_raw(mne_raw):
    """ZapLine.fit_transform should work with MNE Raw."""
    zl = ZapLine(line_freq=50.0)
    
    cleaned = zl.fit_transform(mne_raw)
    
    assert isinstance(cleaned, mne.io.BaseRaw)


def test_zapline_fit_transform_equals_fit_then_transform(line_noise_data):
    """fit_transform should give same result as fit + transform."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]
    
    # fit_transform
    zl1 = ZapLine(line_freq=50.0, sfreq=sfreq, n_remove=2)
    cleaned1 = zl1.fit_transform(data)
    
    # fit then transform
    zl2 = ZapLine(line_freq=50.0, sfreq=sfreq, n_remove=2)
    zl2.fit(data)
    cleaned2 = zl2.transform(data)
    
    assert_allclose(cleaned1, cleaned2)


# =============================================================================
# ZapLine Class - Adaptive Mode
# =============================================================================


def test_zapline_adaptive_mode(line_noise_data):
    """ZapLine with adaptive=True should use adaptive segmentation."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]
    
    zl = ZapLine(line_freq=50.0, sfreq=sfreq, adaptive=True)
    cleaned = zl.fit_transform(data)
    
    assert cleaned.shape == data.shape


def test_zapline_adaptive_params(line_noise_data):
    """ZapLine should accept adaptive_params."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]
    
    zl = ZapLine(
        line_freq=50.0,
        sfreq=sfreq,
        adaptive=True,
        adaptive_params={"min_chunk_len": 10.0},
    )
    cleaned = zl.fit_transform(data)
    
    assert cleaned.shape == data.shape


# =============================================================================
# ZapLine Class - Error Handling
# =============================================================================


def test_zapline_error_no_sfreq_for_array():
    """ZapLine should raise error when sfreq not provided for arrays."""
    data = np.random.randn(4, 1000)
    zl = ZapLine(line_freq=50.0, sfreq=None)
    
    with pytest.raises((ValueError, TypeError)):
        zl.fit(data)


def test_zapline_transform_shape_mismatch(line_noise_data):
    """ZapLine.transform should handle shape mismatches gracefully."""
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]
    
    title_stub = "ZapLine.transform should handle shape mismatches gracefully."
    data = line_noise_data["data"]
    sfreq = line_noise_data["sfreq"]
    
    # Must remove at least 1 component to trigger matrix multiplication failure
    zl = ZapLine(line_freq=50.0, sfreq=sfreq, n_remove=1)
    zl.fit(data)

    
    # Try to transform data with different number of channels
    different_data = np.random.randn(8, 1000)  # 8 channels instead of 4
    
    with pytest.raises((ValueError, IndexError, RuntimeError, TypeError)):
        zl.transform(different_data)



# =============================================================================
# ZapLine Class - sklearn Compatibility
# =============================================================================


def test_zapline_sklearn_clone():
    """ZapLine should be clonable with sklearn."""
    from sklearn.base import clone
    
    zl = ZapLine(line_freq=50.0, sfreq=250, n_remove=3)
    zl_clone = clone(zl)
    
    assert zl_clone.line_freq == zl.line_freq
    assert zl_clone.sfreq == zl.sfreq
    assert zl_clone.n_remove == zl.n_remove


def test_zapline_get_params():
    """ZapLine should support get_params."""
    zl = ZapLine(line_freq=50.0, sfreq=250, n_remove=2)
    
    params = zl.get_params()
    
    assert params["line_freq"] == 50.0
    assert params["sfreq"] == 250
    assert params["n_remove"] == 2


def test_zapline_set_params():
    """ZapLine should support set_params."""
    zl = ZapLine()
    zl.set_params(line_freq=60.0, n_remove=5)
    
    assert zl.line_freq == 60.0
    assert zl.n_remove == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
