import mne
import numpy as np
import pytest

from mne_denoise.dss.denoisers.temporal import SmoothingBias, TimeShiftBias
from mne_denoise.dss.variants.tsr import smooth_dss, time_shift_dss


@pytest.fixture
def slow_data_generator():
    rng = np.random.default_rng(42)
    n_times = 500

    # Slow wave (high autocorrelation)
    t = np.linspace(0, 4 * np.pi, n_times)
    slow_signal = np.sin(t)

    def get_data(shape):
        noise = rng.normal(0, 0.5, shape)  # White noise (low autocorr)
        data = noise.copy()

        # Add slow signal
        if len(shape) == 2:  # (n_ch, n_times)
            data[0] += slow_signal
        elif len(shape) == 3:  # (n_epochs, n_ch, n_times)
            data[:, 0, :] += slow_signal

        return data, slow_signal

    return get_data


def test_tsr_array(slow_data_generator):
    data, slow = slow_data_generator((3, 500))
    dss = time_shift_dss(shifts=10, n_components=3)
    dss.fit(data)

    sources = dss.transform(data)
    # Slow component should be first (highest eigenvalue/score)
    corr = np.abs(np.corrcoef(sources[0], slow)[0, 1])
    assert corr > 0.8


def test_tsr_raw(slow_data_generator):
    data, slow = slow_data_generator((3, 500))
    info = mne.create_info(3, 100, "eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    dss = time_shift_dss(shifts=10, n_components=3)
    dss.fit(raw)

    sources = dss.transform(raw)
    corr = np.abs(np.corrcoef(sources[0], slow)[0, 1])
    assert corr > 0.8


def test_tsr_epochs(slow_data_generator):
    data, slow = slow_data_generator((5, 3, 500))
    info = mne.create_info(3, 100, "eeg")
    epochs = mne.EpochsArray(data, info, verbose=False)

    dss = time_shift_dss(shifts=10, n_components=2)
    dss.fit(epochs)

    sources = dss.transform(epochs)
    assert sources.shape == (5, 2, 500)

    corr = np.abs(np.corrcoef(sources[0, 0], slow)[0, 1])
    assert corr > 0.8


def test_smooth_dss_evoked(slow_data_generator):
    # smooth_dss specifically targets low frequency
    data, slow = slow_data_generator((5, 3, 500))
    info = mne.create_info(3, 100, "eeg")
    epochs = mne.EpochsArray(data, info, verbose=False)
    evoked = epochs.average()

    dss = smooth_dss(window=20, n_components=1)
    dss.fit(evoked)

    src = dss.transform(evoked)
    assert isinstance(src, np.ndarray)
    assert src.shape == (1, 500)

    corr = np.abs(np.corrcoef(src[0], slow)[0, 1])
    assert corr > 0.8


def test_tsr_3d_bias_unit():
    """Test explicit 3D data handling in bias classes."""
    rng = np.random.default_rng(42)
    data_3d = rng.standard_normal((3, 20, 5))  # (ch, times, epochs)

    # 1. TimeShiftBias
    bias = TimeShiftBias(shifts=2)
    out_3d = bias.apply(data_3d)
    assert out_3d.shape == data_3d.shape
    assert out_3d.ndim == 3

    # 2. SmoothingBias
    bias_smooth = SmoothingBias(window=3)
    out_smooth = bias_smooth.apply(data_3d)
    assert out_smooth.shape == data_3d.shape
    assert out_smooth.ndim == 3


def test_tsr_prediction_method(slow_data_generator):
    """Test functionality of prediction method."""
    data, slow = slow_data_generator((3, 500))

    dss = time_shift_dss(shifts=10, method="prediction", n_components=3)
    dss.fit(data)

    sources = dss.transform(data)
    # Should still extract the slow component
    corr = np.abs(np.corrcoef(sources[0], slow)[0, 1])
    assert corr > 0.8
