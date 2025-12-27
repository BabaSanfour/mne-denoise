
import numpy as np
import pytest
from numpy.testing import assert_allclose
import mne
from mne_denoise.dss import ssvep_dss

@pytest.fixture
def ssvep_data_generator():
    rng = np.random.default_rng(42)
    sfreq = 250
    n_times = 500
    times = np.arange(n_times) / sfreq
    f0 = 12
    
    # Signal: 12 Hz sine
    signal = np.sin(2 * np.pi * f0 * times)
    
    def get_data(shape):
        noise = rng.normal(0, 0.5, shape)
        data = noise.copy()
        # Add signal to first channel (broadcasting)
        if len(shape) == 2: # (n_ch, n_times)
            data[0] += signal
        elif len(shape) == 3: # (n_epochs, n_ch, n_times)
            data[:, 0, :] += signal
        return data, sfreq, f0, signal

    return get_data

def test_ssvep_dss_array(ssvep_data_generator):
    data, sfreq, f0, signal = ssvep_data_generator((3, 500))
    
    dss = ssvep_dss(sfreq=sfreq, stim_freq=f0, n_harmonics=1)
    dss.fit(data)
    
    # Check filter shape
    assert dss.filters_.shape == (3, 3) 
    
    # Helper to check recovery
    source = dss.filters_[0] @ data
    corr = np.abs(np.corrcoef(source, signal)[0, 1])
    assert corr > 0.8

def test_ssvep_dss_raw(ssvep_data_generator):
    data, sfreq, f0, signal = ssvep_data_generator((3, 500))
    info = mne.create_info(3, sfreq, 'eeg')
    raw = mne.io.RawArray(data, info, verbose=False)
    
    dss = ssvep_dss(sfreq=sfreq, stim_freq=f0, n_harmonics=1)
    dss.fit(raw)
    
    sources = dss.transform(raw)
    assert sources.shape == (3, 500)
    
    # Check recovery (first component)
    corr = np.abs(np.corrcoef(sources[0], signal)[0, 1])
    assert corr > 0.8

def test_ssvep_dss_epochs(ssvep_data_generator):
    n_epochs = 5
    data, sfreq, f0, signal = ssvep_data_generator((n_epochs, 3, 500))
    info = mne.create_info(3, sfreq, 'eeg')
    epochs = mne.EpochsArray(data, info, verbose=False)
    
    dss = ssvep_dss(sfreq=sfreq, stim_freq=f0, n_harmonics=1, n_components=2)
    dss.fit(epochs)
    
    sources = dss.transform(epochs)
    # Expected: (n_epochs, n_components, n_times)
    assert sources.shape == (n_epochs, 2, 500)
    
    # Verify signal in first component of first epoch
    # Note: signal is identical in all epochs
    corr = np.abs(np.corrcoef(sources[0, 0], signal)[0, 1])
    assert corr > 0.8

def test_ssvep_dss_evoked(ssvep_data_generator):
    n_epochs = 5
    data_epochs, sfreq, f0, signal = ssvep_data_generator((n_epochs, 3, 500))
    info = mne.create_info(3, sfreq, 'eeg')
    epochs = mne.EpochsArray(data_epochs, info, verbose=False)
    evoked = epochs.average()
    
    dss = ssvep_dss(sfreq=sfreq, stim_freq=f0, n_harmonics=1, n_components=2)
    dss.fit(evoked)
    
    source_evoked = dss.transform(evoked)
    assert isinstance(source_evoked, np.ndarray)
    assert source_evoked.shape == (2, 500)
    
    corr = np.abs(np.corrcoef(source_evoked[0], signal)[0, 1])
    assert corr > 0.8
