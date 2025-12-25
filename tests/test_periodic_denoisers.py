"""Unit tests for periodic denoisers (PeakFilterBias, CombFilterBias)."""

import numpy as np

from mne_denoise.dss.denoisers.periodic import PeakFilterBias, CombFilterBias

def test_peak_filter_bias():
    """Test PeakFilterBias extracting a sine wave."""
    sfreq = 1000
    times = np.arange(1000) / sfreq
    freq = 10
    
    # Signal: 10 Hz sine
    signal = np.sin(2 * np.pi * freq * times)
    
    # Noise: different frequency (e.g. 50 Hz)
    noise = np.sin(2 * np.pi * 50 * times) * 2.0
    
    data = (signal + noise)[np.newaxis, :] # (1, n_times)
    
    # Bias towards 10 Hz
    bias = PeakFilterBias(freq=10, sfreq=sfreq, q_factor=50)
    biased_data = bias.apply(data)
    
    # Result should be mostly 10 Hz
    # Correlation with signal should be high
    corr = np.corrcoef(biased_data[0], signal)[0, 1]
    assert corr > 0.9, f"Peak filter failed to extract signal (corr={corr:.3f})"    

def test_comb_filter_bias():
    """Test CombFilterBias extracting fundamental + harmonic."""
    sfreq = 1000
    times = np.arange(1000) / sfreq
    f0 = 12
    
    # Signal: 12 Hz + 24 Hz
    s1 = np.sin(2 * np.pi * f0 * times)
    s2 = 0.5 * np.sin(2 * np.pi * 2 * f0 * times)
    signal = s1 + s2
    
    # Noise at other freq
    noise = np.sin(2 * np.pi * 50 * times) * 2
    
    data = (signal + noise)[np.newaxis, :]
    
    bias = CombFilterBias(fundamental_freq=12, sfreq=sfreq, n_harmonics=2)
    biased_data = bias.apply(data)
    
    corr = np.corrcoef(biased_data[0], signal)[0, 1]
    assert corr > 0.85, f"Comb filter failed (corr={corr:.3f})"
