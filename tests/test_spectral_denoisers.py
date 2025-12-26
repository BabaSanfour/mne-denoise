"""Unit tests for spectral denoisers (BandpassBias, NotchBias)."""

import numpy as np
from scipy.signal import welch

from mne_denoise.dss.denoisers.spectral import (
    BandpassBias, 
    NotchBias, 
    DCTDenoiser, 
    TemporalSmoothnessDenoiser
)

def test_bandpass_bias():
    """Test BandpassBias for rhythm extraction."""
    sfreq = 250
    times = np.arange(1000) / sfreq
    
    # Signal: 10 Hz Alpha
    alpha = np.sin(2 * np.pi * 10 * times)
    # Noise: 50 Hz Line
    line = np.sin(2 * np.pi * 50 * times)
    
    data = (alpha + line)[np.newaxis, :]
    
    # Extract Alpha (8-12 Hz)
    bias = BandpassBias(freq_band=(8, 12), sfreq=sfreq, order=4)
    biased_data = bias.apply(data)
    
    # 1. 50 Hz should be gone
    _, psd = welch(biased_data, fs=sfreq)
    
    corr_alpha = np.corrcoef(biased_data[0], alpha)[0, 1]
    corr_line = np.corrcoef(biased_data[0], line)[0, 1]
    
    assert abs(corr_alpha) > 0.95, "Bandpass should preserve target rhythm"
    assert abs(corr_line) < 0.1, "Bandpass should attenuate out-of-band noise"    

def test_notch_bias_isolation():
    """Test NotchBias isolates the target frequency (for later removal)."""
    sfreq = 250
    times = np.arange(1000) / sfreq
    
    # Main signal (slow)
    slow = np.sin(2 * np.pi * 5 * times)
    # Line noise (50 Hz)
    line = np.sin(2 * np.pi * 50 * times)
    
    data = (slow + line)[np.newaxis, :]
    
    # Bias to FIND the line noise (so we isolate 50 Hz)
    bias = NotchBias(freq=50, sfreq=sfreq, bandwidth=2.0)
    biased_data = bias.apply(data)
    
    # Output should correspond to the line noise, NOT the slow signal
    corr_line = np.corrcoef(biased_data[0], line)[0, 1]
    corr_slow = np.corrcoef(biased_data[0], slow)[0, 1]
    
    assert abs(corr_line) > 0.95, "NotchBias should isolate the target frequency"
    assert abs(corr_slow) < 0.1, "NotchBias should reject other frequencies"

def test_dct_denoiser():
    """Test DCTDenoiser (frequency domain filtering)."""
    # Create signal: low frequency (first few coeffs)
    n = 100
    times = np.linspace(0, 1, n)
    signal_low = np.cos(np.pi * times) # Half-cycle cosine
    
    # Noise: high frequency (checkerboard)
    noise_high = np.ones(n)
    noise_high[::2] = -1
    
    data = signal_low + noise_high
    
    # DCTDenoiser cutoff=0.5 (lowpass)
    denoiser = DCTDenoiser(cutoff_fraction=0.5)
    denoised = denoiser.denoise(data)
    
    # Low freq should be preserved
    # High freq noise (nyquist) should be removed
    corr = np.corrcoef(denoised, signal_low)[0, 1]
    assert corr > 0.99
    
    # RMS check
    noise_rms = np.std(noise_high)
    residual_rms = np.std(denoised - signal_low)
    assert residual_rms < noise_rms * 0.1

def test_temporal_smoothness_denoiser():
    """Test TemporalSmoothnessDenoiser."""
    # Promotes smooth signals.
    n = 100
    times = np.linspace(0, 2*np.pi, n)
    smooth = np.sin(times)
    rough = np.random.randn(n) * 0.5
    
    data = smooth + rough
    
    denoiser = TemporalSmoothnessDenoiser(smoothing_factor=0.5)
    denoised = denoiser.denoise(data)
    
    # Smooth component should dominate
    corr = np.corrcoef(denoised, smooth)[0, 1]
    assert corr > 0.85
    
    # Original data correlation with smooth
    corr_orig = np.corrcoef(data, smooth)[0, 1]
    assert corr > corr_orig, "Denoised signal should be smoother/closer to ground truth"
