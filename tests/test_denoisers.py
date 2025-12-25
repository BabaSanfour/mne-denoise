import pytest
import numpy as np
from numpy.testing import assert_allclose

from mne_denoise.dss.denoisers.artifact import CycleAverageBias



# ----------------------------------------------------------
# Test Base Classes
# ----------------------------------------------------------

def test_base_linear_denoiser():
    """Test LinearDenoiser abstract base class."""
    from mne_denoise.dss.denoisers.base import LinearDenoiser
    
    # Check that we cannot instantiate ABC
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        LinearDenoiser()
        
    # Check robust implementation
    class MockLinear(LinearDenoiser):
        def apply(self, data):
            return data * 2
            
    denoiser = MockLinear()
    data = np.ones((2, 2))
    assert_allclose(denoiser.apply(data), data * 2)
    # Check __call__ alias
    assert_allclose(denoiser(data), data * 2)


def test_base_nonlinear_denoiser():
    """Test NonlinearDenoiser abstract base class."""
    from mne_denoise.dss.denoisers.base import NonlinearDenoiser
    
    # Check that we cannot instantiate ABC
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        NonlinearDenoiser()
        
    # Check robust implementation
    class MockNonlinear(NonlinearDenoiser):
        def denoise(self, source):
            return source ** 2
            
    denoiser = MockNonlinear()
    source = np.array([1, 2, 3])
    assert_allclose(denoiser.denoise(source), [1, 4, 9])
    # Check __call__ alias
    assert_allclose(denoiser(source), [1, 4, 9])


# ----------------------------------------------------------
# Test CycleAverageBias: Quasi-periodic artifact extraction
# ----------------------------------------------------------

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
    
    assert_allclose(biased_data[:, mask], 0, atol=1e-10, err_msg="Biased data should be zero outside windows")
    
    # Extract one window from biased data
    event = events[0]
    biased_epoch = biased_data[:, event + window[0] : event + window[1]]
    
    # Theoretical clean epoch
    clean_epoch = artifact_data[:, event + window[0] : event + window[1]]
    
    # Correlation should be high
    corr = np.corrcoef(biased_epoch.ravel(), clean_epoch.ravel())[0, 1]
    assert corr > 0.95, f"Biased data should correlate with clean artifact (got {corr:.3f})"
    
    # Check shape preservation
    assert biased_data.shape == data.shape

def test_cycle_average_bias_sfreq():
    """Test CycleAverageBias with second-based window."""
    events = [100, 200]
    sfreq = 100.0
    
    # Window: -0.1s to +0.2s -> -10 to +20 samples
    bias = CycleAverageBias(
        event_samples=events, 
        window=(-0.1, 0.2), 
        sfreq=sfreq
    )
    
    assert bias.window == (-10, 20)


# ----------------------------------------------------------
# Test TrialAverageBias: Evoked response enhancement
# ----------------------------------------------------------

def test_trial_average_bias():
    """Test TrialAverageBias on simple 3D data."""
    from mne_denoise.dss.denoisers.evoked import TrialAverageBias
    
    # Create data: (n_channels, n_times, n_epochs)
    n_epochs = 10
    data = np.zeros((1, 5, n_epochs))
    
    # Trials have a constant component (1) + noise
    # We make trial 0 have value 1, trial 1 have value 3 -> mean 2
    data[0, :, :] = 2
    
    bias = TrialAverageBias()
    biased = bias.apply(data)
    
    # Result should be the mean repeated
    expected = np.ones((1, 5, n_epochs)) * 2
    assert_allclose(biased, expected)
    
def test_trial_average_bias_weighted():
    """Test TrialAverageBias with weights."""
    from mne_denoise.dss.denoisers.evoked import TrialAverageBias
    
    # 2 epochs
    data = np.zeros((1, 1, 2))
    data[0, 0, 0] = 10
    data[0, 0, 1] = 20
    
    # Weighted average: 0.8 * 10 + 0.2 * 20 = 8 + 4 = 12
    weights = [0.8, 0.2]
    bias = TrialAverageBias(weights=weights)
    biased = bias.apply(data)
    
    assert_allclose(biased[0, 0, :], 12)

def test_trial_average_bias_errors():
    """Test error handling."""
    from mne_denoise.dss.denoisers.evoked import TrialAverageBias
    
    bias = TrialAverageBias()
    # 2D input should fail (expects epoched)
    data = np.zeros((2, 10))
    with pytest.raises(ValueError, match="requires 3D epoched data"):
        bias.apply(data)


# ----------------------------------------------------------
# Test Periodic Filters: SSVEP / Harmonic extraction
# ----------------------------------------------------------

def test_peak_filter_bias():
    """Test PeakFilterBias extracting a sine wave."""
    from mne_denoise.dss.denoisers.periodic import PeakFilterBias
    
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
    from mne_denoise.dss.denoisers.periodic import CombFilterBias
    
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


# ----------------------------------------------------------
# Test Spectral Filters: Bandpass / Notch Bias
# ----------------------------------------------------------

def test_bandpass_bias():
    """Test BandpassBias for rhythm extraction."""
    from mne_denoise.dss.denoisers.spectral import BandpassBias
    
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
    # Compute power ratio
    from scipy.signal import welch
    _, psd = welch(biased_data, fs=sfreq)
    
    corr_alpha = np.corrcoef(biased_data[0], alpha)[0, 1]
    corr_line = np.corrcoef(biased_data[0], line)[0, 1]
    
    assert abs(corr_alpha) > 0.95, "Bandpass should preserve target rhythm"
    assert abs(corr_line) < 0.1, "Bandpass should attenuate out-of-band noise"    

def test_notch_bias_isolation():
    """Test NotchBias isolates the target frequency (for later removal)."""
    from mne_denoise.dss.denoisers.spectral import NotchBias
    
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


# ----------------------------------------------------------
# Test Spectrogram Denoiser
# ----------------------------------------------------------

def test_spectrogram_denoiser():
    """Test SpectrogramDenoiser for adaptive time-frequency masking."""
    from mne_denoise.dss.denoisers.spectrogram import SpectrogramDenoiser
    
    sfreq = 250
    times = np.arange(1000) / sfreq
    
    # 1. Burst of 50 Hz oscillation in the middle
    # Center at 2s (index 500)
    burst = np.sin(2 * np.pi * 50 * times) * np.exp(-0.5 * (times - 2)**2 / 0.1)
    # 2. Background noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.1, 1000)
    
    data = burst + noise
    data_2d = data[:, np.newaxis]  # (n_times, 1)
    
    # Adaptive thresholding: should enhance the high-amplitude burst
    denoiser = SpectrogramDenoiser(threshold_percentile=95, nperseg=128)
    denoised = denoiser.denoise(data_2d)[:, 0]

    # Extract the burst part of denoised signal
    denoised_burst = denoised[400:600]
    expected_burst = burst[400:600]
    
    corr = np.corrcoef(denoised_burst, expected_burst)[0, 1]
    
    # Should have high correlation with the original burst shape
    assert corr > 0.8, f"Denoised burst should match ground truth (corr={corr:.3f})"
    
    # Check noise suppression in a quiet region (e.g. t=0.0 to 0.5s)
    quiet_in = data[0:100]
    quiet_out = denoised[0:100]
    
    rms_in = np.sqrt(np.mean(quiet_in**2))
    rms_out = np.sqrt(np.mean(quiet_out**2))
    assert rms_out < rms_in, "Denoiser should reduce RMS in quiet regions"

def test_spectrogram_bias():
    """Test SpectrogramBias (Linear) with fixed mask."""
    from scipy import signal
    from mne_denoise.dss.denoisers.spectrogram import SpectrogramBias
    
    sfreq = 100
    times = np.arange(200) / sfreq
    n_times = len(times)
    
    # Create a dummy signal (2 channels)
    # Ch 0: 10 Hz sine
    # Ch 1: Random noise
    data = np.zeros((2, n_times))
    data[0] = np.sin(2 * np.pi * 10 * times)
    data[1] = np.random.randn(n_times)
    
    # Calculate STFT to get mask shape
    nperseg = 64
    f, t, Zxx = signal.stft(data[0], fs=sfreq, nperseg=nperseg)
    
    # Create a mask that keeps only ~10 Hz
    # Freq bins: f[k] = k * fs / nperseg
    # 10 Hz bin index approx: 10 * 64 / 100 = 6.4 -> bin 6 or 7
    mask = np.zeros_like(Zxx, dtype=float)
    freq_indices = np.where((f >= 8) & (f <= 12))[0]
    mask[freq_indices, :] = 1.0
    
    bias = SpectrogramBias(mask=mask, nperseg=nperseg)
    biased_data = bias.apply(data)
    
    # Check shapes
    assert biased_data.shape == data.shape
    
    # Verify effect: 
    # Ch 0 (10 Hz signal) should be largely preserved
    # Ch 1 (White noise) should be low-pass/band-pass filtered
    
    energy_in_0 = np.sum(data[0]**2)
    energy_out_0 = np.sum(biased_data[0]**2)
    
    energy_in_1 = np.sum(data[1]**2)
    energy_out_1 = np.sum(biased_data[1]**2)
    
    # Ratio of preserved energy
    ratio_0 = energy_out_0 / energy_in_0
    ratio_1 = energy_out_1 / energy_in_1
    
    assert ratio_0 > 0.8, "Target signal should be preserved"
    assert ratio_1 < 0.3, "Broadband noise should be filtered"

