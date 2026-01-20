import numpy as np
import pytest
from scipy import signal
from mne_denoise.zapline.plus import dss_zapline_plus
from mne_denoise.zapline.adaptive import find_noise_freqs, segment_data, check_artifact_presence

def test_adaptive_components():
    # Synthetic data
    sfreq = 1000
    times = np.arange(10000) / sfreq # 10s
    data = np.random.randn(4, 10000) * 0.1
    
    # 50 Hz noise
    noise = np.sin(2 * np.pi * 50 * times) * 2.0
    data[:, :] += noise
    
    # Test frequency detection
    freqs = find_noise_freqs(data, sfreq, fmin=45, fmax=55)
    assert len(freqs) > 0
    assert np.isclose(freqs[0], 50, atol=1.0)
    
    # Test artifact presence
    present = check_artifact_presence(data, sfreq, target_freq=50)
    assert present
    
    present_wrong = check_artifact_presence(data, sfreq, target_freq=60)
    assert not present_wrong


def test_zapline_plus_pipeline():
    # Long data with shifting noise
    sfreq = 250
    duration = 120 # 2 minutes
    n_times = int(duration * sfreq)
    times = np.arange(n_times) / sfreq
    n_ch = 4
    data = np.random.randn(n_ch, n_times) * 0.05 # Low background noise
    
    # 0-40s: 50Hz strong
    # 40-80s: 50.1Hz weak
    # 80-120s: No noise
    
    t1 = int(sfreq * 40)
    t2 = int(sfreq * 80)
    
    # Noise 1
    noise1 = np.sin(2 * np.pi * 50.0 * times[:t1]) * 10.0
    # Noise 2 (slightly shifted)
    noise2 = np.sin(2 * np.pi * 50.1 * times[t1:t2]) * 2.0
    
    # Apply strongly to ch 0, weakly to ch 1
    data[0, :t1] += noise1
    data[1, :t1] += noise1 * 0.5
    
    data[0, t1:t2] += noise2
    data[1, t1:t2] += noise2 * 0.5
    
    # Run Zapline Plus
    # Should detect ~50Hz
    # Should segment and clean adaptive
    
    res = dss_zapline_plus(
        data, 
        sfreq, 
        line_freqs=None, # Auto detect
        fmin=45, 
        fmax=55,
        n_remove_params={'sigma': 3.0},
        qa_params={'max_sigma': 4.0}
    )
    
    # Check if lines removed
    cleaned = res.cleaned
    
    # PSD function helper
    def get_power_at(d, freq, fs):
        f, p = signal.welch(d, fs=fs, nperseg=int(fs), axis=-1)
        idx = np.argmin(np.abs(f - freq))
        return np.mean(p[:, idx])
        
    # Check 50Hz power in first segment
    p_orig_1 = get_power_at(data[:, :t1], 50.0, sfreq)
    p_clean_1 = get_power_at(cleaned[:, :t1], 50.0, sfreq)
    
    assert p_clean_1 < p_orig_1 * 0.1 # 90% reduction
    
    # Check 50.1Hz power in second segment
    p_orig_2 = get_power_at(data[:, t1:t2], 50.1, sfreq)
    p_clean_2 = get_power_at(cleaned[:, t1:t2], 50.1, sfreq)
    
    assert p_clean_2 < p_orig_2 * 0.5 # significant reduction
    
    # Check no-noise segment preservation
    p_orig_3 = np.var(data[:, t2:])
    p_clean_3 = np.var(cleaned[:, t2:])
    
    # Variance should be similar (no cleaning needed)
    # Allow small reduction due to random chance correlations
    assert np.isclose(p_clean_3, p_orig_3, rtol=0.2)

if __name__ == "__main__":
    test_adaptive_components()
    test_zapline_plus_pipeline()
    print("All tests passed!")
