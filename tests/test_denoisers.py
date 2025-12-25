import numpy as np
from numpy.testing import assert_allclose

from mne_denoise.dss.denoisers.artifact import CycleAverageBias


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
    data = np.zeros((2, 1000))
    sfreq = 100.0
    
    # Window: -0.1s to +0.2s -> -10 to +20 samples
    bias = CycleAverageBias(
        event_samples=events, 
        window=(-0.1, 0.2), 
        sfreq=sfreq
    )
    
    assert bias.window == (-10, 20)
