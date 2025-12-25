import pytest
import numpy as np
from numpy.testing import assert_allclose

def test_ssvep_dss_pipeline():
    """Test the ssvep_dss convenience function."""
    from mne_denoise.dss import ssvep_dss
    
    sfreq = 250
    times = np.arange(500) / sfreq
    f0 = 12
    
    # Signal: 12 Hz
    signal = np.sin(2 * np.pi * f0 * times)
    # Noise: Random
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.5, (3, 500))
    
    # Mix signal into channel 0
    data = noise.copy()
    data[0] += signal
    
    # Run pipeline
    filters, patterns, evs = ssvep_dss(data, sfreq=sfreq, stim_freq=f0, n_harmonics=1)
    
    assert filters.shape[1] == 3 # (n_comp, n_ch)
    assert len(evs) == 3
    
    # First component should recover signal
    source = filters[0] @ data
    corr = np.corrcoef(source, signal)[0, 1]
    # Note: sign ambiguity in PCA/DSS
    assert abs(corr) > 0.8, f"ssvep_dss failed to recover signal (corr={corr:.3f})"
