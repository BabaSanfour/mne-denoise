"""Unit tests for mne_denoise.viz module."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import mne

from mne_denoise.dss import DSS
from mne_denoise.viz import (
    plot_score_curve,
    plot_spatial_patterns,
    plot_component_summary,
    plot_component_image,
    plot_psd_comparison,
    plot_time_course_comparison,
    plot_spectrogram_comparison,
    plot_power_map,
    plot_denoising_summary,
    plot_zapline_analytics
)
from mne_denoise.viz._utils import _get_info, _get_patterns, _get_scores, _get_components

# Close all figures after each test to save memory
@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close('all')

def test_viz_show(fitted_dss, synthetic_data):
    with pytest.MonkeyPatch.context() as m:
        m.setattr(plt, "show", lambda *args, **kwargs: None)
        plot_score_curve(fitted_dss, show=True)
        plot_spatial_patterns(fitted_dss, show=True)
        # Add comparisons to hit show=True lines in comparison.py
        data = synthetic_data
        epochs_clean = fitted_dss.transform(synthetic_data)
        plot_psd_comparison(data, epochs_clean, show=True)
        plot_time_course_comparison(data, epochs_clean, show=True)



@pytest.fixture(scope='module')
def synthetic_data():
    """Create synthetic epochs with signal and noise."""
    # 5 channels, 100 Hz, 2s duration
    info = mne.create_info(['Fz', 'Cz', 'Pz', 'Oz', 'F3'], 100.0, 'eeg')
    info.set_montage('standard_1020')
    
    n_epochs = 10
    n_times = 200
    data = np.random.randn(n_epochs, 5, n_times)
    
    # Add shared signal (alpha-ish)
    t = np.linspace(0, 2, n_times)
    signal = np.sin(2 * np.pi * 10 * t)
    data[:, 0:3, :] += signal * 0.5
    
    epochs = mne.EpochsArray(data, info)
    return epochs

@pytest.fixture(scope='module')
def fitted_dss(synthetic_data):
    """Return a fitted DSS estimator."""
    def bias_func(d):
        return mne.filter.filter_data(d, 100.0, 8, 12, verbose=False)
        
    dss = DSS(n_components=3, bias=bias_func, return_type='epochs')
    dss.fit(synthetic_data)
    return dss

def test_viz_utils(fitted_dss, synthetic_data):
    """Test internal utility functions."""
    # _get_info
    assert isinstance(_get_info(fitted_dss), mne.Info)
    assert _get_info(fitted_dss, synthetic_data.info) is synthetic_data.info
    
    # Mock estimator attributes
    class MockEst:
        pass
    est = MockEst()
    assert _get_info(est) is None
    est._mne_info = synthetic_data.info
    assert _get_info(est) is synthetic_data.info

    # _handle_picks (if exported or used)
    from mne_denoise.viz._utils import _handle_picks
    picks = _handle_picks(synthetic_data.info, picks=None)
    assert len(picks) == 5
    picks = _handle_picks(synthetic_data.info, picks=[0, 1])
    assert len(picks) == 2
    
    # _get_patterns
    patterns = _get_patterns(fitted_dss)
    assert patterns.shape == (5, 3) # (n_ch, n_comp)
    
    with pytest.raises(ValueError):
        _get_patterns(MockEst())
        
    # _get_filters
    from mne_denoise.viz._utils import _get_filters
    filters = _get_filters(fitted_dss)
    assert filters.shape == (3, 5)
    with pytest.raises(ValueError):
        _get_filters(MockEst())
    
    # _get_scores
    scores = _get_scores(fitted_dss)
    assert len(scores) == 3
    
    est_iter = MockEst()
    est_iter.convergence_info_ = {'dummy': 1}
    assert _get_scores(est_iter) is None
    assert _get_scores(MockEst()) is None
    
    # _get_components
    # Case 1: From estimator (cached or computed via transform)
    # DSS doesn't cache sources by default in fit, but _get_components uses transform
    sources = _get_components(fitted_dss, synthetic_data)
    assert sources.shape == (3, 200, 10) # (n_comp, n_times, n_epochs)
    
    # Test cached sources
    est_cached = MockEst()
    est_cached.sources_ = np.zeros((3, 200, 10))
    # Passing data=None should retrive cached
    assert _get_components(est_cached, data=None) is est_cached.sources_
    
    # Test error w/o data
    with pytest.raises(ValueError, match="Data must be provided"):
        _get_components(fitted_dss, data=None)

    # Test handling of Raw data (2D)
    raw = mne.io.RawArray(synthetic_data.get_data()[0], synthetic_data.info)
    sources_raw = _get_components(fitted_dss, raw)
    assert sources_raw.ndim == 2 # (n_comp, n_times) for Raw

def test_plot_score_curve(fitted_dss):
    """Test score curve plotting."""
    fig = plot_score_curve(fitted_dss, mode='raw', show=False)
    assert isinstance(fig, plt.Figure)
    
    fig = plot_score_curve(fitted_dss, mode='cumulative', show=False)
    assert isinstance(fig, plt.Figure)
    
    fig = plot_score_curve(fitted_dss, mode='ratio', show=False)
    assert isinstance(fig, plt.Figure)
    
    # Test w/ existing axes
    fig, ax = plt.subplots()
    plot_score_curve(fitted_dss, ax=ax, show=False)
    
    # Test w/o scores
    class NoScoreEst:
        pass
    assert plot_score_curve(NoScoreEst(), show=False) is None

def test_plot_spatial_patterns(fitted_dss):
    """Test topomap plotting."""
    fig = plot_spatial_patterns(fitted_dss, show=False)
    assert isinstance(fig, plt.Figure)
    
    # Test subset
    fig = plot_spatial_patterns(fitted_dss, n_components=2, show=False)
    assert isinstance(fig, plt.Figure)
    
    # Test missing info error (mock estimator)
    class MockEst:
        patterns_ = np.zeros((5, 3))
    with pytest.raises(ValueError, match="Info is required"):
        plot_spatial_patterns(MockEst(), show=False)

def test_plot_component_summary(fitted_dss, synthetic_data):
    """Test component summary dashboard."""
    fig = plot_component_summary(fitted_dss, data=synthetic_data, n_components=2, show=False)
    assert isinstance(fig, plt.Figure)
    
    # Test Raw data input
    raw = mne.io.RawArray(synthetic_data.get_data()[0], synthetic_data.info)
    fig = plot_component_summary(fitted_dss, data=raw, n_components=1, show=False)
    assert isinstance(fig, plt.Figure)
    
    
    # Test with list of components
    plot_component_summary(fitted_dss, data=synthetic_data, n_components=[0, 2], show=False)
    
    # Test error w/o sources
    def dummy_bias(d): return d
    est_no_src = DSS(n_components=3, bias=dummy_bias) # Fresh estimator no cache
    with pytest.raises(ValueError, match="Data must be provided"):
         # DSS checks fitted via filters_ usually, but _get_components checks data/sources
         # If we pass fit est but no data:
         plot_component_summary(fitted_dss, data=None, show=False)
         # DSS checks fitted via filters_ usually, but _get_components checks data/sources
         # If we pass fit est but no data:
         plot_component_summary(fitted_dss, data=None, show=False)

def test_plot_component_image(fitted_dss, synthetic_data):
    """Test component image (raster)."""
    fig = plot_component_image(fitted_dss, data=synthetic_data, n_components=2, show=False)
    assert isinstance(fig, plt.Figure)
    
    # Test Raw data input (should handle by treating as 1 epoch)
    raw = mne.io.RawArray(synthetic_data.get_data()[0], synthetic_data.info)
    fig = plot_component_image(fitted_dss, data=raw, n_components=1, show=False)
    assert isinstance(fig, plt.Figure)

def test_comparisons(fitted_dss, synthetic_data):
    """Test all comparison plots."""
    dss = fitted_dss
    
    # Prepare clean data
    # Create copy to avoid modifying fixture if transform did inplace (it shouldn't)
    epochs_clean = dss.transform(synthetic_data)
    
    epochs_clean = dss.transform(synthetic_data)
    
    # Test Raw comparison (coverage for Raw paths)
    raw_orig = mne.io.RawArray(synthetic_data.get_data()[0], synthetic_data.info)
    raw_clean = mne.io.RawArray(epochs_clean.get_data()[0], synthetic_data.info)
    
    # PSD Comparison
    fig, ax = plt.subplots()
    plot_psd_comparison(synthetic_data, epochs_clean, ax=ax, show=False)
    
    # PSD Comparison Raw
    plot_psd_comparison(raw_orig, raw_clean, show=False)

    # Time Course
    plot_time_course_comparison(synthetic_data, epochs_clean, picks=[0], show=False)
    # Raw with start/stop
    plot_time_course_comparison(raw_orig, raw_clean, picks=[0], start=10, stop=50, show=False)
    
    # Power Map with explicit info
    plot_power_map(synthetic_data, epochs_clean, info=synthetic_data.info, show=False)
    # Power Map with Raw and ax
    fig, ax = plt.subplots()
    plot_power_map(raw_orig, raw_clean, ax=ax, show=False)
    
    # Compare with existing ax
    plot_time_course_comparison(synthetic_data, epochs_clean, show=False)
    
    # GFP coverage in summary
    plot_denoising_summary(raw_orig, raw_clean, show=False)
    fig = plot_psd_comparison(synthetic_data, epochs_clean, show=False)
    assert isinstance(fig, plt.Figure)
    # Test average=False
    fig = plot_psd_comparison(synthetic_data, epochs_clean, average=False, show=False)
    assert isinstance(fig, plt.Figure)
    
    # Time Course
    fig = plot_time_course_comparison(synthetic_data, epochs_clean, picks=[0, 1], show=False)
    assert isinstance(fig, plt.Figure)
    
    # Spectrogram
    # Using small freq range/n_freqs for speed
    fig = plot_spectrogram_comparison(synthetic_data, epochs_clean, 
                                    fmin=1, fmax=20, n_freqs=5, show=False)
    assert isinstance(fig, plt.Figure)
    
    # Power Map
    fig = plot_power_map(synthetic_data, epochs_clean, show=False)
    assert isinstance(fig, plt.Figure)
    
    # Summary Dashboard
    fig = plot_denoising_summary(synthetic_data, epochs_clean, show=False)
    assert isinstance(fig, plt.Figure)

def test_zapline_placeholder():
    """Test ZapLine placeholder."""
    # Should currently do nothing or print, but not crash
    assert plot_zapline_analytics(None, show=False) is None
