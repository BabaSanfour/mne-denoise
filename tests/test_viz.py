"""Unit tests for mne_denoise.viz module."""

import matplotlib.pyplot as plt
import mne
import numpy as np
import pytest

from mne_denoise.dss import DSS
from mne_denoise.viz import (
    plot_component_image,
    plot_component_spectrogram,
    plot_component_summary,
    plot_component_time_series,
    plot_denoising_summary,
    plot_evoked_comparison,
    plot_narrowband_scan,
    plot_overlay_comparison,
    plot_power_map,
    plot_psd_comparison,
    plot_score_curve,
    plot_spatial_patterns,
    plot_spectral_psd_comparison,
    plot_spectrogram_comparison,
    plot_tf_mask,
    plot_time_course_comparison,
    plot_zapline_analytics,
)
from mne_denoise.viz._utils import (
    _get_components,
    _get_info,
    _get_patterns,
    _get_scores,
)


# Close all figures after each test to save memory
@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close("all")


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


@pytest.fixture(scope="module")
def synthetic_data():
    """Create synthetic epochs with signal and noise."""
    # 5 channels, 100 Hz, 2s duration
    info = mne.create_info(["Fz", "Cz", "Pz", "Oz", "F3"], 100.0, "eeg")
    info.set_montage("standard_1020")

    n_epochs = 10
    n_times = 200
    data = np.random.randn(n_epochs, 5, n_times)

    # Add shared signal (alpha-ish)
    t = np.linspace(0, 2, n_times)
    signal = np.sin(2 * np.pi * 10 * t)
    data[:, 0:3, :] += signal * 0.5

    epochs = mne.EpochsArray(data, info)
    return epochs


@pytest.fixture(scope="module")
def fitted_dss(synthetic_data):
    """Return a fitted DSS estimator."""

    def bias_func(d):
        return mne.filter.filter_data(d, 100.0, 8, 12, verbose=False)

    dss = DSS(n_components=3, bias=bias_func, return_type="epochs")
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
    assert patterns.shape == (5, 3)  # (n_ch, n_comp)

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
    est_iter.convergence_info_ = {"dummy": 1}
    assert _get_scores(est_iter) is None
    assert _get_scores(MockEst()) is None

    # _get_components
    # Case 1: From estimator (cached or computed via transform)
    # DSS doesn't cache sources by default in fit, but _get_components uses transform
    sources = _get_components(fitted_dss, synthetic_data)
    assert sources.shape == (3, 200, 10)  # (n_comp, n_times, n_epochs)

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
    assert sources_raw.ndim == 2  # (n_comp, n_times) for Raw


def test_plot_score_curve(fitted_dss):
    """Test score curve plotting."""
    fig = plot_score_curve(fitted_dss, mode="raw", show=False)
    assert isinstance(fig, plt.Figure)

    fig = plot_score_curve(fitted_dss, mode="cumulative", show=False)
    assert isinstance(fig, plt.Figure)

    fig = plot_score_curve(fitted_dss, mode="ratio", show=False)
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
    fig = plot_component_summary(
        fitted_dss, data=synthetic_data, n_components=2, show=False
    )
    assert isinstance(fig, plt.Figure)

    # Test Raw data input
    raw = mne.io.RawArray(synthetic_data.get_data()[0], synthetic_data.info)
    fig = plot_component_summary(fitted_dss, data=raw, n_components=1, show=False)
    assert isinstance(fig, plt.Figure)

    # Test with list of components
    plot_component_summary(
        fitted_dss, data=synthetic_data, n_components=[0, 2], show=False
    )

    # Test error w/o sources
    def dummy_bias(d):
        return d

    DSS(n_components=3, bias=dummy_bias)  # Fresh estimator no cache
    with pytest.raises(ValueError, match="Data must be provided"):
        # DSS checks fitted via filters_ usually, but _get_components checks data/sources
        # If we pass fit est but no data:
        plot_component_summary(fitted_dss, data=None, show=False)
        # DSS checks fitted via filters_ usually, but _get_components checks data/sources
        # If we pass fit est but no data:
        plot_component_summary(fitted_dss, data=None, show=False)


def test_plot_component_image(fitted_dss, synthetic_data):
    """Test component image (raster)."""
    fig = plot_component_image(
        fitted_dss, data=synthetic_data, n_components=2, show=False
    )
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
    plot_time_course_comparison(
        raw_orig, raw_clean, picks=[0], start=10, stop=50, show=False
    )

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
    fig = plot_time_course_comparison(
        synthetic_data, epochs_clean, picks=[0, 1], show=False
    )
    assert isinstance(fig, plt.Figure)

    # Spectrogram
    # Using small freq range/n_freqs for speed
    fig = plot_spectrogram_comparison(
        synthetic_data, epochs_clean, fmin=1, fmax=20, n_freqs=5, show=False
    )
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


def test_plot_component_time_series(fitted_dss, synthetic_data):
    """Test stacked time series."""
    fig = plot_component_time_series(fitted_dss, data=synthetic_data, show=False)
    assert isinstance(fig, plt.Figure)

    with pytest.raises(ValueError):
        plot_component_time_series(fitted_dss, data=None, show=False)

    class MockEst:
        sources_ = np.random.randn(3, 100)
        eigenvalues_ = np.array([1.0, 0.5, 0.1])

        def get_params(self):
            return {}

    fig = plot_component_time_series(MockEst(), show=False)
    assert isinstance(fig, plt.Figure)


def test_plot_evoked_comparison(synthetic_data):
    """Test evoked comparison with Bootstrap."""
    epochs_orig = synthetic_data
    epochs_denoised = synthetic_data.copy()  # Just for testing

    fig = plot_evoked_comparison(epochs_orig, epochs_denoised, n_boot=10, show=False)
    assert isinstance(fig, plt.Figure)

    # Test with Evoked (CI should be ignored)
    ev_orig = epochs_orig.average()
    ev_denoised = epochs_denoised.average()
    fig = plot_evoked_comparison(ev_orig, ev_denoised, ci=0.95, show=False)
    assert isinstance(fig, plt.Figure)

    # Test with axes
    fig, ax = plt.subplots()
    plot_evoked_comparison(ev_orig, ev_denoised, ax=ax, show=False)


def test_plot_narrowband_scan():
    """Test narrowband scan visualization."""
    # Create synthetic scan results
    frequencies = np.arange(5, 30, 0.5)
    eigenvalues = np.random.randn(len(frequencies)) * 0.1 + 0.5

    # Add a peak at 10 Hz
    peak_idx = np.argmin(np.abs(frequencies - 10))
    eigenvalues[peak_idx] = 2.0
    eigenvalues[peak_idx - 1 : peak_idx + 2] += np.array([0.5, 0, 0.5])

    # Test basic plot
    fig = plot_narrowband_scan(frequencies, eigenvalues, show=False)
    assert isinstance(fig, plt.Figure)

    # Test with peak frequency highlighted
    fig = plot_narrowband_scan(
        frequencies, eigenvalues, peak_freq=frequencies[peak_idx], show=False
    )
    assert isinstance(fig, plt.Figure)

    # Test with true frequencies marked (for synthetic data validation)
    fig = plot_narrowband_scan(
        frequencies, eigenvalues, true_freqs=[10, 22], show=False
    )
    assert isinstance(fig, plt.Figure)

    # Test with both peak and true frequencies
    fig = plot_narrowband_scan(
        frequencies, eigenvalues, peak_freq=10.0, true_freqs=[10, 22], show=False
    )
    assert isinstance(fig, plt.Figure)

    # Test with existing axes
    fig, ax = plt.subplots()
    fig_ret = plot_narrowband_scan(frequencies, eigenvalues, ax=ax, show=False)
    assert fig_ret is fig


def test_plot_spectral_psd_comparison(fitted_dss, synthetic_data):
    """Test spectral PSD comparison visualization."""
    # Get components
    sources = fitted_dss.transform(synthetic_data)

    # Test with Epochs (3D components)
    fig = plot_spectral_psd_comparison(
        synthetic_data, sources, sfreq=100, peak_freq=10, show=False
    )
    assert isinstance(fig, plt.Figure)

    # Test without peak frequency
    fig = plot_spectral_psd_comparison(synthetic_data, sources, sfreq=100, show=False)
    assert isinstance(fig, plt.Figure)

    # Test with Raw data (2D components)
    raw = mne.io.RawArray(synthetic_data.get_data()[0], synthetic_data.info)
    sources_raw = fitted_dss.transform(raw)
    fig = plot_spectral_psd_comparison(
        raw, sources_raw, sfreq=100, peak_freq=10, show=False
    )
    assert isinstance(fig, plt.Figure)

    # Test with custom frequency range
    fig = plot_spectral_psd_comparison(
        raw, sources_raw, sfreq=100, fmin=5, fmax=20, show=False
    )
    assert isinstance(fig, plt.Figure)


def test_plot_tf_mask():
    """Test TF mask visualization."""
    n_freqs, n_times = 10, 20
    mask = np.random.rand(n_freqs, n_times)
    times = np.arange(n_times)
    freqs = np.arange(n_freqs)

    fig = plot_tf_mask(mask, times, freqs, show=False)
    assert isinstance(fig, plt.Figure)


def test_plot_overlay_comparison(synthetic_data):
    """Test signal overlay comparison."""
    # Create dummy denoised data (scaled version)
    inst_orig = synthetic_data
    inst_denoised = synthetic_data.copy()

    # Test with Epochs
    fig = plot_overlay_comparison(inst_orig, inst_denoised, show=False)
    assert isinstance(fig, plt.Figure)

    # Test with scaling
    fig = plot_overlay_comparison(
        inst_orig, inst_denoised, scale_denoised=True, show=False
    )
    assert isinstance(fig, plt.Figure)

    # Test with start/stop slicing
    fig = plot_overlay_comparison(
        inst_orig, inst_denoised, start=0.1, stop=0.5, show=False
    )
    assert isinstance(fig, plt.Figure)

    # Test with Raw data (flattened inside)
    raw = mne.io.RawArray(synthetic_data.get_data()[0], synthetic_data.info)
    fig = plot_overlay_comparison(raw, raw, show=False)
    assert isinstance(fig, plt.Figure)


def test_plot_component_spectrogram():
    """Test component TFR plotting."""
    sfreq = 100.0
    n_times = 200

    # 1D component (single trial/time series)
    comp_1d = np.random.randn(n_times)
    fig = plot_component_spectrogram(comp_1d, sfreq=sfreq, show=False)
    assert isinstance(fig, plt.Figure)

    # 2D component (1, n_times)
    comp_2d = np.random.randn(1, n_times)
    fig = plot_component_spectrogram(
        comp_2d, sfreq=sfreq, freqs=np.arange(1, 10), show=False
    )
    assert isinstance(fig, plt.Figure)

    # Test with custom ax
    fig, ax = plt.subplots()
    ret_fig = plot_component_spectrogram(comp_1d, sfreq=sfreq, ax=ax, show=False)
    assert ret_fig is fig
