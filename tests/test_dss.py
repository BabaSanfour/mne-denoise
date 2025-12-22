"""Unit tests for DSS module."""

from __future__ import annotations

import numpy as np
import pytest

from mne_denoise.dss import (
    whiten_data,
    compute_whitener,
    compute_dss,
    DSS,
    DSSConfig,
    TrialAverageBias,
    BandpassBias,
    NotchBias,
    CycleAverageBias,
    VarianceMaskDenoiser,
    KurtosisDenoiser,
    IterativeDSS,
    iterative_dss,
    dss_zapline,
    compute_psd_reduction,
)


class TestWhitening:
    """Tests for whitening utilities."""

    def test_whiten_identity_covariance(self) -> None:
        """Whitened data should have approximately identity covariance."""
        rng = np.random.default_rng(42)
        n_channels, n_samples = 16, 5000

        # Create correlated data
        mixing = rng.standard_normal((n_channels, n_channels))
        sources = rng.standard_normal((n_channels, n_samples))
        data = mixing @ sources

        whitened, W, D = whiten_data(data)

        # Check covariance is approximately identity
        cov = whitened @ whitened.T / n_samples
        np.testing.assert_allclose(
            cov, np.eye(whitened.shape[0]), atol=0.1
        )

    def test_whiten_rank_deficient(self) -> None:
        """Whitening should handle rank-deficient data."""
        rng = np.random.default_rng(42)
        n_channels, n_samples = 16, 1000
        true_rank = 8

        # Create rank-deficient data
        sources = rng.standard_normal((true_rank, n_samples))
        mixing = rng.standard_normal((n_channels, true_rank))
        data = mixing @ sources

        whitened, W, D = whiten_data(data)

        # Should auto-detect reduced rank
        assert whitened.shape[0] <= true_rank + 1

    def test_whiten_3d_data(self) -> None:
        """Whitening should work on 3D epoched data."""
        rng = np.random.default_rng(42)
        n_channels, n_times, n_epochs = 8, 100, 20

        data = rng.standard_normal((n_channels, n_times, n_epochs))
        whitened, W, D = whiten_data(data)

        assert whitened.ndim == 3
        assert whitened.shape[1:] == (n_times, n_epochs)

    def test_compute_whitener_matrices(self) -> None:
        """Whitener and dewhitener should be inverses."""
        rng = np.random.default_rng(42)
        n_channels = 8

        # Create covariance
        A = rng.standard_normal((n_channels, n_channels))
        cov = A @ A.T

        W, D, eigenvalues = compute_whitener(cov)

        # W @ D should be approximately identity (up to truncation)
        product = W @ D
        np.testing.assert_allclose(
            product, np.eye(W.shape[0]), atol=1e-10
        )


class TestComputeDSS:
    """Tests for core DSS computation."""

    def test_dss_shape(self) -> None:
        """DSS should return correct shapes."""
        rng = np.random.default_rng(42)
        n_channels, n_samples = 16, 1000

        data = rng.standard_normal((n_channels, n_samples))
        biased = data.copy()  # Identity bias for testing

        filters, patterns, eigenvalues, exp_var = compute_dss(
            data, biased, n_components=5
        )

        assert filters.shape == (5, n_channels)
        assert patterns.shape == (n_channels, 5)
        assert eigenvalues.shape == (5,)
        assert exp_var.shape == (5,)

    def test_dss_recovers_known_signal(self) -> None:
        """DSS should extract a known injected signal."""
        rng = np.random.default_rng(42)
        n_channels, n_samples = 8, 2000
        sfreq = 250

        # Create noise
        noise = rng.standard_normal((n_channels, n_samples))

        # Create a narrowband signal
        t = np.arange(n_samples) / sfreq
        signal_freqs = 10  # 10 Hz sinusoid
        source = np.sin(2 * np.pi * signal_freqs * t)

        # Project to channels with known mixing
        mixing = rng.standard_normal(n_channels)
        mixing = mixing / np.linalg.norm(mixing)
        data = noise + 5 * np.outer(mixing, source)

        # Create bias: bandpass filter around 10 Hz
        from scipy import signal as sig
        sos = sig.butter(4, [8/125, 12/125], btype='band', output='sos')
        biased = sig.sosfiltfilt(sos, data, axis=1)

        filters, patterns, eigenvalues, _ = compute_dss(
            data, biased, n_components=3
        )

        # Top component should correlate with the injected signal
        top_source = filters[0] @ data
        correlation = np.abs(np.corrcoef(top_source, source)[0, 1])
        assert correlation > 0.8, f"Correlation {correlation} too low"

    def test_dss_all_components_reconstruction(self) -> None:
        """Keeping all components should allow perfect reconstruction."""
        rng = np.random.default_rng(42)
        n_channels, n_samples = 8, 500

        data = rng.standard_normal((n_channels, n_samples))
        biased = data.copy()

        filters, patterns, _, _ = compute_dss(data, biased)

        # Transform and inverse transform
        sources = filters @ data
        reconstructed = patterns @ sources

        # Should be able to reconstruct (up to centering)
        data_centered = data - data.mean(axis=1, keepdims=True)
        np.testing.assert_allclose(reconstructed, data_centered, atol=0.5)


class TestDSSClass:
    """Tests for DSS estimator class."""

    def test_fit_transform(self) -> None:
        """DSS class should support fit_transform workflow."""
        rng = np.random.default_rng(42)
        n_channels, n_samples = 8, 1000

        data = rng.standard_normal((n_channels, n_samples))

        dss = DSS(bias='identity', n_components=3)
        sources = dss.fit_transform(data)

        assert sources.shape == (3, n_samples)
        assert dss.filters_ is not None
        assert dss.filters_.shape == (3, n_channels)

    def test_custom_bias_callable(self) -> None:
        """DSS should accept custom callable bias."""
        rng = np.random.default_rng(42)
        n_channels, n_samples = 8, 1000

        data = rng.standard_normal((n_channels, n_samples))

        def my_bias(x):
            return x * 2

        dss = DSS(bias=my_bias, n_components=3)
        sources = dss.fit_transform(data)

        assert sources.shape == (3, n_samples)


class TestDenoisers:
    """Tests for denoiser bias functions."""

    def test_trial_average_bias_shape(self) -> None:
        """TrialAverageBias should preserve input shape."""
        rng = np.random.default_rng(42)
        n_channels, n_times, n_epochs = 8, 100, 20

        data = rng.standard_normal((n_channels, n_times, n_epochs))
        bias = TrialAverageBias()
        biased = bias.apply(data)

        assert biased.shape == data.shape

    def test_trial_average_bias_is_constant_across_trials(self) -> None:
        """Each trial should be identical after averaging."""
        rng = np.random.default_rng(42)
        n_channels, n_times, n_epochs = 4, 50, 10

        data = rng.standard_normal((n_channels, n_times, n_epochs))
        bias = TrialAverageBias()
        biased = bias.apply(data)

        for i in range(n_epochs - 1):
            np.testing.assert_array_equal(biased[:, :, i], biased[:, :, i + 1])

    def test_trial_average_bias_equals_mean(self) -> None:
        """Biased data should equal the trial mean."""
        rng = np.random.default_rng(42)
        n_channels, n_times, n_epochs = 4, 50, 10

        data = rng.standard_normal((n_channels, n_times, n_epochs))
        bias = TrialAverageBias()
        biased = bias.apply(data)

        expected_avg = data.mean(axis=2)
        np.testing.assert_allclose(biased[:, :, 0], expected_avg)

    def test_bandpass_bias_frequency(self) -> None:
        """BandpassBias should isolate target frequency band."""
        sfreq = 250
        n_channels, n_samples = 4, 2500
        t = np.arange(n_samples) / sfreq

        signal_10hz = np.sin(2 * np.pi * 10 * t)
        signal_50hz = np.sin(2 * np.pi * 50 * t)
        data = np.vstack([signal_10hz + signal_50hz] * n_channels)

        bias = BandpassBias(freq_band=(8, 12), sfreq=sfreq)
        biased = bias.apply(data)

        from scipy import signal as sig
        f, psd_orig = sig.welch(data[0], sfreq, nperseg=512)
        f, psd_biased = sig.welch(biased[0], sfreq, nperseg=512)

        idx_10 = np.argmin(np.abs(f - 10))
        idx_50 = np.argmin(np.abs(f - 50))

        assert psd_biased[idx_10] > 0.1 * psd_orig[idx_10]
        assert psd_biased[idx_50] < 0.1 * psd_orig[idx_50]

    def test_notch_bias_callable(self) -> None:
        """NotchBias should work as callable."""
        rng = np.random.default_rng(42)
        n_channels, n_samples = 4, 1000

        data = rng.standard_normal((n_channels, n_samples))
        bias = NotchBias(freq=50, sfreq=250, bandwidth=2)

        biased = bias(data)
        assert biased.shape == data.shape


class TestCycleAverageBias:
    """Tests for CycleAverageBias."""

    def test_cycle_average_shape(self) -> None:
        """CycleAverageBias should preserve shape."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((8, 1000))
        events = np.array([100, 300, 500, 700])

        bias = CycleAverageBias(events, window=(-20, 50))
        biased = bias.apply(data)

        assert biased.shape == data.shape

    def test_cycle_average_extracts_artifact(self) -> None:
        """CycleAverageBias should emphasize periodic artifacts."""
        rng = np.random.default_rng(42)
        n_channels, n_times = 8, 2000

        data = rng.standard_normal((n_channels, n_times)) * 0.5

        artifact_template = np.exp(-np.linspace(0, 3, 30)**2)
        events = np.arange(100, 1900, 200)

        for event in events:
            data[:4, event:event+30] += artifact_template * 2

        bias = CycleAverageBias(events, window=(-5, 30))
        biased = bias.apply(data)

        artifact_power = np.mean(biased[:, events[0]:events[0]+30]**2)
        quiet_power = np.mean(biased[:, 0:30]**2)

        assert artifact_power > quiet_power * 2


class TestNonlinearDenoisers:
    """Tests for nonlinear denoisers."""

    def test_variance_mask_shape(self) -> None:
        """VarianceMaskDenoiser should preserve shape."""
        rng = np.random.default_rng(42)
        source = rng.standard_normal(1000)

        denoiser = VarianceMaskDenoiser(window_samples=50)
        denoised = denoiser.denoise(source)

        assert denoised.shape == source.shape

    def test_kurtosis_denoiser_tanh(self) -> None:
        """KurtosisDenoiser with tanh should work correctly."""
        source = np.array([-2, -1, 0, 1, 2], dtype=float)
        denoiser = KurtosisDenoiser(nonlinearity='tanh')
        denoised = denoiser.denoise(source)

        expected = np.tanh(source)
        np.testing.assert_allclose(denoised, expected)


class TestIterativeDSS:
    """Tests for iterative DSS."""

    def test_iterative_dss_shape(self) -> None:
        """iterative_dss should return correct shapes."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((8, 2000))

        denoiser = KurtosisDenoiser()
        filters, sources, patterns, conv = iterative_dss(
            data, denoiser, n_components=3, max_iter=10
        )

        assert filters.shape == (3, 8)
        assert sources.shape == (3, 2000)
        assert patterns.shape == (8, 3)

    def test_iterative_dss_class(self) -> None:
        """IterativeDSS class should work like sklearn."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 1500))

        denoiser = VarianceMaskDenoiser()
        it_dss = IterativeDSS(denoiser, n_components=5, max_iter=5)

        it_dss.fit(data)
        assert it_dss.filters_ is not None
        assert it_dss.filters_.shape == (5, 10)


class TestZapLine:
    """Tests for DSS-ZapLine."""

    def test_zapline_shape(self) -> None:
        """dss_zapline should return correct shapes."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((16, 5000))

        result = dss_zapline(data, line_freq=50, sfreq=500, n_remove=2)

        assert result.cleaned.shape == data.shape
        assert result.removed.shape == data.shape
        assert result.n_removed == 2

    def test_zapline_removes_line_noise(self) -> None:
        """ZapLine should reduce line noise power."""
        rng = np.random.default_rng(42)
        sfreq = 500
        n_channels, n_times = 16, 10000

        t = np.arange(n_times) / sfreq
        line_noise = 2 * np.sin(2 * np.pi * 50 * t)

        data = rng.standard_normal((n_channels, n_times)) * 0.5
        data[:8] += line_noise

        result = dss_zapline(data, line_freq=50, sfreq=sfreq, n_remove=2)
        metrics = compute_psd_reduction(data, result.cleaned, sfreq, 50)

        assert metrics['reduction_db'] > 5


class TestIntegration:
    """Integration tests."""

    def test_evoked_dss_workflow(self) -> None:
        """Full evoked DSS workflow should improve SNR."""
        rng = np.random.default_rng(42)
        n_channels, n_times, n_epochs = 8, 100, 50

        noise = rng.standard_normal((n_channels, n_times, n_epochs))

        evoked = np.zeros(n_times)
        evoked[40:60] = np.hanning(20)

        mixing = rng.standard_normal(n_channels)
        mixing = mixing / np.linalg.norm(mixing)
        signal = np.outer(mixing, evoked)[:, :, np.newaxis]

        data = noise + signal

        bias = TrialAverageBias()
        dss = DSS(bias=bias, n_components=3)
        sources = dss.fit_transform(data)

        top_source_avg = sources[0].mean(axis=1)
        correlation = np.abs(np.corrcoef(top_source_avg, evoked)[0, 1])

        assert correlation > 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
