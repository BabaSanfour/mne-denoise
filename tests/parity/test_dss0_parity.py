"""Parity test for linear DSS (nt_dss0).

Compares Python compute_dss() against MATLAB NoiseTools nt_dss0.m
to ensure numerical equivalence.
"""

from __future__ import annotations

import numpy as np
import pytest

from mne_denoise.dss import compute_dss
from mne_denoise.dss.denoisers.spectral import BandpassBias

try:
    import matlab.engine
    from . import (
        ParityMetrics,
        close_matlab_engine,
        from_matlab,
        generate_test_data,
        get_matlab_engine,
        to_matlab,
    )

    HAS_MATLAB = True
except ImportError:
    HAS_MATLAB = False


@pytest.fixture(scope="module")
def matlab_engine():
    """Start MATLAB engine for tests."""
    if not HAS_MATLAB:
        pytest.skip("MATLAB Engine not available")
    try:
        eng = get_matlab_engine()
        yield eng
    finally:
        close_matlab_engine()


@pytest.fixture(scope="module")
def test_data():
    """Generate test data."""
    return generate_test_data(
        n_channels=16,
        n_samples=5000,
        sfreq=250.0,
        seed=42,
    )


class TestDSS0Parity:
    """Test parity between Python compute_dss and MATLAB nt_dss0."""

    def test_identity_bias_parity(self, matlab_engine, test_data):
        """Compare with identity bias (biased = original)."""
        data = test_data["continuous"]
        n_channels, n_samples = data.shape

        # Python: compute DSS with identity bias
        py_filters, py_patterns, py_eigs, _ = compute_dss(
            data, data.copy(), n_components=n_channels
        )

        # MATLAB: nt_dss0(data', data')
        # Note: MATLAB uses (samples, channels), Python uses (channels, samples)
        data_mat = to_matlab(data.T)  # Transpose for MATLAB

        result = matlab_engine.nt_dss0(data_mat, data_mat, nargout=3)
        mat_filters = from_matlab(result[0])  # todss
        from_matlab(result[1])  # fromdss
        from_matlab(result[2])  # pwr_ratio (eigenvalues)

        # Compare filters (top components)
        metrics = ParityMetrics("dss0_identity")
        n_compare = min(5, n_channels)

        for i in range(n_compare):
            # Python filters are (n_components, n_channels)
            # MATLAB todss is (n_channels, n_components)
            py_filt = py_filters[i]
            mat_filt = mat_filters[:, i].ravel()

            result = metrics.add_comparison(py_filt, mat_filt, f"filter_{i}")
            print(f"Component {i}: corr={result['correlation']:.4f}")

        summary = metrics.summary()
        print(f"\n{metrics}")

        # Assert high correlation (allowing for sign flips)
        assert summary["mean_correlation"] > 0.99, f"Low correlation: {summary}"
        assert summary["mean_rmse"] < 0.05, f"High RMSE: {summary}"

    def test_bandpass_bias_parity(self, matlab_engine, test_data):
        """Compare with bandpass (alpha) bias."""
        data = test_data["continuous"]
        sfreq = test_data["sfreq"]
        n_channels = data.shape[0]

        # Python: compute DSS with alpha band bias
        bias = BandpassBias(freq_band=(8, 12), sfreq=sfreq)
        biased = bias.apply(data)
        py_filters, py_patterns, py_eigs, _ = compute_dss(
            data, biased, n_components=n_channels
        )

        # MATLAB: apply same filtering then nt_dss0
        data_mat = to_matlab(data.T)

        # Use MATLAB's filter for fairness
        matlab_engine.eval("[b, a] = butter(4, [8 12]/(250/2), 'bandpass');", nargout=0)
        biased_mat = matlab_engine.eval(f"filtfilt(b, a, {data_mat})", nargout=1)

        result = matlab_engine.nt_dss0(data_mat, biased_mat, nargout=3)
        mat_filters = from_matlab(result[0])
        from_matlab(result[2])

        # Compare
        metrics = ParityMetrics("dss0_bandpass")
        n_compare = min(5, n_channels)

        for i in range(n_compare):
            py_filt = py_filters[i]
            mat_filt = mat_filters[:, i].ravel()
            metrics.add_comparison(py_filt, mat_filt, f"filter_{i}")

        summary = metrics.summary()
        print(f"\n{metrics}")

        # Slightly looser tolerance due to filter differences
        assert summary["mean_correlation"] > 0.95, f"Low correlation: {summary}"

    def test_eigenvalue_ordering(self, matlab_engine, test_data):
        """Verify eigenvalues are in same order (descending)."""
        data = test_data["continuous"]

        # Python
        _, _, py_eigs, _ = compute_dss(data, data.copy())

        # MATLAB
        data_mat = to_matlab(data.T)
        result = matlab_engine.nt_dss0(data_mat, data_mat, nargout=3)
        mat_eigs = from_matlab(result[2]).ravel()

        # Both should be descending
        assert np.all(np.diff(py_eigs) <= 0), "Python eigenvalues not descending"
        assert np.all(np.diff(mat_eigs) <= 0), "MATLAB eigenvalues not descending"

        # Compare eigenvalue ratios (normalized)
        py_norm = py_eigs / py_eigs[0]
        mat_norm = mat_eigs / mat_eigs[0]

        corr = np.corrcoef(py_norm, mat_norm[: len(py_norm)])[0, 1]
        print(f"Eigenvalue correlation: {corr:.4f}")
        assert corr > 0.99, f"Eigenvalue mismatch: corr={corr}"


class TestDSS1Parity:
    """Test parity between Python DSS (TrialAverageBias) and MATLAB nt_dss1."""

    def test_evoked_bias_parity(self, matlab_engine, test_data):
        """Compare evoked/ERP-based DSS."""
        from mne_denoise.dss import DSS, TrialAverageBias

        epoched = test_data["epoched"]
        n_channels, n_times, n_epochs = epoched.shape

        # Python: DSS with trial average bias
        bias = TrialAverageBias()
        dss = DSS(bias=bias, n_components=n_channels)
        dss.fit(epoched)
        py_filters = dss.filters_

        # MATLAB: nt_dss1(data)
        # Reshape for MATLAB: (samples, channels, epochs)
        epoched_mat = to_matlab(epoched.transpose(1, 0, 2))

        result = matlab_engine.nt_dss1(epoched_mat, nargout=3)
        mat_filters = from_matlab(result[0])  # todss
        from_matlab(result[2])  # pwr

        # Compare top components
        metrics = ParityMetrics("dss1_evoked")
        n_compare = min(5, n_channels)

        for i in range(n_compare):
            py_filt = py_filters[i]
            mat_filt = mat_filters[:, i].ravel()
            result = metrics.add_comparison(py_filt, mat_filt, f"filter_{i}")
            print(f"Component {i}: corr={result['correlation']:.4f}")

        summary = metrics.summary()
        print(f"\n{metrics}")

        assert summary["mean_correlation"] > 0.95, f"Low correlation: {summary}"

    def test_evoked_component_recovers_signal(self, matlab_engine, test_data):
        """Verify top component correlates with true evoked source."""
        from mne_denoise.dss import DSS, TrialAverageBias

        epoched = test_data["epoched"]
        evoked_mixing = test_data["evoked_mixing"]

        # Python
        bias = TrialAverageBias()
        dss = DSS(bias=bias, n_components=5)
        dss.fit(epoched)

        top_filter = dss.filters_[0]

        # Top filter should correlate with evoked source mixing
        corr = np.abs(np.corrcoef(top_filter, evoked_mixing)[0, 1])
        print(f"Top filter correlation with true evoked: {corr:.4f}")

        # Should be high (evoked is dominant)
        assert corr > 0.7, f"Poor evoked recovery: {corr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
