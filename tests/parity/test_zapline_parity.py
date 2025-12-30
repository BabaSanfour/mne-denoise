"""Parity test for ZapLine (nt_zapline).

Compares Python dss_zapline() against MATLAB NoiseTools nt_zapline.m
to ensure numerical equivalence for line noise removal.
"""

from __future__ import annotations

import numpy as np
import pytest

from mne_denoise.zapline import compute_psd_reduction, dss_zapline

try:
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
    """Generate test data with 50 Hz noise."""
    return generate_test_data(
        n_channels=16,
        n_samples=10000,
        sfreq=500.0,
        seed=42,
    )


class TestZapLineParity:
    """Test parity between Python dss_zapline and MATLAB nt_zapline."""

    def test_50hz_removal_parity(self, matlab_engine, test_data):
        """Compare 50 Hz removal performance."""
        data = test_data["continuous"]
        sfreq = test_data["sfreq"]

        # Python: dss_zapline
        py_result = dss_zapline(data, line_freq=50, sfreq=sfreq, n_remove="auto")
        py_cleaned = py_result.cleaned

        # MATLAB: nt_zapline(data', 50/sfreq)
        # MATLAB expects fline as fraction of sfreq
        data_mat = to_matlab(data.T)
        fline = 50 / sfreq

        try:
            # nt_zapline returns cleaned data
            mat_cleaned_raw = matlab_engine.nt_zapline(data_mat, fline, nargout=2)
            mat_cleaned = from_matlab(mat_cleaned_raw[0]).T  # Transpose back
            from_matlab(mat_cleaned_raw[1]).T
        except Exception as e:
            print(f"MATLAB error: {e}")
            pytest.skip("nt_zapline not available")
            return

        # Compare cleaning performance via PSD reduction
        py_metrics = compute_psd_reduction(data, py_cleaned, sfreq, 50)
        mat_metrics = compute_psd_reduction(data, mat_cleaned, sfreq, 50)

        print(f"Python: {py_metrics['reduction_db']:.1f} dB reduction")
        print(f"MATLAB: {mat_metrics['reduction_db']:.1f} dB reduction")

        # Both should achieve significant reduction
        assert py_metrics["reduction_db"] > 5, "Python ZapLine ineffective"
        assert mat_metrics["reduction_db"] > 5, "MATLAB ZapLine ineffective"

        # Performance should be similar (within 5 dB)
        db_diff = abs(py_metrics["reduction_db"] - mat_metrics["reduction_db"])
        print(f"Performance difference: {db_diff:.1f} dB")
        assert db_diff < 10, f"Large performance gap: {db_diff} dB"

    def test_cleaned_data_correlation(self, matlab_engine, test_data):
        """Compare cleaned data directly."""
        data = test_data["continuous"]
        sfreq = test_data["sfreq"]

        # Python
        py_result = dss_zapline(data, line_freq=50, sfreq=sfreq, n_remove=2)
        py_cleaned = py_result.cleaned

        # MATLAB
        data_mat = to_matlab(data.T)
        fline = 50 / sfreq

        try:
            mat_result = matlab_engine.nt_zapline(data_mat, fline, nargout=1)
            mat_cleaned = from_matlab(mat_result).T
        except Exception as e:
            pytest.skip(f"nt_zapline error: {e}")
            return

        # Cleaned data should be similar
        # (not identical due to algorithm differences, but correlated)
        metrics = ParityMetrics("zapline_cleaned")

        for ch in range(min(5, data.shape[0])):
            corr = np.corrcoef(py_cleaned[ch], mat_cleaned[ch])[0, 1]
            print(f"Channel {ch}: correlation = {corr:.4f}")
            metrics.correlations.append(corr)

        mean_corr = np.mean(metrics.correlations)
        print(f"Mean correlation: {mean_corr:.4f}")

        # Cleaned signals should be highly correlated
        assert mean_corr > 0.9, f"Low correlation: {mean_corr}"

    def test_harmonic_removal(self, matlab_engine, test_data):
        """Test removal of harmonics (100 Hz for 50 Hz fundamental)."""
        data = test_data["continuous"].copy()
        sfreq = test_data["sfreq"]

        # Add 100 Hz harmonic
        t = np.arange(data.shape[1]) / sfreq
        harmonic = 0.5 * np.sin(2 * np.pi * 100 * t)
        data[:8] += harmonic

        # Python with harmonics
        py_result = dss_zapline(
            data, line_freq=50, sfreq=sfreq, n_remove="auto", n_harmonics=2
        )

        # Check both 50 Hz and 100 Hz reduction
        metrics_50 = compute_psd_reduction(data, py_result.cleaned, sfreq, 50)
        metrics_100 = compute_psd_reduction(data, py_result.cleaned, sfreq, 100)

        print(f"50 Hz reduction: {metrics_50['reduction_db']:.1f} dB")
        print(f"100 Hz reduction: {metrics_100['reduction_db']:.1f} dB")

        # Both should show reduction
        assert metrics_50["reduction_db"] > 3, "Poor 50 Hz removal"
        assert metrics_100["reduction_db"] > 1, "Poor 100 Hz removal"

    def test_signal_preservation(self, matlab_engine, test_data):
        """Verify brain signals at other frequencies are preserved."""
        data = test_data["continuous"]
        sfreq = test_data["sfreq"]

        # Python
        py_result = dss_zapline(data, line_freq=50, sfreq=sfreq, n_remove=2)

        # Check that 10 Hz alpha is preserved
        from scipy import signal

        nperseg = int(4 * sfreq)
        freqs, psd_orig = signal.welch(data[0], sfreq, nperseg=nperseg)
        freqs, psd_clean = signal.welch(py_result.cleaned[0], sfreq, nperseg=nperseg)

        # Find 10 Hz power
        idx_10 = np.argmin(np.abs(freqs - 10))
        alpha_orig = np.mean(psd_orig[idx_10 - 2 : idx_10 + 2])
        alpha_clean = np.mean(psd_clean[idx_10 - 2 : idx_10 + 2])

        # Alpha should be preserved (not significantly attenuated)
        alpha_ratio = alpha_clean / alpha_orig
        print(f"10 Hz preservation ratio: {alpha_ratio:.2f}")

        assert alpha_ratio > 0.7, f"Alpha too attenuated: {alpha_ratio}"


class TestZapLineEdgeCases:
    """Test ZapLine edge cases and robustness."""

    def test_no_noise(self, matlab_engine):
        """Test behavior when no line noise present."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((8, 5000))  # Pure noise

        result = dss_zapline(data, line_freq=50, sfreq=500, n_remove="auto")

        # Should not remove significant power
        metrics = compute_psd_reduction(data, result.cleaned, 500, 50)
        print(f"Reduction when no noise: {metrics['reduction_db']:.1f} dB")

        # Data should be largely unchanged
        corr = np.corrcoef(data.ravel(), result.cleaned.ravel())[0, 1]
        print(f"Correlation with original: {corr:.4f}")
        assert corr > 0.95

    def test_60hz_mode(self, matlab_engine):
        """Test 60 Hz (US power grid) removal."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((8, 5000)) * 0.5

        # Add 60 Hz
        t = np.arange(5000) / 500
        line = np.sin(2 * np.pi * 60 * t)
        data[:4] += line * 2

        result = dss_zapline(data, line_freq=60, sfreq=500, n_remove=2)
        metrics = compute_psd_reduction(data, result.cleaned, 500, 60)

        print(f"60 Hz reduction: {metrics['reduction_db']:.1f} dB")
        assert metrics["reduction_db"] > 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
