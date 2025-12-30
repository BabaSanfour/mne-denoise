"""Run all MATLAB parity tests.

Usage:
    conda activate amica310
    python run_parity_tests.py

Requires:
    - MATLAB Engine for Python
    - NoiseTools in MATLAB path
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_all_parity_tests():
    """Run complete parity test suite."""
    print("=" * 60)
    print("DSS MATLAB PARITY TESTS")
    print("=" * 60)

    # Import test infrastructure
    try:
        from mne_denoise.dss import DSS, compute_dss
        from mne_denoise.dss.denoisers import TrialAverageBias
        from mne_denoise.zapline import compute_psd_reduction, dss_zapline
        from tests.parity import (
            ParityMetrics,
            close_matlab_engine,
            from_matlab,
            generate_test_data,
            get_matlab_engine,
            to_matlab,
        )
    except ImportError as e:
        print(f"Import error: {e}")
        return False

    # Start MATLAB Engine
    print("\n1. Starting MATLAB Engine...")
    try:
        eng = get_matlab_engine()
        print("   ✓ MATLAB Engine started")
    except Exception as e:
        print(f"   ✗ Failed to start MATLAB: {e}")
        return False

    # Generate test data
    print("\n2. Generating test data...")
    test_data = generate_test_data(
        n_channels=16,
        n_samples=5000,
        n_epochs=50,
        sfreq=250.0,
        seed=42,
    )
    print(f"   Continuous: {test_data['continuous'].shape}")
    print(f"   Epoched: {test_data['epoched'].shape}")

    results = {}

    # Test 1: nt_dss0 (identity bias)
    print("\n3. Testing nt_dss0 (identity bias)...")
    try:
        data = test_data["continuous"]

        # Python
        py_filters, py_patterns, py_eigs, _ = compute_dss(
            data, data.copy(), n_components=16
        )

        # MATLAB
        data_mat = to_matlab(data.T)
        result = eng.nt_dss0(data_mat, data_mat, nargout=3)
        mat_filters = from_matlab(result[0])
        from_matlab(result[2])

        # Compare
        metrics = ParityMetrics("dss0_identity")
        for i in range(5):
            metrics.add_comparison(py_filters[i], mat_filters[:, i].ravel())

        summary = metrics.summary()
        results["dss0_identity"] = summary
        print(f"   ✓ Mean correlation: {summary['mean_correlation']:.4f}")

    except Exception as e:
        print(f"   ✗ Error: {e}")
        results["dss0_identity"] = {"error": str(e)}

    # Test 2: nt_dss1 (evoked bias)
    print("\n4. Testing nt_dss1 (evoked bias)...")
    try:
        epoched = test_data["epoched"]

        # Python
        bias = TrialAverageBias()
        dss = DSS(bias=bias, n_components=16)
        dss.fit(epoched)
        py_filters = dss.filters_

        # MATLAB
        epoched_mat = to_matlab(epoched.transpose(1, 0, 2))
        result = eng.nt_dss1(epoched_mat, nargout=3)
        mat_filters = from_matlab(result[0])

        # Compare
        metrics = ParityMetrics("dss1_evoked")
        for i in range(5):
            metrics.add_comparison(py_filters[i], mat_filters[:, i].ravel())

        summary = metrics.summary()
        results["dss1_evoked"] = summary
        print(f"   ✓ Mean correlation: {summary['mean_correlation']:.4f}")

    except Exception as e:
        print(f"   ✗ Error: {e}")
        results["dss1_evoked"] = {"error": str(e)}

    # Test 3: nt_zapline
    print("\n5. Testing nt_zapline (50 Hz)...")
    try:
        # Create data with 50 Hz
        data = test_data["continuous"].copy()
        sfreq = 500
        n_samples = data.shape[1]
        t = np.arange(n_samples) / sfreq
        line_noise = 2 * np.sin(2 * np.pi * 50 * t)
        data[:8] += line_noise

        # Python
        py_result = dss_zapline(data, line_freq=50, sfreq=sfreq, n_remove=2)
        py_metrics = compute_psd_reduction(data, py_result.cleaned, sfreq, 50)

        # MATLAB
        data_mat = to_matlab(data.T)
        fline = 50 / sfreq
        mat_result = eng.nt_zapline(data_mat, fline, nargout=1)
        mat_cleaned = from_matlab(mat_result).T
        mat_metrics = compute_psd_reduction(data, mat_cleaned, sfreq, 50)

        results["zapline"] = {
            "py_reduction_db": py_metrics["reduction_db"],
            "mat_reduction_db": mat_metrics["reduction_db"],
            "difference_db": abs(
                py_metrics["reduction_db"] - mat_metrics["reduction_db"]
            ),
        }
        print(f"   ✓ Python: {py_metrics['reduction_db']:.1f} dB")
        print(f"   ✓ MATLAB: {mat_metrics['reduction_db']:.1f} dB")

    except Exception as e:
        print(f"   ✗ Error: {e}")
        results["zapline"] = {"error": str(e)}

    # Close MATLAB
    print("\n6. Closing MATLAB Engine...")
    close_matlab_engine()
    print("   ✓ Done")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, result in results.items():
        if "error" in result:
            print(f"  {test_name}: FAILED - {result['error']}")
            all_passed = False
        elif "mean_correlation" in result:
            passed = result["mean_correlation"] > 0.95
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {test_name}: {status} (corr={result['mean_correlation']:.4f})")
            if not passed:
                all_passed = False
        elif "py_reduction_db" in result:
            passed = result["difference_db"] < 5
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {test_name}: {status} (diff={result['difference_db']:.1f} dB)")
            if not passed:
                all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED")

    return all_passed


if __name__ == "__main__":
    success = run_all_parity_tests()
    sys.exit(0 if success else 1)
