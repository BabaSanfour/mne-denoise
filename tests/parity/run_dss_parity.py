#!/usr/bin/env python
"""DSS Parity Test: Python vs MATLAB NoiseTools.

This script compares our Python DSS implementation against MATLAB NoiseTools
to verify numerical equivalence for publication.

Requires:
- MATLAB Engine for Python
- NoiseTools in MATLAB path

Usage:
    python run_dss_parity.py
    python run_dss_parity.py --no-matlab  # Synthetic test only
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from mne_denoise.dss import compute_dss
from mne_denoise.dss.denoisers.spectral import BandpassBias


def generate_test_data(
    n_channels: int = 16,
    n_samples: int = 5000,
    sfreq: float = 250.0,
    seed: int = 42,
) -> dict:
    """Generate synthetic test data with known ground truth."""
    np.random.seed(seed)
    t = np.arange(n_samples) / sfreq

    # Ground truth sources
    s1 = np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
    s2 = np.sin(2 * np.pi * 20 * t)  # 20 Hz beta
    s3 = np.random.randn(n_samples) * 0.5  # Noise

    sources = np.vstack([s1, s2, s3])
    n_sources = sources.shape[0]

    # Random mixing matrix
    mixing = np.random.randn(n_channels, n_sources)

    # Mixed data
    data = mixing @ sources + 0.3 * np.random.randn(n_channels, n_samples)

    return {
        "data": data,
        "sources": sources,
        "mixing": mixing,
        "sfreq": sfreq,
        "t": t,
    }


def run_python_dss(data: np.ndarray, biased_data: np.ndarray) -> dict:
    """Run Python DSS and return results."""
    start = time.perf_counter()
    filters, patterns, eigenvalues = compute_dss(data, biased_data, n_components=None)
    elapsed = time.perf_counter() - start

    return {
        "filters": filters,
        "patterns": patterns,
        "eigenvalues": eigenvalues,
        "time": elapsed,
    }


def run_matlab_dss(eng, data: np.ndarray, biased_data: np.ndarray) -> dict:
    """Run MATLAB nt_dss0 and return results."""
    import matlab

    # Convert to MATLAB format (samples, channels)
    data_mat = matlab.double(data.T.tolist())
    biased_mat = matlab.double(biased_data.T.tolist())

    start = time.perf_counter()
    result = eng.nt_dss0(data_mat, biased_mat, nargout=3)
    elapsed = time.perf_counter() - start

    # Convert back to numpy
    todss = np.array(result[0])  # (n_channels, n_components)
    fromdss = np.array(result[1])  # (n_components, n_channels)
    pwr = np.array(result[2]).ravel()  # eigenvalues

    return {
        "filters": todss.T,  # Convert to (n_components, n_channels)
        "patterns": fromdss.T,
        "eigenvalues": pwr,
        "time": elapsed,
    }


def compare_results(py_result: dict, mat_result: dict, name: str) -> dict:
    """Compare Python and MATLAB results."""
    n_compare = min(5, len(py_result["eigenvalues"]), len(mat_result["eigenvalues"]))

    # Compare filters
    filter_corrs = []
    for i in range(n_compare):
        py_filt = py_result["filters"][i]
        mat_filt = mat_result["filters"][i]

        # Handle sign ambiguity
        corr = np.corrcoef(py_filt, mat_filt)[0, 1]
        filter_corrs.append(np.abs(corr))

    # Compare eigenvalues
    py_eigs = py_result["eigenvalues"][:n_compare]
    mat_eigs = mat_result["eigenvalues"][:n_compare]

    # Normalize for comparison
    py_eigs_norm = py_eigs / py_eigs[0] if py_eigs[0] > 0 else py_eigs
    mat_eigs_norm = mat_eigs / mat_eigs[0] if mat_eigs[0] > 0 else mat_eigs

    eig_corr = np.corrcoef(py_eigs_norm, mat_eigs_norm)[0, 1]

    return {
        "name": name,
        "filter_correlations": filter_corrs,
        "mean_filter_corr": np.mean(filter_corrs),
        "eigenvalue_corr": eig_corr,
        "python_time": py_result["time"],
        "matlab_time": mat_result["time"],
    }


def print_report(comparisons: list[dict]) -> None:
    """Print parity test report."""
    print("\n" + "=" * 70)
    print("DSS PARITY TEST REPORT: Python vs MATLAB NoiseTools")
    print("=" * 70)

    all_passed = True

    for comp in comparisons:
        passed = comp["mean_filter_corr"] > 0.95 and comp["eigenvalue_corr"] > 0.95
        status = "✅ PASS" if passed else "❌ FAIL"
        all_passed = all_passed and passed

        print(f"\n{comp['name']}:")
        print(f"  Filter correlations: {comp['filter_correlations']}")
        print(f"  Mean filter corr: {comp['mean_filter_corr']:.4f}")
        print(f"  Eigenvalue corr:  {comp['eigenvalue_corr']:.4f}")
        print(f"  Python time: {comp['python_time']:.3f}s")
        print(f"  MATLAB time: {comp['matlab_time']:.3f}s")
        print(f"  Status: {status}")

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL PARITY TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED - Review results above")
    print("=" * 70)


def run_parity_tests(use_matlab: bool = True) -> None:
    """Run complete parity test suite."""
    print("Generating test data...")
    test_data = generate_test_data()
    data = test_data["data"]
    sfreq = test_data["sfreq"]

    comparisons = []

    # Test 1: Identity bias (biased = original)
    print("\n--- Test 1: Identity Bias ---")
    py_ident = run_python_dss(data, data.copy())

    if use_matlab:
        try:
            import matlab.engine

            print("Starting MATLAB engine...")
            eng = matlab.engine.start_matlab()
            eng.addpath(eng.genpath(r"D:\PhD\NoiseTools"), nargout=0)  # Adjust path

            mat_ident = run_matlab_dss(eng, data, data.copy())
            comp = compare_results(py_ident, mat_ident, "Identity Bias (C0 = C1)")
            comparisons.append(comp)
        except ImportError:
            print("MATLAB Engine not available, skipping MATLAB tests")
            use_matlab = False
        except Exception as e:
            print(f"MATLAB error: {e}")
            use_matlab = False

    # Test 2: Bandpass bias (alpha)
    print("\n--- Test 2: Bandpass Bias (8-12 Hz) ---")
    bias = BandpassBias(freq_band=(8, 12), sfreq=sfreq)
    biased = bias.apply(data)
    py_bp = run_python_dss(data, biased)

    if use_matlab:
        # Apply same filter in MATLAB for fair comparison
        eng.eval("[b, a] = butter(4, [8 12]/(250/2), 'bandpass');", nargout=0)
        import matlab

        data_mat = matlab.double(data.T.tolist())
        eng.workspace["data"] = data_mat
        biased_mat = eng.eval("filtfilt(b, a, data)", nargout=1)

        mat_bp = run_matlab_dss(eng, data, np.array(biased_mat).T)
        comp = compare_results(py_bp, mat_bp, "Bandpass Bias (8-12 Hz)")
        comparisons.append(comp)

    # Print report
    if comparisons:
        print_report(comparisons)
    else:
        print("\nNo MATLAB comparison available. Python-only results:")
        print(f"  Python DSS computed {len(py_ident['eigenvalues'])} components")
        print(f"  Top eigenvalues: {py_ident['eigenvalues'][:5]}")

    # Cleanup
    if use_matlab:
        try:
            eng.quit()
        except:
            pass


def run_synthetic_validation() -> None:
    """Run synthetic validation without MATLAB."""
    print("\n" + "=" * 70)
    print("SYNTHETIC DSS VALIDATION (No MATLAB Required)")
    print("=" * 70)

    test_data = generate_test_data()
    data = test_data["data"]
    sources = test_data["sources"]
    mixing = test_data["mixing"]
    sfreq = test_data["sfreq"]

    # Test: Can we recover the 10 Hz source?
    print("\n--- Test: Recover 10 Hz source using bandpass bias ---")
    bias = BandpassBias(freq_band=(8, 12), sfreq=sfreq)
    biased = bias.apply(data)

    filters, patterns, eigenvalues = compute_dss(data, biased)

    # Top component should correlate with first source (10 Hz)
    top_source_recovered = filters[0] @ data
    true_10hz = sources[0]

    corr = np.abs(np.corrcoef(top_source_recovered, true_10hz)[0, 1])
    print(f"  Correlation with true 10 Hz source: {corr:.4f}")

    # Pattern should correlate with mixing column
    pattern_corr = np.abs(np.corrcoef(patterns[:, 0], mixing[:, 0])[0, 1])
    print(f"  Pattern correlation with true mixing: {pattern_corr:.4f}")

    # Assert
    if corr > 0.95 and pattern_corr > 0.9:
        print("  ✅ VALIDATION PASSED")
    else:
        print("  ❌ VALIDATION FAILED")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSS Parity Test")
    parser.add_argument(
        "--no-matlab", action="store_true", help="Skip MATLAB tests, run synthetic only"
    )
    args = parser.parse_args()

    if args.no_matlab:
        run_synthetic_validation()
    else:
        run_parity_tests(use_matlab=True)
