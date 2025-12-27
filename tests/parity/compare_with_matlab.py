"""Compare Python DSS with MATLAB NoiseTools reference results.

Usage:
    1. Run generate_reference.m in MATLAB first
    2. Run this script: python compare_with_matlab.py
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import scipy.io as sio

from mne_denoise.dss import compute_dss
from mne_denoise.dss.denoisers.spectral import BandpassBias


def main():
    print("=" * 60)
    print("DSS PARITY TEST: Python vs MATLAB NoiseTools")
    print("=" * 60)
    
    # Load MATLAB reference
    mat_file = Path(__file__).parent / "matlab_reference" / "dss_reference_results.mat"
    
    if not mat_file.exists():
        print(f"\nERROR: Reference file not found: {mat_file}")
        print("\nPlease run generate_reference.m in MATLAB first:")
        print("  1. Open MATLAB")
        print("  2. cd d:\\PhD\\mne-denoise\\tests\\parity\\matlab_reference")
        print("  3. run generate_reference")
        return False
    
    print(f"\nLoading MATLAB reference: {mat_file}")
    mat_data = sio.loadmat(mat_file)
    
    # Extract data
    data = mat_data['data']  # (n_samples, n_channels)
    n_samples, n_channels = data.shape
    sfreq = float(mat_data['sfreq'][0, 0])
    alpha_mixing = mat_data['alpha_mixing'].ravel()
    
    print(f"Data: {n_samples} samples x {n_channels} channels, {sfreq} Hz")
    
    # MATLAB results
    mat_todss_identity = mat_data['todss_identity']
    mat_pwr_identity = mat_data['pwr_identity'].ravel()
    mat_todss_alpha = mat_data['todss_alpha']
    mat_pwr_alpha = mat_data['pwr_alpha'].ravel()
    
    # Convert data to Python format (channels, samples)
    data_py = data.T
    
    # =========================================================================
    # Test 1: Identity bias
    # =========================================================================
    print("\n" + "-" * 60)
    print("Test 1: DSS with Identity Bias")
    print("-" * 60)
    
    py_filters, py_patterns, py_eigs, _ = compute_dss(
        data_py, data_py.copy(), n_components=n_channels
    )
    
    print(f"Python filters shape: {py_filters.shape}")
    print(f"MATLAB todss shape: {mat_todss_identity.shape}")
    
    print(f"\nTop 3 eigenvalues:")
    print(f"  Python: {py_eigs[:3]}")
    print(f"  MATLAB: {mat_pwr_identity[:3]}")
    
    # Compare filters
    print(f"\nFilter correlation (accounting for sign ambiguity):")
    all_corrs = []
    for i in range(min(5, n_channels)):
        # Python: (n_components, n_channels) - row i
        # MATLAB: (n_channels, n_components) - column i
        py_filt = py_filters[i]
        mat_filt = mat_todss_identity[:, i].ravel()
        
        corr = np.abs(np.corrcoef(py_filt, mat_filt)[0, 1])
        all_corrs.append(corr)
        status = "PASS" if corr > 0.95 else "FAIL"
        print(f"  Component {i}: r = {corr:.4f} [{status}]")
    
    test1_pass = np.mean(all_corrs) > 0.95
    
    # =========================================================================
    # Test 2: Alpha bandpass bias
    # =========================================================================
    print("\n" + "-" * 60)
    print("Test 2: DSS with Alpha Bandpass Bias (8-12 Hz)")
    print("-" * 60)
    
    bias = BandpassBias(freq_band=(8, 12), sfreq=sfreq)
    data_alpha = bias.apply(data_py)
    
    py_filters_alpha, _, py_eigs_alpha, _ = compute_dss(
        data_py, data_alpha, n_components=n_channels
    )
    
    print(f"\nTop 3 eigenvalues:")
    print(f"  Python: {py_eigs_alpha[:3]}")
    print(f"  MATLAB: {mat_pwr_alpha[:3]}")
    
    # Compare top filter with true alpha mixing
    print(f"\nTop filter correlation with true alpha source:")
    py_corr = np.abs(np.corrcoef(py_filters_alpha[0], alpha_mixing)[0, 1])
    mat_corr = np.abs(np.corrcoef(mat_todss_alpha[:, 0], alpha_mixing)[0, 1])
    
    print(f"  Python: r = {py_corr:.4f}")
    print(f"  MATLAB: r = {mat_corr:.4f}")
    
    test2_pass = py_corr > 0.7 and mat_corr > 0.7
    
    # Compare Python vs MATLAB filters
    print(f"\nPython vs MATLAB filter correlation:")
    alpha_corrs = []
    for i in range(min(3, n_channels)):
        py_filt = py_filters_alpha[i]
        mat_filt = mat_todss_alpha[:, i].ravel()
        corr = np.abs(np.corrcoef(py_filt, mat_filt)[0, 1])
        alpha_corrs.append(corr)
        status = "PASS" if corr > 0.8 else "FAIL"
        print(f"  Component {i}: r = {corr:.4f} [{status}]")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"Test 1 (Identity bias):  {'PASS' if test1_pass else 'FAIL'}")
    print(f"  Mean filter correlation: {np.mean(all_corrs):.4f}")
    
    print(f"Test 2 (Alpha bias):     {'PASS' if test2_pass else 'FAIL'}")
    print(f"  Python-MATLAB correlation: {np.mean(alpha_corrs):.4f}")
    print(f"  Python source recovery: {py_corr:.4f}")
    print(f"  MATLAB source recovery: {mat_corr:.4f}")
    
    all_pass = test1_pass and test2_pass
    print(f"\nOverall: {'ALL TESTS PASSED!' if all_pass else 'SOME TESTS FAILED'}")
    
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
