"""
03: MEG Artifact Removal Demo
=============================
Replicates MATLAB demo_MEGprior.m - Artifact removal with prior information.

This script demonstrates the full artifact removal pipeline:
1. Muscular artifacts: Time-windowed mask
2. Ocular artifacts (blinks/saccades): EOG-derived adaptive mask
3. Cardiac artifacts (ECG): Quasi-periodic averaging

This is the "gold standard" demo for DSS artifact removal.
"""

import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy import signal

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mne_denoise.dss import (
    compute_dss,
    IterativeDSS,
    WienerMaskDenoiser,
    QuasiPeriodicDenoiser,
    LinearDenoiser,
)

# Setup
plt.style.use('seaborn-v0_8-whitegrid')
OUT_DIR = "results_03_artifact_removal"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# Custom Mask Denoiser (like MATLAB denoise_mask)
# =============================================================================
class FixedMaskDenoiser:
    """Simple fixed mask denoiser (MATLAB denoise_mask equivalent)."""
    def __init__(self, mask):
        self.mask = mask
    
    def denoise(self, source):
        return source * self.mask


# =============================================================================
# 1. Load MEG Data
# =============================================================================
print("=" * 60)
print("03: MEG Artifact Removal Demo (demo_MEGprior.m)")
print("=" * 60)

data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'MEG_data.mat')
try:
    mat = scipy.io.loadmat(data_path)
    X_full = mat['X']
    sfreq = 160.0
    print(f"Loaded MEG_data.mat: {X_full.shape} @ {sfreq} Hz")
except FileNotFoundError:
    print("MEG_data.mat not found. Cannot run this demo without real data.")
    exit(1)

# Separate MEG and auxiliary channels
X_aux = X_full[122:, :]  # EOG, ECG channels
X_aux[-1, :] = -X_aux[-1, :]  # Make R-peaks positive
X = X_full[:122, :]

n_channels, n_samples = X.shape
t = np.arange(n_samples) / sfreq
print(f"MEG: {n_channels} channels × {n_samples} samples ({n_samples/sfreq:.1f} sec)")
print(f"Aux: {X_aux.shape[0]} channels (EOG, ECG, etc.)")

# =============================================================================
# 2. Visualize Raw Data with Artifacts
# =============================================================================
print("\n--- 2. Raw Data with Visible Artifacts ---")

fig, axes = plt.subplots(10, 1, figsize=(14, 12), sharex=True)

for i in range(7):
    axes[i].plot(t, X[i], linewidth=0.5)
    axes[i].set_ylabel(f'MEG {i+1}', fontsize=8)

# Highlight artifact regions
axes[0].axvspan(10000/sfreq, 13000/sfreq, alpha=0.3, color='red', label='Muscular')
axes[0].legend(loc='upper right')

# Auxiliary channels
axes[7].plot(t, X_aux[0], linewidth=0.5, color='orange')
axes[7].set_ylabel('EOG (blinks)', fontsize=8)
axes[8].plot(t, X_aux[1], linewidth=0.5, color='orange')
axes[8].set_ylabel('EOG (saccades)', fontsize=8)
axes[9].plot(t, X_aux[-1], linewidth=0.5, color='red')
axes[9].set_ylabel('ECG', fontsize=8)

axes[-1].set_xlabel('Time (s)')
plt.suptitle('Raw MEG Data with Artifact Markers')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/raw_with_artifacts.png", dpi=150)
print(f"Saved: {OUT_DIR}/raw_with_artifacts.png")

# =============================================================================
# 3. Muscular Artifact Removal (Time-Windowed Mask)
# =============================================================================
print("\n--- 3. Muscular Artifact Removal ---")
print("   Using mask for samples 10000-13000 (like MATLAB demo)")

# Create binary mask (MATLAB style: ones where artifact is)
mask = np.zeros(n_samples)
mask[10000:13000] = 1.0

# Use linear DSS with mask (PCA approach like MATLAB)
# Apply the mask as a bias function
X_masked = X * mask  # Biased data

# Linear DSS to find components with high power in masked region
filters, patterns, eigenvalues, _ = compute_dss(
    X, X_masked, n_components=5
)

muscular_sources = filters @ X
print(f"Extracted {muscular_sources.shape[0]} muscular artifact components")
print(f"Eigenvalues: {eigenvalues[:5]}")

# Plot muscular components
fig, axes = plt.subplots(5, 1, figsize=(12, 8), sharex=True)
for i in range(5):
    axes[i].plot(t, muscular_sources[i], linewidth=0.5)
    axes[i].set_ylabel(f'Musc {i+1}', fontsize=9)
    axes[i].axvspan(10000/sfreq, 13000/sfreq, alpha=0.3, color='red')
axes[-1].set_xlabel('Time (s)')
plt.suptitle('Muscular Artifact Components (high power in marked region)')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/muscular_components.png", dpi=150)
print(f"Saved: {OUT_DIR}/muscular_components.png")

# =============================================================================
# 4. Ocular Artifact Removal (EOG-Derived Mask)
# =============================================================================
print("\n--- 4. Ocular Artifact Removal ---")
print("   Using EOG channels to create adaptive masks (estimate_mask equivalent)")

def estimate_mask(signal, window_samples=50, threshold_percentile=75):
    """Estimate binary mask based on signal dynamics (MATLAB estimate_mask equivalent)."""
    # Smooth variance
    sig_sq = signal ** 2
    window = np.ones(window_samples) / window_samples
    var_smooth = np.convolve(sig_sq, window, mode='same')
    
    # Threshold
    threshold = np.percentile(var_smooth, threshold_percentile)
    mask = (var_smooth > threshold).astype(float)
    return mask

# Create masks from EOG channels
blink_mask = estimate_mask(X_aux[0])
saccade_mask = estimate_mask(X_aux[1])

# Visualize masks
fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
axes[0].plot(t, X_aux[0] / np.max(np.abs(X_aux[0])), label='EOG (blinks)', alpha=0.7)
axes[0].plot(t, blink_mask, label='Blink mask', linewidth=2)
axes[0].legend()
axes[0].set_ylabel('Normalized')

axes[1].plot(t, X_aux[1] / np.max(np.abs(X_aux[1])), label='EOG (saccades)', alpha=0.7)
axes[1].plot(t, saccade_mask, label='Saccade mask', linewidth=2)
axes[1].legend()
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Normalized')

plt.suptitle('Adaptive Masks from EOG Channels')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eog_masks.png", dpi=150)
print(f"Saved: {OUT_DIR}/eog_masks.png")

# Extract blink component using mask
X_blink_masked = X * blink_mask
filters_blink, _, eig_blink, _ = compute_dss(X, X_blink_masked, n_components=1)
blink_source = filters_blink @ X

# Extract saccade component
X_saccade_masked = X * saccade_mask
filters_saccade, _, eig_saccade, _ = compute_dss(X, X_saccade_masked, n_components=1)
saccade_source = filters_saccade @ X

print(f"Blink component eigenvalue: {eig_blink[0]:.4f}")
print(f"Saccade component eigenvalue: {eig_saccade[0]:.4f}")

# Plot ocular components
fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
axes[0].plot(t, X_aux[0], linewidth=0.5, label='EOG (blinks)')
axes[0].legend()
axes[1].plot(t, blink_source[0], linewidth=0.5, color='red', label='Blink DSS')
axes[1].legend()
axes[2].plot(t, X_aux[1], linewidth=0.5, label='EOG (saccades)')
axes[2].legend()
axes[3].plot(t, saccade_source[0], linewidth=0.5, color='red', label='Saccade DSS')
axes[3].legend()
axes[-1].set_xlabel('Time (s)')
plt.suptitle('Ocular Artifact Components')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/ocular_components.png", dpi=150)
print(f"Saved: {OUT_DIR}/ocular_components.png")

# =============================================================================
# 5. Cardiac Artifact Removal (Quasi-Periodic Averaging)
# =============================================================================
print("\n--- 5. Cardiac Artifact Removal ---")
print("   Using ECG channel for quasi-periodic denoising")

# Detect R-peaks (simple threshold detection like MATLAB demo)
ecg = X_aux[-1]
ecg_threshold = 3 * np.std(ecg)
tr_sig = (ecg > ecg_threshold).astype(int)
tr_diff = np.diff(tr_sig)
triggers = np.where(tr_diff == 1)[0] + 1
triggers = triggers[:-1]  # Remove last (close to end)

# Median cycle length
cycle_lengths = np.diff(triggers)
median_length = int(np.median(cycle_lengths))
print(f"Detected {len(triggers)} R-peaks")
print(f"Median RR interval: {median_length} samples ({median_length/sfreq*1000:.0f} ms)")
print(f"Estimated HR: {60*sfreq/median_length:.0f} bpm")

# Use QuasiPeriodicDenoiser
qp_denoiser = QuasiPeriodicDenoiser(
    peak_distance=int(median_length * 0.8),
    peak_height_percentile=75,
)

# Run iterative DSS
idss_ecg = IterativeDSS(
    denoiser=qp_denoiser.denoise,
    n_components=1,
    method='deflation',
    max_iter=30,
    rank=50,
    verbose=True,
)
idss_ecg.fit(X)
cardiac_source = idss_ecg.transform(X)

# Plot cardiac component
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

axes[0].plot(t, ecg, linewidth=0.5, label='ECG reference')
axes[0].scatter(triggers/sfreq, ecg[triggers], c='red', s=10, label='R-peaks')
axes[0].legend()
axes[0].set_ylabel('ECG')

axes[1].plot(t, cardiac_source[0], linewidth=0.5, color='red', label='Cardiac DSS')
axes[1].legend()
axes[1].set_ylabel('Cardiac comp')

# Correlation
corr = np.corrcoef(ecg, cardiac_source[0])[0, 1]
axes[2].plot(t, ecg / np.std(ecg), alpha=0.7, label='ECG (normalized)')
axes[2].plot(t, cardiac_source[0] / np.std(cardiac_source[0]), alpha=0.7, label='Cardiac DSS (normalized)')
axes[2].set_title(f'Correlation: {np.abs(corr):.3f}')
axes[2].legend()
axes[2].set_xlabel('Time (s)')

plt.suptitle('Cardiac Artifact Extraction (Quasi-Periodic DSS)')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/cardiac_component.png", dpi=150)
print(f"Saved: {OUT_DIR}/cardiac_component.png")

# =============================================================================
# 6. Summary: All Artifact Components
# =============================================================================
print("\n--- 6. Summary: All Artifact Components ---")

fig, axes = plt.subplots(8, 1, figsize=(14, 12), sharex=True)

labels = [
    'Musc 1', 'Musc 2', 'Musc 3',
    'Blinks', 'Saccades',
    'Cardiac',
    'EOG ref', 'ECG ref'
]

# Muscular
for i in range(3):
    axes[i].plot(t, muscular_sources[i], linewidth=0.5)
    axes[i].set_ylabel(labels[i])

# Ocular
axes[3].plot(t, blink_source[0], linewidth=0.5)
axes[3].set_ylabel(labels[3])
axes[4].plot(t, saccade_source[0], linewidth=0.5)
axes[4].set_ylabel(labels[4])

# Cardiac
axes[5].plot(t, cardiac_source[0], linewidth=0.5)
axes[5].set_ylabel(labels[5])

# References
axes[6].plot(t, X_aux[0], linewidth=0.5, color='gray')
axes[6].set_ylabel(labels[6])
axes[7].plot(t, X_aux[-1], linewidth=0.5, color='gray')
axes[7].set_ylabel(labels[7])

axes[-1].set_xlabel('Time (s)')
plt.suptitle('Complete Artifact Decomposition (demo_MEGprior.m equivalent)')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/all_artifacts_summary.png", dpi=150)
print(f"Saved: {OUT_DIR}/all_artifacts_summary.png")

print(f"\n[OK] All results saved to {OUT_DIR}/")
print("\nArtifact removal pipeline complete!")
print("Components extracted:")
print(f"  - 5 muscular (time-windowed mask)")
print(f"  - 1 blink (EOG-derived mask)")
print(f"  - 1 saccade (EOG-derived mask)")
print(f"  - 1 cardiac (quasi-periodic averaging)")
