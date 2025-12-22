"""
02: MEG Blind ICA Demo
======================
Replicates MATLAB demo_MEGblind.m - Blind source separation on MEG data.

This script demonstrates:
1. Loading real MEG data (122 channels + 5 auxiliary)
2. Dimension reduction (whitening to 50 components)
3. Extracting 20 ICA-like sources using tanh nonlinearity
4. Visualizing extracted sources
5. Comparing different nonlinearities on real data
"""

import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy import signal

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mne_denoise.dss import (
    IterativeDSS,
    TanhMaskDenoiser,
    RobustTanhDenoiser,
    GaussDenoiser,
    KurtosisDenoiser,
    beta_tanh,
)

# Setup
plt.style.use('seaborn-v0_8-whitegrid')
OUT_DIR = "results_02_meg_blind"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# 1. Load MEG Data
# =============================================================================
print("=" * 60)
print("02: MEG Blind ICA Demo (demo_MEGblind.m)")
print("=" * 60)

data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'MEG_data.mat')
try:
    mat = scipy.io.loadmat(data_path)
    X_full = mat['X']
    sfreq = 160.0  # Hz (from MATLAB demo)
    print(f"Loaded MEG_data.mat: {X_full.shape} @ {sfreq} Hz")
except FileNotFoundError:
    print("MEG_data.mat not found. Generating synthetic MEG-like data...")
    np.random.seed(42)
    n_channels, n_samples = 127, 17730
    sfreq = 160.0
    t = np.arange(n_samples) / sfreq
    # Simulate some sources
    sources = np.vstack([
        np.sin(2 * np.pi * 10 * t),  # Alpha
        np.sin(2 * np.pi * 20 * t),  # Beta
        signal.sawtooth(2 * np.pi * 1 * t),  # Slow artifact
        np.random.randn(n_samples),  # Noise
    ])
    A = np.random.randn(n_channels, 4)
    X_full = A @ sources + 0.5 * np.random.randn(n_channels, n_samples)

# Separate MEG and auxiliary channels (like MATLAB demo)
X_aux = X_full[122:, :]  # Last 5 channels: EOG, ECG, etc.
if X_aux.shape[0] > 0:
    X_aux[-1, :] = -X_aux[-1, :]  # Make R-peaks positive (MATLAB convention)
X = X_full[:122, :]  # MEG channels only

n_channels, n_samples = X.shape
duration = n_samples / sfreq
print(f"MEG data: {n_channels} channels × {n_samples} samples ({duration:.1f} sec)")
print(f"Auxiliary: {X_aux.shape[0]} channels")

# =============================================================================
# 2. Visualize Raw Data
# =============================================================================
print("\n--- 2. Visualizing Raw Data ---")

t = np.arange(n_samples) / sfreq
fig, axes = plt.subplots(10, 1, figsize=(14, 12), sharex=True)

# Plot first 7 MEG channels
for i in range(7):
    axes[i].plot(t, X[i], linewidth=0.5)
    axes[i].set_ylabel(f'MEG {i+1}', fontsize=8)

# Plot auxiliary channels
aux_labels = ['EOG (blinks)', 'EOG (saccades)', 'ECG']
for i, label in enumerate(aux_labels):
    if i < X_aux.shape[0]:
        axes[7 + i].plot(t, X_aux[i], linewidth=0.5, color='red')
        axes[7 + i].set_ylabel(label, fontsize=8)

axes[-1].set_xlabel('Time (s)')
plt.suptitle('Raw MEG Data with Auxiliary Channels')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/raw_meg_data.png", dpi=150)
print(f"Saved: {OUT_DIR}/raw_meg_data.png")

# =============================================================================
# 3. Blind ICA with Tanh (demo_MEGblind.m equivalent)
# =============================================================================
print("\n--- 3. Blind ICA with TanhMaskDenoiser ---")
print("   (Equivalent to MATLAB demo_MEGblind.m)")

# Parameters from MATLAB demo
wdim = 50  # Reduce dimension to 50 (avoid overfitting)
sdim = 20  # Extract 20 components

denoiser = TanhMaskDenoiser(alpha=1.0)
idss = IterativeDSS(
    denoiser=denoiser.denoise,
    n_components=sdim,
    method='deflation',
    max_iter=100,
    rank=wdim,  # Dimension reduction during whitening
    verbose=True,
    beta=beta_tanh,  # FastICA-style Newton step
)

print(f"Running DSS with rank={wdim}, n_components={sdim}...")
idss.fit(X)
sources = idss.transform(X)

print(f"Extracted {sources.shape[0]} components")

# Plot extracted sources
fig, axes = plt.subplots(sdim, 1, figsize=(14, 16), sharex=True)
for i in range(sdim):
    axes[i].plot(t, sources[i], linewidth=0.5)
    axes[i].set_ylabel(f'S{i+1}', fontsize=8)
axes[-1].set_xlabel('Time (s)')
plt.suptitle(f'Extracted ICA Sources (Blind DSS, {sdim} components)')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/blind_ica_sources.png", dpi=150)
print(f"Saved: {OUT_DIR}/blind_ica_sources.png")

# =============================================================================
# 4. Power Spectral Density of Extracted Sources
# =============================================================================
print("\n--- 4. PSD of Extracted Sources ---")

fig, axes = plt.subplots(4, 5, figsize=(14, 10))
axes = axes.flatten()

for i in range(min(sdim, 20)):
    f, psd = signal.welch(sources[i], sfreq, nperseg=1024)
    axes[i].semilogy(f, psd)
    axes[i].set_xlim([0, 80])
    axes[i].set_title(f'S{i+1}', fontsize=9)
    if i >= 15:
        axes[i].set_xlabel('Freq (Hz)')
    if i % 5 == 0:
        axes[i].set_ylabel('PSD')

plt.suptitle('Power Spectral Density of Extracted Sources')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/source_psd.png", dpi=150)
print(f"Saved: {OUT_DIR}/source_psd.png")

# =============================================================================
# 5. Compare Nonlinearities on Real MEG
# =============================================================================
print("\n--- 5. Comparing Nonlinearities on Real MEG ---")

n_compare = 5  # Components to show
denoisers_compare = {
    'TanhMask': TanhMaskDenoiser(alpha=1.0),
    'RobustTanh': RobustTanhDenoiser(alpha=1.0),
    'Gauss': GaussDenoiser(a=1.0),
    'Kurtosis': KurtosisDenoiser(),
}

fig, axes = plt.subplots(len(denoisers_compare), n_compare, figsize=(14, 8))

for row, (name, denoiser) in enumerate(denoisers_compare.items()):
    print(f"   Running {name}...")
    idss = IterativeDSS(
        denoiser=denoiser.denoise,
        n_components=n_compare,
        method='deflation',
        max_iter=50,
        rank=wdim,
    )
    idss.fit(X)
    src = idss.transform(X)
    
    for col in range(n_compare):
        axes[row, col].plot(t[:3000], src[col, :3000], linewidth=0.5)
        if col == 0:
            axes[row, col].set_ylabel(name, fontsize=9)
        if row == 0:
            axes[row, col].set_title(f'Comp {col+1}')

plt.suptitle('Comparison of Nonlinearities on Real MEG Data', fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/nonlinearity_comparison_meg.png", dpi=150)
print(f"Saved: {OUT_DIR}/nonlinearity_comparison_meg.png")

# =============================================================================
# 6. Spatial Patterns (Topographies)
# =============================================================================
print("\n--- 6. Spatial Patterns ---")

# Get patterns from the best model
patterns = idss.patterns_

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.flatten()

for i in range(min(10, patterns.shape[1])):
    pattern = patterns[:, i]
    # Simple bar plot of pattern (no head shape without MNE)
    axes[i].bar(np.arange(len(pattern)), pattern, width=1.0)
    axes[i].set_title(f'Pattern {i+1}')
    axes[i].set_xlabel('Channel')

plt.suptitle('Spatial Patterns (Mixing Column)')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/spatial_patterns.png", dpi=150)
print(f"Saved: {OUT_DIR}/spatial_patterns.png")

print(f"\n[OK] All results saved to {OUT_DIR}/")
