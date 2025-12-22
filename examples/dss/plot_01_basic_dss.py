"""
01: Artificial Data Demo
=======================
Replicates MATLAB demo_art.m - Basic DSS on artificial data.

This script demonstrates:
1. Loading artificial test data
2. Running DSS with default tanh nonlinearity
3. Comparing different nonlinearities
4. Effect of beta (Newton step) on convergence
5. Effect of gamma (adaptive learning rate)
"""

import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mne_denoise.dss import (
    IterativeDSS,
    TanhMaskDenoiser,
    RobustTanhDenoiser,
    GaussDenoiser,
    KurtosisDenoiser,
    beta_tanh,
    beta_pow3,
    Gamma179,
    GammaPredictive,
)

# Setup
plt.style.use('seaborn-v0_8-whitegrid')
OUT_DIR = "results_01_artificial"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# 1. Load Data
# =============================================================================
print("=" * 60)
print("01: Artificial Data Demo (demo_art.m)")
print("=" * 60)

data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'art_data.mat')
try:
    mat = scipy.io.loadmat(data_path)
    X = mat['X']
    print(f"Loaded art_data.mat: {X.shape}")
except FileNotFoundError:
    print("art_data.mat not found. Generating synthetic data...")
    np.random.seed(42)
    n_samples = 8192
    t = np.arange(n_samples) / 1000
    # Create 3 sources: sinusoid, square wave, noise
    s1 = np.sin(2 * np.pi * 5 * t)
    s2 = np.sign(np.sin(2 * np.pi * 3 * t))
    s3 = np.random.laplace(0, 1, n_samples)
    S = np.vstack([s1, s2, s3])
    A = np.random.randn(5, 3)
    X = A @ S + 0.1 * np.random.randn(5, n_samples)

n_channels, n_samples = X.shape
print(f"Data shape: {n_channels} channels × {n_samples} samples")

# =============================================================================
# 2. Basic DSS with Tanh (like demo_art.m default)
# =============================================================================
print("\n--- 2. Basic DSS with TanhMaskDenoiser ---")

denoiser = TanhMaskDenoiser(alpha=1.0)
idss = IterativeDSS(
    denoiser=denoiser.denoise,
    n_components=n_channels,
    method='deflation',
    max_iter=100,
    verbose=True,
)
idss.fit(X)
sources_tanh = idss.transform(X)

print(f"Extracted {sources_tanh.shape[0]} components")

# Plot results
fig, axes = plt.subplots(n_channels, 1, figsize=(12, 8), sharex=True)
for i in range(n_channels):
    axes[i].plot(sources_tanh[i, :2000])
    axes[i].set_ylabel(f'S{i+1}')
axes[-1].set_xlabel('Samples')
plt.suptitle('Basic DSS with TanhMaskDenoiser (demo_art.m equivalent)')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/basic_tanh_dss.png", dpi=150)
print(f"Saved: {OUT_DIR}/basic_tanh_dss.png")

# =============================================================================
# 3. Compare Different Nonlinearities
# =============================================================================
print("\n--- 3. Comparing Nonlinearities ---")

denoisers = {
    'TanhMask (tanh(s))': TanhMaskDenoiser(alpha=1.0),
    'RobustTanh (s-tanh(s))': RobustTanhDenoiser(alpha=1.0),
    'Gauss (s*exp(-s²/2))': GaussDenoiser(a=1.0),
    'Kurtosis (s³)': KurtosisDenoiser(),
}

fig, axes = plt.subplots(len(denoisers), n_channels, figsize=(14, 10))

for row, (name, denoiser) in enumerate(denoisers.items()):
    idss = IterativeDSS(
        denoiser=denoiser.denoise,
        n_components=n_channels,
        method='deflation',
        max_iter=50,
    )
    idss.fit(X)
    sources = idss.transform(X)
    
    for col in range(n_channels):
        axes[row, col].plot(sources[col, :1000], linewidth=0.8)
        if col == 0:
            axes[row, col].set_ylabel(name, fontsize=9)
        if row == 0:
            axes[row, col].set_title(f'Comp {col+1}')

plt.suptitle('Comparison of Different Nonlinearities', fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/nonlinearity_comparison.png", dpi=150)
print(f"Saved: {OUT_DIR}/nonlinearity_comparison.png")

# =============================================================================
# 4. Effect of Beta (Newton Step)
# =============================================================================
print("\n--- 4. Effect of Beta (Newton Step) ---")

# Without beta
idss_no_beta = IterativeDSS(
    denoiser=TanhMaskDenoiser().denoise,
    n_components=3,
    method='deflation',
    max_iter=50,
    verbose=True,
    beta=None,  # No Newton step
)
idss_no_beta.fit(X)
conv_no_beta = idss_no_beta.convergence_info_

# With beta
idss_with_beta = IterativeDSS(
    denoiser=TanhMaskDenoiser().denoise,
    n_components=3,
    method='deflation',
    max_iter=50,
    verbose=True,
    beta=beta_tanh,  # FastICA Newton step
)
idss_with_beta.fit(X)
conv_with_beta = idss_with_beta.convergence_info_

print(f"\nWithout beta: iterations = {conv_no_beta[:, 0]}")
print(f"With beta:    iterations = {conv_with_beta[:, 0]}")

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(1, 4)
width = 0.35
ax.bar(x - width/2, conv_no_beta[:, 0], width, label='Without beta (gradient)')
ax.bar(x + width/2, conv_with_beta[:, 0], width, label='With beta (Newton)')
ax.set_xlabel('Component')
ax.set_ylabel('Iterations to converge')
ax.set_title('Effect of Beta (Newton Step) on Convergence')
ax.set_xticks(x)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/beta_effect.png", dpi=150)
print(f"Saved: {OUT_DIR}/beta_effect.png")

# =============================================================================
# 5. Effect of Gamma (Adaptive Learning Rate)
# =============================================================================
print("\n--- 5. Effect of Gamma (Adaptive Learning Rate) ---")

# Without gamma
idss_no_gamma = IterativeDSS(
    denoiser=TanhMaskDenoiser().denoise,
    n_components=3,
    max_iter=50,
    gamma=None,
)
idss_no_gamma.fit(X)
conv_no_gamma = idss_no_gamma.convergence_info_

# With Gamma179
gamma_179 = Gamma179()
idss_gamma179 = IterativeDSS(
    denoiser=TanhMaskDenoiser().denoise,
    n_components=3,
    max_iter=50,
    gamma=gamma_179,
)
idss_gamma179.fit(X)
conv_gamma179 = idss_gamma179.convergence_info_

# With GammaPredictive
gamma_pred = GammaPredictive()
idss_gamma_pred = IterativeDSS(
    denoiser=TanhMaskDenoiser().denoise,
    n_components=3,
    max_iter=50,
    gamma=gamma_pred,
)
idss_gamma_pred.fit(X)
conv_gamma_pred = idss_gamma_pred.convergence_info_

print(f"No gamma:         {conv_no_gamma[:, 0]}")
print(f"Gamma179:         {conv_gamma179[:, 0]}")
print(f"GammaPredictive:  {conv_gamma_pred[:, 0]}")

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(1, 4)
width = 0.25
ax.bar(x - width, conv_no_gamma[:, 0], width, label='No gamma (γ=1)')
ax.bar(x, conv_gamma179[:, 0], width, label='Gamma179 (oscillation damping)')
ax.bar(x + width, conv_gamma_pred[:, 0], width, label='GammaPredictive (adaptive)')
ax.set_xlabel('Component')
ax.set_ylabel('Iterations')
ax.set_title('Effect of Gamma (Adaptive Learning Rate)')
ax.set_xticks(x)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/gamma_effect.png", dpi=150)
print(f"Saved: {OUT_DIR}/gamma_effect.png")

print(f"\n[OK] All results saved to {OUT_DIR}/")
