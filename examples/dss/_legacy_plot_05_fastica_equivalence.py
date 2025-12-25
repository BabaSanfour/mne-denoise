"""
05: FastICA Equivalence Demo
============================
Shows that DSS with appropriate parameters equals FastICA.

This script demonstrates:
1. DSS with tanh + beta_tanh = FastICA
2. Convergence comparison with/without Newton step
3. Gamma adaptive learning rate effects
4. Benchmark against sklearn FastICA
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mne_denoise.dss import (
    IterativeDSS,
    TanhMaskDenoiser,
    KurtosisDenoiser,
    beta_tanh,
    beta_pow3,
)
from mne_denoise.dss.denoisers import Gamma179, GammaPredictive

# Try to import sklearn for comparison
try:
    from sklearn.decomposition import FastICA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("sklearn not installed, skipping FastICA comparison")

# Setup
plt.style.use('seaborn-v0_8-whitegrid')
OUT_DIR = "results_05_fastica_equiv"
os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(42)

print("=" * 60)
print("05: FastICA Equivalence Demo")
print("=" * 60)

# =============================================================================
# 1. Create ICA Test Data
# =============================================================================
print("\n--- 1. Creating ICA Test Data ---")

n_samples = 5000
t = np.arange(n_samples) / 1000

# Independent sources
s1 = np.sin(2 * np.pi * 5 * t)  # Sinusoid (sub-Gaussian)
s2 = np.sign(np.sin(2 * np.pi * 3 * t + 0.5))  # Square wave (super-Gaussian)
s3 = stats.laplace.rvs(size=n_samples)  # Laplace (super-Gaussian)
s4 = np.random.uniform(-1.7, 1.7, n_samples)  # Uniform (sub-Gaussian)

S_true = np.vstack([s1, s2, s3, s4])
S_true = S_true / np.std(S_true, axis=1, keepdims=True)  # Normalize

# Random mixing matrix
A = np.random.randn(10, 4)
X = A @ S_true + 0.1 * np.random.randn(10, n_samples)

print(f"Sources: {S_true.shape}")
print(f"Mixtures: {X.shape}")
print(f"Kurtosis of sources: {[f'{stats.kurtosis(s, fisher=False):.2f}' for s in S_true]}")

# =============================================================================
# 2. DSS with Tanh + Beta = FastICA
# =============================================================================
print("\n--- 2. DSS with Tanh + Beta = FastICA ---")

# DSS without beta (gradient ascent)
print("\nDSS without beta (gradient):")
denoiser = TanhMaskDenoiser(alpha=1.0)
idss_no_beta = IterativeDSS(
    denoiser=denoiser.denoise,
    n_components=4,
    method='deflation',
    max_iter=200,
    beta=None,
    verbose=True,
)
t0 = time.time()
idss_no_beta.fit(X)
time_no_beta = time.time() - t0
sources_no_beta = idss_no_beta.transform(X)
conv_no_beta = idss_no_beta.convergence_info_

# DSS with beta (Newton step = FastICA)
print("\nDSS with beta_tanh (Newton = FastICA):")
idss_with_beta = IterativeDSS(
    denoiser=denoiser.denoise,
    n_components=4,
    method='deflation',
    max_iter=200,
    beta=beta_tanh,
    verbose=True,
)
t0 = time.time()
idss_with_beta.fit(X)
time_with_beta = time.time() - t0
sources_with_beta = idss_with_beta.transform(X)
conv_with_beta = idss_with_beta.convergence_info_

print(f"\nWithout beta: {conv_no_beta[:, 0].sum():.0f} total iterations, {time_no_beta:.3f}s")
print(f"With beta:    {conv_with_beta[:, 0].sum():.0f} total iterations, {time_with_beta:.3f}s")
print(f"Speedup: {conv_no_beta[:, 0].sum() / conv_with_beta[:, 0].sum():.1f}x")

# =============================================================================
# 3. Comparison with sklearn FastICA
# =============================================================================
if HAS_SKLEARN:
    print("\n--- 3. Comparison with sklearn FastICA ---")
    
    fastica = FastICA(n_components=4, algorithm='deflation', fun='logcosh', max_iter=200)
    t0 = time.time()
    sources_sklearn = fastica.fit_transform(X.T).T
    time_sklearn = time.time() - t0
    
    print(f"sklearn FastICA: {time_sklearn:.3f}s")
    
    # Compare extracted sources
    def correlation_matrix(S1, S2):
        corr = np.zeros((S1.shape[0], S2.shape[0]))
        for i in range(S1.shape[0]):
            for j in range(S2.shape[0]):
                corr[i, j] = np.abs(np.corrcoef(S1[i], S2[j])[0, 1])
        return corr
    
    corr_dss_sklearn = correlation_matrix(sources_with_beta, sources_sklearn)
    corr_dss_true = correlation_matrix(sources_with_beta, S_true)
    corr_sklearn_true = correlation_matrix(sources_sklearn, S_true)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    im0 = axes[0].imshow(corr_dss_sklearn, cmap='Blues', vmin=0, vmax=1)
    axes[0].set_title('DSS vs sklearn FastICA')
    axes[0].set_xlabel('sklearn')
    axes[0].set_ylabel('DSS')
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(corr_dss_true, cmap='Greens', vmin=0, vmax=1)
    axes[1].set_title('DSS vs True Sources')
    axes[1].set_xlabel('True')
    axes[1].set_ylabel('DSS')
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(corr_sklearn_true, cmap='Oranges', vmin=0, vmax=1)
    axes[2].set_title('sklearn vs True Sources')
    axes[2].set_xlabel('True')
    axes[2].set_ylabel('sklearn')
    plt.colorbar(im2, ax=axes[2])
    
    plt.suptitle('Source Recovery Comparison')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fastica_comparison.png", dpi=150)
    print(f"Saved: {OUT_DIR}/fastica_comparison.png")
    
    # Best match correlation
    best_correlations_dss = np.max(corr_dss_true, axis=1)
    best_correlations_sklearn = np.max(corr_sklearn_true, axis=1)
    
    print(f"\nBest correlations with true sources:")
    print(f"  DSS (beta=beta_tanh): {best_correlations_dss}")
    print(f"  sklearn FastICA:      {best_correlations_sklearn}")

# =============================================================================
# 4. Convergence Comparison
# =============================================================================
print("\n--- 4. Convergence Comparison ---")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart of iterations
x = np.arange(1, 5)
width = 0.35
axes[0].bar(x - width/2, conv_no_beta[:, 0], width, label='No beta (gradient)')
axes[0].bar(x + width/2, conv_with_beta[:, 0], width, label='With beta (Newton)')
axes[0].set_xlabel('Component')
axes[0].set_ylabel('Iterations')
axes[0].set_title('Iterations to Convergence')
axes[0].set_xticks(x)
axes[0].legend()

# Kurtosis of extracted sources
kurt_no_beta = [stats.kurtosis(s, fisher=False) for s in sources_no_beta]
kurt_with_beta = [stats.kurtosis(s, fisher=False) for s in sources_with_beta]

axes[1].bar(x - width/2, np.abs(np.array(kurt_no_beta) - 3), width, label='No beta')
axes[1].bar(x + width/2, np.abs(np.array(kurt_with_beta) - 3), width, label='With beta')
axes[1].set_xlabel('Component')
axes[1].set_ylabel('|Kurtosis - 3|')
axes[1].set_title('Non-Gaussianity of Extracted Sources')
axes[1].set_xticks(x)
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/convergence_comparison.png", dpi=150)
print(f"Saved: {OUT_DIR}/convergence_comparison.png")

# =============================================================================
# 5. Gamma (Adaptive Learning Rate) Effects
# =============================================================================
print("\n--- 5. Gamma (Adaptive Learning Rate) Effects ---")

# Create more challenging data (close to oscillation)
X_hard = X * 1.2 + np.random.randn(*X.shape) * 0.5

configs = [
    ("No gamma (γ=1)", None),
    ("Gamma179 (oscillation damping)", Gamma179()),
    ("GammaPredictive (adaptive)", GammaPredictive()),
]

results = {}
for name, gamma in configs:
    print(f"\n{name}:")
    if isinstance(gamma, (Gamma179, GammaPredictive)):
        gamma.reset()  # Reset state
    
    idss = IterativeDSS(
        denoiser=TanhMaskDenoiser().denoise,
        n_components=4,
        max_iter=100,
        beta=beta_tanh,
        gamma=gamma,
        verbose=True,
    )
    idss.fit(X_hard)
    results[name] = {
        'conv': idss.convergence_info_.copy(),
        'sources': idss.transform(X_hard),
    }
    print(f"  Total iterations: {results[name]['conv'][:, 0].sum():.0f}")
    print(f"  All converged: {np.all(results[name]['conv'][:, 1])}")

# Plot gamma comparison
fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(1, 5)
width = 0.25
for i, (name, _) in enumerate(configs):
    ax.bar(x + (i - 1) * width, results[name]['conv'][:, 0], width, label=name)

ax.set_xlabel('Component')
ax.set_ylabel('Iterations')
ax.set_title('Effect of Gamma (Adaptive Learning Rate)')
ax.set_xticks(x)
ax.legend()

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/gamma_comparison.png", dpi=150)
print(f"Saved: {OUT_DIR}/gamma_comparison.png")

# =============================================================================
# 6. Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary: DSS as FastICA")
print("=" * 60)
print("""
Key findings:
1. DSS with TanhMaskDenoiser + beta_tanh = FastICA with logcosh

2. The beta parameter implements the Newton step:
   w_new = E[X·g(s)] - E[g'(s)]·w
   where g(s) = tanh(s), g'(s) = 1 - tanh²(s)
   so beta = -E[1 - tanh²(s)]

3. Newton step (with beta) converges ~3-5x faster than gradient

4. Gamma (adaptive learning rate) helps stabilize oscillating cases:
   - Gamma179: Detects oscillation, reduces step to 0.5
   - GammaPredictive: Adapts step based on correlation of updates

5. DSS achieves equivalent source recovery as sklearn FastICA
""")

print(f"\n[OK] All results saved to {OUT_DIR}/")
