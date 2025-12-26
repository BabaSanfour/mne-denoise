"""
=====================================
Independent Component Analysis (ICA)
=====================================

This example demonstrates how to use the DSS engine to perform Independent
Component Analysis (ICA) by selecting specific nonlinearities.

Paper Reference:
Särelä & Valpola (2005) Sections 2.4, 2.5 and 3.3 shows that DSS becomes equivalent to
FastICA when using specific denoising functions like `tanh` or `pow3`.

In this demo, we separate mixed signals (Blind Source Separation) using
the `TanhMaskDenoiser` (RobustICA).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from mne_denoise.dss import IterativeDSS
# We import the "ICA" denoisers specifically
from mne_denoise.dss.denoisers import TanhMaskDenoiser, beta_tanh

###############################################################################
# Simulate Independent Sources
# ----------------------------
# We create 3 non-Gaussian sources:
# 1. Sine wave (sub-Gaussian, kurtosis < 0?? actually sine is sub-Gaussian)
# 2. Square wave (sub-Gaussian, kurtosis = -2)
# 3. Sawtooth wave (super-Gaussian? actually dependent on distribution)
# 
# FastICA works on differences in non-Gaussianity.

rng = np.random.default_rng(42)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * np.pi * 1.0 * time)  # Sine
s2 = np.sign(np.sin(2 * np.pi * 3.0 * time))  # Square
s3 = signal.sawtooth(2 * np.pi * 5.0 * time)  # Sawtooth

S = np.c_[s1, s2, s3].T
S += 0.2 * rng.normal(size=S.shape)  # Add mild noise

# Standardize
S /= S.std(axis=1)[:, np.newaxis]

# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = A @ S  # Observed signal

###############################################################################
# Run DSS as FastICA
# ------------------
# To mimic FastICA:
# 1. Use 'symmetric' mode (find all components at once).
# 2. Use 'tanh' nonlinearity (RobustICA).
# 3. Use 'beta' term for Newton convergence speed.

denoiser = TanhMaskDenoiser(alpha=1.0)

# Initialize DSS
dss = IterativeDSS(
    denoiser=denoiser,
    n_components=3,
    method='symmetric',  # FastICA typically uses symmetric orthogonalization
    beta=beta_tanh,      # Critical for FastICA-speed convergence
    max_iter=500,
    tol=1e-5
)

dss.fit(X)
S_estimated = dss.transform(X)

###############################################################################
# Visualization
# -------------

plt.figure(figsize=(10, 8))

models = [X, S, S_estimated]
names = ['Observations (Mixed)', 'True Sources', 'DSS-ICA Recovered']
colors = ['red', 'steelblue', 'orange']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(3, 1, ii)
    plt.title(name)
    for sig in model:
        plt.plot(sig, color=colors[ii-1], alpha=0.75)
    plt.xlim(0, 500)  # Zoom in

plt.tight_layout()
plt.show()

# Print convergence info
print("Convergence Info (Iterations):")
print(dss.convergence_info_)
