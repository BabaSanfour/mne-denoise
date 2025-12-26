"""
=====================================
Adaptive Masking (Wiener / Variance)
=====================================

This example demonstrates how to use the adaptive masking denoisers
(`WienerMaskDenoiser`, `VarianceMaskDenoiser`).

These denoisers are effective for **non-stationary** signals where the 
target activity appears in bursts or transients (e.g., sleep spindles, 
beta bursts, interictal spikes) amidst continuous background noise.

The algorithm estimates the local variance of the signal and suppresses
regions where the variance is low (assumed to be noise).
"""

import numpy as np
import matplotlib.pyplot as plt

from mne_denoise.dss.denoisers import WienerMaskDenoiser

###############################################################################
# Simulate Bursty Data
# --------------------
# We create a synthetic signal that contains three "bursts" of oscillation
# embedded in white noise.

sfreq = 200
duration = 10
times = np.arange(duration * sfreq) / sfreq
n_times = len(times)

# 1. Background noise
rng = np.random.default_rng(42)
noise = rng.normal(0, 0.5, n_times)

# 2. Bursts (simulated as modulated sine waves)
# Burst 1 at 2s, Burst 2 at 5s, Burst 3 at 8s
burst_centers = [2.0, 5.0, 8.0]
burst_signal = np.zeros(n_times)

for center in burst_centers:
    # Gaussian envelope
    env = np.exp(-0.5 * (times - center)**2 / 0.05)
    oscillation = np.sin(2 * np.pi * 15 * times) # 15 Hz spindle
    burst_signal += env * oscillation * 3.0

raw_signal = burst_signal + noise

###############################################################################
# Apply Wiener Mask Denoiser
# --------------------------
# We use `WienerMaskDenoiser` which estimates the local signal-to-noise ratio.
#
# Key parameters:
# - window_samples: Window length for variance calculation. Should cover a 
#   few cycles of the oscillation (e.g., 50 samples @ 200 Hz = 250ms).
# - noise_percentile: Percentile of variance to consider as "noise floor".
#   25th percentile is a conservative estimate (assuming >25% of data is just noise).

denoiser = WienerMaskDenoiser(window_samples=50, noise_percentile=25)
denoised_signal = denoiser.denoise(raw_signal)

###############################################################################
# Visualization
# -------------

plt.figure(figsize=(10, 8))

# Time series
plt.subplot(3, 1, 1)
plt.title("Original Signal (Bursts + Noise)")
plt.plot(times, raw_signal, color='steelblue', lw=1)
plt.xlim(0, 10)

plt.subplot(3, 1, 2)
plt.title("Denoised Signal (Wiener Mask)")
plt.plot(times, denoised_signal, color='darkorange', lw=1)
plt.xlim(0, 10)

# Overlay of envelopes to show suppression
plt.subplot(3, 1, 3)
plt.title("Envelope Comparison (Absolute Value)")
plt.plot(times, np.abs(raw_signal), color='steelblue', alpha=0.3, label='Original')
plt.plot(times, np.abs(denoised_signal), color='darkorange', label='Denoised')
plt.plot(times, np.abs(burst_signal), color='k', linestyle='--', lw=1, label='True Bursts')
plt.legend()
plt.xlim(0, 10)

plt.tight_layout()
plt.show()
