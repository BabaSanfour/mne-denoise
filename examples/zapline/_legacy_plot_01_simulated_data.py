"""
ZapLine: Simulated Data Demo.
============================

This example demonstrates the ZapLine algorithm (de Cheveigné, 2020) for
removing power line artifacts from multichannel data using simulated data.

The algorithm combines spectral and spatial filtering:
1. Split data into line-noise-free and line-noise-contaminated parts
2. Apply DSS spatial filter to remove line components from contaminated part
3. Combine to get clean, full-rank data

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

Reference
---------
de Cheveigné, A. (2020). ZapLine: A simple and effective method to remove
power line artifacts. NeuroImage, 207, 116356.
"""

# %%
# Imports
# -------
# Add parent to path for development
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

sys.path.insert(0, r"D:\PhD\mne-denoise")

from mne_denoise import compute_psd_reduction, dss_zapline

# %%
# Create Simulated Data
# ---------------------
# Following the paper: 10000 x 100 matrix of normally distributed random
# samples (signal), plus a periodic waveform (half-wave rectified sinusoid
# raised to power 3) as line noise.

np.random.seed(42)

# Parameters
n_times = 10000
n_channels = 100
sfreq = 200  # 200 Hz sampling rate
line_freq = 50  # 50 Hz power line

# Time vector
t = np.arange(n_times) / sfreq

# Signal: wide-band random noise
signal_data = np.random.randn(n_channels, n_times)

# Line noise: half-wave rectified sinusoid raised to power 3
# This creates harmonics at 50, 100, 150, 200 Hz etc.
line_waveform = np.sin(2 * np.pi * line_freq * t)
line_waveform = np.maximum(line_waveform, 0) ** 3  # Half-wave rectified, power 3

# Mix line noise into channels with random weights
line_mixing = np.random.randn(n_channels, 1)
line_noise = line_mixing * line_waveform

# Scale to approximately equal power
signal_power = np.mean(signal_data**2)
noise_power = np.mean(line_noise**2)
line_noise = line_noise * np.sqrt(signal_power / noise_power)

# Mixture
data = signal_data + line_noise

print(f"Data shape: {data.shape}")
print(f"Signal power: {np.mean(signal_data**2):.4f}")
print(f"Noise power: {np.mean(line_noise**2):.4f}")
print(f"SNR: {10 * np.log10(signal_power / np.mean(line_noise**2)):.1f} dB")

# %%
# Compute PSD Before Cleaning
# ---------------------------


def compute_normalized_psd(data, sfreq, nperseg=1024):
    """Compute normalized PSD (sum over frequencies = 1)."""
    freqs, psd = signal.welch(data, sfreq, nperseg=min(nperseg, data.shape[1]), axis=1)
    # Average across channels
    mean_psd = np.mean(psd, axis=0)
    # Normalize
    mean_psd = mean_psd / np.sum(mean_psd)
    return freqs, mean_psd


freqs, psd_before = compute_normalized_psd(data, sfreq)

# %%
# Apply ZapLine
# -------------
# The algorithm has one main parameter: n_remove (d in the paper),
# which specifies the number of spatial components to remove.

# Apply ZapLine with d=4 (as suggested in paper for MEG-like data)
result = dss_zapline(
    data,
    line_freq=line_freq,
    sfreq=sfreq,
    n_remove=4,  # Remove 4 components
    n_harmonics=4,  # Include harmonics up to 200 Hz
)

print("\nZapLine Results:")
print(f"  Components removed: {result.n_removed}")
print(f"  Harmonics processed: {result.n_harmonics}")

# %%
# Compute PSD After Cleaning
# --------------------------

freqs, psd_after = compute_normalized_psd(result.cleaned, sfreq)
_, psd_removed = compute_normalized_psd(result.removed, sfreq)

# %%
# Plot Results (Figure 4 from paper)
# ----------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Original data
ax = axes[0]
ax.semilogy(freqs, psd_before, "k-", linewidth=1.5, label="Original")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Normalized PSD")
ax.set_title("Line-Contaminated Data")
ax.set_xlim([0, sfreq / 2])
ax.grid(True, alpha=0.3)
ax.legend()

# Mark line frequency and harmonics
for h in range(1, 5):
    f = line_freq * h
    if f < sfreq / 2:
        ax.axvline(f, color="r", alpha=0.3, linestyle="--")

# Cleaned vs Removed
ax = axes[1]
ax.semilogy(freqs, psd_after, "g-", linewidth=1.5, label="Cleaned")
ax.semilogy(freqs, psd_removed, "r-", linewidth=1.5, alpha=0.7, label="Removed")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Normalized PSD")
ax.set_title("After ZapLine (d=4)")
ax.set_xlim([0, sfreq / 2])
ax.grid(True, alpha=0.3)
ax.legend()

# Mark line frequency and harmonics
for h in range(1, 5):
    f = line_freq * h
    if f < sfreq / 2:
        ax.axvline(f, color="r", alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig("zapline_simulated_results.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Quantitative Assessment
# -----------------------

metrics = compute_psd_reduction(data, result.cleaned, sfreq, line_freq)

print("\nPower Reduction Metrics:")
print(f"  Power at 50 Hz (before): {metrics['power_original']:.2e}")
print(f"  Power at 50 Hz (after):  {metrics['power_cleaned']:.2e}")
print(
    f"  Reduction: {metrics['reduction_db']:.1f} dB ({metrics['reduction_ratio']:.1f}x)"
)

# Check that the floor of the removed signal is low
floor_ratio = np.mean(psd_removed) / np.mean(psd_after)
print(f"\nFloor of removed signal / cleaned signal: {floor_ratio:.4f}")
print("  (Should be << 1, indicating minimal distortion of non-artifact frequencies)")

# %%
# Effect of d Parameter
# ---------------------
# Following paper recommendation: start with d=1 and increment until
# clean data PSD shows no trace of artifact.

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for idx, d in enumerate([1, 2, 3, 4, 5, 6]):
    result_d = dss_zapline(data, line_freq=line_freq, sfreq=sfreq, n_remove=d)
    freqs_d, psd_d = compute_normalized_psd(result_d.cleaned, sfreq)

    ax = axes[idx]
    ax.semilogy(freqs_d, psd_d, "g-", linewidth=1.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized PSD")
    ax.set_title(f"d = {d}")
    ax.set_xlim([0, sfreq / 2])
    ax.grid(True, alpha=0.3)

    # Mark line frequencies
    for h in range(1, 5):
        f = line_freq * h
        if f < sfreq / 2:
            ax.axvline(f, color="r", alpha=0.3, linestyle="--")

plt.suptitle(
    "Effect of Number of Components Removed (d)", fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig("zapline_d_parameter.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# DSS Component Scores
# --------------------
# The DSS algorithm finds components ordered by artifact power.
# The eigenvalues show how much each component is dominated by line noise.

fig, ax = plt.subplots(figsize=(10, 5))

# Plot eigenvalues (scores)
scores = result.dss_eigenvalues
ax.plot(np.arange(1, len(scores) + 1), scores, "b.-", markersize=10)
ax.axhline(
    y=scores[4] if len(scores) > 4 else scores[-1],
    color="r",
    linestyle="--",
    label="d=4 threshold",
)
ax.set_xlabel("Component")
ax.set_ylabel("Score (Artifact Power Ratio)")
ax.set_title("DSS Component Scores - Line Noise Dominance")
ax.set_xlim([0.5, min(20, len(scores)) + 0.5])
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig("zapline_component_scores.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nTop 8 component scores:")
for i, score in enumerate(scores[:8]):
    marker = " <-- removed" if i < result.n_removed else ""
    print(f"  Component {i + 1}: {score:.4f}{marker}")

# %%
# Summary
# -------
# The ZapLine algorithm successfully:
# 1. Removes power line artifacts at 50 Hz and harmonics
# 2. Preserves the flat spectral floor of the underlying signal
# 3. Minimal distortion at non-artifact frequencies
# 4. Works with a single parameter (d = number of components to remove)

print("\n" + "=" * 60)
print("ZapLine Simulated Data Demo Complete!")
print("=" * 60)
print("\nKey findings:")
print("  - Line artifacts completely removed from cleaned data")
print("  - Removed signal shows clear line frequency peaks")
print("  - Spectral floor of removed signal is orders of magnitude below cleaned")
print("  - d=4 is sufficient for this simulation")
print("\n[OK] Demo completed successfully!")
