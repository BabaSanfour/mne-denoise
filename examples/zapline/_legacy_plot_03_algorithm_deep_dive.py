"""
ZapLine: Algorithm Deep Dive.
=============================

This example provides a detailed walkthrough of the ZapLine algorithm,
explaining each step as described in de Cheveigné (2020).

Algorithm Steps:
1. Perfect reconstruction filterbank splits data into two paths
2. Spectral filter removes line artifacts from one path
3. DSS spatial filter removes line artifacts from other path
4. Paths are combined for clean, full-rank output

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
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fft import fft, ifft

sys.path.insert(0, r"D:\PhD\mne-denoise")

from mne_denoise import dss_zapline

# %%
# Create Test Data
# ----------------

np.random.seed(42)

# Parameters
n_times = 5000
n_channels = 32
sfreq = 250
line_freq = 50

t = np.arange(n_times) / sfreq


# Brain signal (pink noise - more realistic than white noise)
def generate_pink_noise(n_channels, n_times):
    """Generate 1/f pink noise."""
    white = np.random.randn(n_channels, n_times)
    # Filter to 1/f
    freqs = np.fft.fftfreq(n_times)
    freqs[0] = 1  # Avoid division by zero
    pink_filter = 1 / np.sqrt(np.abs(freqs))
    pink_filter[0] = 0
    white_fft = fft(white, axis=1)
    pink = np.real(ifft(white_fft * pink_filter, axis=1))
    return pink


brain_signal = generate_pink_noise(n_channels, n_times)

# Line noise with harmonics
line_noise = np.zeros((n_channels, n_times))
for harmonic in [1, 2, 3, 4]:  # 50, 100, 150, 200 Hz
    freq = line_freq * harmonic
    if freq < sfreq / 2:
        amplitude = 1.0 / harmonic  # Decreasing amplitude
        phase = np.random.rand(n_channels, 1) * 2 * np.pi
        mixing = np.random.randn(n_channels, 1) * amplitude
        line_noise += mixing * np.sin(2 * np.pi * freq * t + phase)

# Scale and combine
brain_signal = brain_signal / np.std(brain_signal) * 10e-6  # 10 µV scale
line_noise = line_noise / np.std(line_noise) * 50e-6  # 50 µV scale (strong artifact)

data = brain_signal + line_noise

print(f"Data: {n_channels} channels x {n_times} samples")
print(f"Brain signal RMS: {np.sqrt(np.mean(brain_signal**2)) * 1e6:.1f} µV")
print(f"Line noise RMS: {np.sqrt(np.mean(line_noise**2)) * 1e6:.1f} µV")
print(f"SNR: {10 * np.log10(np.var(brain_signal) / np.var(line_noise)):.1f} dB")

# %%
# Step 1: Perfect Reconstruction Filterbank
# ------------------------------------------
# Split data into two paths using complementary filters:
# - Path 1: Smoothing filter (removes line frequency)
# - Path 2: Residual (contains line noise)

period = int(round(sfreq / line_freq))  # Samples per line cycle

# Smoothing filter: moving average over one period
kernel = np.ones(period) / period

# Apply smoothing (line-free path)
data_smooth = np.apply_along_axis(
    lambda x: np.convolve(x, kernel, mode="same"), axis=1, arr=data
)

# Residual (line-contaminated path)
data_residual = data - data_smooth

print("\nStep 1: Filterbank")
print(f"  Period: {period} samples ({1000 / line_freq:.1f} ms)")
print(f"  Smooth path variance: {np.var(data_smooth):.2e}")
print(f"  Residual path variance: {np.var(data_residual):.2e}")

# %%
# Visualize Filterbank Decomposition
# -----------------------------------

fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

# Show one channel
ch = 0
time_slice = slice(1000, 1500)
t_plot = t[time_slice]

axes[0].plot(t_plot, data[ch, time_slice] * 1e6, "k-", linewidth=1)
axes[0].set_ylabel("Original (µV)")
axes[0].set_title("Step 1: Perfect Reconstruction Filterbank")
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_plot, data_smooth[ch, time_slice] * 1e6, "b-", linewidth=1)
axes[1].set_ylabel("Smooth Path (µV)")
axes[1].set_title("After smoothing filter (line-free)")
axes[1].grid(True, alpha=0.3)

axes[2].plot(t_plot, data_residual[ch, time_slice] * 1e6, "r-", linewidth=1)
axes[2].set_ylabel("Residual Path (µV)")
axes[2].set_xlabel("Time (s)")
axes[2].set_title("Residual (contains line noise)")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("zapline_step1_filterbank.png", dpi=150, bbox_inches="tight")
plt.show()

# Verify perfect reconstruction
reconstruction_error = np.max(np.abs(data - (data_smooth + data_residual)))
print(f"Perfect reconstruction error: {reconstruction_error:.2e}")

# %%
# Step 2: DSS Bias Filter
# -----------------------
# Create a bias filter that emphasizes line frequencies.
# This is used to find spatial components dominated by line noise.

nfft = 1024


def compute_bias_covariance(data_2d, freqs_hz, sfreq, nfft=1024):
    """Compute covariance matrices with bias at specific frequencies."""
    n_ch, n_t = data_2d.shape
    nfft = min(nfft, n_t)
    n_seg = n_t // nfft

    freq_bins = np.fft.fftfreq(nfft, 1 / sfreq)

    # Find frequency bins for line and harmonics
    target_bins = []
    for f in freqs_hz:
        idx = np.argmin(np.abs(freq_bins - f))
        target_bins.append(idx)
        idx_neg = np.argmin(np.abs(freq_bins + f))
        target_bins.append(idx_neg)

    c0 = np.zeros((n_ch, n_ch))  # Baseline
    c1 = np.zeros((n_ch, n_ch))  # Biased

    for seg in range(n_seg):
        segment = data_2d[:, seg * nfft : (seg + 1) * nfft]
        X = fft(segment, axis=1)

        # Baseline: all frequencies
        c0 += np.real(X @ X.conj().T) / nfft

        # Biased: only target frequencies
        X_bias = np.zeros_like(X)
        for idx in target_bins:
            X_bias[:, idx] = X[:, idx]
        c1 += np.real(X_bias @ X_bias.conj().T) / nfft

    c0 /= n_seg
    c1 /= n_seg

    return c0, c1


# Compute bias covariances
harmonic_freqs = [line_freq * h for h in range(1, 5) if line_freq * h < sfreq / 2]
c0, c1 = compute_bias_covariance(data_residual, harmonic_freqs, sfreq, nfft)

print("\nStep 2: Bias Covariance")
print(f"  Target frequencies: {harmonic_freqs} Hz")
print(f"  C0 (baseline) shape: {c0.shape}")
print(f"  C1 (biased) shape: {c1.shape}")
print(f"  C0 trace: {np.trace(c0):.2e}")
print(f"  C1 trace: {np.trace(c1):.2e}")

# %%
# Visualize Covariance Matrices
# -----------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im0 = axes[0].imshow(c0, aspect="auto", cmap="RdBu_r")
axes[0].set_title("C0: Baseline Covariance")
axes[0].set_xlabel("Channel")
axes[0].set_ylabel("Channel")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(c1, aspect="auto", cmap="RdBu_r")
axes[1].set_title("C1: Biased Covariance (line frequencies)")
axes[1].set_xlabel("Channel")
axes[1].set_ylabel("Channel")
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.savefig("zapline_step2_covariance.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Step 3: DSS / Joint Diagonalization
# -----------------------------------
# Solve generalized eigenvalue problem to find components
# that maximize line noise power.

from scipy.linalg import eigh

# Regularize C0
reg = 1e-9
c0_reg = c0 + reg * np.trace(c0) / c0.shape[0] * np.eye(c0.shape[0])

# Solve: C1 * v = lambda * C0 * v
eigvals, eigvecs = eigh(c1, c0_reg)

# Sort by eigenvalue (descending)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# Eigenvalues = scores (ratio of biased to baseline variance)
scores = np.abs(eigvals)

print("\nStep 3: DSS Decomposition")
print(f"  Eigenvalues (top 8): {scores[:8]}")

# %%
# Visualize Component Scores
# --------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scores
ax = axes[0]
ax.bar(np.arange(1, len(scores) + 1), scores)
ax.axvline(4.5, color="r", linestyle="--", label="d=4 cutoff")
ax.set_xlabel("Component")
ax.set_ylabel("Score (Line Noise Power Ratio)")
ax.set_title("DSS Component Scores")
ax.set_xlim([0, n_channels + 1])
ax.legend()
ax.grid(True, alpha=0.3)

# Log scale for better visualization
ax = axes[1]
ax.semilogy(np.arange(1, len(scores) + 1), scores, "b.-", markersize=10)
ax.axvline(4.5, color="r", linestyle="--", label="d=4 cutoff")
ax.set_xlabel("Component")
ax.set_ylabel("Score (log scale)")
ax.set_title("DSS Component Scores (Log Scale)")
ax.set_xlim([0, n_channels + 1])
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("zapline_step3_scores.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Step 4: Project Out Line Components
# -----------------------------------
# Remove top d components from the residual path.

d = 4  # Number of components to remove

# DSS filters (rows = components)
dss_filters = eigvecs.T  # (n_components, n_channels)

# Extract line components
filters_noise = dss_filters[:d]  # Top d filters
sources_noise = filters_noise @ data_residual  # Project onto noise space

# Reconstruct noise estimate
# Using pseudo-inverse for patterns
patterns_noise = np.linalg.pinv(filters_noise).T  # (n_channels, d)
noise_estimate = patterns_noise @ sources_noise

# Clean residual
residual_clean = data_residual - noise_estimate

print("\nStep 4: Projection")
print(f"  Components removed: {d}")
print(f"  Noise estimate variance: {np.var(noise_estimate):.2e}")
print(f"  Clean residual variance: {np.var(residual_clean):.2e}")

# %%
# Visualize Noise Components
# --------------------------

fig, axes = plt.subplots(d, 1, figsize=(14, 2 * d), sharex=True)

for i in range(d):
    axes[i].plot(t[:1000], sources_noise[i, :1000] * 1e6, "r-", linewidth=0.5)
    axes[i].set_ylabel(f"Comp {i + 1}")
    axes[i].grid(True, alpha=0.3)

    # Compute and display dominant frequency
    freqs_fft, psd = signal.welch(sources_noise[i], sfreq, nperseg=512)
    peak_freq = freqs_fft[np.argmax(psd)]
    axes[i].text(
        0.98,
        0.95,
        f"Peak: {peak_freq:.0f} Hz",
        transform=axes[i].transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

axes[-1].set_xlabel("Time (s)")
axes[0].set_title(f"Top {d} Line Noise Components")
plt.tight_layout()
plt.savefig("zapline_step4_components.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Step 5: Combine Paths
# ---------------------
# Clean data = smooth path + cleaned residual

data_clean = data_smooth + residual_clean
data_removed = data - data_clean

print("\nStep 5: Final Output")
print(f"  Clean data variance: {np.var(data_clean):.2e}")
print(f"  Removed variance: {np.var(data_removed):.2e}")

# %%
# Final Comparison
# ----------------

# Compute PSDs
freqs, psd_orig = signal.welch(data, sfreq, nperseg=512, axis=1)
_, psd_clean = signal.welch(data_clean, sfreq, nperseg=512, axis=1)
_, psd_removed = signal.welch(data_removed, sfreq, nperseg=512, axis=1)

psd_orig_mean = np.mean(psd_orig, axis=0)
psd_clean_mean = np.mean(psd_clean, axis=0)
psd_removed_mean = np.mean(psd_removed, axis=0)

# Normalize
psd_orig_norm = psd_orig_mean / np.sum(psd_orig_mean)
psd_clean_norm = psd_clean_mean / np.sum(psd_clean_mean)
psd_removed_norm = psd_removed_mean / np.sum(psd_removed_mean)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.semilogy(freqs, psd_orig_norm, "k-", linewidth=1.5, label="Original")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Normalized PSD")
ax.set_title("Original Data")
ax.grid(True, alpha=0.3)
ax.legend()
for f in harmonic_freqs:
    ax.axvline(f, color="r", alpha=0.3, linestyle="--")

ax = axes[1]
ax.semilogy(freqs, psd_clean_norm, "g-", linewidth=1.5, label="Cleaned")
ax.semilogy(freqs, psd_removed_norm, "r-", linewidth=1.5, alpha=0.7, label="Removed")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Normalized PSD")
ax.set_title(f"After ZapLine (d={d})")
ax.grid(True, alpha=0.3)
ax.legend()
for f in harmonic_freqs:
    ax.axvline(f, color="r", alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig("zapline_step5_final.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Verify with Built-in Function
# -----------------------------

# Now let's verify our manual implementation matches the built-in
result = dss_zapline(data, line_freq=line_freq, sfreq=sfreq, n_remove=d)

print("\nVerification with dss_zapline():")
print(f"  Components removed: {result.n_removed}")

# Compare
correlation = np.corrcoef(data_clean.flatten(), result.cleaned.flatten())[0, 1]
print(f"  Correlation with manual implementation: {correlation:.6f}")

# %%
# Summary
# -------

print("\n" + "=" * 60)
print("ZapLine Algorithm Deep Dive Complete!")
print("=" * 60)
print(
    """
Algorithm Steps:
  1. Filterbank: Split into smooth (line-free) and residual (line-noise) paths
  2. Bias Filter: Compute covariance at line frequencies
  3. DSS: Find spatial components maximizing line noise
  4. Projection: Remove top d components from residual path
  5. Combine: Add cleaned residual to smooth path

Key Properties:
  - Full-rank output (unlike ICA)
  - Full-bandwidth (unlike lowpass filtering)
  - Minimal distortion at non-artifact frequencies
  - Only d dimensions are spectrally filtered
  - Only line-dominated part is spatially filtered

Parameter d:
  - d=1-4 typically sufficient for EEG
  - d=4-8 may be needed for MEG
  - Too small → residual artifacts
  - Too large → more dimensions affected (usually harmless)
"""
)
print("\n[OK] Deep dive complete!")
