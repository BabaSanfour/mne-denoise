r"""
ZapLine: Parameter Tuning and Real Data.
=========================================

This tutorial covers:
1. Parameter exploration (n_remove, nkeep, threshold)
2. Real MEG data from NoiseTools
3. Comparing different parameter settings

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

# %%
# Imports
# -------
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import loadmat

from mne_denoise.zapline import ZapLine, dss_zapline

# %%
# Part 1: n_remove Parameter
# --------------------------
# The `n_remove` parameter controls how many spatial components are removed.
# Too few: line noise remains. Too many: neural signal lost.

print("Part 1: Exploring n_remove parameter")

# Generate synthetic data with multiple line noise sources
sfreq = 1000
duration = 5
n_channels = 16
n_times = int(sfreq * duration)
t = np.arange(n_times) / sfreq

rng = np.random.RandomState(42)

# Create 3 distinct line noise sources (simulating multiple power sources)
n_line_sources = 3
line_patterns = rng.randn(n_channels, n_line_sources)
for i in range(n_line_sources):
    line_patterns[:, i] /= np.linalg.norm(line_patterns[:, i])

# Neural pattern
neural_pattern = rng.randn(n_channels)
neural_pattern /= np.linalg.norm(neural_pattern)

# Generate data
neural_source = np.sin(2 * np.pi * 10 * t)
line_sources = np.zeros((n_line_sources, n_times))
for i in range(n_line_sources):
    phase = rng.uniform(0, 2 * np.pi)
    amp = 2.0 - i * 0.5  # Decreasing amplitude
    line_sources[i] = amp * np.sin(2 * np.pi * 50 * t + phase)

data = np.zeros((n_channels, n_times))
for i in range(n_channels):
    data[i] = neural_pattern[i] * neural_source
    for j in range(n_line_sources):
        data[i] += line_patterns[i, j] * line_sources[j]
    data[i] += rng.randn(n_times) * 0.3

print(f"Data with {n_line_sources} line noise sources")

# %%
# Compare Different n_remove Values
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

n_remove_values = [1, 2, 3, 4, 5, "auto"]

for idx, n_remove in enumerate(n_remove_values):
    row, col = idx // 3, idx % 3
    ax = axes[row, col]
    
    result = dss_zapline(data, line_freq=50, sfreq=sfreq, n_remove=n_remove)
    
    freqs, psd_orig = signal.welch(data, sfreq, nperseg=sfreq)
    freqs, psd_clean = signal.welch(result.cleaned, sfreq, nperseg=sfreq)
    
    ax.semilogy(freqs, np.mean(psd_orig, axis=0), "b-", alpha=0.3, label="Original")
    ax.semilogy(freqs, np.mean(psd_clean, axis=0), "g-", label="Cleaned")
    ax.axvline(50, color="r", linestyle="--", alpha=0.5)
    ax.set_xlim(0, 100)
    ax.set_title(f"n_remove={n_remove} (actual: {result.n_removed})")
    ax.set_xlabel("Frequency (Hz)")
    
    if col == 0:
        ax.set_ylabel("PSD")
    if idx == 0:
        ax.legend()

plt.suptitle("Effect of n_remove Parameter", fontsize=14)
plt.tight_layout()
plt.show()

# %%
# Part 2: nkeep Parameter (High Channel Count)
# ---------------------------------------------
# For data with many channels, the optional `nkeep` parameter reduces
# dimensionality before DSS to avoid overfitting.

print("\nPart 2: nkeep parameter for high-channel data")

# Create high-channel-count data
n_channels_high = 128
data_high = np.zeros((n_channels_high, n_times))

# Multiple line sources with different spatial patterns
n_line_sources = 2
line_patterns_high = rng.randn(n_channels_high, n_line_sources)
for i in range(n_line_sources):
    line_patterns_high[:, i] /= np.linalg.norm(line_patterns_high[:, i])

neural_pattern_high = rng.randn(n_channels_high)
neural_pattern_high /= np.linalg.norm(neural_pattern_high)

for i in range(n_channels_high):
    data_high[i] = neural_pattern_high[i] * neural_source
    for j in range(n_line_sources):
        data_high[i] += line_patterns_high[i, j] * line_sources[j]
    data_high[i] += rng.randn(n_times) * 0.3

print(f"High-channel data: {data_high.shape}")

# %%
# Compare Different nkeep Values
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

nkeep_values = [None, 64, 32, 16]

for idx, nkeep in enumerate(nkeep_values):
    ax = axes[idx]
    
    result = dss_zapline(
        data_high, line_freq=50, sfreq=sfreq, n_remove=2, nkeep=nkeep
    )
    
    freqs, psd_orig = signal.welch(data_high, sfreq, nperseg=sfreq)
    freqs, psd_clean = signal.welch(result.cleaned, sfreq, nperseg=sfreq)
    
    ax.semilogy(freqs, np.mean(psd_orig, axis=0), "b-", alpha=0.3, label="Original")
    ax.semilogy(freqs, np.mean(psd_clean, axis=0), "g-", label="Cleaned")
    ax.axvline(50, color="r", linestyle="--", alpha=0.5)
    ax.set_xlim(0, 100)
    ax.set_title(f"nkeep={nkeep if nkeep else 'All (128)'}")
    ax.set_xlabel("Frequency (Hz)")
    
    if idx == 0:
        ax.set_ylabel("PSD")
        ax.legend()

plt.suptitle("Effect of nkeep Parameter (128 channels)", fontsize=14)
plt.tight_layout()
plt.show()

# %%
# Part 3: Component Scores
# ------------------------
# The eigenvalues (scores) indicate how much each component
# carries line noise. High scores = strong line noise.

print("\nPart 3: Understanding component scores")

result = dss_zapline(data, line_freq=50, sfreq=sfreq, n_remove="auto")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Bar chart of scores
ax = axes[0]
scores = result.dss_eigenvalues
ax.bar(range(len(scores)), scores, color="steelblue")
ax.axhline(np.mean(scores), color="red", linestyle="--", label="Mean")
if result.n_removed > 0:
    ax.axvline(result.n_removed - 0.5, color="green", linestyle="--", 
               label=f"Removed: {result.n_removed}")
ax.set_xlabel("Component")
ax.set_ylabel("Score (eigenvalue)")
ax.set_title("Component Scores")
ax.legend()

# Spatial patterns of top components
ax = axes[1]
patterns = result.dss_patterns
n_show = min(3, patterns.shape[1])
for i in range(n_show):
    ax.plot(patterns[:, i], label=f"Comp {i}", marker="o", alpha=0.7)
ax.set_xlabel("Channel")
ax.set_ylabel("Pattern weight")
ax.set_title("Spatial Patterns (Top 3 Components)")
ax.legend()

plt.tight_layout()
plt.show()

# %%
# Part 4: Real MEG Data (NoiseTools)
# ----------------------------------
# Apply ZapLine to real MEG data from NoiseTools dataset.
# Data from: http://audition.ens.fr/adc/NoiseTools/DATA/

print("\nPart 4: Real MEG Data")

# Find data directory
script_dir = Path(__file__).parent
data_dir = script_dir / "data"

# Load data1.mat (MEG with large near-DC fluctuations)
data1_path = data_dir / "data1.mat"
if data1_path.exists():
    mat = loadmat(str(data1_path))
    meg_data = mat["data"].T  # Transpose to (channels, times)
    sfreq_meg = float(mat["sr"].flatten()[0])
    
    # Demean
    meg_data = meg_data - np.mean(meg_data, axis=1, keepdims=True)
    
    print(f"Loaded data1.mat: {meg_data.shape}, sfreq={sfreq_meg} Hz")
    print("MEG data with large near-DC fluctuations")
else:
    print(f"Data not found: {data1_path}")
    print("Download from: http://audition.ens.fr/adc/NoiseTools/DATA/data1.mat")
    meg_data = None

# %%
# Apply ZapLine to MEG Data
# ^^^^^^^^^^^^^^^^^^^^^^^^^

if meg_data is not None:
    # Apply ZapLine (60 Hz for this dataset)
    result_meg = dss_zapline(
        meg_data, 
        line_freq=60, 
        sfreq=sfreq_meg, 
        n_remove=2,  # As in MATLAB example
    )
    
    print(f"Components removed: {result_meg.n_removed}")

    # %%
    # Compare Before/After for MEG
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Original
    ax = axes[0]
    freqs, psd_orig = signal.welch(meg_data, sfreq_meg, nperseg=int(sfreq_meg))
    ax.semilogy(freqs, np.mean(psd_orig, axis=0))
    ax.axvline(60, color="r", linestyle="--", label="60 Hz")
    ax.axvline(120, color="r", linestyle="--", alpha=0.5, label="120 Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Original MEG")
    ax.legend()
    ax.set_xlim(0, 150)

    # Cleaned
    ax = axes[1]
    freqs, psd_clean = signal.welch(result_meg.cleaned, sfreq_meg, nperseg=int(sfreq_meg))
    ax.semilogy(freqs, np.mean(psd_orig, axis=0), "b-", alpha=0.3, label="Original")
    ax.semilogy(freqs, np.mean(psd_clean, axis=0), "g-", label="Cleaned")
    ax.axvline(60, color="r", linestyle="--", alpha=0.5)
    ax.axvline(120, color="r", linestyle="--", alpha=0.3)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Cleaned MEG")
    ax.legend()
    ax.set_xlim(0, 150)

    plt.tight_layout()
    plt.show()

    # %%
    # Measure Reduction
    # ^^^^^^^^^^^^^^^^^

    idx_60 = np.argmin(np.abs(freqs - 60))
    power_60_orig = np.mean(psd_orig[:, idx_60])
    power_60_clean = np.mean(psd_clean[:, idx_60])
    reduction_db = 10 * np.log10(power_60_orig / power_60_clean)

    print(f"\n=== MEG Results ===")
    print(f"60 Hz power reduction: {reduction_db:.1f} dB")
    print(f"Components removed: {result_meg.n_removed}")

# %%
# Conclusion
# ----------
# Key parameter guidelines:
#
# - **n_remove**: Start with "auto" or 1-3. Increase if line noise remains.
# - **nkeep**: Use for high-channel data (>64). Try 32-64.
# - **threshold**: For "auto" mode. Lower = more aggressive removal.
# - **n_harmonics**: Usually auto-detected. Increase for high sfreq data.
