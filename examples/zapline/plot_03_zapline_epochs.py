r"""
ZapLine: Epoched Data and Real Data Examples.
==============================================

This tutorial demonstrates:
1. ZapLine with epoched MEG data (NoiseTools data3.mat)
2. High-channel MEG data (NoiseTools example_data.mat)  
3. The sklearn-style Transformer API

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

# %%
# Imports
# -------
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import loadmat

from mne_denoise.zapline import ZapLine, dss_zapline

# %%
# Part 1: Synthetic Epoched Data
# ------------------------------
# Create epoched data with line noise to demonstrate ZapLine on trials.

print("Part 1: Synthetic Epoched Data")

# Parameters
sfreq = 500  # Sampling rate
n_epochs = 30
n_channels = 16
n_times = 250  # 0.5 seconds per epoch

rng = np.random.RandomState(42)

# Create spatial patterns
neural_pattern = rng.randn(n_channels)
neural_pattern /= np.linalg.norm(neural_pattern)

line_pattern = rng.randn(n_channels)
line_pattern /= np.linalg.norm(line_pattern)

# Generate epoched data
t = np.arange(n_times) / sfreq
epochs_data = np.zeros((n_epochs, n_channels, n_times))

for i in range(n_epochs):
    # Neural signal (evoked-like)
    neural_source = np.sin(2 * np.pi * 10 * t) * np.exp(-t / 0.2)
    
    # Line noise (constant across epochs, different phase)
    phase = rng.uniform(0, 2 * np.pi)
    line_source = 1.5 * np.sin(2 * np.pi * 50 * t + phase)
    
    for ch in range(n_channels):
        epochs_data[i, ch] = (
            neural_pattern[ch] * neural_source
            + line_pattern[ch] * line_source
            + rng.randn(n_times) * 0.3
        )

print(f"Epochs data shape: {epochs_data.shape}")  # (n_epochs, n_channels, n_times)

# %%
# Apply ZapLine to Epoched Data
# -----------------------------
# ZapLine expects 2D data, so we concatenate epochs for fitting.

print("\nApplying ZapLine to epoched data...")

# Concatenate epochs for ZapLine
data_concat = epochs_data.reshape(n_channels, -1)  # (channels, epochs*times)
print(f"Concatenated shape: {data_concat.shape}")

# Apply ZapLine
result = dss_zapline(data_concat, line_freq=50, sfreq=sfreq, n_remove=1)

# Reshape back to epochs
cleaned_epochs = result.cleaned.reshape(n_epochs, n_channels, n_times)
print(f"Cleaned epochs shape: {cleaned_epochs.shape}")

# %%
# Compare Before/After
# ^^^^^^^^^^^^^^^^^^^^

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Time domain - single trial
ax = axes[0, 0]
ax.plot(t * 1000, epochs_data[0, 0, :], "b-", label="Original")
ax.plot(t * 1000, cleaned_epochs[0, 0, :], "g-", label="Cleaned")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Amplitude")
ax.set_title("Single Trial, Channel 0")
ax.legend()

# Evoked (average across trials)
ax = axes[0, 1]
evoked_orig = np.mean(epochs_data, axis=0)
evoked_clean = np.mean(cleaned_epochs, axis=0)
ax.plot(t * 1000, evoked_orig[0, :], "b-", label="Original")
ax.plot(t * 1000, evoked_clean[0, :], "g-", label="Cleaned")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Amplitude")
ax.set_title("Evoked Response (Mean), Channel 0")
ax.legend()

# PSD - Original
ax = axes[1, 0]
freqs, psd_orig = signal.welch(data_concat, sfreq, nperseg=n_times)
ax.semilogy(freqs, np.mean(psd_orig, axis=0), "b-")
ax.axvline(50, color="r", linestyle="--")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD")
ax.set_title("Original PSD")
ax.set_xlim(0, 100)

# PSD - Cleaned
ax = axes[1, 1]
freqs, psd_clean = signal.welch(result.cleaned, sfreq, nperseg=n_times)
ax.semilogy(freqs, np.mean(psd_orig, axis=0), "b-", alpha=0.3, label="Original")
ax.semilogy(freqs, np.mean(psd_clean, axis=0), "g-", label="Cleaned")
ax.axvline(50, color="r", linestyle="--", alpha=0.5)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD")
ax.set_title("Cleaned PSD")
ax.set_xlim(0, 100)
ax.legend()

plt.tight_layout()
plt.show()

# %%
# Part 2: Using the Transformer API
# ----------------------------------
# The `ZapLine` class provides an sklearn-compatible interface.

print("\nPart 2: Transformer API")

# Create transformer
zl = ZapLine(line_freq=50, sfreq=sfreq, n_remove=1)

# Fit on concatenated data
zl.fit(data_concat)
print(f"Fitted! n_removed_: {zl.n_removed_}")
print(f"scores_: {zl.scores_}")

# Transform
cleaned = zl.transform(data_concat)
print(f"Cleaned shape: {cleaned.shape}")

# fit_transform in one step
cleaned2 = zl.fit_transform(data_concat)
print(f"fit_transform result: {cleaned2.shape}")

# %%
# Part 3: Real MEG Epoched Data (NoiseTools data3.mat)
# ----------------------------------------------------
# MEG epoched data from NoiseTools.
# Shape: (900 times, 151 channels, 30 epochs), sr=300 Hz

print("\nPart 3: Real MEG Epoched Data")

script_dir = Path(__file__).parent
data_dir = script_dir / "data"

# Load data3.mat (MEG epoched)
data3_path = data_dir / "data3.mat"
if data3_path.exists():
    mat = loadmat(str(data3_path))
    meg_epochs = mat["data"]  # (times, channels, epochs) = (900, 151, 30)
    sfreq_meg = float(mat["sr"].flatten()[0])
    
    # Use first 10 epochs as in MATLAB example
    meg_epochs = meg_epochs[:, :, :10]  # (900, 151, 10)
    
    # Transpose to (channels, times*epochs) for ZapLine
    n_times_meg, n_ch_meg, n_ep_meg = meg_epochs.shape
    meg_concat = meg_epochs.transpose(1, 0, 2).reshape(n_ch_meg, -1)  # (151, 9000)
    
    # Demean
    meg_concat = meg_concat - np.mean(meg_concat, axis=1, keepdims=True)
    
    # Scale to reasonable units (MEG data is in Tesla, very small values)
    scale_factor = 1e12  # Convert to pT
    meg_concat = meg_concat * scale_factor
    
    print(f"Loaded data3.mat: {n_ep_meg} epochs, {n_ch_meg} channels, {n_times_meg} times")
    print(f"Concatenated shape: {meg_concat.shape}")
    print(f"Sampling rate: {sfreq_meg} Hz")
    
    # Apply ZapLine (50 Hz)
    result_meg = dss_zapline(
        meg_concat,
        line_freq=50,
        sfreq=sfreq_meg,
        n_remove=2,  # As in MATLAB example
    )
    
    print(f"Components removed: {result_meg.n_removed}")
    
    # Compare PSDs
    fig, ax = plt.subplots(figsize=(10, 5))
    
    freqs, psd_orig = signal.welch(meg_concat, sfreq_meg, nperseg=int(sfreq_meg))
    freqs, psd_clean = signal.welch(result_meg.cleaned, sfreq_meg, nperseg=int(sfreq_meg))
    
    ax.semilogy(freqs, np.mean(psd_orig, axis=0), "b-", alpha=0.7, label="Original")
    ax.semilogy(freqs, np.mean(psd_clean, axis=0), "g-", label="Cleaned")
    ax.axvline(50, color="r", linestyle="--", alpha=0.5, label="50 Hz")
    ax.axvline(100, color="r", linestyle="--", alpha=0.3)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("MEG Epoched Data (data3.mat) - Before/After ZapLine")
    ax.legend()
    ax.set_xlim(0, 150)
    
    plt.show()
    
    # Measure reduction
    idx_50 = np.argmin(np.abs(freqs - 50))
    reduction_db = 10 * np.log10(
        np.mean(psd_orig[:, idx_50]) / np.mean(psd_clean[:, idx_50])
    )
    print(f"50 Hz power reduction: {reduction_db:.1f} dB")

else:
    print(f"Data not found: {data3_path}")
    print("Download from: http://audition.ens.fr/adc/NoiseTools/DATA/data3.mat")

# %%
# Part 4: High-Channel MEG Data (NoiseTools example_data.mat)
# -----------------------------------------------------------
# MEG data with many channels (275), demonstrating nkeep parameter.
# Shape: (3000 times, 275 channels, 30 epochs), sr=600 Hz

print("\nPart 4: High-Channel MEG Data")

example_data_path = data_dir / "example_data.mat"
if example_data_path.exists():
    mat = loadmat(str(example_data_path))
    meg_high = mat["meg"]  # (times, channels, epochs) = (3000, 275, 30)
    sfreq_high = float(mat["sr"].flatten()[0])
    
    # Use first epoch as in MATLAB example
    meg_high = meg_high[:, :, 0].T  # (275, 3000)
    
    # Demean
    meg_high = meg_high - np.mean(meg_high, axis=1, keepdims=True)
    
    # Scale to reasonable units (MEG data is in Tesla, very small values)
    scale_factor = 1e12  # Convert to pT
    meg_high = meg_high * scale_factor
    
    print(f"Loaded example_data.mat: {meg_high.shape}")
    print(f"Sampling rate: {sfreq_high} Hz")
    
    # Apply ZapLine with nkeep (as in MATLAB example)
    result_high = dss_zapline(
        meg_high,
        line_freq=50,
        sfreq=sfreq_high,
        n_remove=6,  # As in MATLAB example
        nkeep=50,    # Reduce dimensionality
    )
    
    print(f"Components removed: {result_high.n_removed}")
    
    # Compare PSDs
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    freqs, psd_orig = signal.welch(meg_high, sfreq_high, nperseg=int(sfreq_high))
    freqs, psd_clean = signal.welch(result_high.cleaned, sfreq_high, nperseg=int(sfreq_high))
    
    ax = axes[0]
    ax.semilogy(freqs, np.mean(psd_orig, axis=0), "b-")
    ax.axvline(50, color="r", linestyle="--", label="50 Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Original MEG (275 channels)")
    ax.legend()
    ax.set_xlim(0, 150)
    
    ax = axes[1]
    ax.semilogy(freqs, np.mean(psd_orig, axis=0), "b-", alpha=0.3, label="Original")
    ax.semilogy(freqs, np.mean(psd_clean, axis=0), "g-", label="Cleaned")
    ax.axvline(50, color="r", linestyle="--", alpha=0.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Cleaned MEG (nkeep=50, n_remove=6)")
    ax.legend()
    ax.set_xlim(0, 150)
    
    plt.tight_layout()
    plt.show()
    
    # Measure reduction
    idx_50 = np.argmin(np.abs(freqs - 50))
    reduction_db = 10 * np.log10(
        np.mean(psd_orig[:, idx_50]) / np.mean(psd_clean[:, idx_50])
    )
    print(f"50 Hz power reduction: {reduction_db:.1f} dB")

else:
    print(f"Data not found: {example_data_path}")
    print("Download from: http://audition.ens.fr/adc/NoiseTools/DATA/example_data.mat")

# %%
# Conclusion
# ----------
# ZapLine handles various data types:
#
# 1. **Epoched Data**: Concatenate epochs, apply ZapLine, reshape back
# 2. **High-Channel Data**: Use nkeep parameter to reduce dimensionality
# 3. **Transformer API**: sklearn-style fit/transform workflow
#
# Key observations from real data:
# - Real MEG data often needs 2-6 components removed
# - nkeep=50 works well for high-channel (>100) data
# - 50 Hz reduction typically >30 dB
