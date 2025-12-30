"""
ZapLine-Plus: Automatic Adaptive Noise Removal.
===============================================

This example demonstrates ZapLine-Plus (Klug & Kloosterman, 2022),
an adaptive extension of ZapLine with:

- Automatic noise frequency detection
- Adaptive data chunking for non-stationary noise
- Automatic component selection via outlier detection
- Adaptive parameter adjustment based on cleaning results

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

Reference
---------
Klug, M., & Kloosterman, N. A. (2022). Zapline-plus: A Zapline extension
for automatic and adaptive removal of frequency-specific noise artifacts
in M/EEG. Human Brain Mapping, 43(9), 2743-2758.
"""

# %%
# Imports
# -------
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

sys.path.insert(0, r"D:\PhD\mne-denoise")

from mne_denoise import dss_zapline, zapline_plus

# %%
# Create Complex Non-Stationary Data
# -----------------------------------
# Simulate data with changing noise characteristics over time,
# as might occur in a mobile EEG experiment.

np.random.seed(42)

# Parameters
sfreq = 250
duration = 120  # 2 minutes
n_channels = 64
n_times = int(sfreq * duration)
t = np.arange(n_times) / sfreq

# Base signal (pink noise)
from scipy.fft import fft, ifft

white = np.random.randn(n_channels, n_times)
freqs_fft = np.fft.fftfreq(n_times)
freqs_fft[0] = 1
pink_filter = 1 / np.sqrt(np.abs(freqs_fft))
pink_filter[0] = 0
brain_signal = np.real(ifft(fft(white, axis=1) * pink_filter, axis=1))
brain_signal = brain_signal / np.std(brain_signal) * 5e-6  # 5 µV

# Non-stationary line noise:
# - First half: 50 Hz at moderate amplitude
# - Second half: 50 Hz at high amplitude + phase shift

line_noise = np.zeros((n_channels, n_times))

# First half
t1 = n_times // 2
mixing1 = np.random.randn(n_channels, 1)
noise1 = np.sin(2 * np.pi * 50 * t[:t1])
line_noise[:, :t1] = mixing1 * noise1 * 20e-6

# Second half (different mixing, higher amplitude)
mixing2 = np.random.randn(n_channels, 1)
noise2 = np.sin(2 * np.pi * 50 * t[t1:] + np.pi / 4)  # Phase shift
line_noise[:, t1:] = mixing2 * noise2 * 50e-6

# Add a mysterious 21 Hz artifact in the second half only
# (like from a VR headset as mentioned in the paper)
mixing_21 = np.random.randn(n_channels, 1)
noise_21 = np.sin(2 * np.pi * 21 * t[t1:])
line_noise[:, t1:] += mixing_21 * noise_21 * 15e-6

# Combine
data = brain_signal + line_noise

print(f"Data: {n_channels} channels x {n_times} samples ({duration}s)")
print("Non-stationary noise with:")
print("  - 50 Hz: moderate in first half, strong in second half")
print("  - 21 Hz: only in second half (simulating VR artifact)")

# %%
# Visualize Non-Stationarity
# --------------------------

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

# Spectrogram of one channel
ax = axes[0]
f_spec, t_spec, Sxx = signal.spectrogram(data[0], sfreq, nperseg=512, noverlap=256)
mask = f_spec < 100
im = ax.pcolormesh(
    t_spec, f_spec[mask], 10 * np.log10(Sxx[mask]), shading="gouraud", cmap="viridis"
)
ax.axhline(50, color="r", linestyle="--", alpha=0.7, label="50 Hz")
ax.axhline(21, color="orange", linestyle="--", alpha=0.7, label="21 Hz")
ax.set_ylabel("Frequency (Hz)")
ax.set_title("Spectrogram - Non-Stationary Line Noise")
ax.legend(loc="upper right")
plt.colorbar(im, ax=ax, label="Power (dB)")

# Time series
ax = axes[1]
ax.plot(t, data[0] * 1e6, "k-", linewidth=0.3)
ax.axvline(t[t1], color="r", linestyle="--", alpha=0.7, label="Condition change")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude (µV)")
ax.set_title("Channel 1 Time Series")
ax.legend()
ax.set_xlim([0, duration])

plt.tight_layout()
plt.savefig("zaplineplus_nonstationarity.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Apply Standard ZapLine (No Chunking)
# ------------------------------------
# First, let's see what happens with standard ZapLine on non-stationary data.

result_basic = dss_zapline(data, line_freq=50, sfreq=sfreq, n_remove="auto")

print("\nStandard ZapLine (single pass):")
print(f"  Components removed: {result_basic.n_removed}")

# The 21 Hz artifact is not removed because we only targeted 50 Hz!

# %%
# Apply ZapLine-Plus (Adaptive)
# -----------------------------

result_plus = zapline_plus(
    data,
    sfreq,
    noisefreqs=None,  # Auto-detect all noise frequencies
    minfreq=17.0,  # Search from 17 Hz
    maxfreq=99.0,  # Up to 99 Hz
    adaptive_nremove=True,
    adaptive_sigma=True,
    chunk_length=0,  # Adaptive chunking
    min_chunk_length=30.0,
    verbose=True,
)

print("\nZapLine-Plus Results:")
print(f"  Detected frequencies: {result_plus.config['noisefreqs']}")
print(f"  Chunks processed: {len(result_plus.chunk_results)}")
print(f"  Final sigma: {result_plus.config['noise_comp_detect_sigma']:.2f}")
print(f"  Power removed: {result_plus.analytics['proportion_removed']:.2%}")

# %%
# Compare Results
# ---------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Compute PSDs
nperseg = 1024
freqs, psd_orig = signal.welch(data, sfreq, nperseg=nperseg, axis=1)
_, psd_basic = signal.welch(result_basic.cleaned, sfreq, nperseg=nperseg, axis=1)
_, psd_plus = signal.welch(result_plus.cleaned, sfreq, nperseg=nperseg, axis=1)

mean_orig = np.mean(psd_orig, axis=0)
mean_basic = np.mean(psd_basic, axis=0)
mean_plus = np.mean(psd_plus, axis=0)

# Original
ax = axes[0, 0]
ax.semilogy(freqs, mean_orig, "k-", linewidth=1.5)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD")
ax.set_title("Original Data")
ax.set_xlim([0, 100])
ax.grid(True, alpha=0.3)
ax.axvline(50, color="r", linestyle="--", alpha=0.5)
ax.axvline(21, color="orange", linestyle="--", alpha=0.5)

# Standard ZapLine
ax = axes[0, 1]
ax.semilogy(freqs, mean_basic, "b-", linewidth=1.5, label="ZapLine")
ax.semilogy(freqs, mean_orig, "k-", linewidth=0.5, alpha=0.3, label="Original")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD")
ax.set_title(f"Standard ZapLine (d={result_basic.n_removed})")
ax.set_xlim([0, 100])
ax.grid(True, alpha=0.3)
ax.axvline(50, color="r", linestyle="--", alpha=0.5)
ax.axvline(21, color="orange", linestyle="--", alpha=0.5, label="21 Hz NOT removed!")
ax.legend()

# ZapLine-Plus
ax = axes[1, 0]
ax.semilogy(freqs, mean_plus, "g-", linewidth=1.5, label="ZapLine-Plus")
ax.semilogy(freqs, mean_orig, "k-", linewidth=0.5, alpha=0.3, label="Original")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD")
ax.set_title("ZapLine-Plus (Adaptive)")
ax.set_xlim([0, 100])
ax.grid(True, alpha=0.3)
ax.axvline(50, color="r", linestyle="--", alpha=0.5)
ax.axvline(21, color="orange", linestyle="--", alpha=0.5)
ax.legend()

# Comparison
ax = axes[1, 1]
ax.semilogy(freqs, mean_basic, "b-", linewidth=1.5, alpha=0.7, label="Standard ZapLine")
ax.semilogy(freqs, mean_plus, "g-", linewidth=1.5, label="ZapLine-Plus")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD")
ax.set_title("Comparison")
ax.set_xlim([0, 100])
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig("zaplineplus_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Analyze Chunks
# --------------

if result_plus.chunk_results:
    print("\nPer-Chunk Analysis:")
    print("-" * 60)

    chunk_freqs = [r["freq"] for r in result_plus.chunk_results]
    chunk_removed = [r["n_removed"] for r in result_plus.chunk_results]

    for i, r in enumerate(result_plus.chunk_results):
        time_start = r["start"] / sfreq
        time_end = r["end"] / sfreq
        print(
            f"Chunk {i + 1}: {time_start:.1f}-{time_end:.1f}s | "
            f"freq={r['freq']:.1f} Hz | removed={r['n_removed']} components"
        )

    # Plot chunk info
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Detected frequencies per chunk
    chunk_times = [
        (r["start"] + r["end"]) / 2 / sfreq for r in result_plus.chunk_results
    ]

    ax = axes[0]
    ax.scatter(chunk_times, chunk_freqs, s=100, c="blue", marker="o")
    ax.set_ylabel("Detected Frequency (Hz)")
    ax.set_title("Per-Chunk Noise Analysis")
    ax.grid(True, alpha=0.3)
    ax.axhline(50, color="r", linestyle="--", alpha=0.5, label="50 Hz target")
    ax.legend()

    # Components removed per chunk
    ax = axes[1]
    ax.bar(
        chunk_times,
        chunk_removed,
        width=duration / len(chunk_times) * 0.8,
        color="green",
        alpha=0.7,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Components Removed")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("zaplineplus_chunks.png", dpi=150, bbox_inches="tight")
    plt.show()

# %%
# Test with Known Line Frequencies
# --------------------------------

result_known = zapline_plus(
    data,
    sfreq,
    noisefreqs=[50.0, 21.0],  # Specify both frequencies
    adaptive_sigma=True,
    verbose=True,
)

print("\nZapLine-Plus with Known Frequencies:")
print(f"  Power removed: {result_known.analytics['proportion_removed']:.2%}")

# Compare
_, psd_known = signal.welch(result_known.cleaned, sfreq, nperseg=nperseg, axis=1)
mean_known = np.mean(psd_known, axis=0)

fig, ax = plt.subplots(figsize=(12, 5))
ax.semilogy(freqs, mean_orig, "k-", linewidth=1, alpha=0.5, label="Original")
ax.semilogy(freqs, mean_plus, "g-", linewidth=1.5, alpha=0.7, label="Auto-detect")
ax.semilogy(freqs, mean_known, "m-", linewidth=1.5, label="Known frequencies")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD")
ax.set_title("ZapLine-Plus: Auto-Detect vs Known Frequencies")
ax.set_xlim([0, 100])
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig("zaplineplus_known_vs_auto.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Summary
# -------

print("\n" + "=" * 60)
print("ZapLine-Plus Demo Complete!")
print("=" * 60)
print(
    f"""
Results:
  Standard ZapLine (50 Hz only):
    - Removed {result_basic.n_removed} components
    - Left 21 Hz artifact untouched!

  ZapLine-Plus (auto-detect):
    - Detected frequencies: {result_plus.config["noisefreqs"]}
    - Processed {len(result_plus.chunk_results)} chunks
    - Removed {result_plus.analytics["proportion_removed"]:.2%} of power

Key Features of ZapLine-Plus:
  1. Automatic noise frequency detection
     - Finds all peaks above threshold (default 4 in 10log10 scale)

  2. Adaptive chunking
     - Detects changes in noise topography
     - Applies different spatial filters per chunk

  3. Per-chunk frequency refinement
     - Finds exact peak within each chunk
     - Handles frequency drift

  4. Adaptive component selection
     - Iterative outlier detection
     - Adjusts threshold if cleaning is too weak/strong

  5. Quality assessment
     - Checks for over/under cleaning
     - Automatically adjusts parameters
"""
)
print("\n[OK] Demo complete!")
