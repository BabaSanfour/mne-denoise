"""
======================================
Spectrogram Denoising (Time-Frequency)
======================================

This example demonstrates how to use :class:`mne_denoise.dss.Spectrogram2DDenoiser`
for denoising signals based on their time-frequency characteristics, as
described in Section 4.1.3 of Särelä & Valpola (2005).

The technique is particularly effective for extracting oscillatory activity
that occurs in bursts (e.g., sleep spindles, beta bursts) from background noise.

The workflow is:
1.  Simulate a signal with transient "spindles".
2.  Compute STFT to visualize the time-frequency landscape.
3.  Apply `Spectrogram2DDenoiser` to mask out low-energy regions (noise).
4.  Reconstruct the clean signal.
"""

# Authors: Sina Esmaeili <sina.esmaeili@umontreal.ca>
#          Hamza Abdelhedi <hamza.abdelhedi@umontreal.ca>

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from mne_denoise.dss.denoisers import SpectrogramDenoiser, SpectrogramBias

###############################################################################
# Simulate Data with Transient Bursts
# -----------------------------------
# We simulate a "spindle" (12 Hz burst) embedded in noise.

rng = np.random.default_rng(42)
sfreq = 100
n_seconds = 10
times = np.arange(n_seconds * sfreq) / sfreq

# Background noise (broadband)
noise = rng.standard_normal(len(times)) * 0.5

# Spindle burst at 2-3s and 7-8s
spindle_freq = 12.0
envelope = np.zeros_like(times)
# Burst 1
mask1 = (times > 2) & (times < 3)
envelope[mask1] = np.hanning(mask1.sum())
# Burst 2
mask2 = (times > 7) & (times < 8)
envelope[mask2] = np.hanning(mask2.sum())

signal_clean = envelope * np.sin(2 * np.pi * spindle_freq * times) * 2.0

data = signal_clean + noise

###############################################################################
# Spectrogram Denoising
# ---------------------
# We use the denoiser to keep only the strongest time-frequency components.
# This works blindly without knowing the spindle frequency beforehand, relying
# on the sparsity of the burst in the TF domain.

# Threshold: Keep only top 5% of TF energy (95th percentile)
denoiser = SpectrogramDenoiser(
    threshold_percentile=95,
    nperseg=128
)
denoised_data = denoiser.denoise(data)

###############################################################################
# Visualization
# -------------

plt.figure(figsize=(10, 10))

# 1. Time Series
plt.subplot(3, 1, 1)
plt.plot(times, data, color='gray', alpha=0.5, label='Noisy Data')
plt.plot(times, signal_clean, color='k', linestyle='--', label='Clean Signal')
plt.plot(times, denoised_data, color='r', alpha=0.8, label='Denoised')
plt.title("Time Domain: Spindle Extraction")
plt.legend()
plt.xlim(0, 10)

# 2. Noisy Spectrogram
f, t, Zxx_noisy = signal.stft(data, fs=sfreq, nperseg=128)
plt.subplot(3, 1, 2)
plt.pcolormesh(t, f, np.abs(Zxx_noisy), shading='gouraud')
plt.title("Noisy Spectrogram")
plt.ylabel('Frequency [Hz]')
plt.ylim(0, 50)

# 3. Denoised Spectrogram
f, t, Zxx_clean = signal.stft(denoised_data, fs=sfreq, nperseg=128)
plt.subplot(3, 1, 3)
plt.pcolormesh(t, f, np.abs(Zxx_clean), shading='gouraud')
plt.title("Denoised Spectrogram (Masked)")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.ylim(0, 50)

plt.tight_layout()
plt.show()


print("Adaptive denoising complete. Spindles should be clearly visible in red.")

###############################################################################
# Part 2: Linear Spectrogram Bias (Fixed Mask)
# --------------------------------------------
# In this section, we demonstrate `SpectrogramBias`, which is a **Linear**
# denoiser. It applies a pre-defined mask to all channels.
#
# Use case: You already know *where* the artifact is in the TF domain
# (e.g. from a previous scan or a different sensor) and want to
# apply that suppression linearly to define a subspace.

# 1. Create a multichannel signal
n_channels = 3
data_multi = np.zeros((n_channels, len(times)))
# Channel 0: Spindle + Noise
data_multi[0] = signal_clean + noise
# Channel 1: Just Noise
data_multi[1] = noise
# Channel 2: Spindle only (clean reference)
data_multi[2] = signal_clean

# 2. Define a Fixed Mask
# Let's say we want to Isolate the 12 Hz band around t=2.5s and t=7.5s.
# We build a mask manually.
f_stft, t_stft, _ = signal.stft(data_multi[0], fs=sfreq, nperseg=128)
mask = np.zeros((len(f_stft), len(t_stft)))

# Find spectral bins for 10-14 Hz
freq_indices = np.where((f_stft >= 10) & (f_stft <= 14))[0]

# Find time indices for t=2-3s and t=7-8s
time_indices_1 = np.where((t_stft >= 2.0) & (t_stft <= 3.0))[0]
time_indices_2 = np.where((t_stft >= 7.0) & (t_stft <= 8.0))[0]

# Set mask to 1 in these regions
for t_idx in np.concatenate([time_indices_1, time_indices_2]):
    mask[freq_indices, t_idx] = 1.0

# 3. Apply Bias
bias = SpectrogramBias(mask=mask, nperseg=128)
biased_data = bias.apply(data_multi)

# Visualization
plt.figure(figsize=(10, 8))

# Channel 0 Original
plt.subplot(3, 1, 1)
plt.plot(times, data_multi[0], color='gray', label='Original (Ch0)')
plt.title("Original Signal (Ch0)")
plt.legend()

# Channel 0 Biased
plt.subplot(3, 1, 2)
plt.plot(times, biased_data[0], color='blue', label='Biased (Ch0)')
plt.plot(times, signal_clean, color='k', linestyle='--', alpha=0.5, label='True Signal')
plt.title("Linearly Biased Signal (Ch0)")
plt.legend()

# Channel 1 Biased (Should be empty as it was just noise)
plt.subplot(3, 1, 3)
plt.plot(times, biased_data[1], color='orange', label='Biased Noise (Ch1)')
plt.title("Biased Noise Channel (Ch1) - Should be near zero")
plt.legend()

plt.tight_layout()
plt.show()

print("Linear bias complete. Noise channel should be suppressed.")
