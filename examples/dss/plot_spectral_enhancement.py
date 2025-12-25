"""
======================================
Spectral (Rhythm) Enhancement
======================================

This example demonstrates how to use :class:`mne_denoise.dss.BandpassBias`
in a DSS pipeline to extract specific brain rhythms (e.g., Alpha waves)
from noisy data.

The workflow is:

1.  Simulate an Alpha rhythm (10 Hz) burst mixed with pink noise and
    line noise.
2.  Apply DSS with ``BandpassBias`` (8-12 Hz) to find components maximizing 
    power in the Alpha band relative to the total power.
3.  Visualize the extracted components and their power spectra.
"""

# Authors: Sina Esmaeili <sina.esmaeili@umontreal.ca>
#          Hamza Abdelhedi <hamza.abdelhedi@umontreal.ca>

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import welch
import mne

from mne_denoise.dss import DSS
from mne_denoise.dss.denoisers.spectral import BandpassBias

###############################################################################
# Simulate Data with Alpha Bursts
# -------------------------------
# We simulate a 10 Hz Alpha rhythm that is present in some channels.

rng = np.random.default_rng(42)
sfreq = 250
n_seconds = 10
n_times = n_seconds * sfreq
n_channels = 16

times = np.arange(n_times) / sfreq

# Generate Pink Noise
noise = np.cumsum(rng.standard_normal((n_channels, n_times)), axis=1)
from scipy.signal import detrend
noise = detrend(noise, axis=1)

# Add 60 Hz line noise (diffeent from Alpha)
line_noise = 0.5 * np.sin(2 * np.pi * 60 * times)
noise += line_noise

# Generate Alpha source (10 Hz)
# Modulate amplitude to create bursts
envelope = np.sin(2 * np.pi * 0.5 * times) ** 2  # 0.5 Hz modulation
alpha_source = envelope * np.sin(2 * np.pi * 10 * times) * 2.0

# Mix into first 3 channels
mixing = rng.standard_normal(n_channels) * 0.1
mixing[0:3] = [1.5, 1.2, 0.8] 
alpha_component = np.outer(mixing, alpha_source)

data = noise + alpha_component

# Create MNE Raw
info = mne.create_info(n_channels, sfreq, "eeg")
raw = mne.io.RawArray(data, info)

print("Data simulated. Alpha bursts (10 Hz) in first 3 channels.")

###############################################################################
# Define Bandpass Bias
# --------------------
# We want to emphasize the 8-12 Hz band.

bias = BandpassBias(
    freq_band=(8, 12),
    sfreq=sfreq,
    order=4
)

###############################################################################
# Apply DSS
# ---------
# We look for components maximizing the ratio of (Alpha power) / (Total power).

dss = DSS(n_components=5, bias=bias)
dss.fit(raw)

print("DSS Explained Variance (Eigenvalues):", dss.eigenvalues_)
# First eigenvalue should be high (Alpha dominance in that component)

###############################################################################
# Analyze Components
# ------------------
# We extract the components and compute their PSD.

sources_data = dss.transform(raw)

comp_info = mne.create_info(5, sfreq, "eeg")
sources_raw = mne.io.RawArray(sources_data[:5], comp_info)

# Plot PSD
fig = sources_raw.compute_psd(fmax=80).plot(show=False)
plt.title("PSD of DSS Components (First is Alpha)")
plt.show()

###############################################################################
# Clean Signal comparison
# -----------------------
# Visualize the first component vs the original noisy channel.

plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(times[:500], raw.get_data()[0, :500])
plt.title("Original Channel 0 (Noisy Alpha)")
plt.ylabel("uV")

plt.subplot(3, 1, 2)
plt.plot(times[:500], sources_data[0, :500])
plt.title("DSS Component 1 (Extracted Alpha)")
plt.ylabel("AU")

plt.subplot(3, 1, 3)
plt.plot(times[:500], envelope[:500] * np.max(sources_data[0,:500]), 'r--', label='Ground Truth Envelope')
plt.plot(times[:500], np.abs(signal.hilbert(sources_data[0]))[:500], 'k', label='Extracted Envelope')
plt.title("Envelope Reconstruction")
plt.legend()

plt.tight_layout()
plt.show()
