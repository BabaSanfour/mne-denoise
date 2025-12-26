"""
======================================
SSVEP / Periodic Signal Enhancement
======================================

This example demonstrates how to use :class:`mne_denoise.dss.PeakFilterBias`
and :class:`mne_denoise.dss.CombFilterBias` in a DSS pipeline to extract
steady-state visual evoked potentials (SSVEP).

The workflow is:

1.  Simulate an SSVEP signal (fundamental + harmonics) mixed with noise.
2.  Apply DSS with ``CombFilterBias`` to find components maximizing power
    at these frequencies.
3.  Visualize the extracted components and their power spectra.
"""

# Authors: Sina Esmaeili <sina.esmaeili@umontreal.ca>
#          Hamza Abdelhedi <hamza.abdelhedi@umontreal.ca>

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import mne

from mne_denoise.dss import DSS
from mne_denoise.dss.denoisers import CombFilterBias

###############################################################################
# Simulate SSVEP Data
# -------------------
# We simulate a 12 Hz SSVEP with harmonics at 24 Hz and 36 Hz.
# Background noise is pink noise (1/f).

rng = np.random.default_rng(42)
sfreq = 250
n_seconds = 10
n_times = n_seconds * sfreq
n_channels = 16

times = np.arange(n_times) / sfreq

# Generate Pink Noise
# Simple approximation: integration of white noise
noise = np.cumsum(rng.standard_normal((n_channels, n_times)), axis=1)
# Detrend to remove massive drift
from scipy.signal import detrend
noise = detrend(noise, axis=1)

# Generate SSVEP source
f_stim = 12.0
ssvep_source = (
    1.0 * np.sin(2 * np.pi * f_stim * times) +
    0.5 * np.sin(2 * np.pi * 2 * f_stim * times) + 
    0.2 * np.sin(2 * np.pi * 3 * f_stim * times)
)

# Mix SSVEP into channels (occipital focus logic, but random here)
# Strong in channel 0 and 1
mixing = rng.standard_normal(n_channels) * 0.1
mixing[0] = 2.0
mixing[1] = 1.5
ssvep_component = np.outer(mixing, ssvep_source)

data = noise + ssvep_component

# Create MNE Raw (RawArray)
info = mne.create_info(n_channels, sfreq, "eeg")
raw = mne.io.RawArray(data, info)

print("Data simulated. 12 Hz SSVEP in channels 0 and 1.")

###############################################################################
# Define Comb Filter Bias
# -----------------------
# We want to emphasize 12 Hz, 24 Hz, and 36 Hz.

bias = CombFilterBias(
    fundamental_freq=f_stim,
    sfreq=sfreq,
    n_harmonics=3,
    q_factor=30
)

###############################################################################
# Apply DSS
# ---------
# We look for components maximizing the ratio of (Comb filtered power) / (Total power).

dss = DSS(n_components=5, bias=bias)
dss.fit(raw)

print("DSS Explained Variance score (Eigenvalues):", dss.eigenvalues_)
# The first eigenvalue should be significantly higher if SSVEP is successfully isolated.

###############################################################################
# Analyze Components
# ------------------
# We extract the components and compute their PSD.

sources_data = dss.transform(raw)
# sources_data: (n_components, n_times)

# Convert to MNE Raw for easy PSD plotting
comp_info = mne.create_info(5, sfreq, "eeg")
sources_raw = mne.io.RawArray(sources_data[:5], comp_info)

# Plot PSD of the first component
fig = sources_raw.compute_psd(fmax=50).plot(show=False)
plt.title("PSD of DSS Components (First is SSVEP)")
plt.show()

###############################################################################
# Comparison with Raw
# -------------------
# Compare time series of raw channel 0 vs DSS component 0.

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(times[:500], raw.get_data()[0, :500])
plt.title("Original Channel 0 (Noisy SSVEP)")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(times[:500], sources_data[0, :500])
plt.title("DSS Component 1 (Clean SSVEP)")
plt.ylabel("Amplitude")
plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()
