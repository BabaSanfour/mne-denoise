"""
======================================
Evoked Response Enhancement using DSS
======================================

This example demonstrates how to use :class:`mne_denoise.dss.TrialAverageBias`
in a DSS pipeline to extract minimal evoked responses from noisy data.

The workflow is:

1.  Simulate a weak evoked response buried in noise.
2.  Apply DSS with ``TrialAverageBias`` to separate reproducible response
    from non-phase-locked noise.
3.  Compare the Signal-to-Noise Ratio (SNR) before and after.
"""

# Authors: Sina Esmaeili <sina.esmaeili@umontreal.ca>
#          Hamza Abdelhedi <hamza.abdelhedi@umontreal.ca>

import numpy as np
import matplotlib.pyplot as plt
import mne

from mne_denoise.dss import DSS
from mne_denoise.dss.denoisers import TrialAverageBias

###############################################################################
# Simulate Data
# -------------
# We simulate 50 trials of an auditory evoked potential (AEP) in high noise.
# The "signal" is a simple damped sinusoid.

rng = np.random.default_rng(42)
n_epochs = 50
n_channels = 10
n_times = 200
sfreq = 100

# Time vector
times = np.arange(n_times) / sfreq

# Define the "True" Evoked Response (N100-like)
# Peak around 100ms (0.1s)
signal = -np.exp(-((times - 0.5) ** 2) / 0.01) * np.sin(2 * np.pi * 5 * times)
# Normalize signal
signal /= np.max(np.abs(signal))

# Create Epochs data: (n_epochs, n_channels, n_times) for MNE
# Note: internal DSS may expect (n_channels, n_times, n_epochs) or
# handle MNE objects directly.
epochs_data = np.zeros((n_epochs, n_channels, n_times))

# Add signal to first few components/channels with varying strength to simulate mixing
mixing = rng.standard_normal(n_channels)
mixing /= np.linalg.norm(mixing)
# Make signal weak
signal_strength = 0.5 

for i in range(n_epochs):
    # Noise: non-phase-locked
    noise = rng.standard_normal((n_channels, n_times)) * 2.0
    
    # Add signal (phase-locked)
    for ch in range(n_channels):
        epochs_data[i, ch, :] = noise[ch] + mixing[ch] * signal * signal_strength

# Create MNE Epochs array
info = mne.create_info(n_channels, sfreq, "eeg")
events = np.array([[i * 1000, 0, 1] for i in range(n_epochs)])
epochs = mne.EpochsArray(epochs_data, info, events=events, tmin=-0.5, verbose=False)

print("Data simulated. Signal is buried in noise.")

# Plot average (Evoked) before denoising
evoked_dirty = epochs.average()
evoked_dirty.plot(show=False, window_title="Original Evoked (Noisy)")

###############################################################################
# Apply DSS
# ---------
# We use ``TrialAverageBias`` to find components that are most consistent
# across trials.

bias = TrialAverageBias()
dss = DSS(n_components=5, bias=bias)

# Fit on epochs
dss.fit(epochs)

print("Explained Variance Ratio (Bias/Data):", dss.eigenvalues_[:5])
# Higher eigenvalue = more reproducible across trials

###############################################################################
# Analyze Components
# ------------------
# The first component should be our evoked response.

# Transform epochs to see the sources
sources_data = dss.transform(epochs)
# sources_data is (n_epochs, n_components, n_times)

# Create info for components
comp_info = mne.create_info(dss.n_components, sfreq, "eeg")
sources = mne.EpochsArray(sources_data, comp_info, events=events, tmin=-0.5, verbose=False)

evoked_sources = sources.average()
evoked_sources.plot(show=False, window_title="DSS Components (Evoked)")

###############################################################################
# Reconstruct Denoised Data
# -------------------------
# We keep only the first component (the most reproducible one) and project back.

keep_mask = np.zeros(dss.n_components, dtype=bool)
keep_mask[0] = True # Keep only the best component

clean_data = dss.inverse_transform(sources_data, component_indices=keep_mask)
# clean_data is (n_epochs, n_channels, n_times) due to MNE structure in transform

clean_epochs = mne.EpochsArray(clean_data, info, events=events, tmin=-0.5, verbose=False)
evoked_clean = clean_epochs.average()

# Verify Improvement
# The clean evoked should look like the original signal template (scaled)
plt.figure()
plt.plot(times, signal, 'k--', label="True Signal", alpha=0.5)
plt.plot(times, evoked_clean.data[0] / np.max(np.abs(evoked_clean.data[0])), label="Denoised (Ch 0)")
plt.title("Normalized Comparison")
plt.legend()
plt.show()

###############################################################################
# Quantitative Comparison
# -----------------------
# We compare the Global Field Power (GFP) of the evoked response.
# The denoised GFP should be cleaner.

plt.figure()
plt.plot(times, evoked_dirty.data.std(axis=0), label="Original GFP")
plt.plot(times, evoked_clean.data.std(axis=0), label="Denoised GFP")
plt.title("Global Field Power")
plt.legend()
plt.show()
