"""
===============================================
Artifact Removal using DSS with Cycle Averaging
===============================================

This example demonstrates how to use :class:`mne_denoise.dss.CycleAverageBias`
in a DSS pipeline to remove repetitive artifacts like heartbeat (ECG) from EEG data.

The workflow is:

1.  Identify artifact events (e.g., R-peaks).
2.  Create a bias function that averages data around these events.
3.  Apply DSS to find components that maximize this average (the artifact).
4.  Remove the artifact components and reconstruct the clean data.
"""

# Authors: Sina Esmaeili <sina.esmaeili@umontreal.ca>
#          Hamza Abdelhedi <hamza.abdelhedi@umontreal.ca>

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import mne

from mne_denoise.dss import DSS
from mne_denoise.dss.denoisers import CycleAverageBias

###############################################################################
# Simulate Data with ECG Artifact
# -------------------------------
# We generate synthetic EEG data (1/f noise) mixed with a strong ECG artifact.
# The artifact typically affects some channels more than others.

rng = np.random.default_rng(42)
sfreq = 250
n_seconds = 10
n_times = n_seconds * sfreq
n_channels = 16

# Create random "EEG" background (1/f noise)
times = np.arange(n_times) / sfreq
eeg_data = np.cumsum(rng.standard_normal((n_channels, n_times)), axis=1)
eeg_data = signal.detrend(eeg_data, axis=1)

# Create synthetic ECG artifact
# Heartbeat every ~0.8s (75 BPM)
ecg_events = np.arange(0.5, n_seconds, 0.8) * sfreq
ecg_events = ecg_events.astype(int)

# Create a "QRS complex" template
duration = 0.1  # seconds
n_samples = int(duration * sfreq)
t_template = np.linspace(-1, 1, n_samples)
template = -5 * np.exp(-100 * t_template**2) + 20 * np.exp(-50 * t_template**2)

# Add artifact to data (linearly mixed)
artifact_source = np.zeros(n_times)
for event in ecg_events:
    start = event - n_samples // 2
    end = start + n_samples
    if start >= 0 and end <= n_times:
        artifact_source[start:end] += template

# Mix artifact into channels (random weights)
# Channels 0 and 1 have strong artifact
mixing = rng.standard_normal(n_channels) * 0.1
mixing[0] = 5.0
mixing[1] = 2.0
artifact_component = np.outer(mixing, artifact_source)

raw_data = eeg_data + artifact_component

# Create MNE Raw object
info = mne.create_info(n_channels, sfreq, "eeg")
raw = mne.io.RawArray(raw_data, info)

print("Data simulated. Strong ECG artifact in channels 0 and 1.")

###############################################################################
# Detect Events
# -------------
# In a real scenario, you would use MNE's ``find_ecg_events`` or ``find_eog_events``
# or load events from annotations. Here, we demonstrate detection using the
# noisy data itself.

# We'll use channel 0 as reference since it has strong artifact
ecg_events_detected, _, _ = mne.preprocessing.find_ecg_events(
    raw, ch_name=str(raw.ch_names[0]), verbose=False
)
event_samples = ecg_events_detected[:, 0]

print(f"Detecting {len(event_samples)} events (True: {len(ecg_events)})")

###############################################################################
# Create Artifact Bias
# --------------------
# We configure the :class:`~mne_denoise.dss.CycleAverageBias`.
# This bias emphasizes signal components that repeat synchronously with the events.

# Window: -100ms to +100ms around peak
bias = CycleAverageBias(
    event_samples=event_samples,
    window=(-0.1, 0.1),
    sfreq=sfreq
)

###############################################################################
# Apply DSS
# ---------
# We ask DSS to find components that are "most reproducible" across these cycles
# (i.e., that maximize the bias function).

dss = DSS(n_components=16, bias=bias)
dss.fit(raw)

# The first component(s) should capture the artifact.
# We can check the explained variance or ratio of biased power.
print("DSS Explained Variance (First 3):", dss.explained_variance_[:3])

###############################################################################
# Remove Artifact and Reconstruct
# -------------------------------
# To clean the data, we reject the top DSS components that capture the artifact.
# Usually visual inspection is needed, but here we know component 0 is the artifact.

keep_mask = np.ones(dss.n_components, dtype=bool)
keep_mask[0] = False  # Remove first component

# Transform to sources
sources_array = dss.transform(raw.get_data())

# Inverse transform excluding the artifact component
clean_data = dss.inverse_transform(sources_array, component_indices=keep_mask)

raw_clean = mne.io.RawArray(clean_data, info)

###############################################################################
# Compare Results
# ---------------
# We plot the original noisy data vs the cleaned data to verify artifact removal.

# Compare Mean Squared Error on the heavily contaminated channel
mse_dirty = np.mean((raw.get_data()[0] - eeg_data[0])**2)
mse_clean = np.mean((raw_clean.get_data()[0] - eeg_data[0])**2)

print(f"MSE Dirty: {mse_dirty:.2f}")
print(f"MSE Clean: {mse_clean:.2f}")

plt.figure(figsize=(10, 6))
plt.plot(times[:500], raw.get_data()[0, :500], label="Original (Noisy)", alpha=0.7)
plt.plot(times[:500], raw_clean.get_data()[0, :500], label="Cleaned", color='k', linewidth=1.5)
plt.title("ECG Artifact Removal")
plt.xlabel("Time (s)")
plt.legend()
plt.show()
