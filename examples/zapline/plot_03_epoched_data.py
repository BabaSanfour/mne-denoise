r"""
ZapLine: Epoched Data and Real Data Examples.
==============================================

This example shows how ZapLine can be applied when the data are naturally
epoched, then extends the same workflow to larger real MEG arrays from the
NoiseTools examples.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

# %%
# Imports
# -------
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from scipy import signal
from scipy.io import loadmat

from mne_denoise.viz import (
    plot_component_cleaning_summary,
    plot_psd_comparison,
)
from mne_denoise.zapline import ZapLine

NOISETOOLS_BASE_URL = "http://audition.ens.fr/adc/NoiseTools/DATA"


def _find_repo_root():
    """Return the repository root for this example."""
    starts = []
    if "__file__" in globals():
        starts.append(Path(__file__).resolve())
    starts.append(Path.cwd().resolve())

    for start in starts:
        current = start if start.is_dir() else start.parent
        for candidate in (current, *current.parents):
            mne_ok = (candidate / "mne_denoise").exists()
            ex_ok = (candidate / "examples").exists()
            if mne_ok and ex_ok:
                return candidate

    raise FileNotFoundError("Could not locate the repository root.")


def _load_or_fetch_data_file(name):
    """Return one ZapLine example data file, downloading it if needed."""
    path = _find_repo_root() / "examples" / "zapline" / "data" / name
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{NOISETOOLS_BASE_URL}/{name}"
    print(f"Downloading {name} to {path}...")
    urlretrieve(url, str(path))
    return path


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
# The important point is that the fit is done on a long 2D signal, then the
# cleaned data are reshaped back to the original epoch structure.

print("\nApplying ZapLine to epoched data...")

# Concatenate epochs for ZapLine
data_concat = epochs_data.reshape(n_channels, -1)  # (channels, epochs*times)
print(f"Concatenated shape: {data_concat.shape}")

# Apply ZapLine
est = ZapLine(line_freq=50, sfreq=sfreq, n_select=1)
est.fit(data_concat)
cleaned = est.transform(data_concat)

# Reshape back to epochs
cleaned_epochs = cleaned.reshape(n_epochs, n_channels, n_times)
print(f"Cleaned epochs shape: {cleaned_epochs.shape}")

# %%
# Compare Before/After
# ^^^^^^^^^^^^^^^^^^^^
# The PSD view should show the narrowband suppression clearly before we inspect
# component summaries.

# Use the reusable viz function for PSD comparison
plot_psd_comparison(data_concat, cleaned, sfreq=sfreq, line_freq=50, show=True)

# %%
# Component Cleaning Summary
# ^^^^^^^^^^^^^^^^^^^^^^^^^^

# Show comprehensive cleaning summary
plot_component_cleaning_summary(
    scores=getattr(est, "scores_", getattr(est, "eigenvalues_", None)),
    selected_count=getattr(est, "n_removed_", 0),
    patterns=getattr(est, "patterns_", None),
    removed=data_concat - cleaned,
    sources=getattr(est, "sources_", None),
    sfreq=sfreq,
    line_freq=50,
    title="Component Cleaning Summary (ZapLine)",
    show=True,
)

# %%
# Part 2: Real MEG Epoched Data (NoiseTools data3.mat)
# ----------------------------------------------------
# MEG epoched data from NoiseTools.
# Shape: (900 times, 151 channels, 30 epochs), sr=300 Hz

print("\nPart 2: Real MEG Epoched Data")

# Load data3.mat (MEG epoched)
data3_path = _load_or_fetch_data_file("data3.mat")
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
est_meg = ZapLine(
    line_freq=50,
    sfreq=sfreq_meg,
    n_select=2,  # As in MATLAB example
)
est_meg.fit(meg_concat)
cleaned_meg = est_meg.transform(meg_concat)

print(f"Components removed: {est_meg.n_removed_}")

# Use the reusable viz functions
# The real-data example follows the same pattern as the synthetic one: inspect
# the spectral change first, then inspect the removed components.
plot_psd_comparison(
    meg_concat,
    cleaned_meg,
    sfreq=sfreq_meg,
    line_freq=50,
    fmax=150,
    show=True,
)

# Measure reduction
nperseg = min(meg_concat.shape[1], int(sfreq_meg * 2))
freqs, psd_orig = signal.welch(meg_concat, sfreq_meg, nperseg=nperseg)
_, psd_clean = signal.welch(cleaned_meg, sfreq_meg, nperseg=nperseg)
idx_50 = np.argmin(np.abs(freqs - 50))
ratio = np.mean(psd_orig[:, idx_50]) / np.mean(psd_clean[:, idx_50])
reduction_db = 10 * np.log10(ratio)
print(f"50 Hz power reduction: {reduction_db:.1f} dB")

# %%
# Part 3: High-Channel MEG Data (NoiseTools example_data.mat)
# -----------------------------------------------------------
# MEG data with many channels (275), demonstrating nkeep parameter.
# Shape: (3000 times, 275 channels, 30 epochs), sr=600 Hz

print("\nPart 3: High-Channel MEG Data")

example_data_path = _load_or_fetch_data_file("example_data.mat")
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

# Apply ZapLine with nkeep
est_high = ZapLine(
    line_freq=50,
    sfreq=sfreq_high,
    n_select=6,  # As in MATLAB example
    nkeep=50,  # Reduce dimensionality
)
est_high.fit(meg_high)
cleaned_high = est_high.transform(meg_high)

print(f"Components removed: {est_high.n_removed_}")

# Use the reusable viz functions
plot_psd_comparison(
    meg_high,
    cleaned_high,
    sfreq=sfreq_high,
    line_freq=50,
    fmax=150,
    show=True,
)

# %%
# Component Cleaning Summary
# ^^^^^^^^^^^^^^^^^^^^^^^^^^

plot_component_cleaning_summary(
    scores=getattr(est_high, "scores_", getattr(est_high, "eigenvalues_", None)),
    selected_count=getattr(est_high, "n_removed_", 0),
    patterns=getattr(est_high, "patterns_", None),
    removed=meg_high - cleaned_high,
    sources=getattr(est_high, "sources_", None),
    sfreq=sfreq_high,
    line_freq=50,
    title="Component Cleaning Summary (ZapLine)",
    show=True,
)

# %%
# Measure Reduction
# ^^^^^^^^^^^^^^^^^

nperseg = min(meg_high.shape[1], int(sfreq_high * 2))
freqs, psd_orig = signal.welch(meg_high, sfreq_high, nperseg=nperseg)
_, psd_clean = signal.welch(cleaned_high, sfreq_high, nperseg=nperseg)
idx_50 = np.argmin(np.abs(freqs - 50))
ratio = np.mean(psd_orig[:, idx_50]) / np.mean(psd_clean[:, idx_50])
reduction_db = 10 * np.log10(ratio)
print(f"50 Hz power reduction: {reduction_db:.1f} dB")

# %%
# Conclusion
# ----------
# ZapLine can be applied to epoched data by concatenating epochs, cleaned on
# high-channel recordings with ``nkeep`` to control dimensionality, and used as
# a standard transformer in a fit/transform workflow.
#
# On real MEG data, removing 2 to 6 components is often sufficient, a value
# like ``nkeep=50`` works well for very high-channel recordings, and the 50 Hz
# attenuation is typically large enough to be obvious in the PSD.
