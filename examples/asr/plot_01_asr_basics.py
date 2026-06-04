r"""
Artifact Subspace Reconstruction: Basic Usage.
==============================================

This example demonstrates standard ASR on a synthetic multichannel EEG signal
with short spatial burst artifacts.
"""

# %%
# Imports
# -------
import matplotlib.pyplot as plt
import numpy as np

from mne_denoise.asr import ASR, compute_asr_qa_metrics
from mne_denoise.viz import plot_signal_overlay

# %%
# Synthetic EEG with Bursts
# -------------------------
sfreq = 250.0
duration = 12.0
n_channels = 8
n_times = int(sfreq * duration)
times = np.arange(n_times) / sfreq
rng = np.random.default_rng(42)

brain = np.zeros((n_channels, n_times))
for ch_idx in range(n_channels):
    phase = rng.uniform(0, 2 * np.pi)
    brain[ch_idx] = (
        0.5 * np.sin(2 * np.pi * 10 * times + phase)
        + 0.2 * np.sin(2 * np.pi * 6 * times + 0.5 * phase)
        + 0.05 * rng.standard_normal(n_times)
    )

data = brain.copy()
burst_mask = np.zeros(n_times, dtype=bool)
spatial = rng.standard_normal((n_channels, 2))
spatial /= np.linalg.norm(spatial, axis=0, keepdims=True)

for onset, stop in ((4.0, 4.8), (8.0, 8.6)):
    start = int(onset * sfreq)
    end = int(stop * sfreq)
    burst_mask[start:end] = True
    data[:, start:end] += spatial @ (8.0 * rng.standard_normal((2, end - start)))

# %%
# Fit and Apply ASR
# -----------------
# Conservative cutoffs around 20 are a typical starting point for adult EEG and
# should still be validated with QC plots and downstream analyses.
asr = ASR(
    sfreq=sfreq,
    cutoff=20.0,
    calibration="auto",
    filter_kind="none",
    max_dims=0.5,
)
clean = asr.fit_transform(data)

print(f"Calibration windows kept: {asr.clean_window_mask_.sum()}")
print(f"Repaired windows: {(asr.n_components_reconstructed_ > 0).sum()}")
print(f"Repaired sample fraction: {asr.fraction_reconstructed_samples_:.2%}")
metrics = compute_asr_qa_metrics(data, clean, asr)
print(f"Variance removed: {metrics['variance_removed_pct']:.2f}%")

# %%
# Plot One Channel
# ----------------
# ``plot_signal_overlay`` draws the before/after pair; we then overlay the
# ground-truth reference and shade the burst windows on the returned axis.
fig = plot_signal_overlay(
    data,
    clean,
    times,
    pick=0,
    scale_after=False,
    before_label="Noisy",
    after_label="ASR cleaned",
    x_label="Time (s)",
    y_label="Amplitude (a.u.)",
    title="ASR burst repair",
    show=False,
)
ax = fig.axes[0]
ax.plot(times, brain[0], color="C2", lw=1.0, alpha=0.8, label="Reference signal")
ax.fill_between(
    times,
    *ax.get_ylim(),
    where=burst_mask,
    color="C3",
    alpha=0.12,
    label="Burst artifact",
)
ax.legend(loc="upper right")
fig.tight_layout()

plt.show()
