r"""
Repairing transient EEG bursts with ASR.
========================================

Can Artifact Subspace Reconstruction (ASR) reduce short, high-amplitude
multichannel bursts while preserving the underlying signal outside the burst
windows? A controlled recording answers both parts of that question because
the clean substrate and artifact mask are known.

The clean calibration segment is supplied separately from the contaminated
recording. We then vary the cutoff over a small illustrative range and measure
artifact residuals and quiet-signal error separately. The best cutoff is
data- and endpoint-dependent; this synthetic trade-off is not a universal
parameter recommendation.

This use case is motivated by standard ASR and its parameter evaluation
:footcite:p:`kothe_jung2016_asr,chang2018_asr,chang2020_asr`.

References
----------
.. footbibliography::
"""

# %%
# Construct a controlled multichannel recording
# ----------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

from mne_denoise.asr import ASR

rng = np.random.default_rng(42)
sfreq = 250.0
duration = 12.0
n_channels = 8
n_times = int(round(sfreq * duration))
times = np.arange(n_times) / sfreq

clean = np.empty((n_channels, n_times), dtype=float)
for channel in range(n_channels):
    phase = rng.uniform(0.0, 2.0 * np.pi)
    clean[channel] = (
        0.5 * np.sin(2.0 * np.pi * 10.0 * times + phase)
        + 0.2 * np.sin(2.0 * np.pi * 6.0 * times + 0.5 * phase)
        + 0.05 * rng.standard_normal(n_times)
    )

artifact_mask = np.zeros(n_times, dtype=bool)
spatial = rng.standard_normal((n_channels, 2))
spatial /= np.linalg.norm(spatial, axis=0, keepdims=True)
contaminated = clean.copy()
for onset, stop in ((4.0, 4.8), (8.0, 8.6)):
    start = int(round(onset * sfreq))
    stop_sample = int(round(stop * sfreq))
    artifact_mask[start:stop_sample] = True
    burst = 8.0 * rng.standard_normal((2, stop_sample - start))
    contaminated[:, start:stop_sample] += spatial @ burst

calibration = clean[:, : int(round(3.0 * sfreq))]


def _residual_rms(data, reference, mask):
    """Return RMS error relative to a reference over selected samples."""
    residual = np.asarray(data)[:, mask] - np.asarray(reference)[:, mask]
    return float(np.sqrt(np.mean(residual**2)))


def _scores(data):
    """Compute the two endpoints used in this example."""
    artifact_before = _residual_rms(contaminated, clean, artifact_mask)
    artifact_after = _residual_rms(data, clean, artifact_mask)
    quiet = ~artifact_mask
    quiet_error = _residual_rms(data, clean, quiet) / _residual_rms(
        clean, np.zeros_like(clean), quiet
    )
    quiet_correlation = float(
        np.corrcoef(data[:, quiet].ravel(), clean[:, quiet].ravel())[0, 1]
    )
    return {
        "artifact_residual_ratio": artifact_after / artifact_before,
        "quiet_relative_error": quiet_error,
        "quiet_correlation": quiet_correlation,
    }


# %%
# Fit ASR and compare a small cutoff range
# -----------------------------------------
# ``calibration="manual"`` makes the clean reference segment explicit. The
# statistics filter is disabled here so the controlled signal remains the
# reference substrate used by the evaluation.
cutoffs = (2.0, 5.0, 20.0)
models = {}
cleaned_by_cutoff = {}
scores_by_cutoff = {}
diagnostics_by_cutoff = {}
for cutoff in cutoffs:
    model = ASR(
        sfreq=sfreq,
        cutoff=cutoff,
        calibration="manual",
        filter_kind="none",
        max_dims=0.5,
        picks=None,
        verbose=False,
    )
    cleaned, diagnostics = model.fit_transform(
        contaminated,
        calibration=calibration,
        return_diagnostics=True,
    )
    models[cutoff] = model
    cleaned_by_cutoff[cutoff] = np.asarray(cleaned)
    scores_by_cutoff[cutoff] = _scores(cleaned_by_cutoff[cutoff])
    diagnostics_by_cutoff[cutoff] = diagnostics

middle_cutoff = 5.0
cleaned = cleaned_by_cutoff[middle_cutoff]
middle_model = models[middle_cutoff]
middle_diagnostics = diagnostics_by_cutoff[middle_cutoff]

print("Cutoff comparison (lower residual/error is better):")
for cutoff in cutoffs:
    scores = scores_by_cutoff[cutoff]
    print(
        f"  cutoff={cutoff:>4.0f}: "
        f"artifact residual ratio={scores['artifact_residual_ratio']:.3f}, "
        f"quiet relative error={scores['quiet_relative_error']:.3f}, "
        f"quiet correlation={scores['quiet_correlation']:.3f}"
    )
print(
    f"ASR QC at cutoff={middle_cutoff:g}: "
    f"{middle_model.fraction_reconstructed_samples_:.1%} of samples "
    f"reconstructed across {middle_diagnostics['n_windows']} windows."
)

# %%
# Plot the attenuation--preservation trade-off
# ---------------------------------------------
fig = plt.figure(figsize=(11, 7), layout="constrained")
grid = fig.add_gridspec(2, 2, height_ratios=(1.3, 1.0))
trace_ax = fig.add_subplot(grid[0, :])
artifact_ax = fig.add_subplot(grid[1, 0])
quiet_ax = fig.add_subplot(grid[1, 1])

trace_ax.plot(times, contaminated[0], color="0.65", lw=0.8, label="contaminated")
trace_ax.plot(times, clean[0], color="k", lw=1.1, label="clean substrate")
trace_ax.plot(times, cleaned[0], color="C0", lw=0.9, label="ASR cleaned")
trace_ax.fill_between(
    times,
    trace_ax.get_ylim()[0],
    trace_ax.get_ylim()[1],
    where=artifact_mask,
    color="C3",
    alpha=0.12,
    label="known burst window",
)
trace_ax.set(
    title="Known bursts are attenuated while quiet signal remains available for a control",
    xlabel="Time (s)",
    ylabel="Amplitude (a.u.)",
)
trace_ax.legend(loc="upper right", ncol=2)

artifact_ax.plot(
    cutoffs,
    [scores_by_cutoff[cutoff]["artifact_residual_ratio"] for cutoff in cutoffs],
    "o-",
    color="C3",
)
artifact_ax.axhline(1.0, color="0.5", ls="--", lw=0.8)
artifact_ax.set(
    title="Artifact endpoint",
    xlabel="ASR cutoff",
    ylabel="burst residual / input residual",
)

quiet_ax.plot(
    cutoffs,
    [scores_by_cutoff[cutoff]["quiet_relative_error"] for cutoff in cutoffs],
    "o-",
    color="C0",
)
quiet_ax.set(
    title="Preservation endpoint",
    xlabel="ASR cutoff",
    ylabel="quiet error / clean-signal RMS",
)

fig.suptitle("Standard ASR: attenuation and preservation are separate endpoints")
plt.show()
