r"""
Adaptive ASR for changing recording statistics.
===============================================

Can adaptive calibration follow a change in the clean recording statistics
while continuing to suppress transient artifacts? This controlled recording
contains a baseline regime, a changed clean regime, and bursts only after the
change. A frozen ASR calibration is compared with one ``AdaptiveASR`` update.

The quiet changed regime is the preservation endpoint. Error in the final
burst windows is the artifact endpoint. The comparison illustrates the
streaming ``fit`` / ``partial_fit`` / ``transform`` lifecycle on one
deterministic substrate; it is not a benchmark of adaptive variants.

The use case is motivated by adaptive ASR research
:footcite:p:`tsai2024_adaptive_asr` and the standard ASR evaluation
:footcite:p:`chang2020_asr`.

References
----------
.. footbibliography::
"""

# %%
# Create three recording regimes
# ------------------------------
import matplotlib.pyplot as plt
import numpy as np

from mne_denoise.asr import ASR, AdaptiveASR

rng = np.random.default_rng(17)
sfreq = 200.0
n_channels = 8
segment_seconds = 12.0
segment_samples = int(round(segment_seconds * sfreq))
n_times = 3 * segment_samples
times = np.arange(n_times) / sfreq
t_segment = np.arange(segment_samples) / sfreq

baseline_mixing = rng.standard_normal((n_channels, 3))
changed_mixing = rng.standard_normal((n_channels, 3))


def _make_sources(time, frequencies, amplitudes):
    """Make a small reproducible source set for one covariance regime."""
    sources = []
    for frequency, amplitude in zip(frequencies, amplitudes):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        sources.append(
            amplitude * np.sin(2.0 * np.pi * frequency * time + phase)
            + 0.05 * rng.standard_normal(time.size)
        )
    return np.asarray(sources)


baseline = baseline_mixing @ _make_sources(
    t_segment,
    frequencies=(10.0, 6.0, 0.8),
    amplitudes=(0.35, 0.2, 0.08),
)
changed = changed_mixing @ _make_sources(
    t_segment,
    frequencies=(3.5, 12.0, 0.8),
    amplitudes=(0.8, 0.45, 0.2),
)

clean = np.hstack((baseline, changed, changed.copy()))
contaminated = clean.copy()
artifact_mask = np.zeros(n_times, dtype=bool)
artifact_spatial = rng.standard_normal(n_channels)
artifact_spatial /= np.linalg.norm(artifact_spatial)
for onset in (26.0, 30.0, 34.0):
    start = int(round(onset * sfreq))
    stop = min(n_times, start + int(round(0.6 * sfreq)))
    artifact_mask[start:stop] = True
    artifact_source = rng.standard_normal(stop - start)
    contaminated[:, start:stop] += 5.0 * np.outer(artifact_spatial, artifact_source)

quiet_changed_mask = np.zeros(n_times, dtype=bool)
quiet_changed_mask[segment_samples : 2 * segment_samples] = True
calibration_stop = segment_samples
changed_stop = 2 * segment_samples


def _relative_error(data, reference, mask):
    """Return RMS error relative to the reference signal."""
    residual = np.asarray(data)[:, mask] - np.asarray(reference)[:, mask]
    denominator = np.sqrt(np.mean(np.asarray(reference)[:, mask] ** 2))
    return float(np.sqrt(np.mean(residual**2)) / denominator)


def _artifact_residual_ratio(data):
    """Return residual artifact RMS after cleaning relative to input."""
    before = _relative_error(contaminated, clean, artifact_mask)
    after = _relative_error(data, clean, artifact_mask)
    return after / before


# %%
# Compare a frozen calibration with one adaptive update
# -----------------------------------------------------
# Both models are initialized on the baseline regime. The adaptive model then
# receives the changed, artifact-free regime through ``partial_fit`` before it
# processes the full stream.
frozen = ASR(
    sfreq=sfreq,
    cutoff=10.0,
    calibration="manual",
    filter_kind="none",
    max_dims=0.5,
    picks=None,
    verbose=False,
)
frozen.fit(clean[:, :calibration_stop])
frozen_clean = np.asarray(frozen.transform(contaminated))

adaptive = AdaptiveASR(
    sfreq=sfreq,
    cutoff=10.0,
    variant="psw",
    window_length=0.5,
    update_window_length=0.1,
    max_dims=0.5,
    picks=None,
    verbose=False,
)
adaptive.fit(clean[:, :calibration_stop])
adaptive.partial_fit(clean[:, calibration_stop:changed_stop])
adaptive_clean = np.asarray(adaptive.transform(contaminated))

scores = {
    "frozen ASR": {
        "artifact_residual_ratio": _artifact_residual_ratio(frozen_clean),
        "changed_quiet_error": _relative_error(frozen_clean, clean, quiet_changed_mask),
    },
    "adaptive ASR": {
        "artifact_residual_ratio": _artifact_residual_ratio(adaptive_clean),
        "changed_quiet_error": _relative_error(
            adaptive_clean, clean, quiet_changed_mask
        ),
    },
}
for name, values in scores.items():
    print(
        f"{name:12s}: artifact residual ratio="
        f"{values['artifact_residual_ratio']:.3f}, "
        f"changed-regime quiet error={values['changed_quiet_error']:.3f}"
    )
print(f"Adaptive calibration states: {len(adaptive.adaptive_update_history_)}")

# %%
# Plot the changed regime and both endpoints
# ------------------------------------------
channel = int(np.argmax(np.abs(artifact_spatial)))
fig = plt.figure(figsize=(11, 7), layout="constrained")
grid = fig.add_gridspec(2, 2, height_ratios=(1.3, 1.0))
trace_ax = fig.add_subplot(grid[0, :])
artifact_ax = fig.add_subplot(grid[1, 0])
quiet_ax = fig.add_subplot(grid[1, 1])

trace_ax.axvspan(
    0.0,
    segment_seconds,
    color="C2",
    alpha=0.08,
    label="baseline calibration regime",
)
trace_ax.axvspan(
    segment_seconds,
    2.0 * segment_seconds,
    color="C1",
    alpha=0.08,
    label="changed quiet regime",
)
trace_ax.axvspan(
    2.0 * segment_seconds,
    3.0 * segment_seconds,
    color="C3",
    alpha=0.08,
    label="changed regime with bursts",
)
trace_ax.plot(times, contaminated[channel], color="0.65", lw=0.75, label="contaminated")
trace_ax.plot(times, clean[channel], color="k", lw=1.0, label="clean substrate")
trace_ax.plot(times, frozen_clean[channel], color="C1", lw=0.8, label="frozen ASR")
trace_ax.plot(times, adaptive_clean[channel], color="C0", lw=0.8, label="adaptive ASR")
trace_ax.set(
    title="A clean covariance change precedes the artifact bursts",
    xlabel="Time (s)",
    ylabel="Amplitude (a.u.)",
)
trace_ax.legend(loc="upper right", ncol=2)

names = list(scores)
artifact_ax.bar(
    names,
    [scores[name]["artifact_residual_ratio"] for name in names],
    color=["C1", "C0"],
)
artifact_ax.axhline(1.0, color="0.5", ls="--", lw=0.8)
artifact_ax.set(
    title="Artifact endpoint",
    ylabel="burst residual / input residual",
)

quiet_ax.bar(
    names,
    [scores[name]["changed_quiet_error"] for name in names],
    color=["C1", "C0"],
)
quiet_ax.set(
    title="Preservation endpoint",
    ylabel="changed-regime error / signal RMS",
)

fig.suptitle("Adaptive ASR after a change in recording statistics")
plt.show()
