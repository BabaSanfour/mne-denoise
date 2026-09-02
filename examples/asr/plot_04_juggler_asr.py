r"""
Juggler ASR when clean windows are scarce.
===========================================

How does reference selection behave when frequent short artifacts leave few
fully clean calibration windows? This controlled recording stresses the
window-based calibration assumption behind standard ASR and compares it with
one public ``JugglerASR`` strategy that selects reference samples point by
point.

The known burst mask provides the artifact endpoint. Recovery toward the clean
substrate outside those bursts provides the preservation endpoint. The
reference fractions and selected masks make the calibration decision visible.
This substrate illustrates the regime that motivates Juggler ASR; it is not a
benchmark of all ASR variants.

The use case is motivated by Juggler's ASR
:footcite:p:`kim2025_juggler_asr` and the standard ASR calibration
evaluation of :footcite:p:`chang2020_asr`.

References
----------
.. footbibliography::
"""

# %%
# Construct a densely contaminated recording
# -------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

from mne_denoise.asr import ASR, JugglerASR

rng = np.random.default_rng(21)
sfreq = 250.0
duration = 14.0
n_channels = 8
n_times = int(round(duration * sfreq))
times = np.arange(n_times) / sfreq

clean = np.empty((n_channels, n_times), dtype=float)
for channel in range(n_channels):
    phase = rng.uniform(0.0, 2.0 * np.pi)
    clean[channel] = (
        0.45 * np.sin(2.0 * np.pi * 10.0 * times + phase)
        + 0.18 * np.sin(2.0 * np.pi * 6.0 * times + 0.4 * phase)
        + 0.05 * rng.standard_normal(n_times)
    )

contaminated = clean.copy()
burst_mask = np.zeros(n_times, dtype=bool)
artifact_spatial = rng.standard_normal((n_channels, 2))
artifact_spatial /= np.linalg.norm(artifact_spatial, axis=0, keepdims=True)
for onset in np.arange(3.0, 11.5, 0.28):
    start = int(round(onset * sfreq))
    stop = min(n_times, start + int(round(0.10 * sfreq)))
    burst_mask[start:stop] = True
    burst_source = 7.0 * rng.standard_normal((2, stop - start))
    contaminated[:, start:stop] += artifact_spatial @ burst_source


def _residual_rms(data, reference, mask):
    """Return RMS error relative to a reference over selected samples."""
    residual = np.asarray(data)[:, mask] - np.asarray(reference)[:, mask]
    return float(np.sqrt(np.mean(residual**2)))


def _scores(data):
    """Compute artifact residual and quiet-signal error."""
    artifact_before = _residual_rms(contaminated, clean, burst_mask)
    quiet = ~burst_mask
    return {
        "artifact_residual_ratio": _residual_rms(data, clean, burst_mask)
        / artifact_before,
        "quiet_relative_error": _residual_rms(data, clean, quiet)
        / _residual_rms(clean, np.zeros_like(clean), quiet),
    }


# %%
# Compare standard and Juggler reference selection
# ------------------------------------------------
standard = ASR(
    sfreq=sfreq,
    cutoff=5.0,
    calibration="auto",
    filter_kind="asr",
    max_dims=0.5,
    picks=None,
    verbose=False,
)
juggler = JugglerASR(
    sfreq=sfreq,
    cutoff=5.0,
    strategy="dbscan",
    filter_kind="asr",
    selection_filter_kind="asr",
    max_dims=0.5,
    picks=None,
    verbose=False,
)

cleaned_standard = np.asarray(standard.fit_transform(contaminated))
cleaned_juggler = np.asarray(juggler.fit_transform(contaminated))

standard_reference = np.asarray(
    standard.calibration_info_["clean_sample_mask"], dtype=bool
)
if standard_reference.shape != (n_times,):
    raise RuntimeError(
        "The ASR calibration mask did not retain the expected sample layout."
    )
juggler_reference = juggler.get_calibration_mask()

scores = {
    "standard ASR": _scores(cleaned_standard),
    "Juggler ASR": _scores(cleaned_juggler),
}
for name, values in scores.items():
    reference_mask = (
        standard_reference if name.startswith("standard") else juggler_reference
    )
    print(
        f"{name:12s}: reference fraction={reference_mask.mean():.1%}, "
        f"artifact residual ratio={values['artifact_residual_ratio']:.3f}, "
        f"quiet relative error={values['quiet_relative_error']:.3f}"
    )

# %%
# Plot the calibration decision and both endpoints
# ------------------------------------------------
channel = int(np.argmax(np.max(np.abs(artifact_spatial), axis=1)))
fig = plt.figure(figsize=(11, 7), layout="constrained")
grid = fig.add_gridspec(2, 2, height_ratios=(1.25, 1.0))
trace_ax = fig.add_subplot(grid[0, :])
artifact_ax = fig.add_subplot(grid[1, 0])
quiet_ax = fig.add_subplot(grid[1, 1])

trace_ax.plot(times, contaminated[channel], color="0.65", lw=0.7, label="contaminated")
trace_ax.plot(times, clean[channel], color="k", lw=1.0, label="clean substrate")
trace_ax.plot(
    times,
    cleaned_standard[channel],
    color="C1",
    lw=0.8,
    label="standard ASR",
)
trace_ax.plot(
    times,
    cleaned_juggler[channel],
    color="C0",
    lw=0.8,
    label="Juggler ASR",
)
trace_ax.fill_between(
    times,
    trace_ax.get_ylim()[0],
    trace_ax.get_ylim()[1],
    where=burst_mask,
    color="C3",
    alpha=0.12,
    label="known burst window",
)
trace_ax.set(
    title="Frequent short bursts leave little fully clean window support",
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
    [scores[name]["quiet_relative_error"] for name in names],
    color=["C1", "C0"],
)
quiet_ax.set(
    title="Preservation endpoint",
    ylabel="quiet error / clean-signal RMS",
)

fig.suptitle("Juggler ASR: pointwise reference selection under dense contamination")
plt.show()

# %%
# Make the reference masks inspectable
# ------------------------------------
fig, ax = plt.subplots(figsize=(11, 2.8), layout="constrained")
mask_rows = (
    ("bursts (ground truth)", burst_mask, "C3"),
    ("standard reference", standard_reference, "C1"),
    ("Juggler reference", juggler_reference, "C0"),
)
for row, (label, mask, color) in enumerate(mask_rows):
    ax.fill_between(
        times,
        row,
        row + 0.8,
        where=mask,
        step="mid",
        color=color,
        alpha=0.75,
    )
    ax.text(-0.02, row + 0.4, label, transform=ax.get_yaxis_transform(), ha="right")
ax.set(
    xlim=(times[0], times[-1]),
    ylim=(0.0, len(mask_rows)),
    yticks=[],
    xlabel="Time (s)",
    title="Calibration masks (shaded = selected/true)",
)
ax.invert_yaxis()
plt.show()
