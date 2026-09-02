r"""
Riemannian ASR on blink-contaminated EEG.
=========================================

Can a robust covariance geometry change the calibration of ASR on a real EEG
recording with eye blinks? This example compares standard ASR with the public
``method="riemannian_windowed"`` backend on the MNE Sample dataset.

Blink coupling to an EOG channel is the artifact endpoint. Samples with a
quiet EOG provide a complementary preservation control: a small change there
is reassuring, but it is not neural ground truth. The comparison illustrates
the calibration difference on one recording; it is not a benchmark establishing
global superiority of either backend.

The use case is motivated by the Riemannian ASR evaluation of
:footcite:p:`blum2019_riemannian_asr` and the standard ASR evaluation of
:footcite:p:`chang2020_asr`.

References
----------
.. footbibliography::
"""

# %%
# Load real EEG with an EOG channel
# ---------------------------------
import matplotlib.pyplot as plt
import mne
import numpy as np

from mne_denoise.asr import ASR

sample_path = mne.datasets.sample.data_path()
raw = mne.io.read_raw_fif(
    sample_path / "MEG" / "sample" / "sample_audvis_raw.fif",
    preload=True,
    verbose="ERROR",
)
raw.pick(["eeg", "eog"]).crop(0.0, 60.0).resample(160.0, verbose="ERROR")
raw.set_eeg_reference("average", verbose="ERROR")
raw.filter(1.0, None, verbose="ERROR")

eeg = raw.get_data(picks="eeg")
eeg_names = [
    raw.ch_names[index] for index in mne.pick_types(raw.info, eeg=True, exclude=[])
]
eog = raw.get_data(picks="eog")[0]


def _absolute_correlations(data, reference):
    """Return absolute channel-wise correlations with an EOG reference."""
    return np.asarray([abs(np.corrcoef(channel, reference)[0, 1]) for channel in data])


eog_coupling_before = _absolute_correlations(eeg, eog)
top_channels = np.argsort(eog_coupling_before)[-8:]
blink_channel_index = int(np.argmax(eog_coupling_before))
blink_channel = eeg_names[blink_channel_index]
quiet_mask = np.abs(eog) <= np.quantile(np.abs(eog), 0.75)
blink_sample = int(np.argmax(np.abs(eog)))
window_start = max(0, blink_sample - int(round(1.0 * raw.info["sfreq"])))
window_stop = min(eeg.shape[1], blink_sample + int(round(2.0 * raw.info["sfreq"])))

print(f"Blink-dominated channel: {blink_channel} (|r|={eog_coupling_before.max():.3f})")
print(f"EOG-quiet preservation control: {quiet_mask.mean():.1%} of samples")

# %%
# Compare standard and Riemannian-windowed calibration
# -----------------------------------------------------
models = {
    "standard": ASR(
        cutoff=20.0,
        method="standard",
        picks="eeg",
        verbose=False,
    ),
    "riemannian": ASR(
        cutoff=20.0,
        method="riemannian_windowed",
        picks="eeg",
        verbose=False,
    ),
}
cleaned = {
    name: model.fit_transform(raw.copy()).get_data(picks="eeg")
    for name, model in models.items()
}


def _mean_eog_coupling(data):
    """Measure residual coupling for the eight most blink-linked channels."""
    return float(np.mean(_absolute_correlations(data[top_channels], eog)))


def _quiet_relative_change(data):
    """Measure change during EOG-quiet samples relative to input RMS."""
    delta = data[:, quiet_mask] - eeg[:, quiet_mask]
    input_rms = float(np.sqrt(np.mean(eeg[:, quiet_mask] ** 2)))
    return float(np.sqrt(np.mean(delta**2)) / input_rms)


coupling = {"raw": _mean_eog_coupling(eeg)}
quiet_change = {"raw": 0.0}
for name, data in cleaned.items():
    coupling[name] = _mean_eog_coupling(data)
    quiet_change[name] = _quiet_relative_change(data)

for name in ("raw", *models):
    print(
        f"  {name:12s}: mean |r(EEG, EOG)|={coupling[name]:.3f}, "
        f"quiet relative change={quiet_change[name]:.3f}"
    )

# %%
# Plot the blink endpoint and the preservation control
# -----------------------------------------------------
times = raw.times
scale = 1e6
fig = plt.figure(figsize=(11, 7), layout="constrained")
grid = fig.add_gridspec(2, 2, height_ratios=(1.3, 1.0))
trace_ax = fig.add_subplot(grid[0, :])
coupling_ax = fig.add_subplot(grid[1, 0])
quiet_ax = fig.add_subplot(grid[1, 1])

trace_ax.plot(
    times[window_start:window_stop],
    scale * eeg[blink_channel_index, window_start:window_stop],
    color="0.55",
    lw=0.8,
    label="input",
)
trace_ax.plot(
    times[window_start:window_stop],
    scale * cleaned["standard"][blink_channel_index, window_start:window_stop],
    color="C1",
    lw=0.9,
    label="standard ASR",
)
trace_ax.plot(
    times[window_start:window_stop],
    scale * cleaned["riemannian"][blink_channel_index, window_start:window_stop],
    color="C0",
    lw=0.9,
    label="Riemannian-windowed ASR",
)
trace_ax.axvspan(
    times[blink_sample] - 0.25,
    times[blink_sample] + 0.25,
    color="C3",
    alpha=0.12,
    label="strongest EOG sample ± 250 ms",
)
trace_ax.set(
    title=f"Blink-dominated channel {blink_channel}",
    xlabel="Time (s)",
    ylabel="Amplitude (µV)",
)
trace_ax.legend(loc="upper right", ncol=2)

labels = list(coupling)
coupling_ax.bar(
    labels,
    [coupling[label] for label in labels],
    color=["0.55", "C1", "C0"],
)
coupling_ax.set(
    title="Artifact endpoint",
    ylabel="mean |correlation with EOG|",
)
coupling_ax.tick_params(axis="x", labelrotation=20)

quiet_ax.bar(
    labels,
    [quiet_change[label] for label in labels],
    color=["0.55", "C1", "C0"],
)
quiet_ax.set(
    title="Preservation control",
    ylabel="quiet relative change",
)
quiet_ax.tick_params(axis="x", labelrotation=20)

fig.suptitle("Riemannian ASR: blink attenuation with an EOG-quiet control")
plt.show()
