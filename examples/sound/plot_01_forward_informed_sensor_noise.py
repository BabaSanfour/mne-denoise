r"""
Forward-informed sensor-noise suppression with SOUND
====================================================

Can SOUND use an anatomical forward model to identify and reduce deliberately
added channel-specific noise while preserving the rest of a recording? This
example uses EEG from the MNE Sample dataset together with the public Sample
forward solution, then plants independent broadband noise in three sensors.

SOUND uses forward-model consistency to estimate channel-specific noise; a
signal that the forward model explains poorly is not thereby proven to be
noise. Forward-model mismatch is a reason to evaluate preservation of the
signal of interest separately. The unmodified average-referenced recording is
the reference substrate for this controlled comparison, not noise-free neural
ground truth.

The use case is motivated by
:footcite:p:`mutanen2018_sound,mutanen2022_source_artifact`.

References
----------
.. footbibliography::
"""

# %%
# Load EEG and an explicit public forward solution
# -------------------------------------------------
import mne
import numpy as np
from matplotlib.patches import Patch

from mne_denoise.qa import rms_change
from mne_denoise.sound import SOUND
from mne_denoise.viz import (
    COLORS,
    FONTS,
    plot_signal_overlay,
    style_axes,
    themed_figure,
    themed_legend,
)

sample_path = mne.datasets.sample.data_path()
raw = mne.io.read_raw_fif(
    sample_path / "MEG" / "sample" / "sample_audvis_raw.fif",
    preload=True,
    verbose="ERROR",
)
forward = mne.read_forward_solution(
    sample_path / "MEG" / "sample" / "sample_audvis-meg-eeg-oct-6-fwd.fif",
    verbose="ERROR",
)

raw.pick("eeg", exclude="bads").crop(0.0, 20.0).resample(200.0, verbose="ERROR")
raw.filter(1.0, 45.0, verbose="ERROR")

# Keep an unreferenced copy for planting sensor-level noise. The Sample file's
# pre-marked bad EEG channel is excluded before building the reference.
unreferenced = raw.copy()
raw.set_eeg_reference("average", projection=True, verbose="ERROR")
raw.apply_proj(verbose="ERROR")

# The unmodified filtered recording is the reference substrate, not a claim of
# noise-free neural ground truth.
reference_data = raw.get_data()
n_channels = reference_data.shape[0]

# %%
# Plant known channel-specific noise
# ----------------------------------
rng = np.random.default_rng(2018)
channel_scale = np.median(np.std(reference_data, axis=1))
corrupted_indices = np.array([5, n_channels // 2, n_channels - 5])
noise_multiplier = 3.0

corrupted_data = unreferenced.get_data().copy()

for index in corrupted_indices:
    corrupted_data[index] += (
        noise_multiplier * channel_scale * rng.standard_normal(raw.n_times)
    )

corrupted = mne.io.RawArray(
    corrupted_data,
    unreferenced.info.copy(),
    first_samp=unreferenced.first_samp,
    verbose=False,
)
corrupted.set_annotations(raw.annotations.copy())
corrupted.set_eeg_reference("average", projection=True, verbose="ERROR")
corrupted.apply_proj(verbose="ERROR")
corrupted_data = corrupted.get_data()
corrupted_channel_names = [raw.ch_names[index] for index in corrupted_indices]
untouched_indices = np.setdiff1d(np.arange(n_channels), corrupted_indices)

# %%
# Fit SOUND and inspect its channel-wise noise estimates
# --------------------------------------------------------
sound = SOUND(
    forward=forward,
    reference="average",
    n_iter=5,
    random_state=0,
    verbose=False,
)
cleaned = sound.fit_transform(corrupted)
cleaned_data = cleaned.get_data()

corrupted_error = rms_change(
    corrupted_data[corrupted_indices],
    reference_data[corrupted_indices],
)
cleaned_error = rms_change(
    cleaned_data[corrupted_indices],
    reference_data[corrupted_indices],
)
reference_scale = np.sqrt(np.mean(reference_data[corrupted_indices] ** 2))
corrupted_relative_error = corrupted_error / reference_scale
cleaned_relative_error = cleaned_error / reference_scale

untouched_change = rms_change(
    cleaned_data[untouched_indices],
    reference_data[untouched_indices],
)
untouched_scale = np.sqrt(np.mean(reference_data[untouched_indices] ** 2))
untouched_relative_change = untouched_change / untouched_scale

rank_order = np.argsort(sound.sigmas_)[::-1]
top_indices = rank_order[:5]
print(f"Planted corrupted channels: {corrupted_channel_names}")
print(f"Corrupted-channel error before SOUND: {corrupted_relative_error:.3f}")
print(f"Corrupted-channel error after SOUND:  {cleaned_relative_error:.3f}")
print(f"Untouched-channel relative change:    {untouched_relative_change:.3f}")
print("Top channels by estimated sigma:")
for position, index in enumerate(top_indices, start=1):
    print(f"  {position}. {raw.ch_names[index]}: {sound.sigmas_[index]:.3e}")
for index in corrupted_indices:
    rank = int(np.flatnonzero(rank_order == index)[0] + 1)
    print(f"Rank of planted channel {raw.ch_names[index]}: {rank}")
print(f"Final convergence value: {sound.convergence_[-1]:.3e}")

# %%
# Plot the fitted channel-noise diagnostic
# ------------------------------------------
sigma_colors = [COLORS["primary"] for _ in range(n_channels)]
for index in corrupted_indices:
    sigma_colors[index] = COLORS["accent"]

fig, ax = themed_figure(figsize=(10.0, 3.8))
ax.bar(np.arange(n_channels), sound.sigmas_, color=sigma_colors)
tick_step = max(1, n_channels // 12)
tick_positions = np.arange(0, n_channels, tick_step)
ax.set_xticks(tick_positions)
ax.set_xticklabels([raw.ch_names[index] for index in tick_positions], rotation=90)
ax.set_xlabel("EEG channel")
ax.set_ylabel("Estimated noise amplitude")
ax.set_title("SOUND channel-noise estimates")
for index in corrupted_indices:
    ax.annotate(
        "planted",
        xy=(index, sound.sigmas_[index]),
        xytext=(index, float(np.max(sound.sigmas_)) * 1.08),
        ha="center",
        va="bottom",
        rotation=90,
        fontsize=FONTS["annotation"],
        color=COLORS["accent"],
        arrowprops={"arrowstyle": "-", "color": COLORS["accent"]},
    )
themed_legend(
    ax,
    handles=[
        Patch(color=COLORS["primary"], label="other channel"),
        Patch(color=COLORS["accent"], label="deliberately corrupted"),
    ],
    loc="upper right",
)
style_axes(ax, grid=True)
fig.tight_layout()

# %%
# Inspect one reconstructed sensor
# ---------------------------------
plot_signal_overlay(
    corrupted,
    cleaned,
    raw.times,
    pick=corrupted_channel_names[0],
    start=2.0,
    stop=4.0,
    scale_after=False,
    before_label="corrupted",
    after_label="SOUND",
    reference=reference_data[corrupted_indices[0]],
    reference_label="unmodified recording",
    x_label="Time (s)",
    y_label="Amplitude (V)",
    title=f"Forward-informed reconstruction at {corrupted_channel_names[0]}",
    show=False,
)
