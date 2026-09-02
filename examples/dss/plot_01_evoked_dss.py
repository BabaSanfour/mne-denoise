r"""
Enhancing a reproducible evoked response with DSS
=================================================

Can trial-average DSS learned on one set of auditory MEG trials enrich
reproducible evoked activity in independent held-out trials while limiting
distortion of the held-out evoked response?

DSS ranks components according to a chosen bias criterion. Here the bias is
trial reproducibility, so the fitted spatial subspace emphasizes activity that
survives averaging across repeated trials. A high score means strong according
to that criterion; it does not identify a neural source by itself
:footcite:p:`sarela2005_dss,decheveigne_simon2008_spatial`.

The before-versus-after comparison is a held-out preservation control, not a
clean-neural ground truth.

References
----------
.. footbibliography::
"""

# %%
# Load one auditory condition from the MNE Sample dataset
# --------------------------------------------------------
import mne
import numpy as np
from mne.datasets import sample

from mne_denoise.dss import DSS, AverageBias
from mne_denoise.viz import plot_component_score_curve, plot_evoked_gfp_comparison

sample_path = sample.data_path(update_path=False)
raw = mne.io.read_raw_fif(
    sample_path / "MEG" / "sample" / "sample_audvis_raw.fif",
    preload=True,
    verbose="ERROR",
)
raw.crop(0.0, 60.0)

# Find events before picking the homogeneous MEG channel type. One auditory
# condition keeps the reproducibility question focused on repeated trials.
events = mne.find_events(raw, stim_channel="STI 014", verbose=False)
auditory_events = events[events[:, 2] == 1]

raw.pick("grad", exclude="bads")
raw.filter(1.0, 40.0, verbose="ERROR")
epochs = mne.Epochs(
    raw,
    auditory_events,
    event_id={"Auditory/Left": 1},
    tmin=-0.2,
    tmax=0.4,
    baseline=(None, 0.0),
    preload=True,
    reject=None,
    verbose=False,
)

# %%
# Split trials before fitting
# ---------------------------
# The single selected Sample condition has a modest number of trials. A
# deterministic alternating split keeps both the training and held-out sets
# broad in time and gives the held-out split-half endpoint two balanced halves.
# The held-out trials are not used by the fit or by any component-count
# decision.
train_epochs = epochs[::2]
held_out_epochs = epochs[1::2]

n_components = 6
n_select = 3
model = DSS(
    bias=AverageBias(axis="epochs"),
    n_components=n_components,
    n_select=n_select,
    component_action="retain",
    verbose=False,
)
model.fit(train_epochs)
cleaned_held_out = model.transform(held_out_epochs)


# %%
# Evaluate held-out reproducibility and evoked preservation
# ----------------------------------------------------------
def _flattened_correlation(first, second):
    """Return correlation after flattening channels and time."""
    first = np.asarray(first, dtype=float).ravel()
    second = np.asarray(second, dtype=float).ravel()
    return float(np.corrcoef(first, second)[0, 1])


before_evoked = held_out_epochs.average()
after_evoked = cleaned_held_out.average()
post_mask = before_evoked.times >= 0.0

# Two independent halves of the held-out set provide the primary repeatability
# endpoint. Both halves are evaluated with the same fitted spatial operator.
n_held_out = len(held_out_epochs)
half = n_held_out // 2
before_half_a = held_out_epochs[:half].average()
before_half_b = held_out_epochs[half:].average()
after_half_a = cleaned_held_out[:half].average()
after_half_b = cleaned_held_out[half:].average()

split_half_before = _flattened_correlation(
    before_half_a.get_data()[:, post_mask],
    before_half_b.get_data()[:, post_mask],
)
split_half_after = _flattened_correlation(
    after_half_a.get_data()[:, post_mask],
    after_half_b.get_data()[:, post_mask],
)
held_out_waveform_correlation = _flattened_correlation(
    before_evoked.get_data()[:, post_mask],
    after_evoked.get_data()[:, post_mask],
)
before_gfp_rms = np.sqrt(np.mean(before_evoked.get_data()[:, post_mask] ** 2))
after_gfp_rms = np.sqrt(np.mean(after_evoked.get_data()[:, post_mask] ** 2))
held_out_gfp_gain = after_gfp_rms / before_gfp_rms

print("Held-out evoked DSS")
print(f"Training trial count: {len(train_epochs)}")
print(f"Held-out trial count: {n_held_out}")
print(f"Channel count: {len(held_out_epochs.ch_names)}")
print(f"n_components: {n_components}")
print(f"n_select: {n_select}")
print(
    "Held-out split-half sizes (first/second held-out order): "
    f"{half}/{n_held_out - half}"
)
print(f"Held-out split-half evoked correlation before: {split_half_before:.4f}")
print(f"Held-out split-half evoked correlation after:  {split_half_after:.4f}")
print(
    "Held-out before-vs-after post-stimulus evoked waveform correlation: "
    f"{held_out_waveform_correlation:.4f}"
)
print(f"Held-out evoked GFP gain (after / before): {held_out_gfp_gain:.4f}")

# %%
# Inspect the held-out evoked result
# ----------------------------------
# The main figure compares only the held-out evoked averages. The optional
# score curve shows the reproducibility-biased ordering used to retain the
# leading components.
plot_evoked_gfp_comparison(
    before_evoked,
    after_evoked,
    times=before_evoked.times,
    ci=None,
    labels=("Held-out input", "Held-out DSS retain"),
    x_label="Time (s)",
    y_label="Sensor RMS (T/m)",
    title="Held-out auditory evoked global field power",
    show=False,
)

plot_component_score_curve(model, mode="ratio", show=False)

# %%
# Interpretation
# --------------
# The component subspace was learned from the training trials, while both
# split-half reproducibility and the before-versus-after evoked comparison use
# held-out trials only. Increased repeatability is evidence that the selected
# subspace follows the specified bias on new trials; it is not proof that every
# retained component is neural. The before-versus-after comparison is a
# preservation control, not clean-neural ground truth. The retained count
# should be checked against the evoked endpoint and the signal of interest in
# an actual study.
