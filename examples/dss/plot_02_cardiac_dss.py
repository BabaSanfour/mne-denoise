r"""
Removing a cardiac-locked artifact with DSS
===========================================

Can a cardiac-locked DSS component learned from one interval attenuate ECG
contamination in independent held-out data while preserving known neural
activity?

This controlled recording contains a known neural substrate, a cardiac artifact,
small background noise, and an explicit ECG channel. The cardiac-removal
operator is fitted only on the contaminated training interval. The same fitted
operator is then applied to held-out contaminated data and to the corresponding
held-out clean reference. Cardiac locking is a bias criterion, not an artifact
label: a reproducible cardiac component can also contain neural signal
:footcite:p:`sarela2005_dss,decheveigne_simon2008_spatial`.

References
----------
.. footbibliography::
"""

# %%
# Construct a controlled ECG-contaminated recording
# --------------------------------------------------
import mne
import numpy as np
from mne.preprocessing import find_ecg_events

from mne_denoise.dss import DSS, CycleAverageBias
from mne_denoise.qa import rms_change
from mne_denoise.viz import plot_evoked_gfp_comparison, plot_signal_overlay

rng = np.random.default_rng(18)
sfreq = 200.0
duration = 60.0
n_times = int(round(sfreq * duration))
times = np.arange(n_times) / sfreq
first_samp = 1_000
n_eeg = 6

neural_sources = np.vstack(
    [
        np.sin(2.0 * np.pi * 10.0 * times),
        0.7 * np.sin(2.0 * np.pi * 1.25 * times + 0.4),
        0.4 * np.sin(2.0 * np.pi * 18.0 * times + 1.1),
    ]
)
neural_topographies = rng.normal(size=(n_eeg, neural_sources.shape[0]))
neural = neural_topographies @ neural_sources
neural /= np.std(neural)

# The ECG channel contains regularly spaced QRS-like events. The neural
# substrate also contains a slow component near the cardiac rate, so the
# cardiac bias is not treated as a guaranteed artifact label.
qrs_samples = np.arange(
    int(sfreq),
    n_times - int(sfreq),
    int(round(0.8 * sfreq)),
)
qrs_offsets = np.arange(-20, 31)
qrs_shape = np.exp(-0.5 * (qrs_offsets / 3.0) ** 2)
qrs_shape -= 0.35 * np.exp(-0.5 * ((qrs_offsets - 8) / 5.0) ** 2)
ecg = np.zeros(n_times)
for sample in qrs_samples:
    ecg[sample - 20 : sample + 31] += qrs_shape

cardiac_topography = np.array([20.0, 14.0, -10.0, 7.0, -5.0, 3.0])
cardiac_artifact = np.outer(cardiac_topography, ecg)
background = 0.08 * rng.standard_normal((n_eeg, n_times))
clean_reference = neural + background
contaminated = clean_reference + cardiac_artifact

ch_names = [f"EEG {index:03d}" for index in range(1, n_eeg + 1)] + ["ECG"]
info = mne.create_info(
    ch_names,
    sfreq,
    ["eeg"] * n_eeg + ["ecg"],
)
ecg_channel = ecg + 0.01 * rng.standard_normal(n_times)
clean_recording = mne.io.RawArray(
    np.vstack([clean_reference, ecg_channel]),
    info.copy(),
    first_samp=first_samp,
    verbose=False,
)
contaminated_recording = mne.io.RawArray(
    np.vstack([contaminated, ecg_channel]),
    info.copy(),
    first_samp=first_samp,
    verbose=False,
)


def _crop_seconds(inst, start, stop):
    """Return a copied half-open temporal interval."""
    return inst.copy().crop(
        tmin=start,
        tmax=stop - 1.0 / sfreq,
    )


# %%
# Detect QRS events and split the temporal intervals
# ---------------------------------------------------
train_recording = _crop_seconds(contaminated_recording, 0.0, 30.0)
held_out_recording = _crop_seconds(contaminated_recording, 30.0, 60.0)
train_clean_recording = _crop_seconds(clean_recording, 0.0, 30.0)
held_out_clean_recording = _crop_seconds(clean_recording, 30.0, 60.0)

train_events, _, _ = find_ecg_events(
    train_recording,
    ch_name="ECG",
    verbose=False,
)
held_out_events, _, _ = find_ecg_events(
    held_out_recording,
    ch_name="ECG",
    verbose=False,
)

train_contaminated = train_recording.copy().pick("eeg", exclude="bads")
held_out_contaminated = held_out_recording.copy().pick("eeg", exclude="bads")
train_reference = train_clean_recording.copy().pick("eeg", exclude="bads")
held_out_reference = held_out_clean_recording.copy().pick("eeg", exclude="bads")

# The bias window is defined in seconds, while the event samples returned by
# MNE are acquisition-numbered. first_samp maps them to this cropped Raw.
bias = CycleAverageBias(
    event_samples=train_events[:, 0],
    window=(-0.10, 0.20),
    window_unit="seconds",
    sfreq=sfreq,
    event_origin="raw",
    first_samp=train_contaminated.first_samp,
)
n_components = 4
n_select = 1
model = DSS(
    bias=bias,
    n_components=n_components,
    n_select=n_select,
    component_action="subtract",
    normalize_input=False,
    verbose=False,
)
model.fit(train_contaminated)
cleaned_held_out = model.transform(held_out_contaminated)
cleaned_held_out_reference = model.transform(held_out_reference)


# %%
# Evaluate held-out artifact attenuation and clean-input preservation
# -------------------------------------------------------------------
def _flattened_correlation(first, second):
    """Return correlation after flattening channels and time."""
    first = np.asarray(first, dtype=float).ravel()
    second = np.asarray(second, dtype=float).ravel()
    return float(np.corrcoef(first, second)[0, 1])


def _event_locked_average(inst, event_samples, window_samples):
    """Average complete event windows using public MNE data access."""
    data = inst.get_data()
    relative_events = np.asarray(event_samples, dtype=int) - int(inst.first_samp)
    start, stop = window_samples
    windows = [data[:, event + start : event + stop] for event in relative_events]
    return np.mean(np.stack(windows), axis=0)


def _event_window_mask(inst, event_samples, window_samples):
    """Mark event windows for the diagnostic overlay."""
    mask = np.zeros(inst.n_times, dtype=bool)
    relative_events = np.asarray(event_samples, dtype=int) - int(inst.first_samp)
    start, stop = window_samples
    for event in relative_events:
        left = max(0, event + start)
        right = min(inst.n_times, event + stop)
        mask[left:right] = True
    return mask


evaluation_window = (
    int(round(-0.10 * sfreq)),
    int(round(0.20 * sfreq)),
)
locked_times = np.arange(*evaluation_window) / sfreq
before_locked = _event_locked_average(
    held_out_contaminated,
    held_out_events[:, 0],
    evaluation_window,
)
after_locked = _event_locked_average(
    cleaned_held_out,
    held_out_events[:, 0],
    evaluation_window,
)

before_locked_rms = np.sqrt(np.mean(before_locked**2))
after_locked_rms = np.sqrt(np.mean(after_locked**2))
attenuation_db = 20.0 * np.log10(before_locked_rms / after_locked_rms)

reference_data = held_out_reference.get_data()
cleaned_reference_data = cleaned_held_out_reference.get_data()
reference_scale = np.sqrt(np.mean(reference_data**2))
clean_input_relative_change = (
    rms_change(reference_data, cleaned_reference_data) / reference_scale
)
clean_input_waveform_correlation = _flattened_correlation(
    reference_data,
    cleaned_reference_data,
)
clean_input_retained_power = np.sum(cleaned_reference_data**2) / np.sum(
    reference_data**2
)

event_mask = _event_window_mask(
    held_out_contaminated,
    held_out_events[:, 0],
    evaluation_window,
)
held_start = int(round(30.0 * sfreq))
held_stop = int(round(60.0 * sfreq))
cardiac_channel_index = int(
    np.argmax(np.sqrt(np.mean(cardiac_artifact[:, held_start:held_stop] ** 2, axis=1)))
)
cardiac_channel = held_out_contaminated.ch_names[cardiac_channel_index]

train_duration = (
    train_contaminated.times[-1] - train_contaminated.times[0] + 1.0 / sfreq
)
held_out_duration = (
    held_out_contaminated.times[-1] - held_out_contaminated.times[0] + 1.0 / sfreq
)
print("Held-out cardiac DSS")
print(f"Training duration: {train_duration:.3f} s")
print(f"Held-out duration: {held_out_duration:.3f} s")
print(f"Training QRS event count: {len(train_events)}")
print(f"Held-out QRS event count: {len(held_out_events)}")
print(f"n_components: {n_components}")
print(f"n_select: {n_select}")
print(f"Held-out QRS-locked RMS attenuation: {attenuation_db:.2f} dB")
print(f"Clean-input relative RMS change: {clean_input_relative_change:.4f}")
print(f"Clean-input waveform correlation: {clean_input_waveform_correlation:.4f}")
print(f"Clean-input retained-power ratio: {clean_input_retained_power:.4f}")

# %%
# Inspect the held-out QRS-locked result
# --------------------------------------
plot_evoked_gfp_comparison(
    before_locked,
    after_locked,
    times=locked_times,
    ci=None,
    labels=("Held-out contaminated", "Held-out DSS subtraction"),
    x_label="Time from QRS (s)",
    y_label="Sensor RMS (a.u.)",
    title="Held-out QRS-locked global field power",
    show=False,
)

first_event_time = (held_out_events[0, 0] - held_out_contaminated.first_samp) / sfreq
plot_signal_overlay(
    held_out_contaminated,
    cleaned_held_out,
    held_out_contaminated.times,
    pick=cardiac_channel,
    start=max(0.0, first_event_time - 0.35),
    stop=min(held_out_contaminated.times[-1], first_event_time + 0.50),
    scale_after=False,
    before_label="held-out contaminated",
    after_label="DSS cardiac subtraction",
    reference=reference_data[cardiac_channel_index],
    reference_label="held-out clean reference",
    highlight_mask=event_mask,
    highlight_label="QRS-locked evaluation window",
    x_label="Time in held-out interval (s)",
    y_label="Amplitude (a.u.)",
    title=f"Cardiac-locked subtraction at {cardiac_channel}",
    show=False,
)

# %%
# Interpretation
# --------------
# The artifact endpoint uses QRS events from the held-out interval, whereas the
# preservation values come from applying the same fitted model to the held-out
# clean-reference substrate. This is an operator-control measurement, not
# evidence that a high correlation proves neural preservation. A component
# reproducible at cardiac events can contain neural signal as well as cardiac
# contamination, so the selected component count remains a scientific choice.
