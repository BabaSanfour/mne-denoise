r"""
Suppressing a TMS-like muscle artifact with SSP-SIR
===================================================

Can SSP-SIR suppress a short high-amplitude TMS-like muscle artifact while
limiting distortion of a known EEG response, when the artifact spatial
direction partly overlaps the neural topography?

This is controlled contamination of a real MNE Sample EEG evoked response, not
validation on real TMS-EEG. The artifact direction is deliberately given
moderate overlap with a clean response topography. SSP-SIR combines artifact
subspace removal with forward-informed reconstruction, so attenuation and
clean-input preservation must be evaluated separately
:footcite:p:`mutanen2016_sspsir,mutanen2024_sspsir_simulation`.
The spatial-overlap interpretation is also discussed in
:footcite:p:`mutanen2022_source_artifact,hernandez_pavon2022_tms_review`.

References
----------
.. footbibliography::
"""

# %%
# Derive a compact average-referenced EEG response from MNE Sample
# ----------------------------------------------------------------
import mne
import numpy as np
from mne.datasets import sample

from mne_denoise.qa import rms_change
from mne_denoise.sspsir import SSPSIR
from mne_denoise.viz import plot_signal_overlay

sample_path = sample.data_path(update_path=False)
raw = mne.io.read_raw_fif(
    sample_path / "MEG" / "sample" / "sample_audvis_raw.fif",
    preload=True,
    verbose="ERROR",
)
raw.crop(0.0, 60.0)
events = mne.find_events(raw, stim_channel="STI 014", verbose=False)
auditory_events = events[events[:, 2] == 1]

raw.pick("eeg", exclude="bads")
raw.filter(1.0, 45.0, verbose="ERROR")
epochs = mne.Epochs(
    raw,
    auditory_events,
    event_id={"Auditory/Left": 1},
    tmin=-0.10,
    tmax=0.30,
    baseline=(None, 0.0),
    preload=True,
    reject=None,
    verbose=False,
)
reference = epochs.average()
reference.set_eeg_reference("average", projection=False, verbose="ERROR")
reference_data = reference.get_data()
sfreq = float(reference.info["sfreq"])
times = reference.times
n_channels = len(reference.ch_names)

forward = mne.read_forward_solution(
    sample_path / "MEG" / "sample" / "sample_audvis-meg-eeg-oct-6-fwd.fif",
    verbose="ERROR",
)

# %%
# Plant one rank-one artifact with known moderate spatial overlap
# ----------------------------------------------------------------
response_window = (0.08, 0.13)
response_mask = (times >= response_window[0]) & (times <= response_window[1])
response_indices = np.flatnonzero(response_mask)
response_gfp = np.sqrt(np.mean(reference_data[:, response_mask] ** 2, axis=0))
response_peak_index = int(response_indices[np.argmax(response_gfp)])
neural_topography = reference_data[:, response_peak_index].copy()
neural_topography -= neural_topography.mean()
neural_topography /= np.linalg.norm(neural_topography)

rng = np.random.default_rng(2024)
orthogonal = rng.standard_normal(n_channels)
orthogonal -= orthogonal.mean()
orthogonal -= neural_topography * np.dot(neural_topography, orthogonal)
orthogonal -= orthogonal.mean()
orthogonal /= np.linalg.norm(orthogonal)

configured_overlap = 0.5
artifact_topography = (
    configured_overlap * neural_topography
    + np.sqrt(1.0 - configured_overlap**2) * orthogonal
)
artifact_topography -= artifact_topography.mean()
artifact_topography /= np.linalg.norm(artifact_topography)
actual_overlap = float(
    np.dot(artifact_topography, neural_topography)
    / (np.linalg.norm(artifact_topography) * np.linalg.norm(neural_topography))
)

high_pass = 100.0
# Choose the burst frequency after reading the Sample sampling frequency. This
# construction is safely above high_pass and below the Sample Nyquist rate.
artifact_frequency = 0.5 * (high_pass + sfreq / 2.0)
artifact_mask = (times >= 0.0) & (times <= 0.05)
artifact_envelope = np.exp(-0.5 * ((times - 0.025) / 0.009) ** 2)
artifact_waveform = np.where(
    artifact_mask,
    artifact_envelope * np.sin(2.0 * np.pi * artifact_frequency * (times - 0.025)),
    0.0,
)
unscaled_artifact = artifact_topography[:, np.newaxis] * artifact_waveform
reference_scale = np.sqrt(np.mean(reference_data**2))
unscaled_artifact_scale = np.sqrt(np.mean(unscaled_artifact[:, artifact_mask] ** 2))
requested_amplitude_ratio = 80.0
artifact = unscaled_artifact * (
    requested_amplitude_ratio * reference_scale / unscaled_artifact_scale
)
contaminated_data = reference_data + artifact
contaminated = mne.EvokedArray(
    contaminated_data,
    reference.info.copy(),
    tmin=reference.times[0],
    nave=reference.nave,
    comment="auditory reference with controlled TMS-like artifact",
    verbose=False,
)
artifact_reference_amplitude_ratio = (
    np.sqrt(np.mean(artifact[:, artifact_mask] ** 2)) / reference_scale
)

# %%
# Fit once on contaminated data and transform both substrates
# -------------------------------------------------------------
model = SSPSIR(
    n_components=1,
    forward=forward,
    art_window=(0.0, 0.05),
    high_pass=high_pass,
    verbose=False,
)
model.fit(contaminated)
cleaned_contaminated = model.transform(contaminated)
cleaned_reference = model.transform(reference)


# %%
# Evaluate artifact attenuation and multiple preservation controls
# ----------------------------------------------------------------
def _flattened_correlation(first, second):
    """Return correlation after flattening channels and time."""
    first = np.asarray(first, dtype=float).ravel()
    second = np.asarray(second, dtype=float).ravel()
    return float(np.corrcoef(first, second)[0, 1])


cleaned_contaminated_data = cleaned_contaminated.get_data()
cleaned_reference_data = cleaned_reference.get_data()
artifact_before = contaminated_data - reference_data
artifact_after = cleaned_contaminated_data - cleaned_reference_data
artifact_before_rms = np.sqrt(np.mean(artifact_before[:, artifact_mask] ** 2))
artifact_after_rms = np.sqrt(np.mean(artifact_after[:, artifact_mask] ** 2))
artifact_residual_ratio = artifact_after_rms / artifact_before_rms
artifact_attenuation_db = 20.0 * np.log10(1.0 / artifact_residual_ratio)

artifact_window_clean_change = (
    rms_change(
        reference_data[:, artifact_mask],
        cleaned_reference_data[:, artifact_mask],
    )
    / reference_scale
)
late_mask = (times >= response_window[0]) & (times <= response_window[1])
late_reference_scale = np.sqrt(np.mean(reference_data[:, late_mask] ** 2))
late_clean_change = (
    rms_change(
        reference_data[:, late_mask],
        cleaned_reference_data[:, late_mask],
    )
    / late_reference_scale
)
late_waveform_correlation = _flattened_correlation(
    reference_data[:, late_mask],
    cleaned_reference_data[:, late_mask],
)
late_gfp_gain = (
    np.sqrt(np.mean(cleaned_reference_data[:, late_mask] ** 2)) / late_reference_scale
)

reference_peak = reference_data[:, response_peak_index]
cleaned_reference_peak = cleaned_reference_data[:, response_peak_index]
peak_topography_correlation = _flattened_correlation(
    reference_peak,
    cleaned_reference_peak,
)
peak_topography_gain = np.linalg.norm(cleaned_reference_peak) / np.linalg.norm(
    reference_peak
)

representative_channel_index = int(np.argmax(np.abs(reference_peak)))
representative_channel = reference.ch_names[representative_channel_index]
print("Controlled TMS-like artifact with SSP-SIR")
print(f"Sample EEG channel count: {n_channels}")
print(f"Sample evoked sampling frequency: {sfreq:.6f} Hz")
print("Artifact window: (0.000, 0.050) s")
print(f"Artifact burst frequency: {artifact_frequency:.3f} Hz")
print(f"Artifact/reference amplitude ratio: {artifact_reference_amplitude_ratio:.3f}")
print(f"Configured topographic overlap: {configured_overlap:.3f}")
print(f"Actual topographic cosine similarity: {actual_overlap:.4f}")
print(f"n_components_: {model.n_components_}")
print(f"M_: {model.M_}")
print(
    "First artifact singular values: "
    f"{np.array2string(model.singular_values_[:4], precision=3)}"
)
print(f"Artifact residual ratio: {artifact_residual_ratio:.4f}")
print(f"Artifact attenuation: {artifact_attenuation_db:.2f} dB")
print(
    "Artifact-window clean-input relative RMS change: "
    f"{artifact_window_clean_change:.4f}"
)
print(f"Late-response clean-input relative RMS change: {late_clean_change:.4f}")
print(f"Late-response waveform correlation: {late_waveform_correlation:.4f}")
print(f"Late-response GFP gain: {late_gfp_gain:.4f}")
print(f"Response-peak topography correlation: {peak_topography_correlation:.4f}")
print(f"Response-peak topography gain: {peak_topography_gain:.4f}")
print(f"Representative channel: {representative_channel}")
print(
    "Representative-channel rule: largest absolute clean response at the selected peak"
)

# %%
# Inspect the signal recovery and removed spatial direction
# -----------------------------------------------------------
plot_signal_overlay(
    contaminated,
    cleaned_contaminated,
    times,
    pick=representative_channel,
    start=-0.02,
    stop=0.18,
    scale_after=False,
    before_label="contaminated",
    after_label="SSP-SIR",
    reference=reference_data[representative_channel_index],
    reference_label="clean reference",
    highlight_mask=artifact_mask,
    highlight_label="TMS-like artifact window",
    x_label="Time (s)",
    y_label="EEG amplitude (V)",
    title=f"SSP-SIR at {representative_channel}",
    show=False,
)

mne.viz.plot_projs_topomap(
    model.projs_,
    reference.info,
    show=False,
)

# %%
# Interpretation
# --------------
# The real Sample evoked response is used only as a clean methodological EEG
# substrate; this is not a validation on real TMS-EEG. The planted artifact is
# rank one, which is why n_components=1 is sufficient for this construction.
# Its direction has deliberate moderate overlap with a real response
# topography, so SSP-SIR can attenuate the artifact while changing the clean
# response. Source-informed reconstruction reduces the signal loss of pure
# projection but does not guarantee distortion-free recovery. Greater
# artifact/neural topographic similarity makes preservation more difficult,
# consistent with the simulation evidence cited above.
