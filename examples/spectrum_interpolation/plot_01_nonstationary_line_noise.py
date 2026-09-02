r"""
Removing nonstationary line noise with spectrum interpolation
===============================================================

Can spectrum interpolation attenuate 60-Hz interference whose amplitude changes
and switches abruptly while limiting distortion of a known time-domain signal?
This controlled multichannel example includes a transient in the clean
substrate, then compares spectrum interpolation with a public IIR notch filter
as a contextual reference.

Spectrum interpolation replaces amplitudes around selected Fourier frequencies
while retaining the original Fourier phase at those bins
:footcite:p:`leske_dalal2019_spectrum`. That implementation property is distinct
from a guarantee of time-domain preservation, so line attenuation and
reconstruction error are evaluated separately here. The paper's nonstationary
line-noise use case motivates the changing envelope and abrupt transitions.

This controlled example illustrates one nonstationary line-noise regime. Filter
choice and parameters should be evaluated against the recording and the
scientific signal of interest.

References
----------
.. footbibliography::
"""

# %%
# Construct a clean substrate and nonstationary line noise
# ---------------------------------------------------------
import mne
import numpy as np
from scipy import signal

from mne_denoise.qa import rms_change, suppression_ratio
from mne_denoise.spectrum_interpolation import SpectrumInterpolation
from mne_denoise.viz import plot_psd_comparison, plot_signal_overlay

sfreq = 500.0
duration = 12.0
n_channels = 6
n_times = int(round(sfreq * duration))
times = np.arange(n_times) / sfreq
rng = np.random.default_rng(20260902)

# The clean reference contains an alpha rhythm, a slower component, broadband
# background, and one short non-line-frequency transient.
channel_phases = rng.uniform(0.0, 2.0 * np.pi, n_channels)
clean_reference = np.empty((n_channels, n_times), dtype=float)
for channel, phase in enumerate(channel_phases):
    clean_reference[channel] = (
        0.35 * np.sin(2.0 * np.pi * 10.0 * times + phase)
        + 0.12 * np.sin(2.0 * np.pi * 2.5 * times + 0.4 * phase)
        + 0.03 * rng.standard_normal(n_times)
    )

transient = (
    0.7
    * np.exp(-0.5 * ((times - 2.65) / 0.035) ** 2)
    * np.sin(2.0 * np.pi * 18.0 * (times - 2.65))
)
transient_pattern = rng.standard_normal(n_channels)
transient_pattern /= np.linalg.norm(transient_pattern)
clean_reference += transient_pattern[:, None] * transient[None, :]

line_pattern = rng.standard_normal(n_channels)
line_pattern /= np.linalg.norm(line_pattern)
line_envelope = np.zeros(n_times, dtype=float)
strong_interval = (times >= 2.5) & (times < 5.5)
lower_interval = (times >= 5.5) & (times < 8.0)
line_envelope[strong_interval] = 4.0 * (
    1.0 + 0.2 * np.sin(2.0 * np.pi * 0.3 * times[strong_interval])
)
line_envelope[lower_interval] = 1.8 * (
    1.0 + 0.15 * np.sin(2.0 * np.pi * 0.2 * times[lower_interval])
)
line_source = line_envelope * np.sin(2.0 * np.pi * 60.0 * times)
line_artifact = line_pattern[:, None] * line_source[None, :]

contaminated = clean_reference + line_artifact

# %%
# Apply spectrum interpolation and a contextual public notch comparator
# ----------------------------------------------------------------------
model = SpectrumInterpolation(
    sfreq=sfreq,
    line_freq=60.0,
    n_harmonics=1,
    verbose=False,
)
cleaned_si = model.fit_transform(contaminated)

# MNE's public notch_filter is used with one explicitly defined IIR/Butterworth
# configuration. It is a contextual comparator, not a universal benchmark.
notch_config = {
    "method": "iir",
    "iir_params": {"order": 4, "ftype": "butter"},
    "notch_widths": 1.0,
    "trans_bandwidth": 1.0,
    "phase": "zero",
}
cleaned_notch = mne.filter.notch_filter(
    contaminated,
    Fs=sfreq,
    freqs=60.0,
    copy=True,
    verbose="ERROR",
    **notch_config,
)

# %%
# Evaluate line attenuation and time-domain preservation separately
# ------------------------------------------------------------------
nperseg = int(2.0 * sfreq)
freqs, psd_contaminated = signal.welch(
    contaminated, fs=sfreq, nperseg=nperseg, axis=-1
)
_, psd_si = signal.welch(cleaned_si, fs=sfreq, nperseg=nperseg, axis=-1)
_, psd_notch = signal.welch(cleaned_notch, fs=sfreq, nperseg=nperseg, axis=-1)

si_suppression = suppression_ratio(
    freqs,
    psd_contaminated,
    psd_si,
    target_freq=60.0,
    bandwidth=1.0,
)
notch_suppression = suppression_ratio(
    freqs,
    psd_contaminated,
    psd_notch,
    target_freq=60.0,
    bandwidth=1.0,
)

reference_scale = np.sqrt(np.mean(clean_reference**2))
si_error = rms_change(cleaned_si, clean_reference)
notch_error = rms_change(cleaned_notch, clean_reference)
si_relative_error = si_error / reference_scale
notch_relative_error = notch_error / reference_scale

# The interval contains both the abrupt line-noise onset and the known transient.
local_mask = (times >= 2.35) & (times <= 2.95)
local_reference_scale = np.sqrt(np.mean(clean_reference[:, local_mask] ** 2))
si_local_error = rms_change(
    cleaned_si[:, local_mask], clean_reference[:, local_mask]
) / local_reference_scale
notch_local_error = rms_change(
    cleaned_notch[:, local_mask], clean_reference[:, local_mask]
) / local_reference_scale

print(f"SpectrumInterpolation target frequencies (Hz): {model.freqs_}")
print(
    "Notch configuration: method='iir', iir_params={'order': 4, "
    "'ftype': 'butter'}, notch_widths=1.0 Hz, trans_bandwidth=1.0 Hz, "
    "phase='zero'"
)
print(f"60-Hz suppression - SpectrumInterpolation: {si_suppression:.2f} dB")
print(f"60-Hz suppression - notch comparator:       {notch_suppression:.2f} dB")
print(
    "Whole-recording relative error - SpectrumInterpolation: "
    f"{si_relative_error:.3f}"
)
print(
    "Whole-recording relative error - notch comparator:       "
    f"{notch_relative_error:.3f}"
)
print(
    "Local onset/transient relative error - SpectrumInterpolation: "
    f"{si_local_error:.3f}"
)
print(
    "Local onset/transient relative error - notch comparator:       "
    f"{notch_local_error:.3f}"
)

# %%
# Compare the spectra and inspect the transition/transient waveform
# ------------------------------------------------------------------
fig_psd = plot_psd_comparison(
    contaminated,
    cleaned_si,
    sfreq=sfreq,
    fmin=0.0,
    fmax=100.0,
    line_freq=60.0,
    show=False,
)
# The public helper expresses the main before/after comparison. Add the notch
# curve to that same axis so the three spectra are not split across figures.
psd_axis = fig_psd.axes[0]
display_mask = freqs <= 100.0
psd_axis.semilogy(
    freqs[display_mask],
    psd_notch.mean(axis=0)[display_mask],
    color="C2",
    linestyle="-.",
    label="IIR notch comparator",
)
psd_axis.legend()
psd_axis.set_title("Nonstationary 60-Hz line noise: spectral comparison")
fig_psd.tight_layout()

representative_channel = int(np.argmax(np.abs(line_pattern)))
plot_signal_overlay(
    contaminated,
    cleaned_si,
    times,
    pick=representative_channel,
    start=2.35,
    stop=2.95,
    reference=clean_reference[representative_channel],
    scale_after=False,
    before_label="contaminated",
    after_label="SpectrumInterpolation",
    reference_label="known clean reference",
    x_label="Time (s)",
    y_label="Amplitude (a.u.)",
    title="Abrupt line-noise onset and known transient",
    show=False,
)
