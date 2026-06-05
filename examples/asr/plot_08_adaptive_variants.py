r"""
Adaptive ASR variants (PSP / PSW / MW).
=======================================

Adaptive ASR (Tsai et al.) keeps standard ASR's burst reconstruction but lets
the clean-subspace model **track non-stationary recordings**. ``mne-denoise``
exposes three calibration rules via :class:`~mne_denoise.asr.AdaptiveASR`:

- ``variant="psp"`` -- plasticity-stabilized (Hebbian) similarity matching;
- ``variant="psw"`` -- plasticity-stabilized whitening (anti-Hebbian);
- ``variant="mw"`` -- moving-window calibration (one calibration per window).

This example compares all three on a recording whose artifact statistics change
half-way through, and plots the moving-window adaptation trajectory. (For the
streaming ``fit`` / ``partial_fit`` / ``transform`` mechanics, see
``plot_03_adaptive_asr.py``.)

References
----------
.. [1] Tsai, B.-Y., et al. Adaptive Artifact Subspace Reconstruction based on
   Hebbian/anti-Hebbian learning networks for enhancing BCI performance.
   (AASR reference implementation.)
.. [2] Pehlevan, C., & Chklovskii, D. B. (2019). Neuroscience-inspired online
   unsupervised learning algorithms. IEEE Signal Processing Magazine, 36(6).
"""

# %%
# Non-stationary synthetic data
# -----------------------------
# Oscillatory brain background + bursts whose amplitude doubles in the second
# half, so a static calibration is sub-optimal and adaptation matters.
import matplotlib.pyplot as plt
import numpy as np

from mne_denoise.asr import AdaptiveASR

rng = np.random.default_rng(11)
sfreq = 200.0
n_channels, n_times = 8, 9000  # 45 s
t = np.arange(n_times) / sfreq
half = n_times // 2

brain = np.zeros((n_channels, n_times))
for ch in range(n_channels):
    phase = rng.uniform(0, 2 * np.pi)
    brain[ch] = 0.6 * np.sin(2 * np.pi * 10.0 * t + phase) + 0.05 * rng.standard_normal(
        n_times
    )

contaminated = brain.copy()
for start in np.arange(400, n_times - 400, 600):
    amp = 6.0 if start < half else 12.0  # statistics shift at the midpoint
    spatial = rng.standard_normal(n_channels)
    spatial /= np.linalg.norm(spatial)
    contaminated[:, start : start + 200] += amp * np.outer(
        spatial, rng.standard_normal(200)
    )


# %%
# Clean with each variant
# -----------------------
# PSP/PSW are streamed (fit on the first third, partial_fit the rest) so their
# adaptive update rule is exercised; MW calibrates per window inside fit.
def stream_clean(variant):
    est = AdaptiveASR(sfreq=sfreq, cutoff=20.0, variant=variant, verbose=False)
    chunks = np.array_split(contaminated, 3, axis=1)
    est.fit(chunks[0])
    for chunk in chunks[1:]:
        est.partial_fit(chunk)
    return np.asarray(est.transform(contaminated))


def scores(cleaned):
    corr = float(np.corrcoef(cleaned.ravel(), brain.ravel())[0, 1])
    snr_before = 10 * np.log10(np.var(brain) / np.var(contaminated - brain))
    snr_after = 10 * np.log10(np.var(brain) / np.var(cleaned - brain))
    return corr, float(snr_after - snr_before)


cleaned = {"psp": stream_clean("psp"), "psw": stream_clean("psw")}

mw = AdaptiveASR(
    sfreq=sfreq, cutoff=20.0, variant="mw", mw_window_length=5.0, verbose=False
)
cleaned["mw"] = np.asarray(mw.fit_transform(contaminated))

for variant, out in cleaned.items():
    corr, dsnr = scores(out)
    print(f"  {variant}:  corr-to-clean={corr:.3f}   SNR gain={dsnr:+.1f} dB")

# %%
# Variant comparison
# ------------------
variants = list(cleaned)
corrs = [scores(cleaned[v])[0] for v in variants]
dsnrs = [scores(cleaned[v])[1] for v in variants]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
ax1.bar(variants, corrs, color="C0")
ax1.set_ylim(0, 1)
ax1.set_ylabel("correlation to clean reference")
ax1.set_title("Signal fidelity")
ax2.bar(variants, dsnrs, color="C2")
ax2.set_ylabel("SNR gain (dB)")
ax2.set_title("Artifact suppression")
fig.suptitle("Adaptive ASR variants on non-stationary data")
fig.tight_layout()

# %%
# Moving-window adaptation trajectory
# -----------------------------------
# How much the threshold matrix T changes from window to window: the spike near
# the midpoint is the MW calibration tracking the artifact-statistics shift.
passed = [d for d in mw.mw_diagnostics_ if d["status"] == "passed"]
t_mats = [d["T"] for d in passed]
deltas = [
    float(np.linalg.norm(t_mats[i] - t_mats[i - 1])) for i in range(1, len(t_mats))
]
centers = [
    (passed[i]["window_start"] + passed[i]["window_stop"]) / 2 / sfreq
    for i in range(1, len(t_mats))
]

fig2, ax = plt.subplots(figsize=(8, 4))
ax.plot(centers, deltas, "o-", color="C3")
ax.axvline(half / sfreq, color="0.5", ls="--", label="statistics shift")
ax.set_xlabel("Time (s)")
ax.set_ylabel(r"$\|T_t - T_{t-1}\|_F$")
ax.set_title("MW-ASR adaptation trajectory")
ax.legend()
fig2.tight_layout()

plt.show()
