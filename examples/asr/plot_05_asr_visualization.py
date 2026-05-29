r"""
Visualizing ASR with mne_denoise.viz
====================================

After cleaning EEG with ASR you usually want to *see* what happened: which
segments were repaired, how aggressive the cleaning was, and how the variants
compare. This example showcases the ``mne_denoise.viz`` ASR plotting helpers on
synthetic burst data, across three backends:

- standard ASR (``method="standard"``),
- per-window Riemannian ASR (``method="riemannian_windowed"``),
- Juggler GEV reference selection (``JugglerASR(strategy="gev")``).

Every ``plot_asr_*`` helper returns ``(fig, ax)``, accepts MNE objects or NumPy
arrays, and honours ``ax=`` / ``show=`` / ``fname=``.
"""

# %%
# Imports and synthetic data
# --------------------------
import matplotlib.pyplot as plt
import numpy as np

from mne_denoise.asr import ASR, JugglerASR
from mne_denoise.viz import (
    plot_asr_cutoff_sweep,
    plot_asr_method_comparison,
    plot_asr_overlay,
    plot_asr_psd_comparison,
    plot_asr_repair_timeline,
    plot_asr_variance_topomap,
)

rng = np.random.default_rng(2026)
sfreq = 250.0
n_channels, n_times = 16, 15000
t = np.arange(n_times) / sfreq

# Oscillatory "brain" background.
brain = np.zeros((n_channels, n_times))
for ch in range(n_channels):
    phase = rng.uniform(0, 2 * np.pi)
    brain[ch] = (
        0.6 * np.sin(2 * np.pi * 10.0 * t + phase)
        + 0.15 * np.sin(2 * np.pi * 6.5 * t + 0.8 * phase)
        + 0.05 * rng.standard_normal(n_times)
    )

# Inject spatially-structured high-amplitude bursts.
contaminated = brain.copy()
for start in np.linspace(1000, n_times - 800, 8).astype(int):
    spatial = rng.standard_normal(n_channels)
    spatial /= np.linalg.norm(spatial)
    temporal = rng.standard_normal(300)
    contaminated[:, start : start + 300] += 12.0 * np.outer(spatial, temporal)

# %%
# Clean with standard ASR and overlay before/after
# -------------------------------------------------
asr = ASR(sfreq=sfreq, cutoff=20.0, picks=None, verbose=False)
cleaned = np.asarray(asr.fit_transform(contaminated))

plot_asr_overlay(
    contaminated,
    cleaned,
    asr,
    pick=0,
    sfreq=sfreq,
    title="Standard ASR — channel 0 (repairs shaded)",
    show=False,
)

# %%
# Repair timeline + PSD before/after
# ----------------------------------
plot_asr_repair_timeline(asr, show=False)
plot_asr_psd_comparison(contaminated, cleaned, sfreq=sfreq, show=False)

# %%
# Per-channel variance reduction
# ------------------------------
# Without a montage this falls back to a bar chart (one bar per channel).
plot_asr_variance_topomap(contaminated, cleaned, show=False)

# %%
# Cutoff sweep across two backends
# --------------------------------
# Build a small sweep table by running each backend at several cutoffs and
# reading the fraction of windows modified + variance reduced.
sweep = []
for variant, kwargs in (
    ("standard", {}),
    ("riemannian_windowed", {"method": "riemannian_windowed"}),
):
    for k in (1.0, 5.0, 20.0, 50.0, 100.0):
        est = ASR(sfreq=sfreq, cutoff=k, picks=None, verbose=False, **kwargs)
        out = np.asarray(est.fit_transform(contaminated))
        diag = est.get_diagnostics()
        var_red = (np.var(contaminated) - np.var(out)) / np.var(contaminated)
        sweep.append(
            {
                "variant": variant,
                "cutoff": k,
                "pct_data_modified": diag.get("fraction_reconstructed_windows", 0.0),
                "pct_variance_reduced": var_red,
            }
        )

plot_asr_cutoff_sweep(sweep, title="Standard vs Riemannian-windowed", show=False)

# %%
# Method comparison: per-channel variance reduction, standard vs Riemannian
# -------------------------------------------------------------------------
rasr = ASR(
    sfreq=sfreq,
    cutoff=20.0,
    method="riemannian_windowed",
    picks=None,
    verbose=False,
)
cleaned_rasr = np.asarray(rasr.fit_transform(contaminated))

var0 = np.var(contaminated, axis=1)
red_standard = (var0 - np.var(cleaned, axis=1)) / var0 * 100
red_rasr = (var0 - np.var(cleaned_rasr, axis=1)) / var0 * 100

plot_asr_method_comparison(
    red_standard,
    red_rasr,
    label_a="standard",
    label_b="riemannian_windowed",
    metric_name="% variance reduced (per channel)",
    show=False,
)

# %%
# Juggler reference fraction for severe contamination
# ---------------------------------------------------
# JugglerASR selects calibration samples point-by-point, which survives heavy
# contamination where the standard clean-windows criterion would fail.
juggler = JugglerASR(
    sfreq=sfreq, cutoff=20.0, strategy="gev", picks=None, verbose=False
)
juggler.fit_transform(contaminated)
print(
    "Juggler GEV reference fraction: "
    f"{juggler.calibration_info_['reference_selected_fraction'] * 100:.1f}%"
)

plt.show()
