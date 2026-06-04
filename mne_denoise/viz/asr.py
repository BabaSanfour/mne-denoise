"""ASR-specific visualization diagnostics.

These helpers cover the few diagnostics that are intrinsic to Artifact
Subspace Reconstruction (per-window repair timeline, calibration / reference
fraction, and the component variance-vs-threshold map) and have no generic
equivalent. For before/after signal overlays, PSD comparison, per-channel
power-ratio topographies, grand averages, and metric scatters, use the generic
:mod:`mne_denoise.viz` helpers (``plot_signal_overlay``, ``plot_psd_comparison``,
``plot_power_ratio_map``, ``plot_grand_average_evokeds``, ``plot_tradeoff_scatter``)
directly -- they work on any denoiser's input/output.
"""

from __future__ import annotations

import numpy as np

from .theme import (
    COLORS,
    get_series_color,
    style_axes,
    themed_figure,
)

try:  # pragma: no cover - matplotlib is a hard dependency of the viz package
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

try:
    import mne
except ImportError:  # pragma: no cover - MNE is a required project dependency
    mne = None

__all__ = [
    "plot_asr_repair_timeline",
    "plot_asr_calibration_fraction",
    "plot_asr_component_reconstruction",
]


# ---------------------------------------------------------------------------
# Shared output helper
# ---------------------------------------------------------------------------


def _finish(fig, ax, *, show: bool, fname: str | None):
    """Apply the save/show convention shared by every plot function."""
    if fname is not None:
        fig.savefig(fname, dpi=fig.dpi, bbox_inches="tight")
    if show and plt is not None:
        plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# 5. Repair timeline
# ---------------------------------------------------------------------------


def plot_asr_repair_timeline(
    estimator,
    *,
    title: str | None = None,
    ax=None,
    show: bool = True,
    fname: str | None = None,
):
    """Per-window count of reconstructed components over time.

    Audits whether ASR surgically repaired brief bursts (good) or modified the
    whole recording (over-cleaning). Reads the fitted estimator's
    ``diagnostics_``.

    Parameters
    ----------
    estimator : ASR | AdaptiveASR | JugglerASR
        A fitted estimator that has run ``transform``.
    title, ax, show, fname
        Standard controls.

    Returns
    -------
    fig, ax
    """
    diag = getattr(estimator, "diagnostics_", None)
    sfreq = float(getattr(estimator, "sfreq_", 0.0) or 0.0)
    if not diag or sfreq <= 0:
        raise ValueError("estimator has no transform diagnostics; run transform first.")
    starts = np.asarray(diag.get("window_starts", []), dtype=float)
    stops = np.asarray(diag.get("window_stops", []), dtype=float)
    counts = np.asarray(diag.get("n_components_reconstructed", []), dtype=float)
    if starts.size == 0:
        raise ValueError("diagnostics contain no processing windows.")
    centers = (starts + stops) / 2.0 / sfreq

    if ax is None:
        fig, ax = themed_figure(figsize=(11, 3.2))
    else:
        fig = ax.figure
    ax.fill_between(centers, counts, step="mid", color=COLORS["primary"], alpha=0.5)
    ax.plot(centers, counts, color=COLORS["primary"], lw=0.8, drawstyle="steps-mid")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Components reconstructed")
    frac = float(np.mean(counts > 0)) * 100 if counts.size else 0.0
    ax.set_title(title or f"ASR repair timeline ({frac:.0f}% of windows modified)")
    style_axes(ax, grid=True)
    fig.tight_layout()
    return _finish(fig, ax, show=show, fname=fname)


# ---------------------------------------------------------------------------
# 6. Calibration / reference fraction
# ---------------------------------------------------------------------------


def plot_asr_calibration_fraction(
    estimators,
    *,
    labels=None,
    title: str | None = None,
    ax=None,
    show: bool = True,
    fname: str | None = None,
):
    """Bar chart of the clean-window / reference-sample fraction per estimator.

    Validates that calibration is sane (Kim 2025 Fig 8): too small a fraction
    means the cutoff or data quality is wrong.

    Parameters
    ----------
    estimators : fitted estimator | sequence of fitted estimators
        Each contributes one bar from its ``calibration_info_``.
    labels : sequence of str, optional
        Bar labels; defaults to the estimator class names.
    title, ax, show, fname
        Standard controls.

    Returns
    -------
    fig, ax
    """
    if not isinstance(estimators, list | tuple):
        estimators = [estimators]
    if labels is None:
        labels = [type(e).__name__ for e in estimators]

    fracs = []
    for e in estimators:
        info = getattr(e, "calibration_info_", {}) or {}
        n_sel = info.get("reference_selected_samples")
        n_cand = info.get("reference_candidate_samples")
        if n_sel is not None and n_cand:
            fracs.append(100.0 * n_sel / n_cand)
            continue
        n_clean = info.get("n_clean_windows")
        n_tot = info.get("n_calibration_windows")
        if n_clean is not None and n_tot:
            fracs.append(100.0 * n_clean / n_tot)
        else:
            fracs.append(np.nan)

    if ax is None:
        fig, ax = themed_figure(figsize=(1.6 * len(labels) + 2, 4.2))
    else:
        fig = ax.figure
    x = np.arange(len(labels))
    ax.bar(x, fracs, color=[get_series_color(i) for i in range(len(labels))])
    for xi, f in zip(x, fracs):
        if np.isfinite(f):
            ax.text(xi, f, f"{f:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Calibration fraction (%)")
    ax.set_title(title or "ASR calibration / reference fraction")
    style_axes(ax, grid=True)
    fig.tight_layout()
    return _finish(fig, ax, show=show, fname=fname)


# ---------------------------------------------------------------------------
# 7. Component-reconstruction map
# ---------------------------------------------------------------------------


def plot_asr_component_reconstruction(
    estimator,
    *,
    title: str | None = None,
    ax=None,
    show: bool = True,
    fname: str | None = None,
):
    """Heatmap of per-window component variances vs thresholds.

    Shows which principal components crossed the rejection threshold in each
    processing window (the rows that are "hot" are the ones ASR reconstructed).

    Parameters
    ----------
    estimator : ASR | AdaptiveASR | JugglerASR
        A fitted estimator that has run ``transform``.
    title, ax, show, fname
        Standard controls.

    Returns
    -------
    fig, ax
    """
    diag = getattr(estimator, "diagnostics_", None)
    sfreq = float(getattr(estimator, "sfreq_", 0.0) or 0.0)
    if not diag:
        raise ValueError("estimator has no transform diagnostics; run transform first.")
    cv = np.asarray(diag.get("component_variances", []), dtype=float)
    ct = np.asarray(diag.get("component_thresholds", []), dtype=float)
    if cv.ndim != 2 or cv.size == 0:
        raise ValueError("diagnostics lack per-window component_variances.")
    # ratio > 1 → component exceeded threshold → reconstructed
    ratio = cv / np.maximum(ct, np.finfo(float).eps)
    starts = np.asarray(diag.get("window_starts", []), dtype=float)
    extent = None
    if starts.size == cv.shape[0] and sfreq > 0:
        extent = [starts[0] / sfreq, starts[-1] / sfreq, 0, cv.shape[1]]

    if ax is None:
        fig, ax = themed_figure(figsize=(11, 3.6))
    else:
        fig = ax.figure
    im = ax.imshow(
        ratio.T,
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=2.0,
        extent=extent,
    )
    fig.colorbar(im, ax=ax, label="variance / threshold")
    ax.set_xlabel("Time (s)" if extent else "Window index")
    ax.set_ylabel("Component")
    ax.set_title(title or "ASR component reconstruction map")
    fig.tight_layout()
    return _finish(fig, ax, show=show, fname=fname)
