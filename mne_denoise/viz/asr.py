"""Visualization helpers for Artifact Subspace Reconstruction (ASR).

These functions render the diagnostics that the ASR literature and the
original MATLAB/Python implementations rely on:

- before/after overlays with repair spans (clean_rawdata ``vis_artifacts``),
- cutoff sweeps and calibration/reference fractions (Chang 2020; Kim 2025),
- per-channel variance topographies (Blum 2019),
- per-window repair timelines and component-reconstruction maps,
- frontal blink-amplitude reduction (Blum 2019 Fig 3),
- grand-average evoked overlays and method-comparison scatters
  (rASRMatlab analysis scripts).

Every function follows the package viz contract: it returns ``(fig, ax)`` or
``(fig, axes)``, accepts MNE objects **or** NumPy arrays where natural, honours
``ax=None`` / ``show=True`` / ``fname=None``, and uses the shared theme.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..utils import extract_data_from_mne
from .theme import (
    COLORS,
    get_series_color,
    style_axes,
    themed_figure,
    themed_legend,
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
    "plot_asr_overlay",
    "plot_asr_cutoff_sweep",
    "plot_asr_psd_comparison",
    "plot_asr_variance_topomap",
    "plot_asr_repair_timeline",
    "plot_asr_calibration_fraction",
    "plot_asr_component_reconstruction",
    "plot_asr_blink_reduction",
    "plot_asr_grand_average",
    "plot_asr_method_comparison",
]

_FRONTAL_HINTS = ("fp1", "fp2", "fpz", "af3", "af4", "af7", "af8", "afz")


# ---------------------------------------------------------------------------
# Shared input helpers
# ---------------------------------------------------------------------------


def _to_array(data: Any) -> tuple[np.ndarray, float | None, Any]:
    """Return ``(2d_array, sfreq, info)`` from an MNE object or ndarray.

    ndarray inputs are returned unchanged with ``sfreq=None`` and ``info=None``.
    """
    if mne is not None and isinstance(
        data, mne.io.BaseRaw | mne.BaseEpochs | mne.Evoked
    ):
        arr, sfreq, mne_type, orig = extract_data_from_mne(data)
        info = getattr(orig, "info", None)
        if mne_type == "epochs" and arr.ndim == 3:
            arr = np.concatenate(list(arr), axis=1)
        return np.asarray(arr, dtype=np.float64), sfreq, info
    return np.asarray(data, dtype=np.float64), None, None


def _finish(fig, ax, *, show: bool, fname: str | None):
    """Apply the save/show convention shared by every plot function."""
    if fname is not None:
        fig.savefig(fname, dpi=fig.dpi, bbox_inches="tight")
    if show and plt is not None:
        plt.show()
    return fig, ax


def _repair_spans_seconds(estimator) -> list[tuple[float, float]]:
    """Extract (onset_s, offset_s) repair spans from a fitted estimator."""
    diag = getattr(estimator, "diagnostics_", None)
    sfreq = float(getattr(estimator, "sfreq_", 0.0) or 0.0)
    if not diag or sfreq <= 0:
        return []
    starts = np.asarray(diag.get("window_starts", []), dtype=float)
    stops = np.asarray(diag.get("window_stops", []), dtype=float)
    counts = np.asarray(diag.get("n_components_reconstructed", []), dtype=float)
    if starts.size == 0 or starts.size != stops.size:
        return []
    spans = []
    for s, e, c in zip(starts, stops, counts if counts.size == starts.size else starts):
        if c >= 1:
            spans.append((float(s) / sfreq, float(e) / sfreq))
    return spans


# ---------------------------------------------------------------------------
# 1. Before/after overlay with repair spans
# ---------------------------------------------------------------------------


def plot_asr_overlay(
    before,
    after,
    estimator=None,
    *,
    pick: int | str = 0,
    sfreq: float | None = None,
    start: float = 0.0,
    duration: float | None = None,
    title: str | None = None,
    ax=None,
    show: bool = True,
    fname: str | None = None,
):
    """Overlay original vs ASR-cleaned signal for one channel, shading repairs.

    Mirrors clean_rawdata's ``vis_artifacts``: the raw trace and the cleaned
    trace are superimposed so a user can see exactly where ASR intervened.
    When a fitted ``estimator`` is supplied, the windows it reconstructed are
    shaded.

    Parameters
    ----------
    before, after : mne.io.BaseRaw | mne.BaseEpochs | ndarray, shape (n_channels, n_times)
        Original and cleaned data.
    estimator : ASR | AdaptiveASR | JugglerASR, optional
        Fitted estimator; used to shade repaired windows from its
        ``diagnostics_``.
    pick : int | str
        Channel index, or channel name (requires an MNE input).
    sfreq : float, optional
        Sampling rate (Hz). Inferred from MNE inputs / the estimator if omitted.
    start, duration : float
        Time window to display, in seconds. ``duration=None`` shows everything.
    title : str, optional
        Axes title.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw into.
    show : bool
        Call ``plt.show()`` when done.
    fname : str, optional
        If given, save the figure here.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    b_arr, b_sf, b_info = _to_array(before)
    a_arr, _, _ = _to_array(after)
    sfreq = sfreq or b_sf or float(getattr(estimator, "sfreq_", 0.0) or 0.0)
    if not sfreq:
        raise ValueError("sfreq could not be inferred; pass sfreq explicitly.")

    ch = pick
    if isinstance(pick, str):
        if b_info is None:
            raise ValueError("Channel-name pick requires an MNE input for `before`.")
        ch = b_info["ch_names"].index(pick)
    ch_label = b_info["ch_names"][ch] if b_info is not None else f"ch{ch}"

    n_times = b_arr.shape[1]
    t = np.arange(n_times) / sfreq
    i0 = int(round(start * sfreq))
    i1 = (
        n_times if duration is None else min(n_times, i0 + int(round(duration * sfreq)))
    )

    if ax is None:
        fig, ax = themed_figure(figsize=(11, 3.4))
    else:
        fig = ax.figure

    ax.plot(
        t[i0:i1],
        b_arr[ch, i0:i1],
        color=COLORS["before"],
        lw=0.7,
        label="Before",
        alpha=0.8,
    )
    ax.plot(
        t[i0:i1], a_arr[ch, i0:i1], color=COLORS["after"], lw=0.9, label="After (ASR)"
    )
    if estimator is not None:
        for onset, offset in _repair_spans_seconds(estimator):
            if offset < t[i0] or onset > t[i1 - 1]:
                continue
            ax.axvspan(onset, offset, color=COLORS["line_marker"], alpha=0.12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{ch_label} amplitude")
    ax.set_title(title or f"ASR overlay — {ch_label}")
    style_axes(ax, grid=True)
    themed_legend(ax, loc="upper right")
    fig.tight_layout()
    return _finish(fig, ax, show=show, fname=fname)


# ---------------------------------------------------------------------------
# 2. Cutoff sweep
# ---------------------------------------------------------------------------


def plot_asr_cutoff_sweep(
    sweep,
    *,
    metric_keys=("pct_data_modified", "pct_variance_reduced"),
    metric_labels=("% data modified", "% variance reduced"),
    variant_key: str = "variant",
    cutoff_key: str = "cutoff",
    title: str | None = None,
    show: bool = True,
    fname: str | None = None,
):
    """Plot ASR metrics vs cutoff, one line per variant (Chang 2020 Fig 2).

    Parameters
    ----------
    sweep : sequence of dict
        Each row has at least ``cutoff_key``, ``variant_key``, and the
        ``metric_keys``. Fractions (0-1) are auto-scaled to percent.
    metric_keys : tuple of str
        Row keys to plot — one subplot per key.
    metric_labels : tuple of str
        Y-axis labels matching ``metric_keys``.
    variant_key, cutoff_key : str
        Row keys for the variant name and cutoff value.
    title, show, fname
        Standard styling/output controls.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of matplotlib.axes.Axes
    """
    rows = [r for r in sweep if r.get("status", "passed") == "passed"]
    if not rows:
        raise ValueError("sweep contains no passed rows to plot.")
    variants = sorted({r[variant_key] for r in rows})
    n = len(metric_keys)
    fig, axes = themed_figure(1, n, figsize=(6.0 * n, 4.2), squeeze=False)
    axes = axes.ravel()
    for ax, key, label in zip(axes, metric_keys, metric_labels):
        for i, variant in enumerate(variants):
            sub = sorted(
                (r for r in rows if r[variant_key] == variant),
                key=lambda r: r[cutoff_key],
            )
            ks = [r[cutoff_key] for r in sub]
            ys = []
            for r in sub:
                v = r.get(key)
                if v is None:
                    v = np.nan
                ys.append(v * 100 if abs(v) <= 1.0 else v)
            ax.plot(ks, ys, marker="o", color=get_series_color(i), label=variant)
        ax.set_xscale("log")
        ax.set_xlabel("Cutoff k (SDs)")
        ax.set_ylabel(label)
        style_axes(ax, grid=True)
        themed_legend(ax, loc="best", fontsize=8)
    fig.suptitle(title or "ASR cutoff sweep")
    fig.tight_layout()
    return _finish(fig, axes, show=show, fname=fname)


# ---------------------------------------------------------------------------
# 3. PSD before/after
# ---------------------------------------------------------------------------


def plot_asr_psd_comparison(
    before,
    after,
    *,
    sfreq: float | None = None,
    fmin: float = 1.0,
    fmax: float = 80.0,
    title: str | None = None,
    ax=None,
    show: bool = True,
    fname: str | None = None,
):
    """Mean PSD before vs after ASR, in dB (Chang 2020 Fig 5).

    Parameters
    ----------
    before, after : mne objects | ndarray, shape (n_channels, n_times)
        Original and cleaned data.
    sfreq : float, optional
        Sampling rate; inferred from MNE inputs if omitted.
    fmin, fmax : float
        Frequency display range in Hz.
    title, ax, show, fname
        Standard controls.

    Returns
    -------
    fig, ax
    """
    from scipy import signal as _sig

    b_arr, b_sf, _ = _to_array(before)
    a_arr, _, _ = _to_array(after)
    sfreq = sfreq or b_sf
    if not sfreq:
        raise ValueError("sfreq could not be inferred; pass sfreq explicitly.")
    nperseg = int(min(b_arr.shape[1], max(256, sfreq * 2)))

    def _mean_db(arr):
        freqs, psd = _sig.welch(arr, fs=sfreq, nperseg=nperseg, axis=1)
        mean = psd.mean(axis=0)
        return freqs, 10.0 * np.log10(mean + 1e-30)

    f, db_before = _mean_db(b_arr)
    _, db_after = _mean_db(a_arr)
    band = (f >= fmin) & (f <= fmax)

    if ax is None:
        fig, ax = themed_figure(figsize=(7.5, 4.4))
    else:
        fig = ax.figure
    ax.plot(f[band], db_before[band], color=COLORS["before"], lw=1.2, label="Before")
    ax.plot(f[band], db_after[band], color=COLORS["after"], lw=1.2, label="After (ASR)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title(title or "ASR PSD comparison")
    style_axes(ax, grid=True)
    themed_legend(ax, loc="best")
    fig.tight_layout()
    return _finish(fig, ax, show=show, fname=fname)


# ---------------------------------------------------------------------------
# 4. Per-channel variance-reduction topomap
# ---------------------------------------------------------------------------


def plot_asr_variance_topomap(
    before,
    after,
    info=None,
    *,
    title: str | None = None,
    ax=None,
    show: bool = True,
    fname: str | None = None,
):
    """Per-channel % variance reduced, on the scalp (Blum 2019 / Chang Fig 6).

    Falls back to a horizontal bar chart when no usable montage is available.

    Parameters
    ----------
    before, after : mne objects | ndarray, shape (n_channels, n_times)
        Original and cleaned data. If MNE objects, their info supplies the
        montage.
    info : mne.Info, optional
        Explicit channel info (overrides the one inferred from ``before``).
    title, ax, show, fname
        Standard controls.

    Returns
    -------
    fig, ax
    """
    b_arr, _, b_info = _to_array(before)
    a_arr, _, _ = _to_array(after)
    info = info or b_info
    var_b = np.var(b_arr, axis=1)
    var_a = np.var(a_arr, axis=1)
    pct = (var_b - var_a) / np.maximum(var_b, np.finfo(float).eps) * 100.0

    positions_ok = False
    if info is not None and mne is not None:
        try:
            picks = mne.pick_types(info, eeg=True, exclude=[])
            if len(picks) == pct.size:
                info = mne.pick_info(info, picks)
            locs = np.array(
                [info["chs"][i]["loc"][:3] for i in range(len(info["chs"]))]
            )
            positions_ok = np.isfinite(locs).all() and np.linalg.norm(locs) > 0
        except Exception:  # pragma: no cover - defensive
            positions_ok = False

    if ax is None:
        fig, ax = themed_figure(figsize=(5.6, 5.0))
    else:
        fig = ax.figure

    if positions_ok:
        vmax = float(max(np.abs(pct).max(), 1.0))
        im, _ = mne.viz.plot_topomap(
            pct,
            info,
            axes=ax,
            show=False,
            cmap="RdBu_r",
            vlim=(-vmax, vmax),
            sensors=True,
            contours=6,
        )
        fig.colorbar(im, ax=ax, label="% variance reduced")
    else:
        names = (
            info["ch_names"]
            if info is not None
            else [f"ch{i}" for i in range(pct.size)]
        )
        order = np.argsort(pct)[::-1]
        ax.barh(np.array(names)[order], pct[order], color=COLORS["primary"])
        ax.invert_yaxis()
        ax.set_xlabel("% variance reduced")
        style_axes(ax, grid=True)
    ax.set_title(title or "ASR per-channel variance reduction")
    fig.tight_layout()
    return _finish(fig, ax, show=show, fname=fname)


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


# ---------------------------------------------------------------------------
# 8. Frontal blink reduction
# ---------------------------------------------------------------------------


def plot_asr_blink_reduction(
    before,
    after,
    *,
    frontal_picks=None,
    percentile: float = 95.0,
    title: str | None = None,
    ax=None,
    show: bool = True,
    fname: str | None = None,
):
    """Frontal peak-to-peak amplitude before vs after ASR (Blum 2019 Fig 3).

    Parameters
    ----------
    before, after : mne objects | ndarray, shape (n_channels, n_times)
        Original and cleaned data.
    frontal_picks : sequence of int | sequence of str, optional
        Channels to summarise. Defaults to frontal channels by name (MNE
        input) or the first two channels (ndarray input).
    percentile : float
        Percentile of the per-channel peak-to-peak distribution to report.
    title, ax, show, fname
        Standard controls.

    Returns
    -------
    fig, ax
    """
    b_arr, _, b_info = _to_array(before)
    a_arr, _, _ = _to_array(after)

    if frontal_picks is None:
        if b_info is not None:
            frontal_picks = [
                i
                for i, n in enumerate(b_info["ch_names"])
                if n.lower() in _FRONTAL_HINTS
            ] or list(range(min(2, b_arr.shape[0])))
        else:
            frontal_picks = list(range(min(2, b_arr.shape[0])))
    elif b_info is not None and all(isinstance(p, str) for p in frontal_picks):
        frontal_picks = [b_info["ch_names"].index(p) for p in frontal_picks]

    idx = np.asarray(frontal_picks, dtype=int)
    before_pp = float(np.percentile(np.ptp(b_arr[idx], axis=1), percentile))
    after_pp = float(np.percentile(np.ptp(a_arr[idx], axis=1), percentile))
    reduction = (before_pp - after_pp) / max(before_pp, np.finfo(float).eps) * 100.0

    if ax is None:
        fig, ax = themed_figure(figsize=(4.6, 4.4))
    else:
        fig = ax.figure
    ax.bar([0, 1], [before_pp, after_pp], color=[COLORS["before"], COLORS["after"]])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Before", "After (ASR)"])
    ax.set_ylabel(f"Frontal peak-to-peak (p{percentile:.0f})")
    ax.set_title(title or f"ASR blink reduction: {reduction:.0f}%")
    style_axes(ax, grid=True)
    fig.tight_layout()
    return _finish(fig, ax, show=show, fname=fname)


# ---------------------------------------------------------------------------
# 9. Grand-average evoked overlay
# ---------------------------------------------------------------------------


def plot_asr_grand_average(
    before_list,
    after_list,
    times=None,
    *,
    pick: int = 0,
    title: str | None = None,
    ax=None,
    show: bool = True,
    fname: str | None = None,
):
    """Grand-average evoked + per-subject overlay, before vs after ASR.

    Reproduces the rASRMatlab ``grandAverageBlink2`` style: thin per-subject
    traces with a bold grand average, before (grey) and after (green).

    Parameters
    ----------
    before_list, after_list : sequence of (n_channels, n_times) arrays | Evoked
        One entry per subject/condition.
    times : ndarray, optional
        Time axis (s). Defaults to sample indices.
    pick : int
        Channel index to display.
    title, ax, show, fname
        Standard controls.

    Returns
    -------
    fig, ax
    """

    def _stack(lst):
        arrs = [_to_array(x)[0] for x in lst]
        return np.stack([a[pick] for a in arrs], axis=0)

    b = _stack(before_list)
    a = _stack(after_list)
    if times is None:
        times = np.arange(b.shape[1])

    if ax is None:
        fig, ax = themed_figure(figsize=(8, 4.4))
    else:
        fig = ax.figure
    for i in range(b.shape[0]):
        ax.plot(times, b[i], color=COLORS["before"], lw=0.4, alpha=0.3)
        ax.plot(times, a[i], color=COLORS["after"], lw=0.4, alpha=0.3)
    ax.plot(times, b.mean(0), color=COLORS["before"], lw=2.2, label="Before (GA)")
    ax.plot(times, a.mean(0), color=COLORS["after"], lw=2.2, label="After (GA)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title or f"ASR grand average (n={b.shape[0]})")
    style_axes(ax, grid=True)
    themed_legend(ax, loc="best")
    fig.tight_layout()
    return _finish(fig, ax, show=show, fname=fname)


# ---------------------------------------------------------------------------
# 10. Method-comparison scatter
# ---------------------------------------------------------------------------


def plot_asr_method_comparison(
    metric_a,
    metric_b,
    *,
    label_a: str = "Method A",
    label_b: str = "Method B",
    metric_name: str = "metric",
    title: str | None = None,
    ax=None,
    show: bool = True,
    fname: str | None = None,
):
    """Per-subject/condition scatter of one metric under two methods.

    Mirrors rASRMatlab ``quickscatter``: points above the identity line favour
    method B, below favour method A.

    Parameters
    ----------
    metric_a, metric_b : array-like
        Paired per-subject metric values under each method.
    label_a, label_b : str
        Axis labels for the two methods.
    metric_name : str
        Name of the metric (used in the axis labels).
    title, ax, show, fname
        Standard controls.

    Returns
    -------
    fig, ax
    """
    a = np.asarray(metric_a, dtype=float)
    b = np.asarray(metric_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("metric_a and metric_b must have the same shape.")

    if ax is None:
        fig, ax = themed_figure(figsize=(5.0, 5.0))
    else:
        fig = ax.figure
    lo = float(np.nanmin([a.min(), b.min()]))
    hi = float(np.nanmax([a.max(), b.max()]))
    pad = 0.05 * (hi - lo + 1e-9)
    ax.plot(
        [lo - pad, hi + pad],
        [lo - pad, hi + pad],
        color=COLORS["muted"],
        lw=1.0,
        ls="--",
        label="identity",
    )
    ax.scatter(a, b, color=COLORS["accent"], s=28, alpha=0.8, edgecolor="white")
    ax.set_xlabel(f"{label_a} {metric_name}")
    ax.set_ylabel(f"{label_b} {metric_name}")
    ax.set_title(title or f"ASR method comparison ({metric_name})")
    style_axes(ax, grid=True)
    themed_legend(ax, loc="best", fontsize=8)
    fig.tight_layout()
    return _finish(fig, ax, show=show, fname=fname)
