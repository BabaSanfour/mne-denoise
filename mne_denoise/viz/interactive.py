"""Interactive component-selection GUI for DSS / ZapLine estimators.

Provides a click-to-toggle dashboard, modelled on
:func:`mne.viz.plot_ica_components`, that plots each component's spatial
pattern, time course and PSD, lets the user click a component to exclude it
from (or restore it to) the clean output, and shows a live before/after
preview (global field power and PSD) of the reconstruction.

The interactivity is pure matplotlib (a ``button_press_event`` callback), so it
works in any interactive backend and is testable headlessly.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

import mne
import numpy as np
from matplotlib.gridspec import GridSpec
from mne.time_frequency import psd_array_welch

from ..utils import extract_data_from_mne
from ._utils import _compute_gfp, _get_components, _get_info, _get_patterns
from .components import _resolve_component_indices
from .theme import COLORS, _finalize_fig, style_axes, themed_figure

_EXCLUDED_COLOR = "#c0392b"  # red: excluded from the clean output
_KEPT_COLOR = "#2c3e50"  # dark: present in the clean output


def _is_zapline(estimator) -> bool:
    """Detect a ZapLine-style estimator (components are noise to remove)."""
    return hasattr(estimator, "_artifact_mixing_") and hasattr(estimator, "n_removed_")


def _as_2d(arr: np.ndarray, n_channels: int) -> np.ndarray:
    """Collapse a 2D/3D signal array to continuous ``(n_channels, n_times)``."""
    a = np.asarray(arr, dtype=float)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        if a.shape[1] == n_channels:  # (n_epochs, n_channels, n_times)
            return np.transpose(a, (1, 0, 2)).reshape(n_channels, -1)
        if a.shape[0] == n_channels:  # (n_channels, n_times, n_epochs)
            return np.transpose(a, (0, 2, 1)).reshape(n_channels, -1)
    raise ValueError(f"Cannot interpret array of shape {a.shape} as channel data.")


def _zapline_continuous(estimator, X):
    """Return ZapLine's continuous data plus reshape metadata."""
    data, _, mne_type, _, _, _ = extract_data_from_mne(
        X, ch_names=getattr(estimator, "_mne_ch_names_", None)
    )
    if data.ndim == 3:
        n_ep, n_ch, n_t = data.shape
        cont = np.transpose(data, (1, 0, 2)).reshape(n_ch, -1)
        return cont, ("epochs", n_ep, n_ch, n_t)
    return data, ("cont", None, data.shape[0], data.shape[1])


def _component_sources(estimator, data: np.ndarray) -> np.ndarray:
    """Component time courses, estimator-aware.

    For DSS these are the signal sources (via ``transform``); for ZapLine they
    are the artifact sources extracted from the residual, because
    ``ZapLine.transform`` returns cleaned data rather than sources.
    """
    if _is_zapline(estimator):
        cont, _ = _zapline_continuous(estimator, data)
        _, residual = estimator._get_smooth_residual(cont, warn=False)
        return estimator.filters_ @ residual
    return _get_components(estimator, data)


def _reconstruct_for_selection(estimator, data, excluded) -> np.ndarray:
    """Clean/reconstruct the sensor signal for the current ``excluded`` set.

    A component in ``excluded`` is left out of the clean output. For DSS the
    non-excluded (kept) components are reconstructed; for ZapLine the excluded
    (noise) components are subtracted from the data.
    """
    excluded = sorted({int(i) for i in excluded})
    if _is_zapline(estimator):
        cont, meta = _zapline_continuous(estimator, data)
        _, residual = estimator._get_smooth_residual(cont, warn=False)
        sources = estimator.filters_ @ residual
        if excluded:
            artifact = estimator._artifact_mixing_[:, excluded] @ sources[excluded]
            cleaned = cont - artifact
        else:
            cleaned = cont.copy()
        kind, n_ep, n_ch, n_t = meta
        if kind == "epochs":
            return cleaned.reshape(n_ch, n_ep, n_t).transpose(1, 0, 2)
        return cleaned

    # DSS: reconstruct the kept (non-excluded) components.
    sources = _get_components(estimator, data)
    n_comp = _get_patterns(estimator).shape[1]
    keep = np.ones(n_comp, dtype=bool)
    if excluded:
        keep[excluded] = False
    rec = estimator.inverse_transform(sources, component_indices=keep)
    if rec.ndim == 3 and rec.shape[0] == _n_channels(estimator):
        # (n_channels, n_times, n_epochs) -> (n_epochs, n_channels, n_times)
        rec = np.transpose(rec, (2, 0, 1))
    return rec


def _n_channels(estimator) -> int:
    """Return the number of channels the estimator reconstructs to."""
    return _get_patterns(estimator).shape[0]


def _resolve_sfreq(estimator, info, sfreq):
    if info is not None:
        return float(info["sfreq"])
    if sfreq is not None:
        return float(sfreq)
    if getattr(estimator, "sfreq", None) is not None:
        return float(estimator.sfreq)
    raise ValueError("sfreq is required when info is not available.")


class ComponentSelector:
    """Interactive component selector returned by :func:`plot_component_selector`.

    Click a component (anywhere in its row) to toggle whether it is excluded
    from the clean output; the before/after preview updates live. The current
    selection is available as :attr:`excluded`, and :meth:`apply` returns the
    clean signal for that selection.

    Parameters
    ----------
    estimator : object
        The fitted DSS- or ZapLine-style estimator.
    data : Raw | Epochs | Evoked | ndarray
        The data the components were computed from (used for the preview and
        for :meth:`apply`).
    fig : matplotlib.figure.Figure
        The dashboard figure.
    excluded : list of int
        Indices of components excluded from the clean output.

    Attributes
    ----------
    excluded : list of int
        Sorted indices currently excluded from the clean output. For DSS this
        starts empty (all components kept); for ZapLine it starts as every
        removed component.

    Notes
    -----
    A future ``plot_sources`` method (a scrollable component-activation browser,
    like :func:`mne.viz.plot_ica_sources`) can be added on this object without
    changing the existing API.
    """

    def __init__(self, estimator, data, fig, excluded):
        self.estimator = estimator
        self.data = data
        self.fig = fig
        self._excluded = {int(i) for i in excluded}
        self._axes_to_comp = {}
        self._titles = {}
        self._preview = None  # populated by the builder
        self._cid = None

    @property
    def excluded(self) -> list[int]:
        """Sorted component indices excluded from the clean output."""
        return sorted(self._excluded)

    def apply(self, data=None) -> np.ndarray:
        """Return the clean signal for the current selection.

        Parameters
        ----------
        data : Raw | Epochs | Evoked | ndarray | None
            Data to clean. If None, the data the selector was built with is
            used.

        Returns
        -------
        cleaned : ndarray
            Cleaned sensor data, shape ``(n_channels, n_times)`` for continuous
            input or ``(n_epochs, n_channels, n_times)`` for epoched input.
        """
        target = self.data if data is None else data
        if target is None:
            raise ValueError("No data available; pass data=... to apply().")
        return _reconstruct_for_selection(self.estimator, target, self._excluded)

    def _toggle(self, comp_idx: int) -> None:
        if comp_idx in self._excluded:
            self._excluded.discard(comp_idx)
        else:
            self._excluded.add(comp_idx)
        self._recolor(comp_idx)
        self._update_preview()
        self.fig.canvas.draw_idle()

    def _recolor(self, comp_idx: int) -> None:
        title = self._titles.get(comp_idx)
        if title is not None:
            excluded = comp_idx in self._excluded
            title.set_color(_EXCLUDED_COLOR if excluded else _KEPT_COLOR)

    def _on_click(self, event) -> None:
        comp_idx = self._axes_to_comp.get(event.inaxes)
        if comp_idx is None:
            return
        self._toggle(comp_idx)

    def _update_preview(self) -> None:
        if self._preview is None or self.data is None:
            return
        n_ch = _n_channels(self.estimator)
        cleaned = _reconstruct_for_selection(self.estimator, self.data, self._excluded)
        after_2d = _as_2d(cleaned, n_ch)

        gfp_after = _compute_gfp(after_2d)
        self._preview["gfp_after"].set_ydata(gfp_after)
        ax_gfp = self._preview["ax_gfp"]
        ax_gfp.relim()
        ax_gfp.autoscale_view()

        psd_after, _ = psd_array_welch(
            after_2d,
            sfreq=self._preview["sfreq"],
            fmin=0,
            fmax=self._preview["psd_fmax"],
            n_fft=self._preview["n_fft"],
            verbose=False,
        )
        psd_after = np.clip(psd_after.mean(axis=0), a_min=1e-30, a_max=None)
        self._preview["psd_after"].set_ydata(psd_after)
        ax_psd = self._preview["ax_psd"]
        ax_psd.relim()
        ax_psd.autoscale_view()


def plot_component_selector(
    estimator,
    data,
    *,
    info=None,
    picks=None,
    n_components=None,
    preview=True,
    sfreq=None,
    psd_fmax=None,
    show=True,
    fname=None,
):
    """Interactive component-selection dashboard for DSS / ZapLine.

    Builds a per-component dashboard (spatial pattern, time course, PSD) with
    click-to-toggle selection and a live before/after preview, and returns a
    :class:`ComponentSelector` exposing the selection and an ``apply`` method.

    Parameters
    ----------
    estimator : object
        A fitted DSS- or ZapLine-style estimator exposing ``patterns_`` and
        ``filters_``.
    data : Raw | Epochs | Evoked | ndarray
        Data used to compute component sources and the preview reconstruction.
    info : mne.Info | None
        Sensor metadata for topomaps. If None, it is read from the estimator
        when available; otherwise the pattern panels show a placeholder.
    picks : array-like of int | None
        Channel indices for the topomaps. Requires ``info``.
    n_components : int | sequence of int | None
        Components to display. If None, up to the first eight are shown.
    preview : bool, default=True
        If True, add a live before/after preview panel (GFP and PSD).
    sfreq : float | None
        Sampling frequency for PSDs when ``info`` is unavailable.
    psd_fmax : float | None
        Maximum frequency (Hz) shown in PSD panels.
    show : bool, default=True
        If True, show the figure.
    fname : path-like | None
        Optional path to save the figure.

    Returns
    -------
    selector : ComponentSelector
        The interactive selector. Read ``selector.excluded`` for the current
        selection and call ``selector.apply(data)`` for the clean signal.

    Raises
    ------
    ValueError
        If ``picks`` is given without ``info``, if no components are available,
        or if neither ``info`` nor ``sfreq`` can resolve a sampling frequency.
    """
    if picks is not None and info is None:
        raise ValueError("info is required when picks is provided.")
    info = _get_info(estimator, info)
    patterns = np.asarray(_get_patterns(estimator))
    sources = _component_sources(estimator, data)
    sfreq_eff = _resolve_sfreq(estimator, info, sfreq)

    n_available = patterns.shape[1]
    indices = _resolve_component_indices(n_components, n_available, default_max=8)
    if not indices:
        raise ValueError("No components available for selection.")

    if psd_fmax is None:
        psd_fmax = min(100.0, sfreq_eff / 2.0)
    psd_fmax = min(float(psd_fmax), sfreq_eff / 2.0)

    # ZapLine removes every available component by default; DSS keeps them all.
    if _is_zapline(estimator):
        excluded0 = list(range(int(getattr(estimator, "n_removed_", n_available))))
    else:
        excluded0 = []

    n_rows = len(indices) + (1 if preview else 0)
    fig, root_ax = themed_figure(figsize=(12, 2.6 * n_rows + 1))
    root_ax.remove()
    gs = GridSpec(n_rows, 3, figure=fig, width_ratios=[1, 2, 1])

    selector = ComponentSelector(estimator, data, fig, excluded0)
    times = np.arange(sources.shape[1])

    for row_idx, comp_idx in enumerate(indices):
        # --- spatial pattern -------------------------------------------------
        ax_topo = fig.add_subplot(gs[row_idx, 0])
        if picks is not None:
            topo_info = mne.pick_info(info, picks)
            if patterns.shape[0] == len(picks):
                topo_data = patterns[:, comp_idx]
            else:
                topo_data = patterns[picks, comp_idx]
            mne.viz.plot_topomap(topo_data, topo_info, axes=ax_topo, show=False)
        else:
            ax_topo.text(0.5, 0.5, "no topomap\ninfo", ha="center", va="center")
            ax_topo.set_axis_off()
        title = ax_topo.set_title(f"Comp {comp_idx}", fontweight="bold")
        selector._titles[comp_idx] = title

        # --- time course -----------------------------------------------------
        ax_time = fig.add_subplot(gs[row_idx, 1])
        if sources.ndim == 3:
            comp = sources[comp_idx]
            ax_time.plot(times, comp.mean(axis=1), color=COLORS["before"])
        else:
            ax_time.plot(times, sources[comp_idx], color=COLORS["before"])
        ax_time.set_title("Time course")
        style_axes(ax_time, grid=True)

        # --- PSD -------------------------------------------------------------
        ax_psd = fig.add_subplot(gs[row_idx, 2])
        if sources.ndim == 3:
            d_flat = sources[comp_idx].T
        else:
            d_flat = sources[comp_idx][np.newaxis, :]
        psd, freqs = psd_array_welch(
            d_flat,
            sfreq=sfreq_eff,
            fmin=0,
            fmax=psd_fmax,
            n_fft=min(2048, d_flat.shape[-1]),
            verbose=False,
        )
        ax_psd.semilogy(
            freqs, np.clip(psd.mean(axis=0), 1e-30, None), color=COLORS["primary"]
        )
        ax_psd.set_title("PSD")
        ax_psd.set_xlim(0, psd_fmax)
        style_axes(ax_psd, grid=True)

        for ax in (ax_topo, ax_time, ax_psd):
            selector._axes_to_comp[ax] = comp_idx

    if preview:
        _build_preview(
            fig, gs, len(indices), selector, data, sfreq_eff, psd_fmax, excluded0
        )

    for comp_idx in indices:
        selector._recolor(comp_idx)

    selector._cid = fig.canvas.mpl_connect("button_press_event", selector._on_click)
    fig.suptitle(
        "Click a component to toggle it (red = excluded from the clean output)",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.93, hspace=0.6, wspace=0.3)

    _finalize_fig(fig, show=show, fname=fname, tight=False)
    return selector


def _build_preview(fig, gs, preview_row, selector, data, sfreq, psd_fmax, excluded0):
    """Add the before/after GFP and PSD preview panels to the dashboard."""
    estimator = selector.estimator
    n_ch = _n_channels(estimator)

    before_2d = _as_2d(_input_array(estimator, data), n_ch)
    cleaned = _reconstruct_for_selection(estimator, data, excluded0)
    after_2d = _as_2d(cleaned, n_ch)
    n_fft = min(2048, before_2d.shape[-1])

    # --- GFP before/after ----------------------------------------------------
    ax_gfp = fig.add_subplot(gs[preview_row, 0:2])
    t = np.arange(before_2d.shape[1])
    ax_gfp.plot(t, _compute_gfp(before_2d), color=COLORS["before"], label="before")
    (gfp_after,) = ax_gfp.plot(
        t, _compute_gfp(after_2d), color=COLORS["after"], label="after"
    )
    ax_gfp.set_title("Preview: global field power")
    ax_gfp.legend(loc="upper right", fontsize=8)
    style_axes(ax_gfp, grid=True)

    # --- PSD before/after ----------------------------------------------------
    ax_psd = fig.add_subplot(gs[preview_row, 2])
    psd_b, freqs = psd_array_welch(
        before_2d, sfreq=sfreq, fmin=0, fmax=psd_fmax, n_fft=n_fft, verbose=False
    )
    psd_a, _ = psd_array_welch(
        after_2d, sfreq=sfreq, fmin=0, fmax=psd_fmax, n_fft=n_fft, verbose=False
    )
    ax_psd.semilogy(freqs, np.clip(psd_b.mean(0), 1e-30, None), color=COLORS["before"])
    (psd_after,) = ax_psd.semilogy(
        freqs, np.clip(psd_a.mean(0), 1e-30, None), color=COLORS["after"]
    )
    ax_psd.set_title("Preview: PSD")
    ax_psd.set_xlim(0, psd_fmax)
    style_axes(ax_psd, grid=True)

    selector._preview = {
        "ax_gfp": ax_gfp,
        "gfp_after": gfp_after,
        "ax_psd": ax_psd,
        "psd_after": psd_after,
        "sfreq": sfreq,
        "psd_fmax": psd_fmax,
        "n_fft": n_fft,
    }


def _input_array(estimator, X) -> np.ndarray:
    """Extract the estimator's channel data from the input as an array."""
    data, _, _, _, _, _ = extract_data_from_mne(
        X, ch_names=getattr(estimator, "_mne_ch_names_", None)
    )
    return np.asarray(data, dtype=float)
