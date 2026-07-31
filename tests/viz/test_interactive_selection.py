"""Tests for the interactive component-selection GUI."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import mne
import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent

import mne_denoise.viz.interactive_selection as interactive_selection
from mne_denoise.dss import DSS, IterativeDSS
from mne_denoise.dss.denoisers import VarianceMaskDenoiser
from mne_denoise.viz import ComponentSelector, plot_component_selector
from mne_denoise.viz.theme import COLORS
from mne_denoise.zapline import ZapLine


def _fake_click(fig, ax, point=(0.5, 0.5)):
    """Simulate a left button-press at a relative point inside ``ax`` (headless)."""
    fig.canvas.draw()
    x, y = ax.transAxes.transform(point)
    event = MouseEvent("button_press_event", fig.canvas, x, y, button=1)
    fig.canvas.callbacks.process("button_press_event", event)


def _axes_for(selector, comp_idx):
    """Return a clickable axes for a component visible on the current page."""
    row = selector._row_for_comp(comp_idx)
    assert row is not None, f"component {comp_idx} is not on the current page"
    return selector._rows[row].ax_time


def _title_for(selector, comp_idx):
    """Return the row-label Text artist for a component on the current page."""
    row = selector._row_for_comp(comp_idx)
    assert row is not None, f"component {comp_idx} is not on the current page"
    return selector._rows[row].ax_topo.title


# ---------------------------------------------------------------------------
# DSS
# ---------------------------------------------------------------------------


def test_selector_builds_for_dss(fitted_dss, synthetic_data):
    """The selector builds and starts with nothing excluded for DSS."""
    sel = plot_component_selector(
        fitted_dss,
        synthetic_data,
        info=synthetic_data.info,
        picks=[0, 1, 2, 3, 4],
        show=False,
    )
    assert isinstance(sel, ComponentSelector)
    assert isinstance(sel.fig, plt.Figure)
    assert sel.excluded == []


def test_selector_click_toggles_dss(fitted_dss, synthetic_data):
    """Clicking a component toggles its exclusion and recolors the title."""
    sel = plot_component_selector(
        fitted_dss,
        synthetic_data,
        info=synthetic_data.info,
        picks=[0, 1, 2, 3, 4],
        show=False,
    )
    ax0 = _axes_for(sel, 0)

    _fake_click(sel.fig, ax0)
    assert sel.excluded == [0]
    color_excluded = _title_for(sel, 0).get_color()

    _fake_click(sel.fig, ax0)
    assert sel.excluded == []
    assert _title_for(sel, 0).get_color() != color_excluded


def test_selector_preview_updates_on_toggle(fitted_dss, synthetic_data):
    """The before/after preview lines change when a component is toggled."""
    sel = plot_component_selector(
        fitted_dss,
        synthetic_data,
        info=synthetic_data.info,
        picks=[0, 1, 2, 3, 4],
        preview=True,
        show=False,
    )
    before = sel._preview.gfp_after.get_ydata().copy()
    _fake_click(sel.fig, _axes_for(sel, 0))
    after = sel._preview.gfp_after.get_ydata()
    assert not np.allclose(before, after)


def test_selector_apply_preserves_epochs_and_restores_mean_dss(
    fitted_dss, synthetic_data
):
    """DSS reconstruction preserves Epochs metadata and the channel mean."""
    sel = plot_component_selector(
        fitted_dss,
        synthetic_data,
        info=synthetic_data.info,
        picks=[0, 1, 2, 3, 4],
        show=False,
    )
    _fake_click(sel.fig, _axes_for(sel, 0))  # exclude component 0

    cleaned = sel.apply(synthetic_data)
    assert isinstance(cleaned, mne.BaseEpochs)
    assert cleaned.ch_names == synthetic_data.ch_names
    assert np.array_equal(cleaned.events, synthetic_data.events)
    assert cleaned.event_id == synthetic_data.event_id

    original = synthetic_data.get_data()
    expected_mean = original.mean(axis=(0, 2))
    assert np.allclose(cleaned.get_data().mean(axis=(0, 2)), expected_mean, atol=1e-12)


def test_selector_full_rank_dss_no_exclusions_returns_input():
    """A complete DSS reconstruction restores the uncentered input."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 600)) + np.arange(4)[:, np.newaxis]
    dss = DSS(
        n_components=4,
        bias=lambda x: x,
        normalize_input=False,
        reg=0,
    ).fit(data)
    selector = plot_component_selector(dss, data, sfreq=100.0, show=False)
    assert selector.excluded == []
    assert np.allclose(selector.apply(), data, atol=1e-10)


def test_selector_ndarray_input_isolated_from_caller_mutation():
    """Cached state must not alias a raw ndarray the caller may later mutate."""
    rng = np.random.default_rng(11)
    data = rng.standard_normal((4, 500))
    dss = DSS(n_components=4, bias=lambda x: x, normalize_input=False).fit(data)
    selector = plot_component_selector(dss, data, sfreq=100.0, show=False)

    cached = selector._state.continuous.copy()
    data[:] = 0.0  # mutate the caller's array in place after construction
    assert np.array_equal(selector._state.continuous, cached)


def test_selector_iterative_dss_reconstructs_selected_sources():
    """IterativeDSS uses the shared DSS component-masking reconstruction."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal((5, 800)) + 2.0
    dss = IterativeDSS(
        VarianceMaskDenoiser(),
        n_components=3,
        max_iter=3,
        random_state=0,
    ).fit(data)
    selector = plot_component_selector(dss, data, sfreq=200.0, show=False)
    selector._toggle(0)
    cleaned = selector.apply()

    normalized = data / dss.channel_norms_[:, np.newaxis]
    sources = dss.filters_ @ (normalized - normalized.mean(axis=1, keepdims=True))
    expected = dss.patterns_[:, 1:] @ sources[1:]
    expected *= dss.channel_norms_[:, np.newaxis]
    expected += data.mean(axis=1, keepdims=True)
    assert np.allclose(cleaned, expected, atol=1e-10)


def test_selector_normalized_dss_apply_matches_inverse_transform():
    """A normalized DSS reconstruction stays in lockstep with inverse_transform."""
    rng = np.random.default_rng(7)
    data = rng.standard_normal((5, 900)) + np.arange(5)[:, np.newaxis]
    dss = DSS(
        n_components=5,
        bias=lambda x: x,
        normalize_input=True,
        reg=0,
    ).fit(data)
    selector = plot_component_selector(dss, data, sfreq=100.0, show=False)
    selector._toggle(1)  # partial exclusion
    selector._toggle(3)

    # Recompute the expectation through the estimator's own public API: zero the
    # excluded source rows, let inverse_transform mix + de-normalize, add the mean.
    masked = selector._state.sources.copy()
    masked[[1, 3]] = 0.0
    expected = dss.inverse_transform(masked) + data.mean(axis=1, keepdims=True)
    assert np.allclose(selector.apply(), expected, atol=1e-10)


def test_selector_linear_dss_preserves_channel_first_epochs_layout():
    """Linear DSS preserves its channel-first three-dimensional array layout."""
    rng = np.random.default_rng(2)
    data = rng.standard_normal((4, 300, 3))
    dss = DSS(n_components=3, bias=lambda x: x, normalize_input=False).fit(data)
    selector = plot_component_selector(dss, data, sfreq=100.0, show=False)
    assert selector.apply().shape == data.shape


def test_selector_iterative_dss_preserves_epochs_first_layout():
    """IterativeDSS preserves its epochs-first three-dimensional array layout."""
    rng = np.random.default_rng(3)
    data = rng.standard_normal((3, 5, 300))
    dss = IterativeDSS(
        VarianceMaskDenoiser(),
        n_components=3,
        max_iter=3,
        random_state=0,
    ).fit(data)
    selector = plot_component_selector(dss, data, sfreq=100.0, show=False)
    assert selector.apply().shape == data.shape


# ---------------------------------------------------------------------------
# ZapLine
# ---------------------------------------------------------------------------


def test_selector_builds_for_zapline(fitted_zapline, zapline_data):
    """For ZapLine, every removed component starts excluded."""
    data, _ = zapline_data
    sel = plot_component_selector(fitted_zapline, data, show=False)
    assert sel.excluded == [0, 1]  # n_remove=2


def test_selector_click_toggles_zapline(fitted_zapline, zapline_data):
    """Clicking a removed component restores it (un-excludes)."""
    data, _ = zapline_data
    sel = plot_component_selector(fitted_zapline, data, show=False)
    _fake_click(sel.fig, _axes_for(sel, 0))
    assert sel.excluded == [1]


def test_selector_apply_matches_transform_zapline(fitted_zapline, zapline_data):
    """apply() with all noise components excluded matches ZapLine.transform."""
    data, _ = zapline_data
    sel = plot_component_selector(fitted_zapline, data, show=False)
    # default already excludes all removed components
    assert np.allclose(sel.apply(data), fitted_zapline.transform(data), atol=1e-9)


def test_selector_zapline_epochs_shapes():
    """The ZapLine path handles epoched (3D) input and returns matching shapes."""
    rng = np.random.default_rng(0)
    sfreq, n_ep, n_ch, n_t = 500.0, 4, 6, 600
    t = np.arange(n_t) / sfreq
    data = rng.standard_normal((n_ep, n_ch, n_t)) * 0.5
    data += 2.0 * np.sin(2 * np.pi * 50 * t)[None, None, :]

    zap = ZapLine(sfreq=sfreq, line_freq=50.0, n_remove=1).fit(data)
    sel = plot_component_selector(zap, data, show=False)
    cleaned = sel.apply(data)
    assert cleaned.shape == (n_ep, n_ch, n_t)


def test_selector_rejects_adaptive_zapline():
    """Adaptive ZapLine has no global components and is rejected explicitly."""
    zap = ZapLine(sfreq=500.0, line_freq=50.0, adaptive=True)
    with pytest.raises(NotImplementedError, match="adaptive ZapLine"):
        plot_component_selector(zap, np.zeros((4, 1000)), show=False)


def test_selector_zapline_preserves_raw_and_unfitted_channels():
    """Applying a selection preserves Raw metadata and channels outside the fit."""
    rng = np.random.default_rng(4)
    sfreq = 500.0
    n_times = 2500
    times = np.arange(n_times) / sfreq
    eeg = rng.standard_normal((3, n_times))
    eeg += np.sin(2 * np.pi * 50 * times)
    stim = np.arange(n_times, dtype=float)[np.newaxis, :]
    info = mne.create_info(["Fz", "Cz", "Pz", "STI 014"], sfreq, ["eeg"] * 3 + ["stim"])
    raw = mne.io.RawArray(np.vstack([eeg, stim]), info, verbose=False)
    raw.set_annotations(mne.Annotations([0.1], [0.2], ["test"]))

    zap = ZapLine(sfreq=sfreq, line_freq=50.0, n_remove=1).fit(raw)
    selector = plot_component_selector(zap, raw, show=False)
    cleaned = selector.apply()

    assert isinstance(cleaned, mne.io.BaseRaw)
    assert cleaned.ch_names == raw.ch_names
    assert cleaned.annotations == raw.annotations
    assert np.array_equal(cleaned.get_data(picks=[3]), stim)


def test_selector_zapline_preserves_evoked():
    """Applying a ZapLine selection preserves Evoked type and metadata."""
    rng = np.random.default_rng(5)
    sfreq = 500.0
    n_times = 2500
    times = np.arange(n_times) / sfreq
    data = rng.standard_normal((4, n_times))
    data += np.sin(2 * np.pi * 50 * times)
    info = mne.create_info(4, sfreq, "eeg")
    evoked = mne.EvokedArray(data, info, tmin=-0.2, nave=7, comment="average")

    zap = ZapLine(sfreq=sfreq, line_freq=50.0, n_remove=1).fit(evoked)
    cleaned = plot_component_selector(zap, evoked, show=False).apply()
    assert isinstance(cleaned, mne.Evoked)
    assert cleaned.nave == evoked.nave
    assert cleaned.comment == evoked.comment
    assert cleaned.times[0] == pytest.approx(evoked.times[0])


# ---------------------------------------------------------------------------
# Options and error paths
# ---------------------------------------------------------------------------


def test_selector_without_preview(fitted_dss, synthetic_data):
    """preview=False builds the dashboard with no preview panel."""
    sel = plot_component_selector(
        fitted_dss,
        synthetic_data,
        info=synthetic_data.info,
        picks=[0, 1, 2, 3, 4],
        preview=False,
        show=False,
    )
    assert sel._preview is None
    # toggling still works (no preview to update)
    _fake_click(sel.fig, _axes_for(sel, 1))
    assert sel.excluded == [1]


def test_selector_n_components_subset(fitted_dss, synthetic_data):
    """A subset of components can be shown."""
    sel = plot_component_selector(
        fitted_dss,
        synthetic_data,
        info=synthetic_data.info,
        picks=[0, 1, 2, 3, 4],
        n_components=[0, 2],
        show=False,
    )
    assert sel._panels.indices == [0, 2]
    assert sel.n_pages == 1
    assert [sel._comp_for_row(row) for row in range(len(sel._rows))] == [0, 2]


def test_selector_picks_without_info_raises(fitted_dss, synthetic_data):
    """picks without info is rejected."""
    with pytest.raises(ValueError, match="info is required"):
        plot_component_selector(fitted_dss, synthetic_data, picks=[0, 1], show=False)


def test_selector_requires_sfreq_for_array_estimator(synthetic_data):
    """Without info or sfreq, the selector cannot resolve a sampling rate."""
    arr = np.asarray(synthetic_data.get_data())[0]  # (n_ch, n_times)
    dss = DSS(n_components=2, bias=lambda x: x).fit(arr)
    with pytest.raises(ValueError, match="sfreq is required"):
        plot_component_selector(dss, arr, info=None, sfreq=None, show=False)


@pytest.mark.parametrize("sfreq", [0, -1, np.nan])
def test_selector_rejects_invalid_sfreq(synthetic_data, sfreq):
    """Sampling frequency must be finite and positive."""
    arr = np.asarray(synthetic_data.get_data())[0]
    dss = DSS(n_components=2, bias=lambda x: x).fit(arr)
    with pytest.raises(ValueError, match="sfreq must be"):
        plot_component_selector(dss, arr, sfreq=sfreq, show=False)


@pytest.mark.parametrize("psd_fmax", [0, -1, np.nan])
def test_selector_rejects_invalid_psd_fmax(fitted_dss, synthetic_data, psd_fmax):
    """PSD upper frequency must be finite and positive."""
    with pytest.raises(ValueError, match="psd_fmax must be"):
        plot_component_selector(
            fitted_dss,
            synthetic_data,
            psd_fmax=psd_fmax,
            show=False,
        )


def test_selector_warns_and_clamps_psd_fmax_above_nyquist(fitted_dss, synthetic_data):
    """psd_fmax above Nyquist warns and is clamped instead of silently ignored."""
    nyquist = synthetic_data.info["sfreq"] / 2.0
    with pytest.warns(UserWarning, match="exceeds the Nyquist frequency"):
        sel = plot_component_selector(
            fitted_dss,
            synthetic_data,
            info=synthetic_data.info,
            picks=[0, 1, 2, 3, 4],
            psd_fmax=nyquist * 10,
            show=False,
        )
    assert sel._preview.psd_fmax == nyquist


def test_selector_rejects_invalid_times(fitted_dss, synthetic_data):
    """Explicit time coordinates must match the component time dimension."""
    with pytest.raises(ValueError, match="times must be"):
        plot_component_selector(
            fitted_dss,
            synthetic_data,
            times=np.arange(5),
            show=False,
        )

    times = synthetic_data.times.copy()
    times[0] = np.nan
    with pytest.raises(ValueError, match="times must contain only finite"):
        plot_component_selector(
            fitted_dss,
            synthetic_data,
            times=times,
            show=False,
        )


def test_selector_rejects_channel_mismatch():
    """Data passed to a selector must match the fitted sensor dimension."""
    data = np.random.default_rng(6).standard_normal((4, 500))
    dss = DSS(n_components=3, bias=lambda x: x).fit(data)
    with pytest.raises(ValueError, match="estimator expects 4"):
        plot_component_selector(
            dss,
            np.zeros((3, 500)),
            sfreq=100.0,
            show=False,
        )


def test_selector_rejects_unsupported_estimator():
    """The selector reports its supported estimator types explicitly."""
    with pytest.raises(TypeError, match="DSS, IterativeDSS, or standard ZapLine"):
        plot_component_selector(object(), np.zeros((2, 100)), sfreq=100, show=False)


def test_selector_rejects_unfitted_estimator():
    """The selector reports an unfitted supported estimator clearly."""
    dss = DSS(n_components=2, bias=lambda x: x)
    with pytest.raises(RuntimeError, match="not fitted"):
        plot_component_selector(dss, np.zeros((2, 100)), sfreq=100, show=False)


def test_selector_apply_without_data_raises(fitted_dss, synthetic_data):
    """apply() needs data when none was retained."""
    sel = plot_component_selector(
        fitted_dss,
        synthetic_data,
        info=synthetic_data.info,
        picks=[0, 1, 2, 3, 4],
        show=False,
    )
    sel.data = None
    with pytest.raises(ValueError, match="No data available"):
        sel.apply()


def test_selector_explicit_sfreq_without_info(synthetic_data):
    """An explicit sfreq is used when no info is available."""
    arr = np.asarray(synthetic_data.get_data())[0]  # (n_ch, n_times)
    dss = DSS(n_components=2, bias=lambda x: x).fit(arr)
    sel = plot_component_selector(dss, arr, info=None, sfreq=100.0, show=False)
    assert isinstance(sel, ComponentSelector)


def test_selector_subset_picks_topomap(fitted_dss, synthetic_data):
    """A picks subset smaller than the fitted channels still renders topomaps."""
    sel = plot_component_selector(
        fitted_dss,
        synthetic_data,
        info=synthetic_data.info,
        picks=[0, 1, 2, 3],
        n_components=[0],
        show=False,
    )
    assert isinstance(sel, ComponentSelector)


def test_selector_empty_components_raises(fitted_dss, synthetic_data):
    """Requesting no components is rejected."""
    with pytest.raises(ValueError, match="No components available"):
        plot_component_selector(
            fitted_dss,
            synthetic_data,
            info=synthetic_data.info,
            picks=[0, 1, 2, 3, 4],
            n_components=[],
            show=False,
        )


def test_selector_click_outside_components_is_noop(fitted_dss, synthetic_data):
    """Clicking outside any component row leaves the selection unchanged."""
    sel = plot_component_selector(
        fitted_dss,
        synthetic_data,
        info=synthetic_data.info,
        picks=[0, 1, 2, 3, 4],
        show=False,
    )
    _fake_click(sel.fig, sel._preview.ax_gfp)  # preview axis, not a component
    assert sel.excluded == []


def test_selector_apply_uses_retained_data(fitted_dss, synthetic_data):
    """apply() with no argument reuses the data the selector was built with."""
    sel = plot_component_selector(
        fitted_dss,
        synthetic_data,
        info=synthetic_data.info,
        picks=[0, 1, 2, 3, 4],
        show=False,
    )
    assert np.allclose(sel.apply().get_data(), sel.apply(synthetic_data).get_data())


def test_selector_click_uses_cached_zapline_sources(
    fitted_zapline, zapline_data, monkeypatch
):
    """Preview updates do not rerun ZapLine residual extraction."""
    data, _ = zapline_data
    selector = plot_component_selector(fitted_zapline, data, show=False)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("residual extraction was not cached")

    monkeypatch.setattr(fitted_zapline, "_get_smooth_residual", fail_if_called)
    _fake_click(selector.fig, _axes_for(selector, 0))


def test_selector_reuses_cached_preview_state(
    fitted_zapline, zapline_data, monkeypatch
):
    """Returning to a previous selection reuses its cached preview arrays."""
    data, _ = zapline_data
    selector = plot_component_selector(fitted_zapline, data, show=False)
    initial_key = frozenset(selector.excluded)
    initial_gfp, initial_psd = selector._preview_cache[initial_key]

    _fake_click(selector.fig, _axes_for(selector, 0))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("PSD preview was not cached")

    monkeypatch.setattr(interactive_selection, "psd_array_welch", fail_if_called)
    _fake_click(selector.fig, _axes_for(selector, 0))

    assert np.array_equal(selector._preview.gfp_after.get_ydata(), initial_gfp)
    assert np.array_equal(selector._preview.psd_after.get_ydata(), initial_psd)


def test_selector_zapline_unexclude_all_returns_input(fitted_zapline, zapline_data):
    """With every noise component restored, ZapLine returns the input unchanged."""
    data, _ = zapline_data
    sel = plot_component_selector(fitted_zapline, data, show=False)
    for comp in (0, 1):
        _fake_click(sel.fig, _axes_for(sel, comp))
    assert sel.excluded == []
    assert np.allclose(sel.apply(data), data, atol=1e-9)


# ---------------------------------------------------------------------------
# Paging
# ---------------------------------------------------------------------------


def _paged_dss_selector(**kwargs):
    """Build a selector whose components span more than one page."""
    rng = np.random.default_rng(5)
    data = rng.standard_normal((6, 900))
    dss = DSS(n_components=6, bias=lambda x: x, normalize_input=False).fit(data)
    kwargs.setdefault("rows_per_page", 2)
    return plot_component_selector(dss, data, sfreq=100.0, show=False, **kwargs)


def test_selector_pages_when_components_exceed_rows():
    """Components beyond one page stay reachable instead of stretching the figure."""
    sel = _paged_dss_selector()
    assert len(sel._rows) == 2
    assert sel.n_pages == 3
    assert sel.page == 0
    assert [sel._comp_for_row(r) for r in range(2)] == [0, 1]

    sel.set_page(2)
    assert [sel._comp_for_row(r) for r in range(2)] == [4, 5]


def test_selector_page_is_clamped_to_valid_range():
    """Out-of-range pages clamp so scroll and key handlers need no bounds checks."""
    sel = _paged_dss_selector()
    sel.set_page(99)
    assert sel.page == sel.n_pages - 1
    sel.set_page(-5)
    assert sel.page == 0


def test_selector_figure_height_is_bounded_by_rows_per_page():
    """Figure height tracks rows_per_page, not the fitted component count."""
    tall = _paged_dss_selector(rows_per_page=2)
    taller = _paged_dss_selector(rows_per_page=6)
    assert tall.fig.get_size_inches()[1] < taller.fig.get_size_inches()[1]
    # Two rows plus the preview must stay within a laptop-sized canvas.
    assert tall.fig.get_size_inches()[1] < 6.0


def test_selector_selection_survives_paging():
    """Toggling on one page must not be lost by navigating to another."""
    sel = _paged_dss_selector()
    _fake_click(sel.fig, _axes_for(sel, 0))
    assert sel.excluded == [0]

    sel.set_page(2)
    assert sel._row_for_comp(0) is None  # component 0 is now off-page
    _fake_click(sel.fig, _axes_for(sel, 5))
    assert sel.excluded == [0, 5]

    sel.set_page(0)
    assert sel.excluded == [0, 5]


def test_selector_offpage_toggle_updates_state_without_styling():
    """A component toggled while off-page still counts toward the selection."""
    sel = _paged_dss_selector()
    sel.set_page(2)
    sel._toggle(0)  # component 0 lives on page 0
    assert sel.excluded == [0]

    sel.set_page(0)
    # Paging back re-applies the excluded styling to the now-visible row.
    assert _title_for(sel, 0).get_color() == COLORS["excluded"]


def test_selector_scroll_and_keys_change_page():
    """Scroll and PageUp/PageDown navigate; Home/End jump to the ends."""
    sel = _paged_dss_selector()
    ax = sel._rows[0].ax_time

    sel._on_scroll(SimpleNamespace(inaxes=ax, step=-1))
    assert sel.page == 1
    sel._on_scroll(SimpleNamespace(inaxes=ax, step=1))
    assert sel.page == 0

    sel._on_key(SimpleNamespace(key="pagedown"))
    assert sel.page == 1
    sel._on_key(SimpleNamespace(key="end"))
    assert sel.page == sel.n_pages - 1
    sel._on_key(SimpleNamespace(key="home"))
    assert sel.page == 0
    sel._on_key(SimpleNamespace(key="q"))  # unrelated key is a no-op
    assert sel.page == 0


def test_selector_scroll_over_preview_does_not_page():
    """The preview panels keep the wheel for themselves."""
    sel = _paged_dss_selector()
    sel._on_scroll(SimpleNamespace(inaxes=sel._preview.ax_gfp, step=-1))
    assert sel.page == 0


def test_selector_rejects_invalid_rows_per_page(fitted_dss, synthetic_data):
    """rows_per_page must be a positive integer."""
    with pytest.raises(ValueError, match="rows_per_page must be a positive integer"):
        plot_component_selector(fitted_dss, synthetic_data, rows_per_page=0, show=False)


def test_selector_trailing_rows_hidden_on_partial_last_page():
    """A short final page hides its unused row slots rather than repeating data."""
    rng = np.random.default_rng(6)
    data = rng.standard_normal((3, 600))
    dss = DSS(n_components=3, bias=lambda x: x, normalize_input=False).fit(data)
    sel = plot_component_selector(dss, data, sfreq=100.0, rows_per_page=2, show=False)

    sel.set_page(1)  # only component 2 remains
    assert sel._comp_for_row(0) == 2
    assert sel._comp_for_row(1) is None
    assert sel._rows[0].ax_time.get_visible()
    assert not sel._rows[1].ax_time.get_visible()


# ---------------------------------------------------------------------------
# Preview reference semantics
# ---------------------------------------------------------------------------


def test_preview_reference_is_the_no_exclusion_reconstruction():
    """The preview baseline is the rank-reduced reconstruction, not the raw input.

    A truncated DSS fit discards a subspace, so comparing against the raw input
    would show a large gap even with nothing excluded and hide the effect of the
    toggles the preview exists to show.
    """
    rng = np.random.default_rng(8)
    data = rng.standard_normal((8, 900))
    dss = DSS(n_components=3, bias=lambda x: x, normalize_input=False).fit(data)
    sel = plot_component_selector(dss, data, sfreq=100.0, show=False)

    assert sel.excluded == []
    reference = sel._preview.ax_gfp.lines[0].get_ydata()
    current = sel._preview.gfp_after.get_ydata()
    # With nothing excluded the two traces must coincide exactly...
    assert np.allclose(reference, current, atol=1e-12)
    # ...while the raw input keeps the discarded subspace and does not.
    raw_gfp = np.sqrt(np.mean(sel._state.continuous**2, axis=0))
    assert not np.allclose(reference, raw_gfp, atol=1e-6)


def test_preview_titles_report_the_reconstruction_rank():
    """A truncated DSS fit says so, so the scale difference is not a mystery."""
    rng = np.random.default_rng(9)
    data = rng.standard_normal((8, 900))
    dss = DSS(n_components=3, bias=lambda x: x, normalize_input=False).fit(data)
    sel = plot_component_selector(dss, data, sfreq=100.0, show=False)
    assert "3/8 comps" in sel._preview.ax_gfp.get_title()

    labels = [line.get_label() for line in sel._preview.ax_gfp.lines]
    assert labels == ["all components kept", "current selection"]


def test_preview_reference_is_the_input_for_zapline(fitted_zapline, zapline_data):
    """ZapLine's no-exclusion reconstruction is the input, so labels say so."""
    data, _ = zapline_data
    sel = plot_component_selector(fitted_zapline, data, show=False)
    reference = sel._preview.ax_gfp.lines[0].get_ydata()
    raw_gfp = np.sqrt(np.mean(data**2, axis=0))
    assert np.allclose(reference, raw_gfp, atol=1e-9)

    labels = [line.get_label() for line in sel._preview.ax_gfp.lines]
    assert labels == ["input", "cleaned"]
    assert "comps" not in sel._preview.ax_gfp.get_title()


# ---------------------------------------------------------------------------
# Row styling and status
# ---------------------------------------------------------------------------


def test_excluded_row_is_tinted_across_all_panels(fitted_dss, synthetic_data):
    """Excluding a component tints the whole row, not only its label."""
    sel = plot_component_selector(
        fitted_dss,
        synthetic_data,
        info=synthetic_data.info,
        picks=[0, 1, 2, 3, 4],
        show=False,
    )
    row = sel._rows[sel._row_for_comp(0)]
    kept_faces = [ax.patch.get_facecolor() for ax in row.axes]

    _fake_click(sel.fig, _axes_for(sel, 0))
    excluded_faces = [ax.patch.get_facecolor() for ax in row.axes]
    assert all(new != old for new, old in zip(excluded_faces, kept_faces))
    # Every panel of the row carries the same tint.
    assert len(set(excluded_faces)) == 1
    assert row.time_line.get_alpha() < 1.0

    _fake_click(sel.fig, _axes_for(sel, 0))
    assert [ax.patch.get_facecolor() for ax in row.axes] == kept_faces
    assert row.time_line.get_alpha() == 1.0


def test_status_header_stays_short_and_tracks_exclusions():
    """The header is one short line; the page position lives in the pager."""
    sel = _paged_dss_selector()
    header = sel._suptitle.get_text()
    assert "\n" not in header
    assert len(header) < 70
    assert "0 of 6 excluded" in header

    _fake_click(sel.fig, _axes_for(sel, 0))
    assert "1 of 6 excluded" in sel._suptitle.get_text()


def test_component_labels_include_the_fitted_eigenvalue(fitted_dss, synthetic_data):
    """Row labels carry the estimator's own metric, not just an index."""
    sel = plot_component_selector(
        fitted_dss,
        synthetic_data,
        info=synthetic_data.info,
        picks=[0, 1, 2, 3, 4],
        show=False,
    )
    label = sel._panels.labels[0]
    assert label.startswith("Comp 0")
    assert "λ" in label


def test_component_labels_omit_metric_when_unavailable():
    """Estimators without usable eigenvalues fall back to a bare index label."""
    rng = np.random.default_rng(10)
    data = rng.standard_normal((4, 600))
    dss = DSS(n_components=4, bias=lambda x: x, normalize_input=False).fit(data)
    dss.eigenvalues_ = None
    sel = plot_component_selector(dss, data, sfreq=100.0, show=False)
    assert sel._panels.labels == [f"Comp {i}" for i in range(4)]


# ---------------------------------------------------------------------------
# Page selector
# ---------------------------------------------------------------------------


def _click_display(fig, x, y):
    """Dispatch a left button-press at absolute display coordinates."""
    fig.canvas.draw()
    event = MouseEvent("button_press_event", fig.canvas, x, y, button=1)
    fig.canvas.callbacks.process("button_press_event", event)


def _click_text(fig, text):
    """Click the centre of a Text artist."""
    bbox = text.get_window_extent()
    _click_display(fig, (bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2)


def test_pager_offers_one_button_per_page():
    """Each page gets its own clickable button while they fit."""
    sel = _paged_dss_selector()
    sel.fig.canvas.draw()
    assert [text.get_text() for text, _ in sel._page_labels] == [
        "Page 1",
        "Page 2",
        "Page 3",
    ]


def _page_button(selector, target):
    """Return the pager Text artist that jumps to ``target``.

    Re-fetched on every use because the pager rebuilds its artists each render.
    """
    selector.fig.canvas.draw()
    return next(text for text, page in selector._page_labels if page == target)


def test_pager_click_changes_page():
    """Clicking a page button jumps straight to that page."""
    sel = _paged_dss_selector()

    _click_text(sel.fig, _page_button(sel, 2))
    assert sel.page == 2
    assert [sel._comp_for_row(r) for r in range(2)] == [4, 5]

    _click_text(sel.fig, _page_button(sel, 0))
    assert sel.page == 0


def test_pager_marks_the_current_page():
    """The active page button is visually distinct from the others."""
    sel = _paged_dss_selector()
    sel.fig.canvas.draw()
    weights = [text.get_fontweight() for text, _ in sel._page_labels]
    assert weights[0] == "bold"
    assert set(weights[1:]) == {"normal"}

    sel.set_page(1)
    weights = [text.get_fontweight() for text, _ in sel._page_labels]
    assert weights[1] == "bold"
    assert weights[0] == "normal"


def test_pager_switches_to_arrows_when_pages_are_many():
    """Beyond the button budget the pager compacts to arrows plus a counter."""
    rng = np.random.default_rng(12)
    data = rng.standard_normal((20, 900))
    dss = DSS(n_components=20, bias=lambda x: x, normalize_input=False).fit(data)
    sel = plot_component_selector(dss, data, sfreq=100.0, rows_per_page=2, show=False)
    sel.fig.canvas.draw()

    assert sel.n_pages == 10
    assert [text.get_text() for text, _ in sel._page_labels] == [
        "◀",
        "Page 1 of 10",
        "▶",
    ]

    _click_text(sel.fig, _page_button(sel, 1))  # the "next" arrow targets page 1
    assert sel.page == 1
    assert sel._page_labels[1][0].get_text() == "Page 2 of 10"


def test_pager_arrow_at_the_edge_is_inert():
    """The back arrow on page 0 points out of range and must do nothing."""
    rng = np.random.default_rng(13)
    data = rng.standard_normal((20, 900))
    dss = DSS(n_components=20, bias=lambda x: x, normalize_input=False).fit(data)
    sel = plot_component_selector(dss, data, sfreq=100.0, rows_per_page=2, show=False)
    sel.fig.canvas.draw()

    back = sel._page_labels[0][0]
    assert sel._page_labels[0][1] == -1
    _click_text(sel.fig, back)
    assert sel.page == 0


def test_pager_hidden_when_everything_fits_on_one_page(fitted_dss, synthetic_data):
    """A single-page decomposition shows no page selector at all."""
    sel = plot_component_selector(
        fitted_dss,
        synthetic_data,
        info=synthetic_data.info,
        picks=[0, 1, 2, 3, 4],
        show=False,
    )
    assert sel.n_pages == 1
    assert not sel._ax_pager.get_visible()
    assert sel._page_labels == []


# ---------------------------------------------------------------------------
# Whole-row click target
# ---------------------------------------------------------------------------


def test_click_in_row_margin_toggles_the_component(fitted_dss, synthetic_data):
    """Row margins must be clickable, not only the three panel boxes.

    ``plot_topomap`` forces an equal aspect, so the topomap axes shrinks to a
    small square and most of the row belongs to no axes at all. Those margins
    are the natural place to click and previously did nothing.
    """
    sel = plot_component_selector(
        fitted_dss,
        synthetic_data,
        info=synthetic_data.info,
        picks=[0, 1, 2, 3, 4],
        show=False,
    )
    sel.fig.canvas.draw()
    row = sel._rows[0]
    topo_bbox = row.ax_topo.bbox
    mid_y = (topo_bbox.y0 + topo_bbox.y1) / 2

    # Far left edge of the figure, level with the first row.
    _click_display(sel.fig, 5, mid_y)
    assert sel.excluded == [0]

    # The gap between the topomap and the time course.
    _click_display(sel.fig, (topo_bbox.x1 + row.ax_time.bbox.x0) / 2, mid_y)
    assert sel.excluded == []

    # The label strip above the topomap.
    _click_display(sel.fig, (topo_bbox.x0 + topo_bbox.x1) / 2, topo_bbox.y1 + 8)
    assert sel.excluded == [0]


def test_click_outside_the_component_block_is_ignored(fitted_dss, synthetic_data):
    """Clicks on the preview or the header must not toggle a component."""
    sel = plot_component_selector(
        fitted_dss,
        synthetic_data,
        info=synthetic_data.info,
        picks=[0, 1, 2, 3, 4],
        preview=True,
        show=False,
    )
    sel.fig.canvas.draw()

    gfp_bbox = sel._preview.ax_gfp.bbox
    _click_display(
        sel.fig, (gfp_bbox.x0 + gfp_bbox.x1) / 2, (gfp_bbox.y0 + gfp_bbox.y1) / 2
    )
    assert sel.excluded == []

    # Just under the suptitle, above every component row.
    _click_display(sel.fig, sel.fig.bbox.x1 / 2, sel.fig.bbox.y1 - 3)
    assert sel.excluded == []


def test_hit_area_is_registered_for_every_row(fitted_dss, synthetic_data):
    """The full-width hit axes participates in row lookup and tinting."""
    sel = plot_component_selector(
        fitted_dss,
        synthetic_data,
        info=synthetic_data.info,
        picks=[0, 1, 2, 3, 4],
        show=False,
    )
    for row, artists in enumerate(sel._rows):
        assert sel._axes_to_row[artists.ax_hit] == row
        assert not artists.ax_hit.axison

    row0 = sel._rows[0]
    _fake_click(sel.fig, _axes_for(sel, 0))
    assert row0.ax_hit.patch.get_facecolor() == row0.ax_time.patch.get_facecolor()
