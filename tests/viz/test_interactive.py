"""Tests for the interactive component-selection GUI."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent

from mne_denoise.dss import DSS
from mne_denoise.viz import ComponentSelector, plot_component_selector
from mne_denoise.viz._utils import _get_components
from mne_denoise.viz.interactive import _as_2d
from mne_denoise.zapline import ZapLine


def _fake_click(fig, ax, point=(0.5, 0.5)):
    """Simulate a left button-press at a relative point inside ``ax`` (headless)."""
    fig.canvas.draw()
    x, y = ax.transAxes.transform(point)
    event = MouseEvent("button_press_event", fig.canvas, x, y, button=1)
    fig.canvas.callbacks.process("button_press_event", event)


def _axes_for(selector, comp_idx):
    return next(ax for ax, c in selector._axes_to_comp.items() if c == comp_idx)


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
    color_excluded = sel._titles[0].get_color()

    _fake_click(sel.fig, ax0)
    assert sel.excluded == []
    assert sel._titles[0].get_color() != color_excluded


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
    before = sel._preview["gfp_after"].get_ydata().copy()
    _fake_click(sel.fig, _axes_for(sel, 0))
    after = sel._preview["gfp_after"].get_ydata()
    assert not np.allclose(before, after)


def test_selector_apply_matches_inverse_transform_dss(fitted_dss, synthetic_data):
    """apply() equals a direct inverse_transform with the excluded comp dropped."""
    sel = plot_component_selector(
        fitted_dss,
        synthetic_data,
        info=synthetic_data.info,
        picks=[0, 1, 2, 3, 4],
        show=False,
    )
    _fake_click(sel.fig, _axes_for(sel, 0))  # exclude component 0

    sources = _get_components(fitted_dss, synthetic_data)
    n_comp = fitted_dss.patterns_.shape[1]
    keep = np.ones(n_comp, dtype=bool)
    keep[0] = False
    expected = np.transpose(
        fitted_dss.inverse_transform(sources, component_indices=keep), (2, 0, 1)
    )
    assert np.allclose(sel.apply(synthetic_data), expected, atol=1e-9)


def test_estimator_plot_components_interactive_and_static(fitted_dss, synthetic_data):
    """DSS.plot_components returns a selector when interactive, else a figure."""
    sel = fitted_dss.plot_components(
        synthetic_data,
        interactive=True,
        preview=True,
        info=synthetic_data.info,
        picks=[0, 1, 2, 3, 4],
        show=False,
    )
    assert isinstance(sel, ComponentSelector)

    fig = fitted_dss.plot_components(
        synthetic_data, info=synthetic_data.info, picks=[0, 1, 2, 3, 4], show=False
    )
    assert isinstance(fig, plt.Figure)


def test_plot_sources_reserved(fitted_dss):
    """plot_sources is reserved for a future scrollable browser."""
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        fitted_dss.plot_sources()


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
    assert set(sel._axes_to_comp.values()) == {0, 2}


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


def test_as_2d_rejects_bad_shape():
    """_as_2d rejects arrays it cannot interpret as channel data."""
    with pytest.raises(ValueError, match="Cannot interpret"):
        _as_2d(np.zeros((2, 3, 4)), n_channels=7)


def test_as_2d_channel_first_layout():
    """_as_2d handles the (n_channels, n_times, n_epochs) layout."""
    out = _as_2d(np.zeros((3, 5, 2)), n_channels=3)
    assert out.shape == (3, 10)


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
    _fake_click(sel.fig, sel._preview["ax_gfp"])  # preview axis, not a component
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
    assert np.allclose(sel.apply(), sel.apply(synthetic_data))


def test_selector_zapline_unexclude_all_returns_input(fitted_zapline, zapline_data):
    """With every noise component restored, ZapLine returns the input unchanged."""
    data, _ = zapline_data
    sel = plot_component_selector(fitted_zapline, data, show=False)
    for comp in (0, 1):
        _fake_click(sel.fig, _axes_for(sel, comp))
    assert sel.excluded == []
    assert np.allclose(sel.apply(data), data, atol=1e-9)
