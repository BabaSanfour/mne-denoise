"""Smoke + contract tests for the ASR visualization module.

Each ``plot_asr_*`` function must: run under the Agg backend without showing,
return ``(fig, ax)`` (or ``(fig, axes)``), accept the documented inputs, and
honour ``ax=`` reuse where applicable. These are not pixel tests — they verify
the API contract and that the diagnostics plumbing is wired correctly.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mne_denoise.asr import ASR, AdaptiveASR, JugglerASR
from mne_denoise.viz import (
    plot_asr_blink_reduction,
    plot_asr_calibration_fraction,
    plot_asr_component_reconstruction,
    plot_asr_cutoff_sweep,
    plot_asr_grand_average,
    plot_asr_method_comparison,
    plot_asr_overlay,
    plot_asr_psd_comparison,
    plot_asr_repair_timeline,
    plot_asr_variance_topomap,
)

SFREQ = 250.0


@pytest.fixture()
def burst_pair():
    """Return (contaminated, clean) 8-channel arrays + a fitted ASR estimator."""
    rng = np.random.default_rng(3)
    n = 8000
    t = np.arange(n) / SFREQ
    clean = np.zeros((8, n))
    for c in range(8):
        clean[c] = 0.6 * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6.28)) + (
            0.05 * rng.standard_normal(n)
        )
    contaminated = clean.copy()
    for s in np.linspace(800, n - 600, 6).astype(int):
        spatial = rng.standard_normal(8)
        spatial /= np.linalg.norm(spatial)
        contaminated[:, s : s + 150] += 10.0 * np.outer(
            spatial, rng.standard_normal(150)
        )
    asr = ASR(sfreq=SFREQ, cutoff=10.0, picks=None, verbose=False)
    cleaned = asr.fit_transform(contaminated)
    return contaminated, np.asarray(cleaned), asr


def _assert_fig_ax(ret):
    fig, ax = ret
    assert isinstance(fig, Figure)
    # ax may be a single Axes or an array of Axes
    first = ax.ravel()[0] if hasattr(ax, "ravel") else ax
    assert isinstance(first, Axes)
    plt.close(fig)


def test_plot_asr_overlay(burst_pair):
    contaminated, cleaned, asr = burst_pair
    _assert_fig_ax(
        plot_asr_overlay(contaminated, cleaned, asr, sfreq=SFREQ, show=False)
    )


def test_plot_asr_overlay_ax_reuse(burst_pair):
    contaminated, cleaned, asr = burst_pair
    fig, ax = plt.subplots()
    out_fig, out_ax = plot_asr_overlay(
        contaminated, cleaned, asr, sfreq=SFREQ, ax=ax, show=False
    )
    assert out_ax is ax
    assert out_fig is fig
    plt.close(fig)


def test_plot_asr_cutoff_sweep():
    sweep = [
        {
            "variant": "standard",
            "cutoff": k,
            "pct_data_modified": 1.0 / k,
            "pct_variance_reduced": 0.9 / k,
        }
        for k in (1.0, 5.0, 20.0, 50.0)
    ] + [
        {
            "variant": "riemannian_windowed",
            "cutoff": k,
            "pct_data_modified": 0.8 / k,
            "pct_variance_reduced": 0.7 / k,
        }
        for k in (1.0, 5.0, 20.0, 50.0)
    ]
    _assert_fig_ax(plot_asr_cutoff_sweep(sweep, show=False))


def test_plot_asr_psd_comparison(burst_pair):
    contaminated, cleaned, _ = burst_pair
    _assert_fig_ax(
        plot_asr_psd_comparison(contaminated, cleaned, sfreq=SFREQ, show=False)
    )


def test_plot_asr_variance_topomap_bar_fallback(burst_pair):
    """No montage -> bar-chart fallback path."""
    contaminated, cleaned, _ = burst_pair
    _assert_fig_ax(plot_asr_variance_topomap(contaminated, cleaned, show=False))


def test_plot_asr_repair_timeline(burst_pair):
    _, _, asr = burst_pair
    _assert_fig_ax(plot_asr_repair_timeline(asr, show=False))


def test_plot_asr_calibration_fraction(burst_pair):
    _, _, asr = burst_pair
    juggler = JugglerASR(sfreq=SFREQ, cutoff=10.0, picks=None, verbose=False)
    juggler.fit_transform(burst_pair[0])
    _assert_fig_ax(
        plot_asr_calibration_fraction(
            [asr, juggler], labels=["standard", "juggler"], show=False
        )
    )


def test_plot_asr_component_reconstruction(burst_pair):
    _, _, asr = burst_pair
    _assert_fig_ax(plot_asr_component_reconstruction(asr, show=False))


def test_plot_asr_blink_reduction(burst_pair):
    contaminated, cleaned, _ = burst_pair
    _assert_fig_ax(
        plot_asr_blink_reduction(
            contaminated, cleaned, frontal_picks=[0, 1], show=False
        )
    )


def test_plot_asr_grand_average():
    rng = np.random.default_rng(5)
    n = 500
    before = [rng.standard_normal((4, n)) for _ in range(6)]
    after = [b * 0.7 for b in before]
    _assert_fig_ax(
        plot_asr_grand_average(before, after, times=np.arange(n) / SFREQ, show=False)
    )


def test_plot_asr_method_comparison():
    rng = np.random.default_rng(6)
    a = rng.uniform(0, 1, 12)
    b = a + rng.normal(0, 0.1, 12)
    _assert_fig_ax(
        plot_asr_method_comparison(
            a,
            b,
            label_a="standard",
            label_b="rASR",
            metric_name="correlation",
            show=False,
        )
    )


def test_plot_asr_method_comparison_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        plot_asr_method_comparison(np.zeros(5), np.zeros(6), show=False)


def test_adaptive_estimator_works_with_timeline():
    """The viz functions accept any fitted ASR-family estimator."""
    rng = np.random.default_rng(7)
    n = 6000
    t = np.arange(n) / SFREQ
    X = np.zeros((8, n))
    for c in range(8):
        X[c] = 0.6 * np.sin(2 * np.pi * 10 * t) + 0.05 * rng.standard_normal(n)
    for s in (1500, 3500):
        X[:, s : s + 150] += 8.0 * np.outer(
            rng.standard_normal(8), rng.standard_normal(150)
        )
    aasr = AdaptiveASR(
        sfreq=SFREQ, cutoff=10.0, variant="psw", picks=None, verbose=False
    )
    aasr.fit_transform(X)
    _assert_fig_ax(plot_asr_repair_timeline(aasr, show=False))
