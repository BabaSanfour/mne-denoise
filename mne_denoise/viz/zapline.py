"""ZapLine-specific visualization functions.

Provides reusable plotting utilities for ZapLine analysis and results.

Functions
---------
plot_psd_comparison
    Compare power spectral density before and after cleaning.
plot_component_scores
    Visualize DSS component eigenvalues with removal threshold.
plot_spatial_patterns
    Display spatial patterns of noise components.
plot_cleaning_summary
    Combined multi-panel summary figure.
plot_zapline_analytics
    Legacy analytics dashboard.

Authors Sina Esmaeili (sina.esmaeili@umontreal.ca)
        Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

if TYPE_CHECKING:
    pass


def plot_psd_comparison(
    data_before: np.ndarray,
    data_after: np.ndarray,
    sfreq: float,
    line_freq: float | None = None,
    fmax: float = 100.0,
    ax: plt.Axes | None = None,
    show: bool = True,
) -> plt.Axes:
    """Compare power spectral density before and after cleaning.

    Parameters
    ----------
    data_before : ndarray, shape (n_channels, n_times)
        Original data before cleaning.
    data_after : ndarray, shape (n_channels, n_times)
        Cleaned data after ZapLine.
    sfreq : float
        Sampling frequency in Hz.
    line_freq : float | None
        Line noise frequency to mark. If None, no vertical line is drawn.
    fmax : float
        Maximum frequency to display.
    ax : Axes | None
        Matplotlib axes. If None, creates new figure.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    ax : Axes
        The matplotlib axes with the plot.

    Examples
    --------
    >>> from mne_denoise.viz import plot_psd_comparison
    >>> plot_psd_comparison(data, cleaned, sfreq=1000, line_freq=50)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    nperseg = min(data_before.shape[1], int(sfreq * 2))
    freqs, psd_before = signal.welch(data_before, sfreq, nperseg=nperseg)
    _, psd_after = signal.welch(data_after, sfreq, nperseg=nperseg)

    ax.semilogy(freqs, np.mean(psd_before, axis=0), "b-", alpha=0.5, label="Before")
    ax.semilogy(freqs, np.mean(psd_after, axis=0), "g-", label="After")

    if line_freq is not None:
        ax.axvline(
            line_freq, color="r", linestyle="--", alpha=0.7, label=f"{line_freq} Hz"
        )
        # Mark harmonics if within range
        for h in range(2, 5):
            if line_freq * h < fmax:
                ax.axvline(line_freq * h, color="r", linestyle="--", alpha=0.3)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Power Spectral Density: Before vs After")
    ax.set_xlim(0, fmax)
    ax.legend()

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_component_scores(
    estimator,
    ax: plt.Axes | None = None,
    show: bool = True,
) -> plt.Axes:
    """Visualize DSS component eigenvalues with removal threshold.

    Parameters
    ----------
    estimator : ZapLine
        Fitted ZapLine estimator with eigenvalues_ attribute.
    ax : Axes | None
        Matplotlib axes. If None, creates new figure.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    ax : Axes
        The matplotlib axes with the plot.

    Examples
    --------
    >>> from mne_denoise.viz import plot_component_scores
    >>> zapline = ZapLine(sfreq=1000, line_freq=50).fit(data)
    >>> plot_component_scores(zapline)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    scores = getattr(estimator, "eigenvalues_", None)
    if scores is None or len(scores) == 0:
        ax.text(
            0.5,
            0.5,
            "No scores available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        return ax

    ax.bar(range(len(scores)), scores, color="steelblue", edgecolor="navy", alpha=0.8)
    ax.axhline(
        np.mean(scores), color="red", linestyle="--", linewidth=1.5, label="Mean"
    )

    n_removed = getattr(estimator, "n_removed_", 0)
    if n_removed > 0:
        ax.axvline(
            n_removed - 0.5,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Removed: {n_removed}",
        )

    ax.set_xlabel("Component")
    ax.set_ylabel("Score (eigenvalue)")
    ax.set_title("Component Scores")
    ax.legend()

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_spatial_patterns(
    estimator,
    n_patterns: int = 3,
    ax: plt.Axes | None = None,
    show: bool = True,
) -> plt.Axes:
    """Display spatial patterns of noise components.

    Parameters
    ----------
    estimator : ZapLine
        Fitted ZapLine estimator with patterns_ attribute.
    n_patterns : int
        Number of top patterns to display.
    ax : Axes | None
        Matplotlib axes. If None, creates new figure.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    ax : Axes
        The matplotlib axes with the plot.

    Examples
    --------
    >>> from mne_denoise.viz import plot_spatial_patterns
    >>> zapline = ZapLine(sfreq=1000, line_freq=50).fit(data)
    >>> plot_spatial_patterns(zapline, n_patterns=3)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    patterns = getattr(estimator, "patterns_", None)
    if patterns is None or patterns.size == 0:
        ax.text(
            0.5,
            0.5,
            "No patterns available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        return ax

    n_show = min(n_patterns, patterns.shape[1])
    colors = plt.cm.tab10(np.linspace(0, 1, n_show))

    for i in range(n_show):
        ax.plot(
            patterns[:, i],
            label=f"Component {i}",
            marker="o",
            markersize=4,
            alpha=0.8,
            color=colors[i],
        )

    ax.set_xlabel("Channel")
    ax.set_ylabel("Pattern weight")
    ax.set_title(f"Spatial Patterns (Top {n_show} Components)")
    ax.legend()
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def plot_cleaning_summary(
    data_before: np.ndarray,
    data_after: np.ndarray,
    estimator,
    sfreq: float,
    line_freq: float | None = None,
    show: bool = True,
) -> plt.Figure:
    """Create a combined multi-panel cleaning summary.

    Parameters
    ----------
    data_before : ndarray, shape (n_channels, n_times)
        Original data before cleaning.
    data_after : ndarray, shape (n_channels, n_times)
        Cleaned data after ZapLine.
    estimator : ZapLine
        Fitted ZapLine estimator.
    sfreq : float
        Sampling frequency in Hz.
    line_freq : float | None
        Line noise frequency.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    fig : Figure
        The matplotlib figure.

    Examples
    --------
    >>> from mne_denoise.viz import plot_cleaning_summary
    >>> zapline = ZapLine(sfreq=1000, line_freq=50)
    >>> cleaned = zapline.fit_transform(data)
    >>> plot_cleaning_summary(data, cleaned, zapline, sfreq=1000, line_freq=50)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # PSD comparison
    plot_psd_comparison(
        data_before, data_after, sfreq, line_freq=line_freq, ax=axes[0, 0], show=False
    )

    # Component scores
    plot_component_scores(estimator, ax=axes[0, 1], show=False)

    # Spatial patterns
    plot_spatial_patterns(estimator, ax=axes[1, 0], show=False)

    # Statistics text
    ax = axes[1, 1]
    ax.axis("off")

    stats = []
    if line_freq is not None:
        stats.append(f"Line Frequency: {line_freq:.1f} Hz")

    n_removed = getattr(estimator, "n_removed_", 0)
    stats.append(f"Components Removed: {n_removed}")

    n_harmonics = getattr(estimator, "n_harmonics_", None)
    if n_harmonics is not None:
        stats.append(f"Harmonics: {n_harmonics}")

    # Compute power reduction at line frequency
    if line_freq is not None:
        nperseg = min(data_before.shape[1], int(sfreq * 2))
        freqs, psd_before = signal.welch(data_before, sfreq, nperseg=nperseg)
        _, psd_after = signal.welch(data_after, sfreq, nperseg=nperseg)
        idx = np.argmin(np.abs(freqs - line_freq))
        power_before = np.mean(psd_before[:, idx])
        power_after = np.mean(psd_after[:, idx])
        if power_after > 0:
            reduction_db = 10 * np.log10(power_before / power_after)
            stats.append(f"Power Reduction: {reduction_db:.1f} dB")

    ax.text(
        0.1,
        0.8,
        "\n".join(stats),
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "lightgray", "alpha": 0.5},
    )
    ax.set_title("Summary Statistics")

    plt.suptitle("ZapLine Cleaning Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_zapline_analytics(result, sfreq=None, show=True):
    """Plot ZapLine cleaning analytics (legacy function).

    Parameters
    ----------
    result : ZapLine | dict
        Result from ZapLine estimator or result dictionary.
    sfreq : float | None
        Sampling frequency (unused, kept for compatibility).
    show : bool
        Whether to show the figure.

    Returns
    -------
    fig : Figure
        Matplotlib figure handle.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Plot 1: Component scores
    ax = axes[0]
    scores = None
    if hasattr(result, "eigenvalues_"):
        scores = result.eigenvalues_
    elif hasattr(result, "dss_eigenvalues"):
        scores = result.dss_eigenvalues
    elif isinstance(result, dict) and "eigenvalues" in result:
        scores = result["eigenvalues"]

    if scores is not None and isinstance(scores, np.ndarray) and scores.size > 0:
        ax.bar(range(len(scores)), scores, color="steelblue")
        ax.axhline(np.mean(scores), color="red", linestyle="--", label="Mean")

        n_rem = 0
        if hasattr(result, "n_removed_"):
            n_rem = result.n_removed_
        elif hasattr(result, "n_removed"):
            n_rem = result.n_removed

        if n_rem > 0:
            ax.axvline(
                n_rem - 0.5, color="green", linestyle="--", label=f"Removed: {n_rem}"
            )
        ax.set_xlabel("Component")
        ax.set_ylabel("Score (eigenvalue)")
        ax.set_title("Component Scores")
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "No scores available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    # Plot 2: Removed signal power
    ax = axes[1]
    if hasattr(result, "removed") and result.removed is not None:
        removed_power = np.mean(result.removed**2, axis=1)
        ax.bar(range(len(removed_power)), removed_power, color="salmon")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Mean Squared Amplitude")
        ax.set_title("Removed Power per Channel")
    else:
        ax.text(
            0.5,
            0.5,
            "No removal data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    # Plot 3: Summary stats
    ax = axes[2]
    ax.axis("off")
    stats_text = []
    if hasattr(result, "line_freq"):
        stats_text.append(f"Line Frequency: {result.line_freq:.1f} Hz")
    if hasattr(result, "n_harmonics"):
        stats_text.append(f"Harmonics: {result.n_harmonics}")

    if hasattr(result, "n_removed_"):
        stats_text.append(f"Components Removed: {result.n_removed_}")
    elif hasattr(result, "n_removed"):
        stats_text.append(f"Components Removed: {result.n_removed}")

    if hasattr(result, "cleaned") and hasattr(result, "removed"):
        total_var = np.var(result.cleaned + result.removed)
        removed_var = np.var(result.removed)
        if total_var > 0:
            pct_removed = 100 * removed_var / total_var
            stats_text.append(f"Variance Removed: {pct_removed:.2f}%")

    ax.text(
        0.1,
        0.8,
        "\n".join(stats_text),
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
    )
    ax.set_title("Summary")

    plt.tight_layout()

    if show:
        plt.show()

    return fig
