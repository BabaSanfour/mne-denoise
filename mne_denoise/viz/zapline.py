"""ZapLine-specific visualization functions.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_zapline_analytics(result, sfreq=None, show=True):
    """Plot ZapLine cleaning analytics.
    
    Parameters
    ----------
    result : ZapLineResult
        Result from dss_zapline or ZapLinePlusResult.
    sfreq : float, optional
        Sampling frequency for PSD computation. Required if result
        doesn't have this info.
    show : bool
        Whether to show the figure. Default True.
        
    Returns
    -------
    fig : Figure
        Matplotlib figure handle.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Plot 1: Component scores
    ax = axes[0]
    if hasattr(result, "dss_eigenvalues") and result.dss_eigenvalues is not None:
        scores = result.dss_eigenvalues
        if isinstance(scores, np.ndarray) and scores.size > 0:
            ax.bar(range(len(scores)), scores, color="steelblue")
            ax.axhline(np.mean(scores), color="red", linestyle="--", label="Mean")
            if result.n_removed > 0:
                ax.axvline(result.n_removed - 0.5, color="green", linestyle="--", 
                          label=f"Removed: {result.n_removed}")
            ax.set_xlabel("Component")
            ax.set_ylabel("Score (eigenvalue)")
            ax.set_title("Component Scores")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No scores available", ha="center", va="center",
                   transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "No scores available", ha="center", va="center",
               transform=ax.transAxes)
    
    # Plot 2: Removed signal power
    ax = axes[1]
    if hasattr(result, "removed") and result.removed is not None:
        removed_power = np.mean(result.removed ** 2, axis=1)
        ax.bar(range(len(removed_power)), removed_power, color="salmon")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Mean Squared Amplitude")
        ax.set_title("Removed Power per Channel")
    else:
        ax.text(0.5, 0.5, "No removal data", ha="center", va="center",
               transform=ax.transAxes)
    
    # Plot 3: Summary stats
    ax = axes[2]
    ax.axis("off")
    stats_text = []
    if hasattr(result, "line_freq"):
        stats_text.append(f"Line Frequency: {result.line_freq:.1f} Hz")
    if hasattr(result, "n_harmonics"):
        stats_text.append(f"Harmonics: {result.n_harmonics}")
    if hasattr(result, "n_removed"):
        stats_text.append(f"Components Removed: {result.n_removed}")
    if hasattr(result, "cleaned") and hasattr(result, "removed"):
        total_var = np.var(result.cleaned + result.removed)
        removed_var = np.var(result.removed)
        if total_var > 0:
            pct_removed = 100 * removed_var / total_var
            stats_text.append(f"Variance Removed: {pct_removed:.2f}%")
    
    ax.text(0.1, 0.8, "\n".join(stats_text), transform=ax.transAxes,
           fontsize=11, verticalalignment="top", fontfamily="monospace")
    ax.set_title("Summary")
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig
