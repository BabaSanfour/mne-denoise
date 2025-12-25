"""Evoked response bias for DSS.

Implements trial averaging to enhance stimulus-locked activity.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

import numpy as np

from .base import LinearDenoiser


class TrialAverageBias(LinearDenoiser):
    """Bias function for evoked response enhancement.

    Applies trial averaging to emphasize reproducible evoked responses
    while canceling non-phase-locked noise. This maximizes the
    reproducibility of the response across trials.

    References
    ----------
    Särelä & Valpola (2005). Denoising Source Separation. J. Mach. Learn. Res., 6, 233-272.
    Section 4.1.4 "DENOISING OF QUASIPERIODIC SIGNALS":

    de Cheveigné & Simon (2008). Denoising based on spatial filtering. J. Neurosci. Methods.
    Bias function

    Parameters
    ----------
    weights : array-like, optional
        Weights for each trial. If None, uniform weighting.

    Examples
    --------
    >>> epochs_data = np.random.randn(64, 100, 50)  # channels x times x trials
    >>> bias = TrialAverageBias()
    >>> biased = bias.apply(epochs_data)
    >>> # biased has same shape, but each trial is replaced by the average
    """

    def __init__(self, weights: np.ndarray | None = None) -> None:
        self.weights = weights

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply trial averaging bias.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times, n_epochs)
            Epoched data with trials in the last dimension.

        Returns
        -------
        biased : ndarray, same shape as input
            Data where each trial is replaced by the weighted average
            across trials.
        """
        if data.ndim != 3:
            raise ValueError(
                f"TrialAverageBias requires 3D epoched data "
                f"(n_channels, n_times, n_epochs), got shape {data.shape}"
            )

        n_channels, n_times, n_epochs = data.shape

        if self.weights is not None:
            # Weighted average
            weights = np.asarray(self.weights)
            if weights.shape[0] != n_epochs:
                raise ValueError(
                    f"weights length ({len(weights)}) must match "
                    f"n_epochs ({n_epochs})"
                )
            weights = weights / weights.sum()  # Normalize
            avg = np.tensordot(data, weights, axes=(2, 0))  # (n_ch, n_times)
        else:
            # Simple mean
            avg = data.mean(axis=2)

        # Broadcast average to all trials
        biased = np.broadcast_to(avg[:, :, np.newaxis], data.shape).copy()

        return biased
