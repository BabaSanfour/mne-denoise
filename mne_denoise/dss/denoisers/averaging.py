"""Averaging bias functions for DSS.

Implements trial/epoch and group/dataset averaging to enhance reproducible patterns.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

References
----------
.. [1] Särelä & Valpola (2005). Denoising Source Separation. J. Mach. Learn. Res., 6, 233-272.
.. [2] de Cheveigné & Simon (2008). Denoising based on spatial filtering. J. Neurosci. Methods.
.. [3] de Cheveigné & Parra (2014). Joint denoising source separation. NeuroImage, 98, 489-496.
"""

from __future__ import annotations

import numpy as np

from .base import LinearDenoiser


class AverageBias(LinearDenoiser):
    """Bias function for finding repeatable components via averaging.

    Maximizes the reproducibility of patterns across trials (epochs) or
    datasets (subjects). This LinearDenoiser covers:
    - Trial averaging (axis='epochs'): for evoked response enhancement
    - Dataset averaging (axis='datasets'): for group-level repeatability (JDSS)

    Parameters
    ----------
    axis : str
        Dimension to average over:
        - 'epochs' (default): Average across trials. Input shape: (n_channels, n_times, n_epochs)
        - 'datasets': Average across datasets/subjects. Input shape: (n_datasets, n_channels, n_times)
    weights : array-like, optional
        Weights for averaging. If None, uniform weighting.

    Examples
    --------
    >>> from mne_denoise.dss.denoisers import AverageBias
    >>> # For evoked response enhancement (like old TrialAverageBias)
    >>> epochs_data = np.random.randn(64, 100, 50)  # channels x times x trials
    >>> bias = AverageBias(axis="epochs")
    >>> biased = bias.apply(epochs_data)

    >>> # For group-level repeatability (like old JDSS)
    >>> group_data = np.random.randn(10, 64, 100)  # subjects x channels x times
    >>> bias = AverageBias(axis="datasets")
    >>> biased = bias.apply(group_data)

    References
    ----------
    Särelä & Valpola (2005). Section 4.1.4 "DENOISING OF QUASIPERIODIC SIGNALS"
    de Cheveigné & Parra (2014). Joint denoising source separation.
    """

    def __init__(self, axis: str = "epochs", weights: np.ndarray | None = None) -> None:
        if axis not in ("epochs", "datasets"):
            raise ValueError(f"axis must be 'epochs' or 'datasets', got {axis!r}")
        self.axis = axis
        self.weights = weights

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply averaging bias.

        Parameters
        ----------
        data : ndarray
            Input data.
            - For axis='epochs': shape (n_channels, n_times, n_epochs)
            - For axis='datasets': shape (n_datasets, n_channels, n_times)

        Returns
        -------
        biased : ndarray, same shape as input
            Data where each slice is replaced by the weighted average.
        """
        if self.axis == "epochs":
            return self._apply_epochs(data)
        else:  # datasets
            return self._apply_datasets(data)

    def _apply_epochs(self, data: np.ndarray) -> np.ndarray:
        """Average across epochs (last axis)."""
        if data.ndim != 3:
            raise ValueError(
                f"AverageBias(axis='epochs') requires 3D data "
                f"(n_channels, n_times, n_epochs), got shape {data.shape}"
            )

        n_channels, n_times, n_epochs = data.shape

        if self.weights is not None:
            weights = np.asarray(self.weights)
            if weights.shape[0] != n_epochs:
                raise ValueError(
                    f"weights length ({len(weights)}) must match n_epochs ({n_epochs})"
                )
            weights = weights / weights.sum()
            avg = np.tensordot(data, weights, axes=(2, 0))  # (n_ch, n_times)
        else:
            avg = data.mean(axis=2)

        # Broadcast average to all epochs
        biased = np.broadcast_to(avg[:, :, np.newaxis], data.shape).copy()
        return biased

    def _apply_datasets(self, data: np.ndarray) -> np.ndarray:
        """Average across datasets."""
        if data.ndim != 3:
            raise ValueError("AverageBias(axis='datasets') requires 3D data.")

        # Typically, for group DSS (JDSS), the input data shape might be
        # (n_datasets, n_channels, n_times). We assume axis=0 corresponds to datasets.

        n_datasets, n_channels, n_times = data.shape

        if self.weights is not None:
            weights = np.asarray(self.weights)
            if weights.shape[0] != n_datasets:
                raise ValueError(
                    f"weights length ({len(weights)}) must match n_datasets ({n_datasets})"
                )
            weights = weights / weights.sum()
            avg = np.tensordot(weights, data, axes=(0, 0))  # (n_ch, n_times)
        else:
            avg = data.mean(axis=0)

        # Broadcast average to all datasets
        biased = np.broadcast_to(avg[np.newaxis, :, :], data.shape).copy()
        return biased
