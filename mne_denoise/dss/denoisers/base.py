"""Base classes for DSS denoiser functions.

Provides abstract interfaces for linear and nonlinear bias functions
that can be plugged into the DSS pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class LinearDenoiser(ABC):
    """Base class for linear bias functions.

    Linear denoisers apply a deterministic transformation to the data
    that emphasizes a signal of interest. The DSS algorithm then finds
    spatial filters that maximize the ratio of biased to baseline variance.

    Subclasses must implement the `apply` method.
    """

    @abstractmethod
    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply bias transformation to data.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
            Input data.

        Returns
        -------
        biased : ndarray, same shape as input
            Biased data with signal of interest emphasized.
        """
        pass

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Allow using denoiser as a callable."""
        return self.apply(data)


class NonlinearDenoiser(ABC):
    """Base class for nonlinear/adaptive denoiser functions.

    Nonlinear denoisers operate on source time series rather than
    sensor data. They are used in the iterative DSS algorithm where
    the denoising function is applied to the current source estimate
    at each iteration.

    Examples include variance-based masking, kurtosis maximization,
    and other adaptive transformations.
    """

    @abstractmethod
    def denoise(self, source: np.ndarray) -> np.ndarray:
        """Apply nonlinear denoising to source time series.

        Parameters
        ----------
        source : ndarray, shape (n_times,) or (n_times, n_epochs)
            Source time series (single component).

        Returns
        -------
        denoised : ndarray, same shape as input
            Denoised source with enhanced signal characteristics.
        """
        pass

    def __call__(self, source: np.ndarray) -> np.ndarray:
        """Allow using denoiser as a callable."""
        return self.denoise(source)
