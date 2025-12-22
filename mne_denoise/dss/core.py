"""Core linear DSS algorithm implementation.

This module implements the one-shot (linear) DSS algorithm as described in
Särelä & Valpola (2005) and used in NoiseTools nt_dss0.m.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np

from .whitening import compute_whitener


@dataclass
class DSSConfig:
    """Configuration for DSS computation.

    Parameters
    ----------
    n_components : int, optional
        Number of DSS components to retain. If None, keep all.
    rank : int, optional
        Rank for whitening stage. If None, auto-determined.
    reg : float
        Regularization threshold for eigenvalue cutoff. Default 1e-9.
    store_filters : bool
        Whether to store spatial filters. Default True.
    store_patterns : bool
        Whether to store spatial patterns. Default True.
    """

    n_components: Optional[int] = None
    rank: Optional[int] = None
    reg: float = 1e-9
    store_filters: bool = True
    store_patterns: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


def compute_dss(
    data: np.ndarray,
    biased_data: np.ndarray,
    *,
    n_components: Optional[int] = None,
    rank: Optional[int] = None,
    reg: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute DSS spatial filters from baseline and biased covariances.

    This implements the core DSS algorithm EXACTLY matching NoiseTools nt_dss0.m:
    1. Compute baseline covariance C0 from data
    2. Compute biased covariance C1 from biased_data
    3. PCA + whitening from C0 (nt_pcarot)
    4. Apply whitening to C1: c2 = N' * topcs1' * c1 * topcs1 * N
    5. Eigendecomposition of c2 (nt_pcarot)
    6. DSS matrix: todss = topcs1 * N * topcs2
    7. Normalize so components have unit variance

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
        Original (baseline) data.
    biased_data : ndarray, shape (n_channels, n_times) or same as data
        Data after applying bias function. Should emphasize signal of interest.
    n_components : int, optional
        Number of DSS components to return. If None, return all.
    rank : int, optional
        Rank for whitening stage. If None, auto-determined from data.
    reg : float
        Regularization threshold. Default 1e-9.

    Returns
    -------
    dss_filters : ndarray, shape (n_components, n_channels)
        DSS spatial filters. Apply as: sources = dss_filters @ data
    dss_patterns : ndarray, shape (n_channels, n_components)
        DSS spatial patterns (for visualization/interpretation).
    eigenvalues : ndarray, shape (n_components,)
        DSS eigenvalues (biased power per component).
    explained_variance : ndarray, shape (n_components,)
        Variance explained by each component.

    Notes
    -----
    The DSS filters maximize the ratio of biased variance to baseline variance.
    Components are ordered by descending eigenvalue (most biased first).

    References
    ----------
    .. [1] Särelä & Valpola (2005). Denoising Source Separation. JMLR.
    .. [2] NoiseTools nt_dss0.m implementation.
    """
    if data.shape != biased_data.shape:
        raise ValueError(
            f"data and biased_data must have same shape, "
            f"got {data.shape} and {biased_data.shape}"
        )

    # Handle 3D epoched data by flattening
    original_shape = data.shape
    if data.ndim == 3:
        n_channels, n_times, n_epochs = data.shape
        data = data.reshape(n_channels, -1)
        biased_data = biased_data.reshape(n_channels, -1)
    elif data.ndim != 2:
        raise ValueError(f"Data must be 2D or 3D, got {data.ndim}D")

    n_channels, n_samples = data.shape

    # Remove mean
    data = data - data.mean(axis=1, keepdims=True)
    biased_data = biased_data - biased_data.mean(axis=1, keepdims=True)

    # Compute baseline covariance C0
    c0 = data @ data.T / n_samples

    # Compute biased covariance C1
    c1 = biased_data @ biased_data.T / n_samples

    # =========================================================================
    # STEP 1: PCA and whitening from C0 (matches nt_pcarot)
    # =========================================================================
    c0_sym = (c0 + c0.T) / 2  # Ensure symmetry
    evs1, topcs1 = np.linalg.eigh(c0_sym)

    # Sort descending (MATLAB: sort(diag(S)', 'descend'))
    idx = np.argsort(evs1)[::-1]
    evs1 = evs1[idx]
    topcs1 = topcs1[:, idx]

    # Take absolute value of eigenvalues (nt_dss0 line 37: evs1=abs(evs1))
    evs1 = np.abs(evs1)

    # Apply threshold (nt_dss0 line 41: idx=find(evs1/max(evs1)>keep2))
    max_ev = np.max(evs1)
    if max_ev < 1e-15:
        raise ValueError("Covariance matrix has no significant variance")

    keep_mask = evs1 / max_ev > reg

    # Apply rank constraint
    if rank is not None:
        keep_mask[rank:] = False

    n_keep = np.sum(keep_mask)
    if n_keep == 0:
        raise ValueError("No components above regularization threshold")

    # Truncate
    evs1 = evs1[keep_mask]
    topcs1 = topcs1[:, keep_mask]

    # =========================================================================
    # STEP 2: Apply PCA and whitening to biased covariance
    # MATLAB: N = diag(sqrt(1./evs1)); c2 = N' * topcs1' * c1 * topcs1 * N
    # =========================================================================
    N = np.diag(np.sqrt(1.0 / evs1))
    c2 = N.T @ topcs1.T @ c1 @ topcs1 @ N

    # Ensure symmetry
    c2 = (c2 + c2.T) / 2

    # =========================================================================
    # STEP 3: Second eigendecomposition (nt_pcarot on c2)
    # =========================================================================
    evs2, topcs2 = np.linalg.eigh(c2)

    # Sort descending
    idx2 = np.argsort(evs2)[::-1]
    evs2 = evs2[idx2]
    topcs2 = topcs2[:, idx2]

    # =========================================================================
    # STEP 4: Build DSS matrix
    # MATLAB: todss = topcs1 * N * topcs2
    # =========================================================================
    todss = topcs1 @ N @ topcs2

    # =========================================================================
    # STEP 5: Normalize so components have unit variance
    # MATLAB: N2 = diag(todss' * c0 * todss); todss = todss * diag(1./sqrt(N2))
    # =========================================================================
    N2 = np.diag(todss.T @ c0 @ todss)
    N2 = np.where(N2 > 1e-15, N2, 1.0)  # Prevent division by zero
    todss = todss @ np.diag(1.0 / np.sqrt(N2))

    # =========================================================================
    # STEP 6: Truncate to n_components
    # =========================================================================
    if n_components is None:
        n_components = todss.shape[1]
    else:
        n_components = min(n_components, todss.shape[1])

    todss = todss[:, :n_components]
    eigenvalues = evs2[:n_components]

    # =========================================================================
    # Convert to our convention: filters are (n_components, n_channels)
    # MATLAB todss is (n_channels, n_components), we want (n_components, n_channels)
    # =========================================================================
    dss_filters = todss.T

    # DSS patterns: for interpretation
    # patterns = C0 @ todss, normalized columns
    dss_patterns = c0 @ todss
    pattern_norms = np.sqrt(np.sum(dss_patterns**2, axis=0))
    pattern_norms = np.where(pattern_norms > 1e-15, pattern_norms, 1.0)
    dss_patterns = dss_patterns / pattern_norms

    # Compute explained variance
    sources = dss_filters @ data
    explained_variance = np.var(sources, axis=1)

    return dss_filters, dss_patterns, eigenvalues, explained_variance


class DSS:
    """Denoising Source Separation estimator.

    Scikit-learn style API for applying DSS to multichannel data.

    Parameters
    ----------
    bias : str or callable
        Bias function specification. Can be:
        - 'identity': No bias (for testing)
        - callable: Function that takes data and returns biased data
    n_components : int, optional
        Number of DSS components to retain.
    rank : int, optional
        Rank for whitening. If None, auto-determined.
    reg : float
        Regularization threshold. Default 1e-9.

    Attributes
    ----------
    filters_ : ndarray, shape (n_components, n_channels)
        Fitted DSS spatial filters.
    patterns_ : ndarray, shape (n_channels, n_components)
        Fitted DSS spatial patterns.
    eigenvalues_ : ndarray, shape (n_components,)
        DSS eigenvalues.
    explained_variance_ : ndarray, shape (n_components,)
        Variance explained by each component.

    Examples
    --------
    >>> # Simple bandpass bias example
    >>> dss = DSS(bias=my_bandpass_filter, n_components=5)
    >>> dss.fit(data)
    >>> sources = dss.transform(data)
    >>> cleaned = dss.inverse_transform(sources[:3])  # Keep top 3
    """

    def __init__(
        self,
        bias: Union[str, Callable[[np.ndarray], np.ndarray]] = "identity",
        *,
        n_components: Optional[int] = None,
        rank: Optional[int] = None,
        reg: float = 1e-9,
    ) -> None:
        self.bias = bias
        self.n_components = n_components
        self.rank = rank
        self.reg = reg

        # Fitted attributes
        self.filters_: Optional[np.ndarray] = None
        self.patterns_: Optional[np.ndarray] = None
        self.eigenvalues_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None

    def _apply_bias(self, data: np.ndarray) -> np.ndarray:
        """Apply the bias function to data."""
        if self.bias == "identity":
            return data.copy()
        elif callable(self.bias):
            return self.bias(data)
        else:
            raise ValueError(f"Unknown bias type: {self.bias}")

    def fit(self, data: np.ndarray) -> "DSS":
        """Fit DSS filters from data.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
            Training data.

        Returns
        -------
        self : DSS
            The fitted estimator.
        """
        # Apply bias to get biased data
        biased_data = self._apply_bias(data)

        # Compute DSS
        filters, patterns, eigenvalues, explained_var = compute_dss(
            data,
            biased_data,
            n_components=self.n_components,
            rank=self.rank,
            reg=self.reg,
        )

        self.filters_ = filters
        self.patterns_ = patterns
        self.eigenvalues_ = eigenvalues
        self.explained_variance_ = explained_var

        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply DSS filters to data.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
            Data to transform.

        Returns
        -------
        sources : ndarray, shape (n_components, n_times) or (n_components, n_times, n_epochs)
            DSS component time series.
        """
        if self.filters_ is None:
            raise RuntimeError("DSS not fitted. Call fit() first.")

        original_shape = data.shape
        if data.ndim == 3:
            n_channels, n_times, n_epochs = data.shape
            data_2d = data.reshape(n_channels, -1)
        else:
            data_2d = data
            n_times = data.shape[1]

        # Center data
        data_2d = data_2d - data_2d.mean(axis=1, keepdims=True)

        # Apply filters
        sources = self.filters_ @ data_2d

        # Reshape if needed
        if len(original_shape) == 3:
            sources = sources.reshape(self.filters_.shape[0], n_times, n_epochs)

        return sources

    def inverse_transform(
        self,
        sources: np.ndarray,
        *,
        component_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Reconstruct data from DSS components.

        Parameters
        ----------
        sources : ndarray, shape (n_components, n_times) or subset
            DSS component time series to reconstruct from.
        component_indices : array-like, optional
            Indices of components being passed (if subset). Required if
            sources has fewer components than filters_.

        Returns
        -------
        reconstructed : ndarray, shape (n_channels, n_times)
            Reconstructed sensor-space data.
        """
        if self.patterns_ is None:
            raise RuntimeError("DSS not fitted. Call fit() first.")

        # Handle component subset
        n_sources = sources.shape[0]
        if component_indices is not None:
            patterns = self.patterns_[:, component_indices]
        elif n_sources < self.patterns_.shape[1]:
            # Assume first n_sources components
            patterns = self.patterns_[:, :n_sources]
        else:
            patterns = self.patterns_

        # Reshape for 3D
        original_shape = sources.shape
        if sources.ndim == 3:
            n_comp, n_times, n_epochs = sources.shape
            sources_2d = sources.reshape(n_comp, -1)
        else:
            sources_2d = sources
            n_times = sources.shape[1]

        # Reconstruct: X_rec = patterns @ sources
        reconstructed = patterns @ sources_2d

        if len(original_shape) == 3:
            reconstructed = reconstructed.reshape(patterns.shape[0], n_times, n_epochs)

        return reconstructed

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit DSS and return transformed data.

        Parameters
        ----------
        data : ndarray
            Training data.

        Returns
        -------
        sources : ndarray
            DSS component time series.
        """
        return self.fit(data).transform(data)
