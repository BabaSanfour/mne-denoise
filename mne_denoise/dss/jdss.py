"""Joint Denoising Source Separation (JDSS).

Implements the "repeatability" formulation of JDSS (de Cheveigné & Parra, 2014).
Finds components that are maximally reproducible across multiple datasets
(e.g., subjects, blocks, trials).

The objective is to maximize the ratio of "Signal Variance" (grand average)
to "Total Variance" (mean of individual covariances).

References
----------
.. [1] de Cheveigné, A., & Parra, L. C. (2014). Joint denoising source separation.
       NeuroImage, 98, 489-496.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Optional MNE support
try:
    import mne
    from mne.epochs import BaseEpochs
    from mne.evoked import Evoked
    from mne.io import BaseRaw
except ImportError:
    mne = None

from .utils import compute_covariance
from .linear import compute_dss


def compute_jdss(
    datasets: List[np.ndarray],
    n_components: Optional[int] = None,
    reg: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Joint DSS filters to maximize repeatability across datasets.

    Parameters
    ----------
    datasets : list of ndarray
        List of datasets. Each dataset must be (n_channels, n_times).
        All datasets must have the same number of channels and times.
        Ideally, they should be time-aligned (e.g., same stimulus).
    n_components : int, optional
        Number of components to keep.
    reg : float, optional
        Regularization for covariance inverting.

    Returns
    -------
    filters : ndarray, shape (n_components, n_channels)
        Spatial filters.
    patterns : ndarray, shape (n_channels, n_components)
        Spatial patterns (topographies).
    eigenvalues : ndarray, shape (n_components,)
        Eigenvalues (repeatability scores).
    """
    n_datasets = len(datasets)
    n_channels, n_times = datasets[0].shape
    
    # Validation
    for i, d in enumerate(datasets):
        if d.shape != (n_channels, n_times):
            raise ValueError(f"Dataset {i} shape mismatch. Expected {(n_channels, n_times)}, got {d.shape}")

    # Stack datasets: (n_datasets, n_channels, n_times)
    stacked = np.array(datasets)
    
    # 1. Compute Total Covariance (Average of individual covariances)
    # C_tot = 1/N * sum( Cov(X_i) )
    # Assuming centered data is handled by compute_covariance
    covariances = [compute_covariance(d) for d in datasets]
    cov_total = np.mean(covariances, axis=0)
    
    # 2. Compute Signal Covariance (Covariance of Grand Average)
    # X_bar = 1/N * sum(X_i)
    grand_average = np.mean(stacked, axis=0)
    cov_signal = compute_covariance(grand_average)
    
    # 3. Solve Generalized Eigenvalue Problem
    # We use the existing compute_dss function which solves exactly this problem:
    # Maximize (w' * cov_biased * w) / (w' * cov_baseline * w)
    # Here: Baseline = cov_total, Biased = cov_signal
    
    # Note: Ratio approaches 1 for perfect repeatability (signal = total).
    # Ideally, we maximize cov_signal relative to cov_total.
    
    filters, patterns, eigenvalues = compute_dss(
        covariance_baseline=cov_total,
        covariance_biased=cov_signal,
        n_components=n_components,
        reg=reg
    )
    
    return filters, patterns, eigenvalues


class JDSS(BaseEstimator, TransformerMixin):
    """Joint Denoising Source Separation (JDSS) Estimator.

    Finds spatial filters that extract components repeatable across datasets.
    
    Parameters
    ----------
    n_components : int, optional
        Number of components to keep.
    reg : float, optional
        Regularization parameter.

    Attributes
    ----------
    filters_ : array, shape (n_components, n_channels)
        The spatial filters.
    patterns_ : array, shape (n_channels, n_components)
        The spatial patterns (topographies).
    eigenvalues_ : array, shape (n_components,)
        scores reflecting repeatability.
    """

    def __init__(self, n_components: Optional[int] = None, reg: float = 1e-9):
        self.n_components = n_components
        self.reg = reg
        
        self.filters_ = None
        self.patterns_ = None
        self.mixing_ = None
        self.eigenvalues_ = None


    def fit(self, X: Union[List, np.ndarray], y=None) -> "JDSS":
        """Compute JDSS filters from multiple datasets.

        Parameters
        ----------
        X : list of arrays or 3D array
            Training data.
            - If list: [dataset1, dataset2, ...] where each is (n_channels, n_times).
            - If 3D array: (n_datasets, n_channels, n_times).
            - MNE Epochs/Evoked objects can be passed in a list.
        """
        # Prepare data as list of 2D arrays (n_channels, n_times)
        datasets = self._validate_input(X)
        
        self.filters_, self.patterns_, self.eigenvalues_ = compute_jdss(
            datasets, n_components=self.n_components, reg=self.reg
        )
        
        # Compute mixing matrix for reconstruction
        # (Patterns from compute_dss are normalized for visualization)
        self.mixing_ = np.linalg.pinv(self.filters_)

        
        return self

    def transform(self, X: Union[np.ndarray, List]) -> Union[np.ndarray, List]:
        """Apply JDSS filters to data.

        Parameters
        ----------
        X : array or list
            Data to transform.
            - If 2D array (n_channels, n_times): Transform single dataset.
            - If 3D array (n_datasets, n_ch, n_times): Transform each dataset.
            - If list: Transform each dataset in list.

        Returns
        -------
        out : array or list
            Transformed sources.
        """
        if self.filters_ is None:
            raise RuntimeError("JDSS not fitted.")
            
        if isinstance(X, list):
            return [self.filters_ @ d for d in X]
        elif isinstance(X, np.ndarray):
            if X.ndim == 2:
                return self.filters_ @ X
            elif X.ndim == 3:
                # (n_datasets, n_channels, n_times) -> (n_datasets, n_components, n_times)
                # filters is (n_comp, n_ch)
                # Contract over axis 1 of X (n_ch) with axis 1 of filters_ (n_ch)
                return np.tensordot(X, self.filters_, axes=(1, 1)).transpose(0, 2, 1)

        
        raise TypeError("Unsupported input type.")

    def inverse_transform(self, sources: Union[np.ndarray, List]) -> Union[np.ndarray, List]:
        """Reconstruct data from sources.

        Parameters
        ----------
        sources : array or list
            Source time series.
            - If 2D (n_components, n_times): Single dataset.
            - If 3D (n_datasets, n_components, n_times): Multiple datasets.
            - If list: List of 2D arrays.

        Returns
        -------
        reconstructed : array or list
            Reconstructed data in sensor space.
        """
        if self.mixing_ is None:
            raise RuntimeError("JDSS not fitted.")

        
        if isinstance(sources, list):
            return [self.mixing_ @ s for s in sources]
        elif isinstance(sources, np.ndarray):
            if sources.ndim == 2:
                return self.mixing_ @ sources
            elif sources.ndim == 3:
                # (n_datasets, n_components, n_times) -> (n_datasets, n_channels, n_times)
                # mixing_ is (n_ch, n_comp)
                # Contract over axis 1 of sources (n_comp) with axis 1 of mixing_ (n_comp)
                return np.tensordot(sources, self.mixing_, axes=(1, 1)).transpose(0, 2, 1)


        
        raise TypeError("Unsupported input type.")

    def _validate_input(self, X) -> List[np.ndarray]:
        """Convert input to list of 2D numpy arrays."""
        datasets = []
        if isinstance(X, (list, tuple)):
            for item in X:
                datasets.append(self._to_numpy(item))
        elif isinstance(X, np.ndarray) and X.ndim == 3:
            # Assume (n_datasets, n_channels, n_times)
            for i in range(X.shape[0]):
                datasets.append(X[i])
        else:
            raise ValueError("Input must be list of datasets or 3D array.")
        return datasets

    def _to_numpy(self, item) -> np.ndarray:
        """Extract numpy array from MNE objects or return array."""
        if isinstance(item, np.ndarray):
            return item
        if mne:
            if isinstance(item, (BaseRaw, Evoked)):
                return item.get_data()
            if isinstance(item, BaseEpochs):
                 # Epochs: (n_epochs, n_ch, n_times) -> flatten or average?
                 # JDSS usually works on averaged data (ERPs) or continuous blocks.
                 # If user passes Epochs, we likely treated as single dataset? 
                 # Or reshape to (n_ch, -1)?
                 # Let's assume user passes "List of Epochs" meaning each Epochs obj is a subject.
                 # We'll concatenate epochs or average?
                 # JDSS requires time-alignment. Concatenating epochs breaks alignment structure 
                 # unless it's "blocks". 
                 # Simplest: reshape to (n_ch, n_times_total)
                 data = item.get_data() # (n_epo, n_ch, n_times)
                 # Concatenate trials -> (n_ch, n_epo*n_time)
                 return np.hstack([data[i] for i in range(data.shape[0])])
        return np.asarray(item)
