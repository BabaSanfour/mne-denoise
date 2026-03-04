"""Core linear DSS algorithm and Estimator.

This module contains:
1. `compute_dss`: The core mathematical implementation of Linear DSS.
2. `DSS`: The Scikit-learn estimator compatible with MNE-Python objects or NumPy arrays.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

References
----------
.. [1] Särelä & Valpola (2005). Denoising Source Separation. J. Mach. Learn. Res., 6, 233-272.
.. [2] de Cheveigné & Simon (2008). Denoising based on spatial filtering. J. Neurosci. Methods.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

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

from .._logging import set_log_level_from_verbose
from ..utils import extract_data_from_mne, reconstruct_mne_object
from .denoisers import LinearDenoiser
from .utils import compute_covariance
from .utils.segmentation import CovarianceSegmenter, FixedWindowSegmenter
from .utils.whitening import (
    apply_covariance_transform,
    apply_spatial_transform,
    compute_data_covariance_whitener,
    compute_mne_sensor_whitener,
    map_spatial_matrices_to_sensor_space,
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. Core Algorithm
# -----------------------------------------------------------------------------


def compute_dss(
    covariance_baseline: np.ndarray,
    covariance_biased: np.ndarray,
    *,
    n_components: int | None = None,
    rank: int | None = None,
    reg: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute DSS spatial filters from baseline and biased covariances.

    This implements the core Linear DSS algorithm as described in Särelä & Valpola (2005) [1]_.

    The algorithm finds a linear transform (spatial filters) that maximizes the
    biased variance (signal) relative to total/baseline variance (noise).

    The process corresponds to Equation 7 in de Cheveigné & Simon (2008) [2]_:

    .. math:: \\tilde{S}(t) = P Q R_2 N_2 R_1 N_1 S(t)

    where:

    *   **N1** (Initial Normalization): Handled externally (e.g. ``DSS(normalize_input=True)``).
        Ensures equal weight for each sensor.
    *   **R1** (First PCA): Rotation derived from baseline covariance (Sphering/Whitening PCA).
        Discards components with negligible power.
    *   **N2** (Whitening): Normalization to obtain orthonormal "spatially whitened" vectors.
    *   **R2** (Second PCA): Rotation derived from biased covariance in the whitened space.
    *   **Q** (Selector): Selection of the top ``n_components`` with highest bias score.
    *   **P** (Projection): Projection back to sensor space (Spatial Patterns).

    Parameters
    ----------
    covariance_baseline : ndarray
        Baseline covariance.
    covariance_biased : ndarray
        Biased covariance.
    n_components : int, optional
        Number of DSS components to return (The **Q** selector step). If None, return all.
    rank : int, optional
        Rank for whitening stage. If None, auto-determined from data.
    reg : float
        Regularization threshold. Default 1e-9.

    Returns
    -------
    dss_filters : ndarray, shape (n_components, n_channels)
        DSS spatial filters (unmixing matrix transposed).
        Corresponds to the combined transform :math:`Q R_2 N_2 R_1`.
        Apply as: ``sources = dss_filters @ data``.
    dss_patterns : ndarray, shape (n_channels, n_components)
        DSS spatial patterns (mixing matrix).
        Corresponds to the projection matrix **P**.
    eigenvalues : ndarray, shape (n_components,)
        DSS eigenvalues (ratio of biased power to baseline power).

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.dss import compute_dss, compute_covariance
    >>> # Generate synthetic data (n_channels, n_times)
    >>> data = np.random.randn(10, 1000)
    >>> # Compute covariances
    >>> cov_baseline = compute_covariance(data)
    >>> # Biased covariance: trial-averaged standard example or filtering
    >>> cov_biased = compute_covariance(data)  # Just a placeholder
    >>> # Compute DSS
    >>> filters, patterns, evs = compute_dss(cov_baseline, cov_biased, n_components=5)

    See Also
    --------
    DSS : Estimator class for linear DSS.

    References
    ----------
    .. [1] Särelä, J., & Valpola, H. (2005). Denoising source separation.
           Journal of Machine Learning Research, 6, 233-272.
    .. [2] de Cheveigné, A., & Simon, J. Z. (2008). Denoising based on spatial filtering.
           Journal of Neuroscience Methods, 171(2), 331-339.
    """
    # Check shapes
    if covariance_baseline.shape != covariance_biased.shape:
        raise ValueError(
            f"Covariance shapes mismatch: {covariance_baseline.shape} vs {covariance_biased.shape}"
        )

    n_channels = covariance_baseline.shape[0]
    if covariance_baseline.shape != (n_channels, n_channels):
        raise ValueError(f"Covariance must be square, got {covariance_baseline.shape}")

    # STEP 1 + 2: derive and apply the shared baseline-covariance whitener.
    whitener, _, eigenvalues_white = compute_data_covariance_whitener(
        covariance_baseline,
        rank=rank,
        reg=reg,
    )
    n_keep = eigenvalues_white.size
    max_ev = eigenvalues_white[0]

    if n_keep < n_channels // 4:
        logger.warning(
            "DSS: only %d/%d components kept after rank reduction "
            "(reg=%g, max_eigval=%.3g, smallest_kept_eigval=%.3g). "
            "This is common for MEG data with a large dynamic range "
            "(e.g., raw CTF magnetometers in Tesla). Consider passing "
            "normalize_input=True to DSS, lowering reg, or fitting "
            "homogeneous channel types separately instead of mixing channels "
            "with different physical units.",
            int(n_keep),
            int(n_channels),
            float(reg),
            float(max_ev),
            float(eigenvalues_white[n_keep - 1]),
        )

    covariance_whitened = apply_covariance_transform(whitener, covariance_biased)

    # =========================================================================
    # STEP 3: PCA on whitened covariance_biased -> defines R2
    # =========================================================================
    eigenvalues_biased, eigenvectors_biased = np.linalg.eigh(covariance_whitened)

    # Sort descending
    idx2 = np.argsort(eigenvalues_biased)[::-1]
    eigenvalues_biased = eigenvalues_biased[idx2]
    eigenvectors_biased = eigenvectors_biased[:, idx2]

    # =========================================================================
    # STEP 4: Build DSS matrix (filters = R2 * N2 * R1)
    # =========================================================================
    unmixing_matrix = whitener.T @ eigenvectors_biased

    # =========================================================================
    # STEP 5: Normalize so components have unit variance
    # =========================================================================
    norm_factor = np.diag(unmixing_matrix.T @ covariance_baseline @ unmixing_matrix)
    norm_factor = np.where(norm_factor > 1e-15, norm_factor, 1.0)
    unmixing_matrix = unmixing_matrix @ np.diag(1.0 / np.sqrt(norm_factor))

    # =========================================================================
    # STEP 6: Truncate to n_components
    # =========================================================================
    if n_components is None:
        n_components = unmixing_matrix.shape[1]
    else:
        n_components = min(n_components, unmixing_matrix.shape[1])

    unmixing_matrix = unmixing_matrix[:, :n_components]
    eigenvalues = eigenvalues_biased[:n_components]

    # =========================================================================
    # Convert to our convention: filters are (n_components, n_channels)
    # Corresponds to Q selector on the rows of the combined matrix.
    # =========================================================================
    dss_filters = unmixing_matrix.T

    # DSS patterns: L2-normalized for topographic visualization (Haufe et al. 2014)
    dss_patterns = covariance_baseline @ unmixing_matrix
    pattern_norms = np.sqrt(np.sum(dss_patterns**2, axis=0))
    pattern_norms = np.where(pattern_norms > 1e-15, pattern_norms, 1.0)
    dss_patterns = dss_patterns / pattern_norms

    return dss_filters, dss_patterns, eigenvalues


# -----------------------------------------------------------------------------
# 2. Scikit-Learn Estimator
# -----------------------------------------------------------------------------


class DSS(BaseEstimator, TransformerMixin):
    """Denoising Source Separation (DSS) Transformer.

    Implements DSS as a scikit-learn compatible transformer that fits natively
    on MNE-Python objects (Raw, Epochs, Evoked) or numpy arrays.

    Parameters
    ----------
    n_components : int, optional
        Number of DSS components to keep. If None, keep all.
    bias : LinearDenoiser
        Bias function to define the signal of interest. Must be an instance of
        `mne_denoise.dss.LinearDenoiser` (e.g. `BandpassBias`, `TrialAverageBias`)
        or a callable that takes data and returns biased data.
    n_select : int | 'auto' | None, default=None
        Number of significant components to auto-select after fitting.
        If ``'auto'``, uses the method specified by ``selection_method``
        to determine significant components. The result is stored
        in :attr:`n_selected_`.
        If ``int``, uses that exact number.
        If ``None`` (default), no automatic selection is performed.
    selection_method : {'combined', 'outlier', 'ratio', 'max_gap'}, default='combined'
        Algorithm for automatic component selection when ``n_select='auto'``:

        - ``'outlier'``: Iterative outlier removal (mean + sigma × std).
          Works best when eigenvalue contrast is high (e.g., ZapLine with
          smoothing). Uses ``selection_threshold`` as the sigma parameter.
        - ``'ratio'``: Eigenvalue ratio test (scree test). Finds the first
          drop ≥ ``selection_threshold`` between consecutive eigenvalues.
          Works well for moderate eigenvalue contrast.
        - ``'max_gap'``: Maximum gap method. Finds the position of the
          biggest drop in the eigenvalue spectrum and uses it as the
          cutpoint. Most lenient method; works for weak artifacts.
        - ``'combined'`` (default): Cascade of all methods — outlier first,
          then ratio, then max_gap — returning the first non-zero result.
    selection_threshold : float, default=3.0
        Threshold for automatic component selection.
        For ``'outlier'`` method: sigma for outlier detection
        (components with eigenvalue > mean + sigma × std).
        For ``'ratio'`` method: minimum ratio between consecutive
        eigenvalues (default 3.0 means a 3× drop).
        For ``'combined'``: uses 3.0 for outlier, 2.0 for ratio fallback.
    rank : int or dict, optional
        Rank of the data for whitening. If None, rank is estimated automatically.
    reg : float
        Regularization for covariance estimation. Default 1e-9.
    normalize_input : bool
        If True, normalize input data channel-wise (L2 norm) before fitting/transforming.
        Useful when mixing sensors with different scales (e.g. MAG and GRAD). Default True.
        Ignored when ``whiten=True`` (the whitener handles the scaling).
    cov_method : str
        Method for covariance estimation.
        For MNE objects, passed as `method` to `mne.compute_covariance`.
        For NumPy arrays, passed as `method` to `mne_denoise.utils.compute_covariance`.
        Default 'empirical'.
    cov_kws : dict, optional
        Additional keywords options for covariance estimation.
        For MNE objects, passed to `mne.compute_covariance` (e.g. `{'tstep': 0.1, 'rank': 'info'}`).
        For NumPy arrays, passed to `mne_denoise.utils.compute_covariance` (e.g. `{'shrinkage': 0.1}`).
    smooth : SmoothingBias | int | None, default=None
        Optional smoothing decomposition before DSS, inspired by ZapLine.
        When set, data is decomposed into ``smooth + residual`` and DSS
        is fitted/applied on the **residual** only.  This dramatically
        increases eigenvalue contrast for narrowband artifacts because
        DSS no longer competes against broadband EEG variance.

        - If ``SmoothingBias`` instance: used directly.
        - If ``int``: interpreted as the smoothing window in samples
          (e.g., ``int(sfreq / line_freq)`` for line noise).
        - If ``None`` (default): no smoothing, DSS is applied to the
          full data (original behavior).
    segmented : bool, default=False
        If ``True``, data is split into segments and DSS is fitted
        independently per segment.  This handles **non-stationary**
        artifacts whose spatial or spectral profile changes over
        time.  Requires :meth:`fit_transform`; calling :meth:`fit`
        alone raises an error.
    segmenter : CovarianceSegmenter | FixedWindowSegmenter | None, default=None
        Segmentation strategy.  If ``None`` and ``segmented=True``,
        a :class:`CovarianceSegmenter` is created automatically
        (requires ``sfreq`` to be determinable from the input or
        from the bias function).
    max_prop_remove : float | None, default=None
        Maximum proportion of channels that can be removed per segment.
        E.g. ``0.2`` caps ``n_selected`` at ``int(n_channels × 0.2)``.
        Safety valve to prevent over-cleaning.
    min_select : int, default=0
        Minimum components to select when ``n_select='auto'`` and
        the artifact is present.  Guarantees a floor on cleaning
        strength.  Only effective when ``segmented=True``.
    return_type : {'sources', 'epochs', 'raw'}
        Type of object to return from `transform`. 'sources' returns a numpy array
        of DSS components. 'epochs'/'raw' returns the denoised input object.
    whiten : bool, default=False
        If True, decompose all data channel types jointly (e.g. mag + grad + eeg)
        instead of isolating a single homogeneous type. The data is whitened
        before the DSS bias/covariance step and un-whitened on reconstruction, so
        channels with different physical units no longer contaminate one another.
    noise_cov : mne.Covariance | None, default=None
        Noise covariance used to build the whitener when ``whiten=True`` (MNE
        inputs only). If None, MNE inputs are scaled by channel type, matching
        MNE's ICA pre-whitening fallback; NumPy arrays are scaled per channel.
        Ignored when ``whiten=False``.
    verbose : bool | str | int | None, default=None
        Control logging verbosity.

    Attributes
    ----------
    filters_ : array, shape (n_components, n_channels)
        The spatial filters (un-mixing matrix).
    patterns_ : array, shape (n_channels, n_components)
        The spatial patterns (mixing matrix).
    eigenvalues_ : array, shape (n_components,)
        The power of each component in the biased data (bias score).
    n_selected_ : int | None
        Number of significant components detected by automatic selection.
        Only set when ``n_select`` is not ``None``. Use this to determine
        how many components to remove/keep in downstream processing.
    segment_results_ : list of dict | None
        Per-segment metadata when ``segmented=True``.  Each dict
        contains ``'start'``, ``'end'``, ``'n_selected'``,
        ``'eigenvalues'``, and ``'patterns'``.

    Examples
    --------
    >>> from mne_denoise.dss import DSS, BandpassBias
    >>> from mne_denoise.dss.denoisers import TrialAverageBias
    >>> # Create a bias (e.g. emphasize 10Hz oscillations)
    >>> bias = BandpassBias(sfreq=250, freq=10, bandwidth=2)
    >>> # Initialize DSS
    >>> dss = DSS(bias=bias, n_components=3)
    >>> # Fit on data (MNE Raw/Epochs or NumPy)
    >>> dss.fit(raw_data)
    >>> # Extract sources
    >>> sources = dss.transform(raw_data)
    >>> # Or return denoised data
    >>> dss.return_type = "raw"
    >>> denoised_raw = dss.transform(raw_data)

    See Also
    --------
    compute_dss : Functional interface for computing DSS solutions.
    """

    def __init__(
        self,
        bias: LinearDenoiser | Callable,
        n_components: int | None = None,
        n_select: int | str | None = None,
        selection_method: str = "combined",
        selection_threshold: float = 3.0,
        rank: int | dict | None = None,
        reg: float = 1e-9,
        normalize_input: bool = True,
        cov_method: str = "empirical",
        cov_kws: dict | None = None,
        smooth: LinearDenoiser | int | None = None,
        segmented: bool = False,
        segmenter: CovarianceSegmenter | FixedWindowSegmenter | None = None,
        max_prop_remove: float | None = None,
        min_select: int = 0,
        return_type: str = "sources",
        whiten: bool = False,
        noise_cov=None,
        verbose: bool | str | int | None = None,
    ) -> None:
        self.n_components = n_components
        self.bias = bias
        self.n_select = n_select
        self.selection_method = selection_method
        self.selection_threshold = selection_threshold
        self.rank = rank
        self.reg = reg
        self.normalize_input = normalize_input
        self.cov_method = cov_method
        self.cov_kws = cov_kws
        self.smooth = smooth
        self.segmented = segmented
        self.segmenter = segmenter
        self.max_prop_remove = max_prop_remove
        self.min_select = min_select
        self.return_type = return_type
        self.whiten = whiten
        self.noise_cov = noise_cov
        self.verbose = verbose
        set_log_level_from_verbose(self.verbose)

        # Fitted attributes
        self.filters_: np.ndarray | None = None
        self.patterns_: np.ndarray | None = None
        self.mixing_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.channel_norms_: np.ndarray | None = None
        self.n_selected_: np.ndarray | None = None
        self.segment_results_: list | None = None
        self._whitener_: np.ndarray | None = None
        self._dewhitener_: np.ndarray | None = None
        self._smoother = None  # Resolved SmoothingBias instance
        self._mne_info = None

    def _resolve_smoother(self):
        """Resolve the ``smooth`` parameter to a ``SmoothingBias`` instance."""
        from .denoisers.temporal import SmoothingBias

        if self.smooth is None:
            self._smoother = None
        elif isinstance(self.smooth, int):
            self._smoother = SmoothingBias(window=self.smooth, iterations=1)
        elif isinstance(self.smooth, SmoothingBias):
            self._smoother = self.smooth
        elif hasattr(self.smooth, "apply"):
            # Duck-type: any LinearDenoiser with .apply() method
            self._smoother = self.smooth
        else:
            raise TypeError(
                f"smooth must be SmoothingBias, int, or None, "
                f"got {type(self.smooth)}"
            )

    def _decompose_smooth(self, data: np.ndarray):
        """Decompose data into smooth and residual components.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times) or (n_ch, n_times, n_ep)
            Input data.

        Returns
        -------
        data_smooth : ndarray
            Smoothed (low-frequency / broadband) component.
        data_residual : ndarray
            Residual (narrowband / artifact) component.
        """
        if self._smoother is None:
            return None, data

        data_smooth = self._smoother.apply(data)
        data_residual = data - data_smooth
        return data_smooth, data_residual

    def fit(
        self,
        X: BaseRaw | BaseEpochs | Evoked | np.ndarray,
        y=None,
        weights: np.ndarray | None = None,
    ) -> DSS:
        """Compute DSS spatial filters.

        Parameters
        ----------
        X : Raw | Epochs | Evoked | array
            The data to fit.
            - If array, shape must be:
              - `(n_channels, n_times)` for continuous data.
              - `(n_channels, n_times, n_epochs)` for epoch data (evoked DSS).
              - `(n_datasets, n_channels, n_times)` for group data (Joint DSS).
            Note: For group DSS, you must reshape your list of datasets into a 3D array before fitting.
        y : None
            Ignored.
        weights : array, shape (n_times,), optional
             Sample weights for covariance computation. Only used if input is numpy array
             or if internal logic supports weighted covariance for MNE objects.

        Returns
        -------
        self : DSS
            The fitted transformer.
        """
        set_log_level_from_verbose(self.verbose)
        if self.segmented:
            raise RuntimeError(
                "Segmented mode requires simultaneous fit and transform. "
                "Use fit_transform() instead."
            )

        if self.whiten:
            # Joint multi-sensor decomposition: the whitener replaces the
            # channel-wise normalization and the homogeneous-type isolation.
            self._fit_whitened(X, weights=weights)
            self.mixing_ = self.patterns_
            return self

        if self.normalize_input:
            X_norm = self._normalize(X, fit=True)
        else:
            X_norm = X

        # Resolve smoothing (if configured)
        self._resolve_smoother()

        # If smoothing is enabled, decompose and fit on residual only
        if self._smoother is not None:
            data, _, mne_type, _ = extract_data_from_mne(X_norm)
            if mne_type == "epochs":
                data = np.transpose(data, (1, 2, 0))

            _, data_residual = self._decompose_smooth(data)
            # Fit DSS on residual (always numpy path)
            self._fit_numpy(data_residual, weights=weights)
        elif mne is not None and isinstance(X_norm, BaseRaw | BaseEpochs | Evoked):
            self._fit_mne(X_norm, weights=weights)
        elif isinstance(X_norm, np.ndarray):
            self._fit_numpy(X_norm, weights=weights)
        else:
            raise TypeError(f"Unsupported input type: {type(X_norm)}")

        # Compute mixing matrix (pseudoinverse of filters)
        self.mixing_ = np.linalg.pinv(self.filters_)

        # Automatic component selection
        if self.n_select is not None and self.eigenvalues_ is not None:
            self.n_selected_ = self.auto_select()

        return self

    def auto_select(self, threshold=None, method=None):
        """Automatically determine the number of significant DSS components.

        Supports multiple selection strategies:

        - **outlier**: Iterative outlier removal (mean + sigma × std).
          Best for high eigenvalue contrast (e.g., after smoothing).
        - **ratio**: Eigenvalue ratio / scree test. Finds the first large
          drop between consecutive eigenvalues. For moderate contrast.
        - **max_gap**: Maximum gap method. Finds the *biggest* drop in
          the eigenvalue spectrum. Most lenient; works for weak artifacts.
        - **combined**: Cascade — outlier → ratio → max_gap — returns
          the first non-zero result.

        This method is called automatically during :meth:`fit` when
        ``n_select`` is set. It can also be called manually after fitting
        with a different threshold or method.

        Parameters
        ----------
        threshold : float | None
            Override the threshold.  If ``None``, uses
            ``self.selection_threshold``.
        method : {'outlier', 'ratio', 'max_gap', 'combined'} | None
            Override the selection method.  If ``None``, uses
            ``self.selection_method``.

        Returns
        -------
        n_selected : int
            Number of significant components detected.

        Raises
        ------
        RuntimeError
            If the estimator has not been fitted yet.

        Examples
        --------
        >>> dss = DSS(bias=my_bias, n_components=30)
        >>> dss.fit(raw)
        >>> n = dss.auto_select(threshold=2.5, method='outlier')
        >>> print(f"{n} significant components at sigma=2.5")
        """
        if self.eigenvalues_ is None:
            raise RuntimeError("DSS not fitted. Call fit() first.")

        from .utils.selection import (
            eigenvalue_ratio_selection,
            iterative_outlier_removal,
            max_gap_selection,
        )

        threshold = threshold if threshold is not None else self.selection_threshold
        method = method if method is not None else self.selection_method

        if isinstance(self.n_select, int):
            return min(self.n_select, len(self.eigenvalues_))

        if method == "outlier":
            return iterative_outlier_removal(self.eigenvalues_, threshold)
        elif method == "ratio":
            return eigenvalue_ratio_selection(self.eigenvalues_, threshold)
        elif method == "max_gap":
            return max_gap_selection(self.eigenvalues_, min_ratio=min(threshold, 1.2))
        elif method == "combined":
            # Tier 1: Outlier removal (strict — needs high contrast)
            n = iterative_outlier_removal(self.eigenvalues_, threshold)
            if n > 0:
                return n
            # Tier 2: Ratio test (moderate — needs a clear drop)
            ratio_th = min(threshold, 2.0)
            n = eigenvalue_ratio_selection(self.eigenvalues_, ratio_th)
            if n > 0:
                return n
            # Tier 3: Max gap (lenient — finds the biggest drop wherever)
            n = max_gap_selection(self.eigenvalues_, min_ratio=1.2)
            return n
        else:
            raise ValueError(
                f"Unknown selection method '{method}'. "
                "Choose from 'outlier', 'ratio', 'max_gap', or 'combined'."
            )

    def _normalize(
        self, X: BaseRaw | BaseEpochs | Evoked | np.ndarray, fit: bool = False
    ) -> BaseRaw | BaseEpochs | Evoked | np.ndarray:
        """Normalize data channel-wise.

        This mimics MNE's Scaling capabilities, ensuring channels with different
        units (e.g. MAG vs GRAD) contribute equally.
        """
        # Helper to get numpy data
        is_mne = False
        mne_type = None
        if mne is not None and isinstance(X, BaseRaw | BaseEpochs | Evoked):
            data = X.get_data()
            is_mne = True
            if isinstance(X, BaseEpochs):
                mne_type = "epochs"
                # MNE Epochs: (n_epochs, n_channels, n_times) -> (n_channels, n_times, n_epochs)
                data = np.transpose(data, (1, 2, 0))
            elif isinstance(X, Evoked):
                mne_type = "evoked"
            else:
                mne_type = "raw"
        else:
            data = X

        # Now data is always (n_channels, ...) for both 2D and 3D
        orig_shape = data.shape
        if data.ndim == 3:
            n_ch, n_times, n_epochs = data.shape
            data_2d = data.reshape(n_ch, -1)
        else:
            n_ch, n_times = data.shape
            data_2d = data

        if fit:
            # unique norms per channel
            self.channel_norms_ = np.linalg.norm(data_2d, axis=1)
            # Avoid division by zero
            self.channel_norms_ = np.where(
                self.channel_norms_ > 0, self.channel_norms_, 1.0
            )

        # Apply normalization
        data_norm = data_2d / self.channel_norms_[:, np.newaxis]

        # Reshape back
        if len(orig_shape) == 3:
            data_norm = data_norm.reshape(orig_shape)

        if is_mne:
            if mne_type == "raw":
                out = mne.io.RawArray(data_norm, X.info.copy(), verbose=False)
                # Preserve annotations
                if hasattr(X, "annotations") and X.annotations is not None:
                    out.set_annotations(X.annotations)
                return out
            elif mne_type == "epochs":
                # Transpose back to MNE format: (n_ch, n_times, n_epochs) -> (n_epochs, n_ch, n_times)
                data_norm = np.transpose(data_norm, (2, 0, 1))
                out = mne.EpochsArray(
                    data_norm,
                    X.info.copy(),
                    events=getattr(X, "events", None),
                    tmin=getattr(X, "tmin", 0),
                    event_id=getattr(X, "event_id", None),
                    verbose=False,
                )
                # Preserve metadata
                if hasattr(X, "metadata") and X.metadata is not None:
                    out.metadata = X.metadata.copy()
                return out
            else:  # Evoked
                out = mne.EvokedArray(
                    data_norm,
                    X.info.copy(),
                    tmin=getattr(X, "tmin", 0),
                    comment=getattr(X, "comment", ""),
                    nave=getattr(X, "nave", 1),
                    verbose=False,
                )
                return out
        else:
            return data_norm

    def _apply_bias(self, data: np.ndarray) -> np.ndarray:
        """Apply bias function to data."""
        if hasattr(self.bias, "apply"):
            return self.bias.apply(data)
        else:
            return self.bias(data)

    def _fit_mne(
        self,
        inst: BaseRaw | BaseEpochs | Evoked,
        weights: np.ndarray | None = None,
    ) -> None:
        """Fit using MNE objects."""
        self.info_ = inst.info

        if weights is not None:
            # If weights provided, extract data and use numpy path
            data = inst.get_data()
            self._fit_numpy(data, weights=weights)
            return

        method = self.cov_method
        kws = self.cov_kws.copy() if self.cov_kws else {}
        # Set defaults if not in kws
        kws.setdefault("rank", self.rank)
        kws.setdefault("verbose", False)

        data, _, mne_type, _, picks, ch_names = extract_data_from_mne(inst)
        self._mne_ch_names_ = ch_names

        # MNE covariance computation requires the inst object to match the array
        if picks is not None:
            inst = inst.copy().pick(picks)

        if mne_type == "epochs":
            # DSS transpose preference
            data = np.transpose(data, (1, 2, 0))

        biased_data = self._apply_bias(data)

        if isinstance(inst, BaseEpochs):
            biased_data = np.transpose(biased_data, (2, 0, 1))

        if isinstance(inst, BaseRaw):
            kws.setdefault("tstep", 2.0)
            baseline_cov = mne.compute_raw_covariance(inst, method=method, **kws)
            biased_inst = mne.io.RawArray(biased_data, inst.info, verbose=False)
            biased_cov = mne.compute_raw_covariance(biased_inst, method=method, **kws)

        elif isinstance(inst, BaseEpochs):
            baseline_cov = mne.compute_covariance(inst, method=method, **kws)
            biased_inst = mne.EpochsArray(biased_data, inst.info, verbose=False)
            biased_cov = mne.compute_covariance(biased_inst, method=method, **kws)

        else:  # Evoked - use numpy path since MNE doesn't support Evoked covariance
            self._fit_numpy(data, weights=weights)
            return

        # Extract data from MNE covariances
        self.filters_, self.patterns_, self.eigenvalues_ = compute_dss(
            covariance_baseline=baseline_cov.data,
            covariance_biased=biased_cov.data,
            n_components=self.n_components,
            reg=self.reg,
        )

        # Calculate explained variance from filters and baseline covariance
        # Diag(filters @ baseline_cov.data @ filters.T)
        sources_cov = self.filters_ @ baseline_cov.data @ self.filters_.T
        self.explained_variance_ = np.diag(sources_cov)

    def _fit_numpy(self, X: np.ndarray, weights: np.ndarray | None = None) -> None:
        """Fit using numpy arrays."""
        biased_X = self._apply_bias(X)

        method = self.cov_method
        kws = self.cov_kws.copy() if self.cov_kws else {}

        baseline_cov = compute_covariance(X, method=method, weights=weights, **kws)
        biased_cov = compute_covariance(biased_X, method=method, weights=weights, **kws)

        # Use rank if provided (compute from covariance if not)
        rank = None
        if self.rank is not None and isinstance(self.rank, int):
            rank = self.rank
            # If rank is a dict (MNE style), ignore for numpy

        self.filters_, self.patterns_, self.eigenvalues_ = compute_dss(
            covariance_baseline=baseline_cov,
            covariance_biased=biased_cov,
            n_components=self.n_components,
            rank=rank,
            reg=self.reg,
        )

        # Calculate explained variance
        sources_cov = self.filters_ @ baseline_cov @ self.filters_.T
        self.explained_variance_ = np.diag(sources_cov)

    def _fit_whitened(
        self,
        X: BaseRaw | BaseEpochs | Evoked | np.ndarray,
        weights: np.ndarray | None = None,
    ) -> None:
        """Fit DSS on all data channels jointly after whitening.

        The whitener ``W`` is baked into ``filters_`` and its inverse into
        ``patterns_``/``mixing_`` so that ``transform`` and ``inverse_transform``
        operate in sensor units without any further change.
        """
        method = self.cov_method
        kws = self.cov_kws.copy() if self.cov_kws else {}
        # The NumPy covariance helper does not accept MNE-only options.
        for key in ("rank", "verbose", "tstep"):
            kws.pop(key, None)

        data, _, _, orig_inst, _, ch_names = extract_data_from_mne(
            X,
            auto_pick="data",
            channel_first_epochs=True,
        )
        self._mne_ch_names_ = ch_names
        self.info_ = orig_inst.info if orig_inst is not None else None
        self._mne_info = self.info_

        data_w = self._prewhiten_sensor_data(
            data,
            info=self.info_,
            ch_names=ch_names,
        )
        biased_w = self._apply_bias(data_w)

        baseline_cov = compute_covariance(data_w, method=method, weights=weights, **kws)
        biased_cov = compute_covariance(biased_w, method=method, weights=weights, **kws)

        rank = self.rank if isinstance(self.rank, int) else None
        filters_w, patterns_w, self.eigenvalues_ = compute_dss(
            baseline_cov,
            biased_cov,
            n_components=self.n_components,
            rank=rank,
            reg=self.reg,
        )

        # Store the fitted spatial matrices in the original sensor coordinates.
        self.filters_, self.patterns_ = map_spatial_matrices_to_sensor_space(
            filters_w,
            patterns_w,
            whitener=self._whitener_,
            dewhitener=self._dewhitener_,
        )
        self.explained_variance_ = np.diag(filters_w @ baseline_cov @ filters_w.T)

    def _prewhiten_sensor_data(
        self,
        data: np.ndarray,
        *,
        info=None,
        ch_names: list[str] | None = None,
    ) -> np.ndarray:
        """Fit the configured sensor whitener and apply it to data."""
        whitener, dewhitener = compute_mne_sensor_whitener(
            data,
            info=info,
            ch_names=ch_names,
            noise_cov=self.noise_cov,
            rank=self.rank,
        )
        self._whitener_ = whitener
        self._dewhitener_ = dewhitener
        return apply_spatial_transform(whitener, data)

    def transform(
        self, X: BaseRaw | BaseEpochs | Evoked | np.ndarray
    ) -> np.ndarray | BaseRaw | BaseEpochs | Evoked:
        """Apply DSS spatial filters.

        Parameters
        ----------
        X : Raw | Epochs | Evoked | array
            Data to transform.
            - If array, must match the shape convention used in fit (see fit docstring).

        Returns
        -------
        out : array | Raw | Epochs | Evoked
            If return_type='sources', returns the source time series.
            If return_type='raw'/'epochs'/'evoked', returns the reconstructed data (denoised)
            projected back to sensor space (keeping n_components).
        """
        set_log_level_from_verbose(self.verbose)
        if self.filters_ is None:
            raise RuntimeError("DSS not fitted. Call fit() first.")

        if self.normalize_input and not self.whiten:
            # Apply normalization using fitted norms
            X_in = self._normalize(X, fit=False)
        else:
            X_in = X

        # Helper to extract data
        data, _, mne_type, orig_inst, picks, _ = extract_data_from_mne(
            X_in, ch_names=getattr(self, "_mne_ch_names_", None)
        )

        # DSS internal convention for Epochs: (n_channels, n_times, n_epochs)
        if mne_type == "epochs":
            data = np.transpose(data, (1, 2, 0))

        # If smoothing is enabled, project the residual (not full data)
        if self._smoother is not None:
            data_smooth, data_for_dss = self._decompose_smooth(data)
        else:
            data_smooth = None
            data_for_dss = data

        orig_shape = data.shape
        if data_for_dss.ndim == 3:
            n_ch, n_times, n_epochs = data_for_dss.shape
            data_2d = data_for_dss.reshape(n_ch, -1)
        else:
            n_ch, n_times = data_for_dss.shape
            data_2d = data_for_dss

        # Center using mean on data_2d
        # DSS implies zero-mean assumption for correct projection
        mean_ = data_2d.mean(axis=1, keepdims=True)
        data_centered = data_2d - mean_

        sources = self.filters_ @ data_centered

        if self.return_type == "sources":
            if len(orig_shape) == 3:
                sources = sources.reshape(
                    self.n_components or sources.shape[0], n_times, n_epochs
                )
                if mne_type == "epochs":
                    # Return as (n_epochs, n_components, n_times)
                    return sources.transpose(2, 0, 1)
            return sources

        # Use only kept components
        n_keep = self.n_components if self.n_components else self.filters_.shape[0]
        # mixing shape: (n_channels, n_components)
        rec = self.mixing_[:, :n_keep] @ sources[:n_keep]
        rec += mean_

        # Add back smooth component if it was separated
        if data_smooth is not None:
            smooth_2d = (
                data_smooth.reshape(data_smooth.shape[0], -1)
                if data_smooth.ndim == 3
                else data_smooth
            )
            rec = rec + smooth_2d

        # Reshape to original
        if len(orig_shape) == 3:
            rec = rec.reshape(orig_shape)  # (n_ch, n_times, n_epochs)

        # De-normalization
        if self.normalize_input and not self.whiten:
            if len(orig_shape) == 3:  # (n_ch, n_times, n_epochs)
                rec = rec * self.channel_norms_[:, np.newaxis, np.newaxis]
            else:  # (n_ch, n_times)
                rec = rec * self.channel_norms_[:, np.newaxis]

        # Prepare for reconstruction (transpose back if needed)
        if mne_type == "epochs":
            rec = np.transpose(rec, (2, 0, 1))

        return reconstruct_mne_object(
            rec, orig_inst, mne_type, picks=picks, verbose=False
        )

    def inverse_transform(
        self, sources: np.ndarray, component_indices: np.ndarray | None = None
    ) -> np.ndarray:
        """Transform sources back to sensor space.

        Parameters
        ----------
        sources : array, shape (n_components, n_times)
            The latent sources.
        component_indices : array-like of bool or int, optional
            Indices of components to keep. If None, keep all.

        Returns
        -------
        reconstructed : array, shape (n_channels, n_times)
            The reconstructed sensor space data.
        """
        if self.filters_ is None:
            raise RuntimeError("DSS not fitted. Call fit() first.")
        is_epochs_mne = False

        if sources.ndim == 3:
            # Determine orientation: sources from transform() are
            # (n_comps, n_times, n_epochs) for numpy or (n_epochs, n_comps, n_times) for MNE epochs
            # Use shape[0] vs mixing_.shape[1] to detect MNE epoch format
            n_comp_fit = self.mixing_.shape[1]
            if sources.shape[0] != n_comp_fit and sources.shape[1] == n_comp_fit:
                # MNE epochs format: (n_epochs, n_comps, n_times) -> (n_comps, n_times, n_epochs)
                sources_internal = np.transpose(sources, (1, 2, 0))
                is_epochs_mne = True
            else:
                sources_internal = sources
        else:
            sources_internal = sources

        n_comp_sources = sources_internal.shape[0]
        patterns = self.mixing_[:, :n_comp_sources]

        if component_indices is not None:
            # Make a copy to avoid modifying input
            sources_used = sources_internal.copy()
            mask = np.array(component_indices)

            # Handle boolean mask
            if mask.dtype == bool:
                if len(mask) != n_comp_sources:
                    raise ValueError(
                        f"Mask length {len(mask)} != n_sources {n_comp_sources}"
                    )
                sources_used[~mask] = 0
            else:
                # Handle integer indices
                # Create a boolean mask from indices
                full_mask = np.zeros(n_comp_sources, dtype=bool)
                full_mask[mask] = True
                sources_used[~full_mask] = 0

            rec_internal = np.tensordot(patterns, sources_used, axes=(1, 0))
        else:
            rec_internal = np.tensordot(patterns, sources_internal, axes=(1, 0))

        if is_epochs_mne:
            # rec_internal: (n_ch, n_times, n_epochs) -> (n_epochs, n_ch, n_times)
            rec = np.transpose(rec_internal, (2, 0, 1))
        else:
            rec = rec_internal

        if self.normalize_input and not self.whiten:
            # rec is (n_epochs, n_ch, n_times) OR (n_ch, n_times, n_epochs) OR (n_ch, n_times)
            if is_epochs_mne:
                rec = rec * self.channel_norms_[np.newaxis, :, np.newaxis]
            elif rec.ndim == 3:  # (n_ch, n_times, n_epochs)
                rec = rec * self.channel_norms_[:, np.newaxis, np.newaxis]
            else:  # (n_ch, n_times)
                rec = rec * self.channel_norms_[:, np.newaxis]

        return rec

    # -----------------------------------------------------------------
    # Segmented mode
    # -----------------------------------------------------------------

    def fit_transform(
        self, X, y=None, **fit_params
    ):
        """Fit and transform data in one step.

        In **segmented mode** (``segmented=True``), the data is split into
        segments and each segment gets its own independent DSS fit +
        cleaning pass.  This is the only entry-point for segmented
        processing because ``fit()`` alone is not meaningful when
        filters differ per segment.

        In standard mode, this is equivalent to
        ``self.fit(X).transform(X)``.

        Parameters
        ----------
        X : Raw | Epochs | Evoked | ndarray
            The data to process.
        y : None
            Ignored.
        **fit_params
            Additional keyword arguments forwarded to :meth:`fit`.

        Returns
        -------
        X_out : ndarray | Raw | Epochs | Evoked
            In segmented mode, returns cleaned data (same type as input).
            In standard mode with ``return_type='sources'``, returns DSS
            source time-series.  With any other ``return_type``, returns
            cleaned (denoised) data produced by subtracting the artifact
            captured by the first ``n_selected_`` components.
        """
        if not self.segmented:
            self.fit(X, **fit_params)

            if self.return_type == "sources":
                return self.transform(X)

            # ── Denoise via artifact subtraction ──
            data, _, mne_type, orig_inst = extract_data_from_mne(X)

            n_remove = self.n_selected_ if self.n_selected_ is not None else 0
            if n_remove > 0:
                # Temporarily switch to get source time-series
                saved_rt = self.return_type
                self.return_type = "sources"
                try:
                    sources = self.transform(X)
                finally:
                    self.return_type = saved_rt

                artifact = self.inverse_transform(
                    sources, component_indices=np.arange(n_remove)
                )
                cleaned = data - artifact
            else:
                cleaned = data

            return reconstruct_mne_object(
                cleaned, orig_inst, mne_type, verbose=False
            )

        # --- segmented mode ---
        data, extracted_sfreq, mne_type, orig_inst = extract_data_from_mne(X)

        # Determine sfreq
        sfreq = extracted_sfreq
        if sfreq is None and hasattr(self.bias, "sfreq"):
            sfreq = self.bias.sfreq
        if sfreq is None:
            raise ValueError(
                "Cannot determine sfreq for segmented mode. "
                "Pass an MNE object or use a bias with a .sfreq attribute."
            )

        # Handle epochs: concatenate into continuous
        is_epochs = False
        if data.ndim == 3:
            is_epochs = True
            n_ep, n_ch, n_t = data.shape
            data_cont = np.transpose(data, (1, 0, 2)).reshape(n_ch, -1)
        else:
            data_cont = data

        # Resolve smoother once
        self._resolve_smoother()

        # Run segmented processing
        cleaned = self._run_segmented(data_cont, sfreq)

        # Reshape back if epochs
        if is_epochs:
            cleaned = cleaned.reshape(n_ch, n_ep, n_t).transpose(1, 0, 2)

        return reconstruct_mne_object(cleaned, orig_inst, mne_type, verbose=False)

    def _resolve_segmenter(self, sfreq: float):
        """Resolve the segmenter parameter.

        If ``self.segmenter`` is ``None``, creates a default
        :class:`CovarianceSegmenter` with optional bandpass from the
        bias function.

        Parameters
        ----------
        sfreq : float
            Sampling frequency in Hz.

        Returns
        -------
        segmenter : CovarianceSegmenter | FixedWindowSegmenter
        """
        if self.segmenter is not None:
            return self.segmenter

        # Build a default CovarianceSegmenter
        bandpass = None
        # If the bias has a target frequency, focus segmentation around it
        if hasattr(self.bias, "freq") and self.bias.freq is not None:
            f = float(self.bias.freq)
            bandpass = (max(1.0, f - 3), min(sfreq / 2 - 1, f + 3))

        return CovarianceSegmenter(
            sfreq=sfreq,
            min_chunk_len=30.0,
            bandpass=bandpass,
        )

    def _run_segmented(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        """Run segmented fit-transform on continuous data.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times)
            Continuous data.
        sfreq : float
            Sampling frequency.

        Returns
        -------
        cleaned : ndarray, shape (n_channels, n_times)
            Cleaned data (segments are concatenated).
        """
        import logging

        logger = logging.getLogger(__name__)

        segmenter = self._resolve_segmenter(sfreq)
        segments = segmenter.segment(data)

        if not segments:
            raise ValueError(
                "Segmenter returned no segments. Check segmenter settings "
                "and data length."
            )

        logger.info(
            f"Segmented DSS: {len(segments)} segment(s) "
            f"over {data.shape[1] / sfreq:.1f}s"
        )

        self.segment_results_ = []
        cleaned_chunks = []
        per_segment_n_removed = []

        for seg_idx, (start, end) in enumerate(segments):
            chunk = data[:, start:end]
            result = self._process_segment(chunk)

            cleaned_chunks.append(result["cleaned"])
            per_segment_n_removed.append(result["n_selected"])

            # Store per-segment metadata
            self.segment_results_.append(
                {
                    "start": start,
                    "end": end,
                    "n_selected": result["n_selected"],
                    "eigenvalues": result["eigenvalues"],
                    "patterns": result["patterns"],
                }
            )

            # Keep last segment's filters/patterns as representative
            if result["eigenvalues"] is not None:
                self.eigenvalues_ = result["eigenvalues"]
            if result["patterns"] is not None:
                self.patterns_ = result["patterns"]
            if result["filters"] is not None:
                self.filters_ = result["filters"]
                self.mixing_ = np.linalg.pinv(self.filters_)

        self.n_selected_ = max(per_segment_n_removed) if per_segment_n_removed else 0
        return np.concatenate(cleaned_chunks, axis=1)

    def _process_segment(self, chunk: np.ndarray) -> dict:
        """Process a single segment: fit DSS, select components, clean.

        Parameters
        ----------
        chunk : ndarray, shape (n_channels, n_times)
            Data segment.

        Returns
        -------
        result : dict
            Contains 'cleaned', 'n_selected', 'eigenvalues', 'patterns',
            'filters'.
        """
        n_channels = chunk.shape[0]

        # Create a fresh DSS for this segment (non-segmented)
        seg_dss = DSS(
            bias=self.bias,
            n_components=self.n_components,
            n_select=self.n_select,
            selection_method=self.selection_method,
            selection_threshold=self.selection_threshold,
            rank=self.rank if isinstance(self.rank, int | type(None)) else None,
            reg=self.reg,
            normalize_input=self.normalize_input,
            cov_method=self.cov_method,
            cov_kws=self.cov_kws,
            smooth=self.smooth,
            segmented=False,  # Do NOT recurse
        )

        seg_dss.fit(chunk)
        n_sel = seg_dss.n_selected_ if seg_dss.n_selected_ is not None else 0

        # Apply caps
        if self.max_prop_remove is not None:
            n_sel = min(n_sel, int(n_channels * self.max_prop_remove))
        n_sel = max(n_sel, self.min_select)

        # Clean the segment
        cleaned = self._clean_segment(chunk, seg_dss, n_sel)

        return {
            "cleaned": cleaned,
            "n_selected": n_sel,
            "eigenvalues": seg_dss.eigenvalues_,
            "patterns": seg_dss.patterns_,
            "filters": seg_dss.filters_,
        }

    def _clean_segment(
        self, data: np.ndarray, fitted_dss: DSS, n_remove: int
    ) -> np.ndarray:
        """Clean a segment by projecting out *n_remove* DSS components.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times)
            Segment data.
        fitted_dss : DSS
            A fitted DSS instance (with ``filters_``, ``mixing_``, etc.).
        n_remove : int
            Number of components to remove.

        Returns
        -------
        cleaned : ndarray, shape (n_channels, n_times)
        """
        if n_remove <= 0 or fitted_dss.filters_ is None:
            return data.copy()

        # Smoothing decomposition (if configured)
        if fitted_dss._smoother is not None:
            data_smooth, data_residual = fitted_dss._decompose_smooth(data)
        else:
            data_smooth = np.zeros_like(data)
            data_residual = data

        # Center residual before projection (DSS assumes zero-mean)
        mean_ = data_residual.mean(axis=1, keepdims=True)
        residual_centered = data_residual - mean_

        # Project residual through the top n_remove DSS filters
        sources = fitted_dss.filters_[:n_remove] @ residual_centered
        artifact = fitted_dss.mixing_[:, :n_remove] @ sources

        return data_smooth + (data_residual - artifact)
