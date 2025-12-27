"""Core nonlinear/iterative DSS algorithm and Estimator.

This module contains:
1. `iterative_dss`: The core mathematical implementation of Nonlinear DSS.
2. `IterativeDSS`: The Scikit-learn estimator compatible with MNE-Python objects or NumPy arrays.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

References
----------
.. [1] Särelä & Valpola (2005). Denoising Source Separation. J. Mach. Learn. Res., 6, 233-272.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import numpy as np

from .utils.whitening import whiten_data


def iterative_dss_one(
    X_whitened: np.ndarray,
    denoiser: Callable[[np.ndarray], np.ndarray],
    *,
    w_init: Optional[np.ndarray] = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    alpha: Union[float, Callable[[np.ndarray], float], None] = None,
    beta: Union[float, Callable[[np.ndarray], float], None] = None,
    gamma: Union[float, Callable[[np.ndarray, np.ndarray, int], float], None] = None,
) -> Tuple[np.ndarray, np.ndarray, int, bool]:
    """Fixed-point iteration for extracting a single DSS component.

    This implements Algorithm 1 from Särelä & Valpola (2005) [1]_ with optional
    Newton step acceleration (FastICA equivalence).
    
    The algorithm finds a spatial filter **w** that maximizes the objective:
    
    .. math:: J(w) = E[f(w^T X)^2]
    
    where f(·) is the nonlinear denoising function.
    
    **Algorithm**::
    
        Initialize: w = random unit vector
        
        Repeat until converged:
            1. source = w @ X                    # Project data → 1D source
            2. source_denoised = f(source)       # Apply nonlinearity
            3. source_denoised *= alpha          # (optional) signal normalization
            4. w_new = E[X · source_denoised]    # Gradient direction
            5. w_new += beta · w                 # (optional) Newton step
            6. w_new = normalize(w_new)          # Unit norm constraint
            7. w = w_old + gamma·(w_new - w_old) # (optional) relaxation
            8. Check convergence: |w · w_old| ≈ 1

    Parameters
    ----------
    X_whitened : ndarray, shape (n_components, n_times)
        Whitened data matrix. Must have identity covariance.
    denoiser : callable
        Nonlinear denoising function f(s) → s_denoised.
        Examples: TanhMaskDenoiser, GaussDenoiser, WienerMaskDenoiser.
    w_init : ndarray, shape (n_components,), optional
        Initial weight vector. If None, random initialization.
    max_iter : int
        Maximum iterations. Default 100.
    tol : float
        Convergence tolerance. Default 1e-6.
    alpha : float or callable, optional
        Signal normalization factor applied after denoising:
        ``source_denoised *= alpha``.
        Useful for denoisers with different output variance.
    beta : float or callable, optional
        Spectral shift parameter for Newton-like acceleration.
        For tanh denoiser: ``beta = -E[1 - tanh(s)²]`` (use ``beta_tanh``).
        For cubic denoiser: ``beta = -3`` (use ``beta_pow3``).
    gamma : float or callable, optional
        Learning rate / relaxation parameter. Controls step size:
        ``w = w_old + gamma · (w_new - w_old)``.
        Default None (gamma=1, full step).

    Returns
    -------
    w : ndarray, shape (n_components,)
        Optimal spatial filter (unit norm).
    source : ndarray, shape (n_times,)
        Extracted source time series.
    n_iter : int
        Number of iterations performed.
    converged : bool
        Whether the algorithm converged within max_iter.

    References
    ----------
    .. [1] Särelä & Valpola (2005). Denoising Source Separation. JMLR, 6, 233-272.
    """
    n_components, n_times = X_whitened.shape
    
    # Initialize weight vector
    if w_init is not None:
        w = w_init.copy()
    else:
        rng = np.random.default_rng()
        w = rng.standard_normal(n_components)
    
    # Normalize
    norm = np.linalg.norm(w)
    if norm < 1e-12:
        w = np.ones(n_components) / np.sqrt(n_components)
    else:
        w = w / norm
    
    converged = False
    n_iter = 0
    
    for iteration in range(max_iter):
        n_iter = iteration + 1
        w_old = w.copy()
        
        # Step 2: Extract source
        source = w @ X_whitened  # (n_times,)
        
        # Step 3: Apply denoiser
        source_denoised = denoiser(source)
        
        # Apply alpha (signal normalization)
        if alpha is not None:
            if callable(alpha):
                step_alpha = alpha(source)
            else:
                step_alpha = alpha
            source_denoised = step_alpha * source_denoised
        
        # Calculate beta step if applicable
        step_beta = 0.0
        if beta is not None:
            if callable(beta):
                step_beta = beta(source)
            else:
                step_beta = beta
        
        # Step 4: Update weights
        # w_new = E[X * f(s)] + beta * w
        # Standard DSS: w_new = E[X * f(s)]
        # FastICA Newton: beta = -E[f'(s)]
        gradient_part = X_whitened @ source_denoised / n_times
        w_new = gradient_part + step_beta * w
        
        # Step 5: Normalize
        norm = np.linalg.norm(w_new)
        if norm < 1e-12:
            # Denoiser killed the signal, reinitialize
            rng = np.random.default_rng(iteration)
            w = rng.standard_normal(n_components)
            w = w / np.linalg.norm(w)
            continue
        
        w_normalized = w_new / norm
        
        # Apply gamma (learning rate / relaxation)
        if gamma is not None:
            if callable(gamma):
                step_gamma = gamma(w_normalized, w_old, iteration)
            else:
                step_gamma = gamma
            # w = w_old + gamma * (w_new - w_old)
            w = w_old + step_gamma * (w_normalized - w_old)
            # Re-normalize after relaxation
            w = w / np.linalg.norm(w)
        else:
            w = w_normalized
        
        # Check convergence (using abs because sign can flip)
        correlation = np.abs(np.dot(w, w_old))
        if 1 - correlation < tol:
            converged = True
            break
    
    # Final source extraction
    source = w @ X_whitened
    
    return w, source, n_iter, converged


def iterative_dss(
    data: np.ndarray,
    denoiser: Callable[[np.ndarray], np.ndarray],
    n_components: int,
    *,
    method: str = 'deflation',
    rank: Optional[int] = None,
    reg: float = 1e-9,
    max_iter: int = 100,
    tol: float = 1e-6,
    w_init: Optional[np.ndarray] = None,
    verbose: bool = False,
    alpha: Union[float, Callable, None] = None,
    beta: Union[float, Callable, None] = None,
    gamma: Union[float, Callable, None] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract multiple DSS components using iterative (nonlinear) algorithm.

    This implements the Iterative DSS algorithm from Särelä & Valpola (2005) [1]_.
    Unlike linear DSS which uses a closed-form eigendecomposition, iterative DSS
    uses fixed-point iteration with a nonlinear denoising function.
    
    **Algorithm Overview**::
    
        1. Center data: X = X - mean(X)
        2. Whiten data: X_white = Whitener @ X  (identity covariance)
        3. Extract components using deflation or symmetric method:
           
           Deflation (sequential):
               For each component i = 1..n_components:
                   w_i = iterative_dss_one(X_deflated)
                   Orthogonalize w_i against w_1..w_{i-1}
                   Deflate: X_deflated -= w_i @ w_i.T @ X_deflated
           
           Symmetric (parallel):
               Initialize W = [w_1, ..., w_n] randomly
               Repeat until converged:
                   Update all w_i simultaneously
                   W = symmetric_orthogonalize(W)
        
        4. Convert filters to sensor space: filters = W @ Whitener

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
        Input multichannel data.
    denoiser : callable or NonlinearDenoiser
        Nonlinear denoising function f(s) → s_denoised.
        Examples: TanhMaskDenoiser, GaussDenoiser, WienerMaskDenoiser.
    n_components : int
        Number of components to extract.
    method : {'deflation', 'symmetric'}
        Component extraction method:
        
        - ``'deflation'``: Extract one-by-one, orthogonalizing after each.
          More stable, but order-dependent.
        - ``'symmetric'``: Update all simultaneously, then orthogonalize.
          Order-independent, may be less stable.
          
        Default ``'deflation'``.
    rank : int, optional
        Rank for whitening. If None, auto-determined from eigenvalue threshold.
    reg : float
        Regularization for whitening eigenvalue cutoff. Default 1e-9.
    max_iter : int
        Maximum iterations per component. Default 100.
    tol : float
        Convergence tolerance. Default 1e-6.
    w_init : ndarray, shape (n_components, n_whitened), optional
        Initial weight matrix. If None, random initialization.
    verbose : bool
        Print convergence info. Default False.
    alpha : float or callable, optional
        Signal normalization factor (see ``iterative_dss_one``).
    beta : float or callable, optional
        Newton step parameter (see ``iterative_dss_one``).
    gamma : float or callable, optional
        Learning rate / relaxation (see ``iterative_dss_one``).

    Returns
    -------
    filters : ndarray, shape (n_components, n_channels)
        DSS spatial filters in sensor space.
        Apply as: ``sources = filters @ data``.
    sources : ndarray, shape (n_components, n_times)
        Extracted source time series.
    patterns : ndarray, shape (n_channels, n_components)
        Spatial patterns for visualization / reconstruction.
        Reconstruct as: ``data_recon = patterns @ sources``.
    convergence_info : ndarray, shape (n_components, 2)
        ``[n_iterations, converged]`` for each component.

    See Also
    --------
    iterative_dss_one : Single component extraction.
    IterativeDSS : Sklearn-style estimator wrapper.

    References
    ----------
    .. [1] Särelä & Valpola (2005). Denoising Source Separation. JMLR, 6, 233-272.
    """
    # Handle 3D epoched data
    if data.ndim == 3:
        n_channels, n_times, n_epochs = data.shape
        data = data.reshape(n_channels, -1)
    elif data.ndim != 2:
        raise ValueError(f"Data must be 2D or 3D, got {data.ndim}D")
    
    n_channels, n_samples = data.shape
    
    # Center data
    data = data - data.mean(axis=1, keepdims=True)
    
    # Whiten data
    X_whitened, whitener, dewhitener = whiten_data(data, rank=rank, reg=reg)
    n_whitened = X_whitened.shape[0]
    
    # Limit components to whitened dimension
    n_components = min(n_components, n_whitened)
    
    if method == 'deflation':
        filters_whitened, sources, convergence_info = _iterative_dss_deflation(
            X_whitened, denoiser, n_components,
            max_iter=max_iter, tol=tol, w_init=w_init, verbose=verbose,
            alpha=alpha, beta=beta, gamma=gamma
        )
    elif method == 'symmetric':
        filters_whitened, sources, convergence_info = _iterative_dss_symmetric(
            X_whitened, denoiser, n_components,
            max_iter=max_iter, tol=tol, w_init=w_init, verbose=verbose,
            alpha=alpha, beta=beta, gamma=gamma
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'deflation' or 'symmetric'")
    
    # Convert filters from whitened to sensor space
    # filters_whitened: (n_components, n_whitened)
    # whitener: (n_whitened, n_channels)
    # sensor_filter = whitened_filter @ whitener
    filters = filters_whitened @ whitener  # (n_components, n_channels)
    
    # Compute patterns using covariance
    # patterns = C @ filters.T, normalized
    C = data @ data.T / n_samples
    patterns = C @ filters.T
    pattern_norms = np.linalg.norm(patterns, axis=0)
    pattern_norms = np.where(pattern_norms > 1e-12, pattern_norms, 1.0)
    patterns = patterns / pattern_norms
    
    return filters, sources, patterns, convergence_info


def _iterative_dss_deflation(
    X_whitened: np.ndarray,
    denoiser: Callable,
    n_components: int,
    *,
    max_iter: int,
    tol: float,
    w_init: Optional[np.ndarray],
    verbose: bool,
    alpha: Union[float, Callable, None] = None,
    beta: Union[float, Callable, None] = None,
    gamma: Union[float, Callable, None] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract components one-by-one using deflation.
    
    **Algorithm**::
    
        For i = 1..n_components:
            1. w_i = iterative_dss_one(X_deflated)     # Extract one component
            2. Orthogonalize: w_i -= W_prev.T @ (W_prev @ w_i)
            3. Normalize: w_i = w_i / ||w_i||
            4. s_i = w_i @ X_whitened                  # Extract source
            5. Deflate: X_deflated -= w_i @ w_i.T @ X_deflated
    
    Parameters
    ----------
    X_whitened : ndarray, shape (n_whitened, n_times)
        Whitened data with identity covariance.
    denoiser : callable
        Nonlinear denoising function.
    n_components : int
        Number of components to extract.
    max_iter, tol : int, float
        Convergence parameters passed to ``iterative_dss_one``.
    w_init : ndarray, optional
        Initial weight matrix.
    verbose : bool
        Print progress.
    alpha, beta, gamma : optional
        Passed to ``iterative_dss_one``.
    
    Returns
    -------
    W : ndarray, shape (n_components, n_whitened)
        Weight matrix (spatial filters in whitened space).
    sources : ndarray, shape (n_components, n_times)
        Extracted source time series.
    convergence_info : ndarray, shape (n_components, 2)
        [n_iter, converged] per component.
    """
    n_whitened, n_times = X_whitened.shape
    
    # Storage
    W = np.zeros((n_components, n_whitened))
    sources = np.zeros((n_components, n_times))
    convergence_info = np.zeros((n_components, 2))
    
    X_deflated = X_whitened.copy()
    
    for i in range(n_components):
        # Get initial weight
        if w_init is not None and i < w_init.shape[0]:
            w_i = w_init[i]
        else:
            w_i = None
        
        # Run single-component iteration
        w, source, n_iter, converged = iterative_dss_one(
            X_deflated, denoiser,
            w_init=w_i, max_iter=max_iter, tol=tol,
            alpha=alpha, beta=beta, gamma=gamma
        )
        
        if verbose:
            status = "converged" if converged else "max_iter"
            print(f"  Component {i+1}: {n_iter} iterations ({status})")
        
        # Orthogonalize against previous components (vectorized)
        if i > 0:
            W_prev = W[:i]  # (i, n_whitened)
            w = w - W_prev.T @ (W_prev @ w)
            norm = np.linalg.norm(w)
            if norm < 1e-12:
                if verbose:
                    print(f"  Component {i+1}: degenerate, using random")
                rng = np.random.default_rng(i)
                w = rng.standard_normal(n_whitened)
                w = w - W_prev.T @ (W_prev @ w)
                norm = np.linalg.norm(w)
            w = w / norm
        
        W[i] = w
        sources[i] = w @ X_whitened
        convergence_info[i] = [n_iter, float(converged)]
        
        # Deflate: remove component from data
        outer = np.outer(w, w)
        X_deflated = X_deflated - outer @ X_deflated
    
    return W, sources, convergence_info


def _iterative_dss_symmetric(
    X_whitened: np.ndarray,
    denoiser: Callable,
    n_components: int,
    *,
    max_iter: int,
    tol: float,
    w_init: Optional[np.ndarray],
    verbose: bool,
    alpha: Union[float, Callable, None] = None,
    beta: Union[float, Callable, None] = None,
    gamma: Union[float, Callable, None] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract components simultaneously with symmetric orthogonalization.
    
    **Algorithm**::
    
        Initialize: W = random (n_components, n_whitened)
        W = symmetric_orthogonalize(W)
        
        Repeat until converged:
            For each i:
                s_i = W[i] @ X
                s_denoised = f(s_i)
                W[i] = E[X · s_denoised] + beta · W[i]
            W = symmetric_orthogonalize(W)    # (W @ W.T)^{-1/2} @ W
            Check max |1 - |W[i] · W_old[i]|| < tol
    
    Parameters
    ----------
    X_whitened : ndarray, shape (n_whitened, n_times)
        Whitened data with identity covariance.
    denoiser : callable
        Nonlinear denoising function.
    n_components : int
        Number of components to extract.
    max_iter, tol : int, float
        Convergence parameters.
    w_init : ndarray, optional
        Initial weight matrix.
    verbose : bool
        Print progress.
    alpha, beta, gamma : optional
        Iteration parameters.
    
    Returns
    -------
    W : ndarray, shape (n_components, n_whitened)
        Weight matrix (spatial filters in whitened space).
    sources : ndarray, shape (n_components, n_times)
        Extracted source time series.
    convergence_info : ndarray, shape (n_components, 2)
        [n_iter, converged] (same for all components in symmetric).
    """
    n_whitened, n_times = X_whitened.shape
    
    # Initialize weight matrix
    if w_init is not None:
        W = w_init[:n_components, :n_whitened].copy()
    else:
        rng = np.random.default_rng()
        W = rng.standard_normal((n_components, n_whitened))
    
    # Symmetric orthogonalization (decorrelation)
    W = _symmetric_orthogonalize(W)
    
    convergence_info = np.zeros((n_components, 2))
    
    for iteration in range(max_iter):
        W_old = W.copy()
        
        # Update all components
        for i in range(n_components):
            source = W[i] @ X_whitened
            source_denoised = denoiser(source)
            
            # Apply alpha (signal normalization)
            if alpha is not None:
                if callable(alpha):
                    step_alpha = alpha(source)
                else:
                    step_alpha = alpha
                source_denoised = step_alpha * source_denoised
            
            gradient_part = X_whitened @ source_denoised / n_times
            
            # Calculate beta step if applicable
            step_beta = 0.0
            if beta is not None:
                if callable(beta):
                    step_beta = beta(source)
                else:
                    step_beta = beta
            
            W[i] = gradient_part + step_beta * W[i]
        
        # Symmetric orthogonalization
        W = _symmetric_orthogonalize(W)
        
        # Check convergence
        max_change = 0.0
        for i in range(n_components):
            correlation = np.abs(np.dot(W[i], W_old[i]))
            max_change = max(max_change, 1 - correlation)
        
        if max_change < tol:
            if verbose:
                print(f"  Symmetric: converged at iteration {iteration + 1}")
            convergence_info[:, 0] = iteration + 1
            convergence_info[:, 1] = 1.0
            break
    else:
        if verbose:
            print(f"  Symmetric: max iterations ({max_iter})")
        convergence_info[:, 0] = max_iter
        convergence_info[:, 1] = 0.0
    
    # Extract final sources
    sources = W @ X_whitened
    
    return W, sources, convergence_info


def _symmetric_orthogonalize(W: np.ndarray) -> np.ndarray:
    """Symmetric orthogonalization using (W * W.T)^{-1/2} * W."""
    # EVD of W @ W.T
    D, E = np.linalg.eigh(W @ W.T)
    # Handle numerical issues
    D = np.maximum(D, 1e-12)
    # W_orth = E @ diag(1/sqrt(D)) @ E.T @ W
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
    W_orth = E @ D_inv_sqrt @ E.T @ W
    return W_orth


class IterativeDSS:
    """Iterative (Nonlinear) Denoising Source Separation Transformer.

    Implements Iterative DSS as a scikit-learn compatible transformer that 
    fits natively on MNE-Python objects (Raw, Epochs) or numpy arrays.
    
    Unlike linear DSS which uses closed-form eigendecomposition, Iterative DSS
    uses fixed-point iteration with a nonlinear denoising function, making it
    equivalent to FastICA when using ICA contrast functions (tanh, gauss, cube).

    Parameters
    ----------
    denoiser : callable or NonlinearDenoiser
        Nonlinear denoising function f(s) → s_denoised. Must be an instance of
        `mne_denoise.dss.NonlinearDenoiser` (e.g. `TanhMaskDenoiser`, 
        `WienerMaskDenoiser`) or a callable.
    n_components : int
        Number of components to extract.
    method : {'deflation', 'symmetric'}
        Component extraction method:
        
        - ``'deflation'``: Extract one-by-one, orthogonalizing after each.
        - ``'symmetric'``: Update all simultaneously, then orthogonalize.
        
        Default ``'deflation'``.
    rank : int, optional
        Rank for whitening. If None, auto-determined from eigenvalue threshold.
    reg : float
        Regularization for whitening. Default 1e-9.
    max_iter : int
        Maximum iterations per component. Default 100.
    tol : float
        Convergence tolerance. Default 1e-6.
    verbose : bool
        Print convergence info. Default False.
    alpha : float or callable, optional
        Signal normalization factor applied after denoising.
    beta : float or callable, optional
        Spectral shift (Newton step) for faster convergence.
        Use ``beta_tanh`` for TanhMaskDenoiser, ``beta_pow3`` for cubic.
    gamma : float or callable, optional
        Learning rate / relaxation parameter.

    Attributes
    ----------
    filters_ : ndarray, shape (n_components, n_channels)
        The spatial filters (un-mixing matrix). Apply as: ``sources = filters_ @ data``.
    patterns_ : ndarray, shape (n_channels, n_components)
        The spatial patterns (mixing matrix). Reconstruct as: ``data = patterns_ @ sources``.
    sources_ : ndarray, shape (n_components, n_times)
        Extracted sources from fit data.
    convergence_info_ : ndarray, shape (n_components, 2)
        [n_iterations, converged] for each component.

    Examples
    --------
    >>> from mne_denoise.dss import IterativeDSS
    >>> from mne_denoise.dss.denoisers import TanhMaskDenoiser, beta_tanh
    >>> 
    >>> # With numpy array
    >>> dss = IterativeDSS(TanhMaskDenoiser(), n_components=5, beta=beta_tanh)
    >>> dss.fit(data)
    >>> sources = dss.transform(data)
    >>> 
    >>> # With MNE Raw object
    >>> dss.fit(raw)
    >>> sources = dss.transform(raw.get_data())

    See Also
    --------
    DSS : Linear DSS transformer.
    iterative_dss : Functional API.
    """

    def __init__(
        self,
        denoiser: Callable[[np.ndarray], np.ndarray],
        n_components: int,
        *,
        method: str = 'deflation',
        rank: Optional[int] = None,
        reg: float = 1e-9,
        max_iter: int = 100,
        tol: float = 1e-6,
        verbose: bool = False,
        alpha: Union[float, Callable, None] = None,
        beta: Union[float, Callable, None] = None,
        gamma: Union[float, Callable, None] = None,
    ) -> None:
        self.denoiser = denoiser
        self.n_components = n_components
        self.method = method
        self.rank = rank
        self.reg = reg
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Fitted attributes
        self.filters_: Optional[np.ndarray] = None
        self.patterns_: Optional[np.ndarray] = None
        self.sources_: Optional[np.ndarray] = None
        self.convergence_info_: Optional[np.ndarray] = None
        self._mne_info = None

    def fit(self, X) -> 'IterativeDSS':
        """Compute Iterative DSS spatial filters.

        Parameters
        ----------
        X : Raw | Epochs | ndarray
            The data to fit. Accepts:
            
            - ``mne.io.Raw``: Continuous data
            - ``mne.Epochs``: Epoched data
            - ``ndarray``: Shape (n_channels, n_times) or (n_channels, n_times, n_epochs)

        Returns
        -------
        self : IterativeDSS
            The fitted transformer.
        """
        # Handle MNE objects
        try:
            import mne
            is_mne = isinstance(X, (mne.io.BaseRaw, mne.BaseEpochs))
        except ImportError:
            is_mne = False
        
        if is_mne:
            self._mne_info = X.info
            data = X.get_data()
        else:
            data = X
        
        filters, sources, patterns, conv_info = iterative_dss(
            data, self.denoiser, self.n_components,
            method=self.method, rank=self.rank, reg=self.reg,
            max_iter=self.max_iter, tol=self.tol, verbose=self.verbose,
            alpha=self.alpha, beta=self.beta, gamma=self.gamma
        )
        
        self.filters_ = filters
        self.patterns_ = patterns
        self.sources_ = sources
        self.convergence_info_ = conv_info
        
        return self

    def transform(self, X) -> np.ndarray:
        """Apply fitted filters to extract sources.

        Parameters
        ----------
        X : Raw | Epochs | ndarray
            Data to transform. Same formats as ``fit()``.

        Returns
        -------
        sources : ndarray, shape (n_components, n_times) or (n_components, n_times, n_epochs)
            Extracted source time series.
        """
        if self.filters_ is None:
            raise RuntimeError("IterativeDSS not fitted. Call fit() first.")
        
        # Handle MNE objects
        try:
            import mne
            if isinstance(X, (mne.io.BaseRaw, mne.BaseEpochs)):
                data = X.get_data()
            else:
                data = X
        except ImportError:
            data = X
        
        # Handle 3D
        original_shape = data.shape
        if data.ndim == 3:
            n_ch, n_times, n_epochs = data.shape
            data_2d = data.reshape(n_ch, -1)
        else:
            data_2d = data
            n_times = data.shape[1]
        
        # Center
        data_2d = data_2d - data_2d.mean(axis=1, keepdims=True)
        
        # Apply filters
        sources = self.filters_ @ data_2d
        
        if len(original_shape) == 3:
            sources = sources.reshape(self.n_components, n_times, n_epochs)
        
        return sources

    def inverse_transform(self, sources: np.ndarray) -> np.ndarray:
        """Reconstruct data from sources.

        Parameters
        ----------
        sources : ndarray, shape (n_sources, n_times)
            Source time series. Can use fewer sources than fitted.

        Returns
        -------
        reconstructed : ndarray, shape (n_channels, n_times)
            Reconstructed data: ``patterns_[:, :n_sources] @ sources``.
        """
        if self.patterns_ is None:
            raise RuntimeError("IterativeDSS not fitted. Call fit() first.")
        
        n_sources = sources.shape[0]
        patterns = self.patterns_[:, :n_sources]
        
        return patterns @ sources

    def fit_transform(self, X) -> np.ndarray:
        """Fit and transform in one step.

        Parameters
        ----------
        X : Raw | Epochs | ndarray
            Data to fit and transform.

        Returns
        -------
        sources : ndarray
            Extracted sources.
        """
        return self.fit(X).transform(X)

