"""Nonlinear/iterative DSS algorithm implementation.

This module implements the iterative DSS algorithm from Särelä & Valpola (2005)
for extracting sources using nonlinear denoising functions.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import numpy as np

from .whitening import whiten_data, compute_whitener


def iterative_dss_one(
    X_whitened: np.ndarray,
    denoiser: Callable[[np.ndarray], np.ndarray],
    *,
    w_init: Optional[np.ndarray] = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    beta: Union[float, Callable[[np.ndarray], float], None] = None,
    gamma: Union[float, Callable[[np.ndarray, np.ndarray, int], float], None] = None,
) -> Tuple[np.ndarray, np.ndarray, int, bool]:
    """Fixed-point iteration for single DSS component.

    Implements Algorithm 1 from Särelä & Valpola (2005) with optional Newton step.

    Parameters
    ----------
    X_whitened : ndarray, shape (n_components, n_times)
        Whitened data matrix.
    denoiser : callable
        Nonlinear denoising function f(s) -> s_denoised.
    w_init : ndarray, optional
        Initial weight vector.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    beta : float or callable, optional
        Spectral shift parameter for Newton-like update (FastICA style).
        If provided, the update becomes:
        w_new = E[X * f(s)] + beta * w
        where beta can be a fixed float or a function beta(source).
        For Tanh denoiser, beta = -E[1 - tanh(s)^2] yields FastICA's Newton step.
    gamma : float or callable, optional
        Learning rate / relaxation parameter. Controls step size:
        w = w_old + gamma * (w_new - w_old)
        If callable, signature is gamma(w_new, w_old, iteration) -> float.
        Use gamma_179 or gamma_predictive for adaptive learning rate.
        Default None (gamma=1, no relaxation).

    Returns
    -------
    w, source, n_iter, converged
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
    beta: Union[float, Callable, None] = None,
    gamma: Union[float, Callable, None] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract multiple DSS components using iterative algorithm.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
        Input data matrix.
    denoiser : callable or NonlinearDenoiser
        Nonlinear denoising function.
    n_components : int
        Number of components to extract.
    method : str
        Component extraction method:
        - 'deflation': Extract one-by-one with orthogonal projection
        - 'symmetric': Simultaneous extraction with Gram-Schmidt
        Default 'deflation'.
    rank : int, optional
        Rank for whitening. If None, auto-determined.
    reg : float
        Regularization for whitening. Default 1e-9.
    max_iter : int
        Maximum iterations per component. Default 100.
    tol : float
        Convergence tolerance. Default 1e-6.
    w_init : ndarray, shape (n_components, n_whitened), optional
        Initial weight matrix. If None, random initialization.
    verbose : bool
        Print convergence info. Default False.

    Returns
    -------
    filters : ndarray, shape (n_components, n_channels)
        DSS spatial filters in sensor space.
    sources : ndarray, shape (n_components, n_times)
        Extracted source time series.
    patterns : ndarray, shape (n_channels, n_components)
        Spatial patterns for visualization.
    convergence_info : ndarray, shape (n_components, 2)
        [n_iterations, converged] for each component.

    Notes
    -----
    Unlike linear DSS, iterative DSS can capture nonlinear dependencies
    in the data through the choice of denoising function.

    References
    ----------
    .. [1] Särelä & Valpola (2005). Denoising Source Separation. JMLR.
    """
    # Handle 3D epoched data
    original_shape = data.shape
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
            beta=beta, gamma=gamma
        )
    elif method == 'symmetric':
        filters_whitened, sources, convergence_info = _iterative_dss_symmetric(
            X_whitened, denoiser, n_components,
            max_iter=max_iter, tol=tol, w_init=w_init, verbose=verbose,
            beta=beta, gamma=gamma
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
    beta: Union[float, Callable, None] = None,
    gamma: Union[float, Callable, None] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract components one-by-one using deflation."""
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
            w_init=w_i, max_iter=max_iter, tol=tol, beta=beta, gamma=gamma
        )
        
        if verbose:
            status = "converged" if converged else "max_iter"
            print(f"  Component {i+1}: {n_iter} iterations ({status})")
        
        # Orthogonalize against previous components
        if i > 0:
            for j in range(i):
                w = w - np.dot(w, W[j]) * W[j]
            norm = np.linalg.norm(w)
            if norm < 1e-12:
                if verbose:
                    print(f"  Component {i+1}: degenerate, using random")
                rng = np.random.default_rng(i)
                w = rng.standard_normal(n_whitened)
                for j in range(i):
                    w = w - np.dot(w, W[j]) * W[j]
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
    beta: Union[float, Callable, None] = None,
    gamma: Union[float, Callable, None] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract components simultaneously with symmetric orthogonalization."""
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
    """Scikit-learn style estimator for iterative/nonlinear DSS.

    Parameters
    ----------
    denoiser : callable or NonlinearDenoiser
        Nonlinear denoising function.
    n_components : int
        Number of components to extract.
    method : str
        'deflation' or 'symmetric'. Default 'deflation'.
    rank : int, optional
        Rank for whitening.
    reg : float
        Regularization for whitening. Default 1e-9.
    max_iter : int
        Maximum iterations. Default 100.
    tol : float
        Convergence tolerance. Default 1e-6.
    verbose : bool
        Print convergence info. Default False.

    Attributes
    ----------
    filters_ : ndarray, shape (n_components, n_channels)
        Fitted spatial filters.
    patterns_ : ndarray, shape (n_channels, n_components)
        Fitted spatial patterns.
    sources_ : ndarray, shape (n_components, n_times)
        Extracted sources from fit data.
    convergence_info_ : ndarray
        Convergence info for each component.
    beta : float or callable, optional
        Spectral shift parameter (Newton step term).
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
        self.beta = beta
        self.gamma = gamma
        
        # Fitted attributes
        self.filters_: Optional[np.ndarray] = None
        self.patterns_: Optional[np.ndarray] = None
        self.sources_: Optional[np.ndarray] = None
        self.convergence_info_: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> 'IterativeDSS':
        """Fit iterative DSS from data.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times) or (n_channels, n_times, n_epochs)
            Training data.

        Returns
        -------
        self : IterativeDSS
            Fitted estimator.
        """
        filters, sources, patterns, conv_info = iterative_dss(
            data, self.denoiser, self.n_components,
            method=self.method, rank=self.rank, reg=self.reg,
            max_iter=self.max_iter, tol=self.tol, verbose=self.verbose,
            beta=self.beta, gamma=self.gamma
        )
        
        self.filters_ = filters
        self.patterns_ = patterns
        self.sources_ = sources
        self.convergence_info_ = conv_info
        
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply fitted filters to new data.

        Parameters
        ----------
        data : ndarray
            Data to transform.

        Returns
        -------
        sources : ndarray
            Extracted sources.
        """
        if self.filters_ is None:
            raise RuntimeError("IterativeDSS not fitted. Call fit() first.")
        
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
        sources : ndarray, shape (n_components, n_times)
            Source time series.

        Returns
        -------
        reconstructed : ndarray, shape (n_channels, n_times)
            Reconstructed data.
        """
        if self.patterns_ is None:
            raise RuntimeError("IterativeDSS not fitted. Call fit() first.")
        
        # patterns: (n_channels, n_components)
        # sources: (n_components, n_times)
        n_sources = sources.shape[0]
        patterns = self.patterns_[:, :n_sources]
        
        return patterns @ sources

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data."""
        return self.fit(data).transform(data)
