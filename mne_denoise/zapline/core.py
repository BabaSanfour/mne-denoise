"""DSS-based ZapLine implementation for line noise removal.

Implements the ZapLine algorithm (de Cheveigné, 2020) and ZapLine-Plus
(Klug & Kloosterman, 2022) using the DSS framework.

References
----------
de Cheveigné, A. (2020). ZapLine: A simple and effective method to remove
    power line artifacts. NeuroImage, 207, 116356.

Klug, M., & Kloosterman, N. A. (2022). Zapline-plus: A Zapline extension
    for automatic and adaptive removal of frequency-specific noise artifacts
    in M/EEG. Human Brain Mapping, 43(9), 2743-2758.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import signal
from scipy.fft import fft

# =============================================================================
# Result dataclasses
# =============================================================================


@dataclass
class ZapLineResult:
    """Results from DSS-ZapLine processing.

    Attributes
    ----------
    cleaned : ndarray
        Cleaned data with line noise removed.
    removed : ndarray
        Removed noise components.
    n_removed : int
        Number of components removed.
    dss_filters : ndarray
        DSS spatial filters for line noise.
    dss_eigenvalues : ndarray
        DSS eigenvalues (noise power per component).
    line_freq : float
        Target line frequency.
    n_harmonics : int
        Number of harmonics processed.
    """

    cleaned: np.ndarray
    removed: np.ndarray
    n_removed: int
    dss_filters: np.ndarray
    dss_eigenvalues: np.ndarray
    line_freq: float
    n_harmonics: int = 1


@dataclass
class ZapLinePlusResult:
    """Results from ZapLine-Plus processing.

    Attributes
    ----------
    cleaned : ndarray
        Cleaned data.
    removed : ndarray
        Total removed noise.
    config : dict
        Final configuration used.
    analytics : dict
        Analytics and diagnostics.
    chunk_results : list
        Per-chunk cleaning results.
    """

    cleaned: np.ndarray
    removed: np.ndarray
    config: Dict[str, Any]
    analytics: Dict[str, Any]
    chunk_results: List[Dict[str, Any]] = field(default_factory=list)


# =============================================================================
# Core ZapLine implementation (de Cheveigné 2020)
# =============================================================================


def _smooth_data(data: np.ndarray, period: int, n_iterations: int = 1) -> np.ndarray:
    """Smooth data by moving average over one period of line frequency.

    This is equivalent to MATLAB's nt_smooth - it cancels the line frequency
    and its harmonics while preserving lower frequencies.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Input data.
    period : int
        Period in samples (sfreq / line_freq).
    n_iterations : int
        Number of smoothing iterations.

    Returns
    -------
    smoothed : ndarray
        Smoothed data.
    """
    if period < 1:
        return data.copy()

    kernel = np.ones(period) / period
    smoothed = data.copy()

    for _ in range(n_iterations):
        # Apply moving average along time axis
        smoothed = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode="same"), axis=1, arr=smoothed
        )

    return smoothed


def _bias_fft(
    data: np.ndarray,
    freqs: np.ndarray,
    sfreq: float,
    nfft: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute covariance matrices biased at specific frequencies.

    This is equivalent to MATLAB's nt_bias_fft - computes the baseline
    covariance (c0) and the biased covariance at target frequencies (c1).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Input data.
    freqs : ndarray
        Target frequencies in Hz.
    sfreq : float
        Sampling frequency.
    nfft : int
        FFT size.

    Returns
    -------
    c0 : ndarray, shape (n_channels, n_channels)
        Baseline covariance.
    c1 : ndarray, shape (n_channels, n_channels)
        Biased covariance at target frequencies.
    """
    n_channels, n_times = data.shape

    # Compute FFT
    nfft = min(nfft, n_times)
    n_segments = n_times // nfft

    if n_segments < 1:
        n_segments = 1
        nfft = n_times

    # Baseline covariance
    c0 = np.zeros((n_channels, n_channels))
    # Biased covariance
    c1 = np.zeros((n_channels, n_channels))

    # Frequency indices for target frequencies
    freq_bins = np.fft.fftfreq(nfft, 1 / sfreq)
    target_indices = []
    for f in freqs:
        idx = np.argmin(np.abs(freq_bins - f))
        if idx not in target_indices:
            target_indices.append(idx)
        # Also include negative frequency
        idx_neg = np.argmin(np.abs(freq_bins + f))
        if idx_neg not in target_indices:
            target_indices.append(idx_neg)

    # Process segments
    for seg in range(n_segments):
        start = seg * nfft
        end = start + nfft
        segment = data[:, start:end]

        # Compute FFT
        X = fft(segment, axis=1)

        # Baseline: covariance of all frequencies
        c0 += np.real(X @ X.conj().T) / nfft

        # Biased: covariance of target frequencies only
        X_bias = np.zeros_like(X)
        for idx in target_indices:
            X_bias[:, idx] = X[:, idx]

        c1 += np.real(X_bias @ X_bias.conj().T) / nfft

    # Average over segments
    c0 /= n_segments
    c1 /= n_segments

    return c0, c1


def dss_zapline(
    data: np.ndarray,
    line_freq: float,
    sfreq: float,
    *,
    n_remove: Union[int, str] = "auto",
    n_harmonics: Optional[int] = None,
    nfft: int = 1024,
    nkeep: Optional[int] = None,
    rank: Optional[int] = None,
    reg: float = 1e-9,
    threshold: float = 3.0,
) -> ZapLineResult:
    """Remove line noise using DSS-based spatial filtering.

    Implements the ZapLine algorithm (de Cheveigné, 2020):
    1. Smooth data to isolate low-frequency content
    2. Compute residual (line noise + high-freq)
    3. Use FFT-based bias for DSS at line frequency harmonics
    4. Find spatial components that maximize line noise variance
    5. Project out top components from original data

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Input data with line noise.
    line_freq : float
        Line noise frequency in Hz (e.g., 50 or 60).
    sfreq : float
        Sampling frequency in Hz.
    n_remove : int or 'auto'
        Number of components to remove. If 'auto', determine using
        outlier detection. Default 'auto'.
    n_harmonics : int, optional
        Number of harmonics to include. If None, use all harmonics
        up to Nyquist.
    nfft : int
        FFT size for bias computation. Default 1024.
    nkeep : int, optional
        Number of PCA components to keep before DSS. Helps avoid
        overfitting with many channels.
    rank : int, optional
        Rank for DSS whitening.
    reg : float
        Regularization for DSS. Default 1e-9.
    threshold : float
        Z-score threshold for auto component selection. Default 3.0.

    Returns
    -------
    result : ZapLineResult
        Container with cleaned data and metadata.

    Examples
    --------
    >>> # Remove 50 Hz line noise
    >>> result = dss_zapline(eeg_data, line_freq=50, sfreq=1000)
    >>> cleaned = result.cleaned

    >>> # Remove 60 Hz with all harmonics
    >>> result = dss_zapline(eeg_data, line_freq=60, sfreq=1000)

    Notes
    -----
    This implementation follows the original MATLAB nt_zapline algorithm:
    - Uses moving average smoothing to separate low-freq from residual
    - Uses FFT-based bias covariance (nt_bias_fft)
    - Applies DSS (nt_dss0) to find line-noise components
    - Projects out noise using time-shift regression (nt_tsr)
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(
            f"Data must be 2D (n_channels, n_times), got shape {data.shape}"
        )

    n_channels, n_times = data.shape
    nyquist = sfreq / 2

    # Compute number of harmonics if not specified
    if n_harmonics is None:
        n_harmonics = int(np.floor(nyquist / line_freq))
    else:
        max_harmonics = int(np.floor(nyquist / line_freq))
        n_harmonics = min(n_harmonics, max_harmonics)

    # Step 1: Smooth data to get low-frequency component
    # Period = samples per line frequency cycle
    period = int(round(sfreq / line_freq))
    data_smooth = _smooth_data(data, period, n_iterations=1)

    # Step 2: Residual contains line noise and high frequencies
    data_residual = data - data_smooth

    # Step 3: Reduce dimensionality if requested (avoid overfitting)
    if nkeep is not None and nkeep < n_channels:
        # PCA reduction
        residual_centered = data_residual - data_residual.mean(axis=1, keepdims=True)
        cov = residual_centered @ residual_centered.T / n_times
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1][:nkeep]
        pca_filters = eigvecs[:, idx].T  # (nkeep, n_channels)
        data_reduced = pca_filters @ data_residual  # (nkeep, n_times)
    else:
        data_reduced = data_residual
        pca_filters = None

    # Step 4: Compute FFT-based bias covariance matrices
    harmonic_freqs = np.array([line_freq * (h + 1) for h in range(n_harmonics)])
    harmonic_freqs = harmonic_freqs[harmonic_freqs < nyquist]

    c0, c1 = _bias_fft(data_reduced, harmonic_freqs, sfreq, nfft)

    # Step 5: Apply DSS with covariance matrices
    # Regularize c0
    c0_reg = c0 + reg * np.trace(c0) / c0.shape[0] * np.eye(c0.shape[0])

    # Solve generalized eigenvalue problem using scipy
    from scipy.linalg import eigh as scipy_eigh

    try:
        eigvals, eigvecs = scipy_eigh(c1, c0_reg)
    except np.linalg.LinAlgError:
        # Fallback: use standard eigenvalue problem
        c0_inv = np.linalg.pinv(c0_reg)
        eigvals, eigvecs = np.linalg.eigh(c0_inv @ c1)

    # Sort by eigenvalue (descending)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    dss_filters = eigvecs[:, idx].T  # (n_components, n_reduced)

    # Eigenvalues are the scores (ratio of biased to baseline variance)
    scores = np.abs(eigvals)

    # Step 6: Determine number of components to remove
    if n_remove == "auto":
        n_remove = _iterative_outlier_removal(scores, threshold)
        # Cap at 1/5 of components
        n_remove = min(n_remove, len(scores) // 5)
        n_remove = max(n_remove, 1)  # Remove at least 1
    else:
        n_remove = min(int(n_remove), len(scores))

    if n_remove == 0:
        return ZapLineResult(
            cleaned=data.copy(),
            removed=np.zeros_like(data),
            n_removed=0,
            dss_filters=dss_filters,
            dss_eigenvalues=scores,
            line_freq=line_freq,
            n_harmonics=n_harmonics,
        )

    # Step 7: Project out line noise components
    # Extract noise components
    filters_noise = dss_filters[:n_remove]  # (n_remove, n_reduced)
    noise_sources = filters_noise @ data_reduced  # (n_remove, n_times)

    # Time-shift regression to project out noise
    # This is equivalent to nt_tsr
    if pca_filters is not None:
        # Map back to original channel space
        # patterns_noise = (pca_filters.T @ filters_noise.T).T
        noise_in_original = pca_filters.T @ filters_noise.T @ noise_sources
    else:
        noise_in_original = filters_noise.T @ noise_sources

    # Regress out noise from residual
    residual_clean = _regress_out(data_residual, noise_in_original.T)

    # Reconstruct clean signal
    cleaned = data_smooth + residual_clean
    removed = data - cleaned

    # Map DSS filters to original space for output
    if pca_filters is not None:
        dss_filters_original = dss_filters @ pca_filters
    else:
        dss_filters_original = dss_filters

    return ZapLineResult(
        cleaned=cleaned,
        removed=removed,
        n_removed=n_remove,
        dss_filters=dss_filters_original,
        dss_eigenvalues=scores,
        line_freq=line_freq,
        n_harmonics=n_harmonics,
    )


def _regress_out(data: np.ndarray, regressors: np.ndarray) -> np.ndarray:
    """Regress out components from data.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Original data.
    regressors : ndarray, shape (n_times, n_regressors)
        Regressors to remove.

    Returns
    -------
    cleaned : ndarray, shape (n_channels, n_times)
        Data with regressors removed.
    """
    # Solve: data = coeffs @ regressors.T + residual
    # coeffs = data @ regressors @ inv(regressors.T @ regressors)
    reg_cov = regressors.T @ regressors
    reg_cov += 1e-10 * np.trace(reg_cov) / reg_cov.shape[0] * np.eye(reg_cov.shape[0])

    try:
        coeffs = data @ regressors @ np.linalg.inv(reg_cov)
    except np.linalg.LinAlgError:
        coeffs = data @ regressors @ np.linalg.pinv(reg_cov)

    cleaned = data - coeffs @ regressors.T
    return cleaned


def _iterative_outlier_removal(scores: np.ndarray, sigma: float = 3.0) -> int:
    """Detect outliers iteratively using mean + sigma threshold.

    This is equivalent to MATLAB's iterative_outlier_removal.

    Parameters
    ----------
    scores : ndarray
        Component scores.
    sigma : float
        Sigma threshold. Default 3.0.

    Returns
    -------
    n_outliers : int
        Number of outliers detected.
    """
    scores = np.asarray(scores)
    n_outliers = 0
    remaining = scores.copy()

    while len(remaining) > 2:
        mean_val = np.mean(remaining)
        std_val = np.std(remaining)

        if std_val < 1e-12:
            break

        threshold = mean_val + sigma * std_val
        outliers = remaining > threshold

        if not np.any(outliers):
            break

        n_outliers += np.sum(outliers)
        remaining = remaining[~outliers]

    return n_outliers


def _auto_select_components(eigenvalues: np.ndarray, threshold: float) -> int:
    """Select number of components using outlier detection."""
    return _iterative_outlier_removal(eigenvalues, threshold)


# =============================================================================
# ZapLine-Plus implementation (Klug & Kloosterman 2022)
# =============================================================================


def zapline_plus(
    data: np.ndarray,
    sfreq: float,
    *,
    noisefreqs: Optional[Union[str, List[float]]] = "line",
    minfreq: float = 17.0,
    maxfreq: float = 99.0,
    adaptive_nremove: bool = True,
    fixed_nremove: int = 1,
    detection_winsize: float = 6.0,
    coarse_freq_detect_power_diff: float = 4.0,
    search_individual_noise: bool = True,
    noise_comp_detect_sigma: float = 3.0,
    adaptive_sigma: bool = True,
    minsigma: float = 2.5,
    maxsigma: float = 5.0,
    chunk_length: float = 0,
    min_chunk_length: float = 30.0,
    nkeep: Optional[int] = None,
    verbose: bool = True,
) -> ZapLinePlusResult:
    """Adaptive ZapLine-Plus for automatic line noise removal.

    Implements the full ZapLine-Plus algorithm (Klug & Kloosterman, 2022):
    1. Automatic noise frequency detection
    2. Adaptive chunking based on noise stationarity
    3. Per-chunk frequency refinement
    4. Automatic component selection with outlier detection
    5. Adaptive parameter adjustment for optimal cleaning

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Input data with line noise.
    sfreq : float
        Sampling frequency in Hz.
    noisefreqs : str or list of float, optional
        Noise frequencies to remove. If 'line', auto-detect 50/60 Hz.
        If None or empty, auto-detect all noise frequencies.
    minfreq : float
        Minimum frequency to search for noise. Default 17.0.
    maxfreq : float
        Maximum frequency to search for noise. Default 99.0.
    adaptive_nremove : bool
        Use adaptive component selection. Default True.
    fixed_nremove : int
        Fixed number of components to remove (minimum if adaptive). Default 1.
    detection_winsize : float
        Window size in Hz for peak detection. Default 6.0.
    coarse_freq_detect_power_diff : float
        Power threshold for noise detection (in 10*log10). Default 4.0.
    search_individual_noise : bool
        Search for peak frequency in each chunk. Default True.
    noise_comp_detect_sigma : float
        Sigma threshold for outlier detection. Default 3.0.
    adaptive_sigma : bool
        Adapt sigma based on cleaning results. Default True.
    minsigma : float
        Minimum sigma. Default 2.5.
    maxsigma : float
        Maximum sigma. Default 5.0.
    chunk_length : float
        Fixed chunk length in seconds. If 0, use adaptive chunking. Default 0.
    min_chunk_length : float
        Minimum chunk length for adaptive chunking. Default 30.0.
    nkeep : int, optional
        PCA reduction before DSS.
    verbose : bool
        Print progress. Default True.

    Returns
    -------
    result : ZapLinePlusResult
        Container with cleaned data, config, and analytics.

    Examples
    --------
    >>> # Auto-detect and remove line noise
    >>> result = zapline_plus(eeg_data, sfreq=500)
    >>> cleaned = result.cleaned

    >>> # Remove 50 Hz specifically
    >>> result = zapline_plus(eeg_data, sfreq=500, noisefreqs=[50.0])

    References
    ----------
    Klug, M., & Kloosterman, N. A. (2022). Zapline-plus: A Zapline extension
    for automatic and adaptive removal of frequency-specific noise artifacts
    in M/EEG. Human Brain Mapping, 43(9), 2743-2758.
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(
            f"Data must be 2D (n_channels, n_times), got shape {data.shape}"
        )

    n_channels, n_times = data.shape

    if verbose:
        print("ZapLine-Plus: Removing frequency artifacts using DSS")
        print(f"  Data: {n_channels} channels x {n_times} samples @ {sfreq} Hz")

    # Step 1: Determine noise frequencies
    if noisefreqs == "line":
        noisefreqs = _detect_line_frequency(data, sfreq, minfreq, maxfreq)
        if verbose:
            print(f"  Detected line frequency: {noisefreqs[0]} Hz")
    elif noisefreqs is None or len(noisefreqs) == 0:
        noisefreqs = _find_noise_frequencies(
            data,
            sfreq,
            minfreq,
            maxfreq,
            detection_winsize,
            coarse_freq_detect_power_diff,
        )
        if verbose:
            print(f"  Detected noise frequencies: {noisefreqs}")
    else:
        # We know it's a list/tuple of floats here, not "line"
        noisefreqs = list(noisefreqs)  # type: ignore

    if not noisefreqs:
        if verbose:
            print("  No noise frequencies detected")
        return ZapLinePlusResult(
            cleaned=data.copy(),
            removed=np.zeros_like(data),
            config={"noisefreqs": []},
            analytics={"noise_detected": False},
        )

    # Step 2: Process each noise frequency
    cleaned = data.copy()
    total_removed = np.zeros_like(data)
    all_chunk_results = []

    current_sigma = noise_comp_detect_sigma
    current_fixed = fixed_nremove

    for freq in noisefreqs:
        if verbose:
            print(f"\n  Processing {freq} Hz...")

        # Determine chunks
        chunks = _determine_chunks(
            cleaned, sfreq, freq, chunk_length, min_chunk_length, detection_winsize
        )

        if verbose:
            print(f"    {len(chunks)} chunk(s)")

        # Clean each chunk with adaptive parameters
        for max_iter in range(8):  # Max 8 adaptation iterations
            chunk_results = []
            cleaned_chunks = []

            for i, (start, end) in enumerate(chunks):
                chunk_data = cleaned[:, start:end]

                # Detect peak frequency in chunk
                if search_individual_noise:
                    chunk_freq = _refine_chunk_frequency(
                        chunk_data, sfreq, freq, detection_winsize
                    )
                else:
                    chunk_freq = freq

                # Apply ZapLine to chunk
                if adaptive_nremove:
                    n_remove_arg: Union[int, str] = "auto"
                else:
                    n_remove_arg = int(current_fixed)

                result = dss_zapline(
                    chunk_data,
                    chunk_freq,
                    sfreq,
                    n_remove=n_remove_arg,
                    threshold=current_sigma,
                    nkeep=nkeep,
                )

                # Ensure minimum removal
                if result.n_removed < current_fixed:
                    result = dss_zapline(
                        chunk_data,
                        chunk_freq,
                        sfreq,
                        n_remove=current_fixed,
                        nkeep=nkeep,
                    )

                cleaned_chunks.append(result.cleaned)
                chunk_results.append(
                    {
                        "start": start,
                        "end": end,
                        "freq": chunk_freq,
                        "n_removed": result.n_removed,
                        "eigenvalues": result.dss_eigenvalues,
                    }
                )

            # Reconstruct full data
            for i, (start, end) in enumerate(chunks):
                cleaned[:, start:end] = cleaned_chunks[i]

            # Check cleaning quality
            if not adaptive_sigma:
                break

            assessment = _assess_cleaning(data, cleaned, sfreq, freq, detection_winsize)

            if assessment["too_strong"] and current_sigma < maxsigma:
                current_sigma = min(maxsigma, current_sigma + 0.25)
                current_fixed = max(1, current_fixed - 1)
                if verbose:
                    print(
                        f"    Cleaning too strong, relaxing (sigma={current_sigma:.2f})"
                    )
                continue

            if assessment["too_weak"] and current_sigma > minsigma:
                current_sigma = max(minsigma, current_sigma - 0.25)
                current_fixed += 1
                if verbose:
                    print(
                        f"    Cleaning too weak, strengthening (sigma={current_sigma:.2f})"
                    )
                continue

            break

        all_chunk_results.extend(chunk_results)

        if verbose:
            mean_removed = np.mean([r["n_removed"] for r in chunk_results])
            print(f"    Removed {mean_removed:.1f} components on average")

    total_removed = data - cleaned

    # Compute analytics
    analytics = _compute_analytics(data, cleaned, sfreq, noisefreqs)

    config = {
        "noisefreqs": noisefreqs,
        "noise_comp_detect_sigma": current_sigma,
        "fixed_nremove": current_fixed,
        "adaptive_nremove": adaptive_nremove,
        "adaptive_sigma": adaptive_sigma,
        "chunk_length": chunk_length,
        "min_chunk_length": min_chunk_length,
    }

    if verbose:
        print(f"\n  Proportion of power removed: {analytics['proportion_removed']:.4f}")
        for freq in noisefreqs:
            ratio_key = f"ratio_noise_{int(freq)}"
            if ratio_key in analytics:
                print(
                    f"  Noise/surroundings ratio at {freq} Hz: {analytics[ratio_key]:.2f}"
                )

    return ZapLinePlusResult(
        cleaned=cleaned,
        removed=total_removed,
        config=config,
        analytics=analytics,
        chunk_results=all_chunk_results,
    )


# =============================================================================
# Helper functions
# =============================================================================


def _detect_line_frequency(
    data: np.ndarray,
    sfreq: float,
    minfreq: float,
    maxfreq: float,
) -> List[float]:
    """Detect dominant line frequency (50 or 60 Hz)."""
    n_times = data.shape[1]
    nperseg = min(n_times, int(4 * sfreq))

    freqs, psd = signal.welch(data, sfreq, nperseg=nperseg, axis=1)
    mean_psd = np.mean(psd, axis=0)

    candidates = [50.0, 60.0]
    powers = []

    for freq in candidates:
        if freq < minfreq or freq > maxfreq:
            powers.append(-np.inf)
            continue

        mask = (freqs >= freq - 1) & (freqs <= freq + 1)
        if np.any(mask):
            powers.append(float(np.mean(mean_psd[mask])))
        else:
            powers.append(-np.inf)

    return [candidates[np.argmax(powers)]]


def _find_noise_frequencies(
    data: np.ndarray,
    sfreq: float,
    minfreq: float,
    maxfreq: float,
    detection_winsize: float,
    power_diff_thresh: float,
) -> List[float]:
    """Find all noise frequencies above threshold."""
    n_times = data.shape[1]
    nperseg = min(n_times, int(4 * sfreq))

    freqs, psd = signal.welch(data, sfreq, nperseg=nperseg, axis=1)
    # Geometric mean (log space)
    psd_log = 10 * np.log10(np.maximum(psd, 1e-30))
    mean_psd_log = np.mean(psd_log, axis=0)

    noise_freqs = []
    search_start = minfreq

    while search_start < maxfreq:
        # Find peak in current window
        mask = (freqs >= search_start) & (
            freqs <= min(search_start + detection_winsize, maxfreq)
        )
        if not np.any(mask):
            break

        window_psd = mean_psd_log[mask]
        window_freqs = freqs[mask]

        # Center power (excluding middle third)
        third = len(window_psd) // 3
        if third < 1:
            break
        center_psd = np.mean(np.concatenate([window_psd[:third], window_psd[-third:]]))

        # Check for peak
        peak_idx = np.argmax(window_psd)
        peak_power = window_psd[peak_idx]

        if peak_power - center_psd > power_diff_thresh:
            noise_freqs.append(float(window_freqs[peak_idx]))
            search_start = window_freqs[peak_idx] + detection_winsize / 2
        else:
            search_start += detection_winsize / 2

    return noise_freqs


def _determine_chunks(
    data: np.ndarray,
    sfreq: float,
    freq: float,
    chunk_length: float,
    min_chunk_length: float,
    detection_winsize: float,
) -> List[Tuple[int, int]]:
    """Determine chunk boundaries based on noise stationarity."""
    n_times = data.shape[1]

    if chunk_length > 0:
        # Fixed chunk length
        chunk_samples = int(chunk_length * sfreq)
        chunks = []
        for start in range(0, n_times, chunk_samples):
            end = min(start + chunk_samples, n_times)
            chunks.append((start, end))
        return chunks

    # Adaptive chunking based on covariance stationarity
    min_samples = int(min_chunk_length * sfreq)
    segment_samples = int(sfreq)  # 1 second segments

    if n_times < 2 * min_samples:
        return [(0, n_times)]

    # Bandpass around target frequency
    nyquist = sfreq / 2
    low = max((freq - detection_winsize / 2) / nyquist, 0.01)
    high = min((freq + detection_winsize / 2) / nyquist, 0.99)

    try:
        sos = signal.butter(4, [low, high], btype="band", output="sos")
        data_bp = signal.sosfiltfilt(sos, data, axis=1)
    except Exception:
        return [(0, n_times)]

    # Compute covariance per segment
    n_segments = n_times // segment_samples
    if n_segments < 3:
        return [(0, n_times)]

    covs = []
    for i in range(n_segments):
        start = i * segment_samples
        end = min((i + 1) * segment_samples, n_times)
        segment = data_bp[:, start:end]
        cov = np.cov(segment)
        covs.append(cov)

    # Compute distances between consecutive covariances
    distances = []
    for i in range(len(covs) - 1):
        diff = covs[i + 1] - covs[i]
        distances.append(np.linalg.norm(diff, "fro"))
    distances = np.array(distances)

    if len(distances) < 2 or np.std(distances) < 1e-10:
        return [(0, n_times)]

    # Find peaks (chunk boundaries)
    prominence = np.quantile(distances, 0.95)
    min_distance = max(1, int(min_chunk_length / 1.0))  # 1 second segments

    peaks, _ = signal.find_peaks(
        distances, prominence=prominence, distance=min_distance
    )

    # Build chunks
    boundaries = [0]
    for peak in peaks:
        boundary = min(n_times, (peak + 1) * segment_samples)
        if boundary - boundaries[-1] >= min_samples:
            boundaries.append(boundary)

    if n_times - boundaries[-1] < min_samples:
        boundaries[-1] = n_times
    else:
        boundaries.append(n_times)

    chunks = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    return chunks if chunks else [(0, n_times)]


def _refine_chunk_frequency(
    data: np.ndarray,
    sfreq: float,
    target_freq: float,
    detection_winsize: float,
) -> float:
    """Find peak frequency within chunk around target."""
    n_times = data.shape[1]
    nperseg = min(n_times, int(sfreq))

    freqs, psd = signal.welch(data, sfreq, nperseg=nperseg, axis=1)
    psd_log = 10 * np.log10(np.maximum(psd, 1e-30))
    mean_psd = np.mean(psd_log, axis=0)

    # Search within narrow window
    mask = (freqs >= target_freq - 0.5) & (freqs <= target_freq + 0.5)
    if not np.any(mask):
        return target_freq

    peak_idx = np.argmax(mean_psd[mask])
    return float(freqs[mask][peak_idx])


def _assess_cleaning(
    original: np.ndarray,
    cleaned: np.ndarray,
    sfreq: float,
    freq: float,
    detection_winsize: float,
) -> Dict[str, bool]:
    """Assess if cleaning is too weak or too strong."""
    n_times = original.shape[1]
    nperseg = min(n_times, int(4 * sfreq))

    freqs, psd_clean = signal.welch(cleaned, sfreq, nperseg=nperseg, axis=1)
    psd_log = 10 * np.log10(np.maximum(psd_clean, 1e-30))
    mean_psd = np.mean(psd_log, axis=0)

    # Check around noise frequency
    mask = (freqs >= freq - detection_winsize / 2) & (
        freqs <= freq + detection_winsize / 2
    )
    if not np.any(mask):
        return {"too_weak": False, "too_strong": False}

    window_psd = mean_psd[mask]
    third = max(1, len(window_psd) // 3)

    center = np.mean(np.concatenate([window_psd[:third], window_psd[-third:]]))
    lower_quantile = np.mean(
        [np.quantile(window_psd[:third], 0.05), np.quantile(window_psd[-third:], 0.05)]
    )

    upper_thresh = center + 2 * (center - lower_quantile)
    lower_thresh = center - 2 * (center - lower_quantile)

    # Check for remaining peak (too weak) or notch (too strong)
    freq_mask = (freqs >= freq - 0.05) & (freqs <= freq + 0.05)
    below_mask = (freqs >= freq - 0.4) & (freqs <= freq + 0.1)

    too_weak = False
    too_strong = False

    if np.any(freq_mask):
        above = np.mean(mean_psd[freq_mask] > upper_thresh)
        too_weak = bool(above > 0.005)

    if np.any(below_mask):
        below = np.mean(mean_psd[below_mask] < lower_thresh)
        too_strong = bool(below > 0.005)

    return {"too_weak": too_weak, "too_strong": too_strong}


def _compute_analytics(
    original: np.ndarray,
    cleaned: np.ndarray,
    sfreq: float,
    noisefreqs: List[float],
) -> Dict[str, Any]:
    """Compute cleaning analytics."""
    n_times = original.shape[1]
    nperseg = min(n_times, int(4 * sfreq))

    freqs, psd_orig = signal.welch(original, sfreq, nperseg=nperseg, axis=1)
    _, psd_clean = signal.welch(cleaned, sfreq, nperseg=nperseg, axis=1)

    psd_orig_log = 10 * np.log10(np.maximum(psd_orig, 1e-30))
    psd_clean_log = 10 * np.log10(np.maximum(psd_clean, 1e-30))

    # Proportion of power removed (in log space = geometric mean)
    proportion_removed = 1 - 10 ** (
        (np.mean(psd_clean_log) - np.mean(psd_orig_log)) / 10
    )

    analytics = {
        "proportion_removed": float(proportion_removed),
        "noise_detected": True,
    }

    # Per-frequency analytics
    for freq in noisefreqs:
        freq_mask = (freqs >= freq - 0.5) & (freqs <= freq + 0.5)
        surround_mask = ((freqs >= freq - 3) & (freqs <= freq - 1)) | (
            (freqs >= freq + 1) & (freqs <= freq + 3)
        )

        if np.any(freq_mask) and np.any(surround_mask):
            ratio_orig = np.mean(psd_orig[:, freq_mask]) / np.mean(
                psd_orig[:, surround_mask]
            )
            ratio_clean = np.mean(psd_clean[:, freq_mask]) / np.mean(
                psd_clean[:, surround_mask]
            )

            analytics[f"ratio_noise_{int(freq)}_raw"] = float(ratio_orig)
            analytics[f"ratio_noise_{int(freq)}"] = float(ratio_clean)

    return analytics


# =============================================================================
# Convenience functions
# =============================================================================


def dss_zapline_adaptive(
    data: np.ndarray,
    sfreq: float,
    *,
    line_freq: Optional[float] = None,
    min_freq: float = 47.0,
    max_freq: float = 63.0,
    n_harmonics: Optional[int] = None,
    bandwidth: float = 2.0,
    threshold: float = 3.0,
    max_iter: int = 5,
    min_remove: int = 1,
    max_remove_fraction: float = 0.2,
) -> ZapLineResult:
    """Adaptive DSS-ZapLine with automatic frequency detection.

    Simple adaptive wrapper around dss_zapline that iteratively
    removes line noise until no significant noise remains.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        Input data.
    sfreq : float
        Sampling frequency in Hz.
    line_freq : float, optional
        Line frequency. If None, auto-detect from 50/60 Hz.
    min_freq : float
        Minimum search frequency. Default 47.0.
    max_freq : float
        Maximum search frequency. Default 63.0.
    n_harmonics : int, optional
        Number of harmonics. If None, use all up to Nyquist.
    bandwidth : float
        Filter bandwidth in Hz. Default 2.0.
    threshold : float
        Component selection threshold. Default 3.0.
    max_iter : int
        Maximum adaptation iterations. Default 5.
    min_remove : int
        Minimum components to remove. Default 1.
    max_remove_fraction : float
        Maximum fraction of components to remove. Default 0.2.

    Returns
    -------
    result : ZapLineResult
        Final cleaning result.
    """
    # Auto-detect line frequency if not specified
    if line_freq is None:
        detected = _detect_line_frequency(data, sfreq, min_freq, max_freq)
        line_freq = detected[0] if detected else 50.0

    n_channels = data.shape[0]
    max_remove = max(1, int(n_channels * max_remove_fraction))

    current_data = data.copy()
    total_removed = np.zeros_like(data)
    last_result = None

    for iteration in range(max_iter):
        result = dss_zapline(
            current_data,
            line_freq,
            sfreq,
            n_remove="auto",
            n_harmonics=n_harmonics,
            threshold=threshold,
        )

        total_removed = total_removed + result.removed
        current_data = result.cleaned
        last_result = result

        if result.n_removed == 0:
            break

        if result.n_removed >= max_remove:
            break

    return ZapLineResult(
        cleaned=current_data,
        removed=total_removed,
        n_removed=last_result.n_removed if last_result else 0,
        dss_filters=last_result.dss_filters if last_result else np.array([]),
        dss_eigenvalues=last_result.dss_eigenvalues if last_result else np.array([]),
        line_freq=line_freq,
        n_harmonics=last_result.n_harmonics if last_result else 1,
    )


def compute_psd_reduction(
    original: np.ndarray,
    cleaned: np.ndarray,
    sfreq: float,
    line_freq: float,
    *,
    bandwidth: float = 2.0,
) -> Dict[str, float]:
    """Compute PSD reduction metrics at line frequency.

    Parameters
    ----------
    original : ndarray
        Original data.
    cleaned : ndarray
        Cleaned data.
    sfreq : float
        Sampling frequency.
    line_freq : float
        Line frequency.
    bandwidth : float
        Bandwidth for power computation.

    Returns
    -------
    metrics : dict
        Dictionary with:
        - 'power_original': Mean power at line freq before
        - 'power_cleaned': Mean power at line freq after
        - 'reduction_db': Power reduction in dB
        - 'reduction_ratio': Power ratio (original/cleaned)
    """
    n_times = original.shape[1]
    nperseg = min(n_times, int(4 * sfreq))

    freqs, psd_orig = signal.welch(original, sfreq, nperseg=nperseg, axis=1)
    _, psd_clean = signal.welch(cleaned, sfreq, nperseg=nperseg, axis=1)

    # Average across channels
    psd_orig = np.mean(psd_orig, axis=0)
    psd_clean = np.mean(psd_clean, axis=0)

    # Power at line frequency
    mask = (freqs >= line_freq - bandwidth / 2) & (freqs <= line_freq + bandwidth / 2)
    if not np.any(mask):
        return {
            "power_original": np.nan,
            "power_cleaned": np.nan,
            "reduction_db": np.nan,
            "reduction_ratio": np.nan,
        }

    power_orig = np.mean(psd_orig[mask])
    power_clean = np.mean(psd_clean[mask])

    if power_clean < 1e-15:
        reduction_db = np.inf
        reduction_ratio = np.inf
    else:
        reduction_db = 10 * np.log10(power_orig / power_clean)
        reduction_ratio = float(power_orig / power_clean)

    return {
        "power_original": float(power_orig),
        "power_cleaned": float(power_clean),
        "reduction_db": float(reduction_db),
        "reduction_ratio": float(reduction_ratio),
    }
