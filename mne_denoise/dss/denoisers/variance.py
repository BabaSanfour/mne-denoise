"""Variance-based denoiser functions for nonlinear DSS.

Implements adaptive denoising functions from Särelä & Valpola (2005)
for use with iterative DSS algorithms.

References
----------
.. [1] Särelä & Valpola (2005). Denoising Source Separation. JMLR 6:233-272.
"""

from __future__ import annotations

from typing import Optional, Callable

import numpy as np
from scipy import ndimage, signal

from .base import NonlinearDenoiser


# =============================================================================
# Paper-faithful implementations from Särelä & Valpola 2005
# =============================================================================


class WienerMaskDenoiser(NonlinearDenoiser):
    """Adaptive Wiener mask denoiser (Särelä & Valpola 2005, Eq. 7).

    The core nonlinear DSS denoiser from the paper. Estimates time-varying
    signal variance and applies soft Wiener-style masking:

        m(t) = σ²_signal(t) / [σ²_signal(t) + σ²_noise]
        s⁺(t) = s(t) · m(t)

    This is adaptive/nonlinear because the mask is estimated from the data.
    Ideal for bursty, non-stationary signals (spindles, beta bursts, 
    intermittent artifacts).

    Parameters
    ----------
    window_samples : int
        Window size for local variance estimation. Default 50.
    noise_percentile : float
        Percentile of local variance used to estimate noise floor.
        Lower values = more aggressive denoising. Default 25.
    min_gain : float
        Minimum mask value (prevents complete zeroing). Default 0.01.
    noise_variance : float, optional
        If provided, use this fixed noise variance instead of estimating.

    Examples
    --------
    >>> denoiser = WienerMaskDenoiser(window_samples=50)
    >>> denoised = denoiser.denoise(source)

    Notes
    -----
    From the paper: "estimate the denoising specifications from the data...
    makes the denoising nonlinear or adaptive" (Section 3.1).
    """

    def __init__(
        self,
        window_samples: int = 50,
        noise_percentile: float = 25.0,
        *,
        min_gain: float = 0.01,
        noise_variance: Optional[float] = None,
    ) -> None:
        self.window_samples = max(3, window_samples)
        self.noise_percentile = noise_percentile
        self.min_gain = min_gain
        self.noise_variance = noise_variance

    def denoise(self, source: np.ndarray) -> np.ndarray:
        """Apply Wiener mask denoising.

        Parameters
        ----------
        source : ndarray, shape (n_times,) or (n_times, n_epochs)
            Source time series.

        Returns
        -------
        denoised : ndarray, same shape as input
            Wiener-masked source.
        """
        if source.ndim == 1:
            return self._denoise_1d(source)
        elif source.ndim == 2:
            n_times, n_epochs = source.shape
            denoised = np.zeros_like(source)
            for ep in range(n_epochs):
                denoised[:, ep] = self._denoise_1d(source[:, ep])
            return denoised
        else:
            raise ValueError(f"Source must be 1D or 2D, got {source.ndim}D")

    def _denoise_1d(self, source: np.ndarray) -> np.ndarray:
        """Apply Wiener mask to 1D source."""
        n_samples = len(source)
        window = min(self.window_samples, n_samples // 2)

        # Estimate local signal variance: σ²(t) = E[s²] - E[s]²
        source_sq = source ** 2
        local_mean_sq = ndimage.uniform_filter1d(source_sq, size=window, mode='reflect')
        local_mean = ndimage.uniform_filter1d(source, size=window, mode='reflect')
        local_var = np.maximum(local_mean_sq - local_mean ** 2, 0)

        # Estimate noise variance (from quiet periods)
        if self.noise_variance is not None:
            noise_var = self.noise_variance
        else:
            # Use percentile of local variance as noise floor estimate
            noise_var = np.percentile(local_var, self.noise_percentile)
            noise_var = max(noise_var, 1e-15)  # Prevent division by zero

        # Wiener mask: m(t) = σ²_signal / (σ²_signal + σ²_noise)
        # where σ²_signal = max(0, local_var - noise_var)
        signal_var = np.maximum(local_var - noise_var, 0)
        mask = signal_var / (signal_var + noise_var + 1e-15)
        
        # Apply minimum gain
        mask = np.maximum(mask, self.min_gain)

        return source * mask


class TanhMaskDenoiser(NonlinearDenoiser):
    """Tanh mask denoiser (ICA-style, Särelä & Valpola 2005, Section 3.2).

    Saturating nonlinearity that is robust to outliers. The paper shows this
    is equivalent to a bounded denoising mask:

        s⁺(t) = tanh(α·s(t))

    Which can be interpreted as mask m(t) = tanh(α·s)/s that saturates
    for large values, preventing outlier amplification.

    Parameters
    ----------
    alpha : float
        Scaling factor controlling saturation point. Larger alpha = 
        faster saturation. Default 1.0.
    normalize : bool
        If True, normalize source to unit variance before applying tanh.
        Default True.

    Examples
    --------
    >>> denoiser = TanhMaskDenoiser(alpha=1.0)
    >>> denoised = denoiser.denoise(source)

    Notes
    -----
    From the paper: "Using tanh... gives a more robust denoising rule, 
    similar to shrinkage rules in denoising" (Section 3.2).

    The connection to ICA: tanh nonlinearity in FastICA can be interpreted
    as a denoising mask that bounds the output, making it robust to 
    super-Gaussian outliers.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        *,
        normalize: bool = True,
    ) -> None:
        self.alpha = alpha
        self.normalize = normalize

    def denoise(self, source: np.ndarray) -> np.ndarray:
        """Apply tanh mask denoising.

        Parameters
        ----------
        source : ndarray
            Source time series.

        Returns
        -------
        denoised : ndarray
            Tanh-transformed source.
        """
        if self.normalize:
            # Normalize to unit variance for consistent alpha interpretation
            std = np.std(source)
            if std > 1e-15:
                source_normalized = source / std
            else:
                return source
            denoised = np.tanh(self.alpha * source_normalized)
            # Scale back
            denoised = denoised * std
        else:
            denoised = np.tanh(self.alpha * source)

        return denoised


class RobustTanhDenoiser(NonlinearDenoiser):
    """Robust tanh denoiser (MATLAB dss_1-0 compatible).

    Implements the robust nonlinearity used in the MATLAB DSS package:
        s_new = s - tanh(s)

    This effectively subtracts the 'saturated' part of the signal,
    emphasizing the 'tails' or 'outliers' differently than standard tanh.
    
    Parameters
    ----------
    alpha : float
        Scaling factor inside tanh. Default 1.0.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def denoise(self, source: np.ndarray) -> np.ndarray:
        """Apply robust tanh denoising: s - tanh(alpha * s)."""
        return source - np.tanh(self.alpha * source)


class GaussDenoiser(NonlinearDenoiser):
    """FastICA Gaussian nonlinearity (MATLAB denoise_fica_gauss.m).

    Implements: s_new = s * exp(-a * s² / 2)

    Good for sub-Gaussian sources with lighter tails than Gaussian.
    This is FastICA's G2 nonlinearity.

    Parameters
    ----------
    a : float
        Scaling constant. Default 1.0.
    """

    def __init__(self, a: float = 1.0) -> None:
        self.a = a

    def denoise(self, source: np.ndarray) -> np.ndarray:
        """Apply Gaussian nonlinearity: s * exp(-a * s² / 2)."""
        s2 = source ** 2
        return source * np.exp(-self.a * s2 / 2)


class SkewDenoiser(NonlinearDenoiser):
    """FastICA skewness nonlinearity (MATLAB denoise_fica_skew.m).

    Implements: s_new = s²

    For asymmetric/skewed sources. This is FastICA's G3 nonlinearity.
    Simple but effective for highly skewed distributions.
    """

    def denoise(self, source: np.ndarray) -> np.ndarray:
        """Apply skewness nonlinearity: s²."""
        return source ** 2


class DCTDenoiser(NonlinearDenoiser):
    """DCT domain denoiser (MATLAB denoise_dct.m).

    Applies a mask in the DCT (Discrete Cosine Transform) domain.
    Useful for frequency-selective denoising without explicit bandpass.

    s_new = IDCT(mask * DCT(s))

    Parameters
    ----------
    mask : ndarray or None
        DCT domain mask. Must have same length as signal, or will be
        expanded/truncated. If None, creates lowpass mask.
    cutoff_fraction : float
        If mask is None, this fraction of DCT coefficients are kept.
        Default 0.5 (lowpass, keep first 50% of coefficients).
    """

    def __init__(
        self, 
        mask: Optional[np.ndarray] = None, 
        cutoff_fraction: float = 0.5
    ) -> None:
        self.mask = mask
        self.cutoff_fraction = cutoff_fraction
        self._cached_mask = None
        self._cached_len = None

    def denoise(self, source: np.ndarray) -> np.ndarray:
        """Apply DCT filtering."""
        from scipy.fftpack import dct, idct
        
        n = len(source)
        
        # Create or retrieve mask
        if self.mask is not None:
            if len(self.mask) == n:
                mask = self.mask
            else:
                # Resample mask to match signal length
                mask = np.interp(
                    np.linspace(0, 1, n),
                    np.linspace(0, 1, len(self.mask)),
                    self.mask
                )
        else:
            # Create lowpass mask if not cached or length changed
            if self._cached_mask is None or self._cached_len != n:
                cutoff = int(n * self.cutoff_fraction)
                mask = np.zeros(n)
                mask[:cutoff] = 1.0
                self._cached_mask = mask
                self._cached_len = n
            else:
                mask = self._cached_mask
        
        # DCT -> mask -> IDCT
        dct_coeffs = dct(source, type=2, norm='ortho')
        dct_filtered = dct_coeffs * mask
        return idct(dct_filtered, type=2, norm='ortho')


class Spectrogram2DDenoiser(NonlinearDenoiser):
    """2D spectrogram denoiser (MATLAB dss_2dmask.m concept).

    Applies masking in the time-frequency domain for spectrogram-based
    denoising. Useful for isolating specific time-frequency regions.

    Parameters
    ----------
    mask : ndarray, shape (n_freqs, n_times), optional
        2D mask to apply. If None, uses magnitude-based adaptive mask.
    nperseg : int
        Segment length for STFT. Default 256.
    noverlap : int, optional
        Overlap between segments. Default nperseg // 2.
    threshold_percentile : float
        For adaptive masking, threshold below this percentile. Default 50.
    """

    def __init__(
        self,
        mask: Optional[np.ndarray] = None,
        nperseg: int = 256,
        noverlap: Optional[int] = None,
        threshold_percentile: float = 50.0,
    ) -> None:
        self.mask = mask
        self.nperseg = nperseg
        self.noverlap = noverlap if noverlap is not None else nperseg // 2
        self.threshold_percentile = threshold_percentile

    def denoise(self, source: np.ndarray) -> np.ndarray:
        """Apply 2D spectrogram masking."""
        # STFT
        f, t, Zxx = signal.stft(
            source, nperseg=self.nperseg, noverlap=self.noverlap
        )
        
        if self.mask is not None:
            # Use provided mask (resize if needed)
            if self.mask.shape == Zxx.shape:
                mask_2d = self.mask
            else:
                # Basic resize via interpolation
                from scipy.ndimage import zoom
                zoom_factors = (
                    Zxx.shape[0] / self.mask.shape[0],
                    Zxx.shape[1] / self.mask.shape[1]
                )
                mask_2d = zoom(self.mask, zoom_factors, order=1)
        else:
            # Adaptive magnitude-based mask
            magnitude = np.abs(Zxx)
            threshold = np.percentile(magnitude, self.threshold_percentile)
            mask_2d = (magnitude > threshold).astype(float)
        
        # Apply mask
        Zxx_masked = Zxx * mask_2d
        
        # ISTFT
        _, reconstructed = signal.istft(
            Zxx_masked, nperseg=self.nperseg, noverlap=self.noverlap
        )
        
        # Match length
        if len(reconstructed) > len(source):
            reconstructed = reconstructed[:len(source)]
        elif len(reconstructed) < len(source):
            reconstructed = np.pad(
                reconstructed, (0, len(source) - len(reconstructed))
            )
        
        return reconstructed


# =============================================================================
# Beta helper functions for FastICA-style Newton updates
# =============================================================================

def beta_tanh(source: np.ndarray) -> float:
    """Compute beta for tanh denoiser (FastICA Newton step).

    beta = -E[1 - tanh²(s)]

    Use with `iterative_dss(..., beta=beta_tanh)` for FastICA-equivalent
    convergence speed.

    Parameters
    ----------
    source : ndarray
        Current source estimate.

    Returns
    -------
    beta : float
        Spectral shift value.
    """
    return -np.mean(1 - np.tanh(source) ** 2)


def beta_pow3(source: np.ndarray) -> float:
    """Compute beta for kurtosis/pow3 denoiser (FastICA Newton step).

    beta = -3 (constant for pow3 nonlinearity)

    Use with `iterative_dss(..., beta=beta_pow3)` for FastICA-equivalent
    convergence.

    Parameters
    ----------
    source : ndarray
        Current source estimate (unused, beta is constant).

    Returns
    -------
    beta : float
        Spectral shift value (-3).
    """
    return -3.0


def beta_gauss(source: np.ndarray, a: float = 1.0) -> float:
    """Compute beta for Gaussian denoiser (FastICA Newton step).

    beta = -E[(1 - a*s²) * exp(-a*s²/2)]

    Parameters
    ----------
    source : ndarray
        Current source estimate.
    a : float
        Scaling constant (same as GaussDenoiser.a).

    Returns
    -------
    beta : float
        Spectral shift value.
    """
    s2 = source ** 2
    return -np.mean((1 - a * s2) * np.exp(-a * s2 / 2))


# =============================================================================
# Gamma helper functions for adaptive learning rate
# =============================================================================

class Gamma179:
    """179-rule adaptive gamma (MATLAB gamma_179.m).

    Detects oscillation by comparing consecutive weight deltas.
    If the angle between deltas is > 90°, reduces gamma to 0.5.

    Usage
    -----
    >>> gamma_fn = Gamma179()
    >>> idss = IterativeDSS(..., gamma=gamma_fn)
    """

    def __init__(self):
        self.gamma = 1.0
        self.deltaw = None

    def __call__(
        self, w_new: np.ndarray, w_old: np.ndarray, iteration: int
    ) -> float:
        """Compute adaptive gamma."""
        if iteration <= 2:
            self.gamma = 1.0
            if iteration == 2:
                self.deltaw = w_old - w_new
        elif self.gamma != 1.0:
            deltaw_old = self.deltaw
            self.deltaw = w_old - w_new
            
            # Check angle between consecutive deltas
            limit = 0.0  # cos(90°)
            norm_prod = np.linalg.norm(self.deltaw) * np.linalg.norm(deltaw_old)
            if norm_prod > 1e-12:
                cos_angle = np.dot(self.deltaw, deltaw_old) / norm_prod
                if cos_angle <= limit:
                    self.gamma = 0.5
        else:
            # First time after iteration 2
            deltaw_old = self.deltaw
            self.deltaw = w_old - w_new
        
        return self.gamma

    def reset(self):
        """Reset state for new component."""
        self.gamma = 1.0
        self.deltaw = None


class GammaPredictive:
    """Predictive controller adaptive gamma (MATLAB gamma_predictive.m).

    Adjusts gamma based on correlation between consecutive weight deltas.
    More aggressive than gamma_179.

    Usage
    -----
    >>> gamma_fn = GammaPredictive()
    >>> idss = IterativeDSS(..., gamma=gamma_fn)
    """

    def __init__(self, min_gamma: float = 0.5):
        self.gamma = 1.0
        self.deltaw = None
        self.min_gamma = min_gamma

    def __call__(
        self, w_new: np.ndarray, w_old: np.ndarray, iteration: int
    ) -> float:
        """Compute adaptive gamma using predictive controller."""
        if iteration <= 2:
            self.gamma = 1.0
            if iteration == 2:
                self.deltaw = w_old - w_new
        else:
            deltaw_old = self.deltaw
            self.deltaw = w_old - w_new
            
            # Predictive update
            norm_sq = np.dot(deltaw_old, deltaw_old)
            if norm_sq > 1e-12:
                self.gamma = self.gamma + np.dot(self.deltaw, deltaw_old) / norm_sq
                if self.gamma < self.min_gamma:
                    self.gamma = self.min_gamma
        
        return self.gamma

    def reset(self):
        """Reset state for new component."""
        self.gamma = 1.0
        self.deltaw = None


class QuasiPeriodicDenoiser(NonlinearDenoiser):
    """Quasi-periodic denoiser via cycle averaging (Särelä & Valpola 2005, Sec 3.4).

    For signals with repeating structure (ECG, respiration, periodic artifacts):
    1. Detect peaks/cycles in the source
    2. Segment into individual cycles
    3. Time-warp cycles to common length
    4. Average to create template
    5. Replace each cycle with time-warped template

    This is "gold" for ECG artifact removal and makes a great demo.

    Parameters
    ----------
    peak_distance : int
        Minimum distance between peaks in samples. Default 100.
    peak_height_percentile : float
        Percentile of signal for peak detection threshold. Default 75.
    warp_length : int, optional
        Length to warp each cycle to. If None, use median cycle length.
    smooth_template : bool
        If True, smooth the template. Default True.

    Examples
    --------
    >>> # For ECG-like signal at 250 Hz (peaks ~1 sec apart)
    >>> denoiser = QuasiPeriodicDenoiser(peak_distance=200)
    >>> denoised = denoiser.denoise(ecg_source)

    Notes
    -----
    From the paper: "detect peaks, chop cycles, time-warp, average, 
    replace each cycle by the average" (Section 3.4).
    """

    def __init__(
        self,
        peak_distance: int = 100,
        peak_height_percentile: float = 75.0,
        *,
        warp_length: Optional[int] = None,
        smooth_template: bool = True,
    ) -> None:
        self.peak_distance = max(10, peak_distance)
        self.peak_height_percentile = peak_height_percentile
        self.warp_length = warp_length
        self.smooth_template = smooth_template

    def denoise(self, source: np.ndarray) -> np.ndarray:
        """Apply quasi-periodic denoising.

        Parameters
        ----------
        source : ndarray, shape (n_times,) or (n_times, n_epochs)
            Source time series with quasi-periodic structure.

        Returns
        -------
        denoised : ndarray, same shape as input
            Denoised source with cycles replaced by template.
        """
        if source.ndim == 1:
            return self._denoise_1d(source)
        elif source.ndim == 2:
            n_times, n_epochs = source.shape
            denoised = np.zeros_like(source)
            for ep in range(n_epochs):
                denoised[:, ep] = self._denoise_1d(source[:, ep])
            return denoised
        else:
            raise ValueError(f"Source must be 1D or 2D, got {source.ndim}D")

    def _denoise_1d(self, source: np.ndarray) -> np.ndarray:
        """Apply quasi-periodic denoising to 1D source."""
        n_samples = len(source)

        # Step 1: Detect peaks
        height_threshold = np.percentile(np.abs(source), self.peak_height_percentile)
        peaks, _ = signal.find_peaks(
            np.abs(source),
            height=height_threshold,
            distance=self.peak_distance,
        )

        if len(peaks) < 3:
            # Not enough cycles, return original
            return source

        # Step 2: Determine cycle boundaries (midpoints between peaks)
        boundaries = np.zeros(len(peaks) + 1, dtype=int)
        boundaries[0] = 0
        boundaries[-1] = n_samples
        for i in range(1, len(peaks)):
            boundaries[i] = (peaks[i - 1] + peaks[i]) // 2

        # Step 3: Extract cycles and determine warp length
        cycles = []
        cycle_lengths = []
        for i in range(len(peaks)):
            start = boundaries[i]
            end = boundaries[i + 1]
            if end > start:
                cycles.append(source[start:end])
                cycle_lengths.append(end - start)

        if len(cycles) < 2:
            return source

        # Warp length: use provided or median
        if self.warp_length is not None:
            warp_len = self.warp_length
        else:
            warp_len = int(np.median(cycle_lengths))
        warp_len = max(10, warp_len)

        # Step 4: Time-warp all cycles to common length and average
        warped_cycles = []
        for cycle in cycles:
            if len(cycle) >= 3:
                # Resample to warp_len
                warped = np.interp(
                    np.linspace(0, 1, warp_len),
                    np.linspace(0, 1, len(cycle)),
                    cycle
                )
                warped_cycles.append(warped)

        if len(warped_cycles) < 2:
            return source

        # Average to create template
        template = np.mean(warped_cycles, axis=0)

        # Optional smoothing
        if self.smooth_template:
            smooth_window = max(3, warp_len // 20)
            template = ndimage.uniform_filter1d(template, size=smooth_window, mode='reflect')

        # Step 5: Replace each cycle with time-warped template
        denoised = np.zeros_like(source)
        for i, cycle in enumerate(cycles):
            start = boundaries[i]
            end = boundaries[i + 1]
            cycle_len = end - start
            
            if cycle_len >= 3:
                # Warp template back to original cycle length
                warped_template = np.interp(
                    np.linspace(0, 1, cycle_len),
                    np.linspace(0, 1, warp_len),
                    template
                )
                # Match amplitude to original cycle
                scale = np.std(cycle) / (np.std(warped_template) + 1e-15)
                offset = np.mean(cycle) - np.mean(warped_template) * scale
                denoised[start:end] = warped_template * scale + offset

        return denoised


# =============================================================================
# Original denoisers (kept for backward compatibility)
# =============================================================================


class VarianceMaskDenoiser(NonlinearDenoiser):
    """Nonlinear denoiser using local variance masking.

    Identifies high-variance regions in the source time series and
    weights them higher, effectively emphasizing transient activity.
    Useful for extracting non-stationary sources.

    Parameters
    ----------
    window_samples : int
        Window size for local variance computation. Default 100.
    percentile : float
        Percentile threshold for high-variance mask. Default 75.
    soft : bool
        If True, use soft weighting based on variance magnitude.
        If False, use binary mask. Default True.

    Examples
    --------
    >>> denoiser = VarianceMaskDenoiser(window_samples=50, percentile=80)
    >>> denoised_source = denoiser.denoise(source)

    See Also
    --------
    WienerMaskDenoiser : Paper-faithful Wiener mask implementation.
    """

    def __init__(
        self,
        window_samples: int = 100,
        percentile: float = 75.0,
        *,
        soft: bool = True,
    ) -> None:
        self.window_samples = max(3, window_samples)
        self.percentile = percentile
        self.soft = soft

    def denoise(self, source: np.ndarray) -> np.ndarray:
        """Apply variance-based masking to source time series."""
        if source.ndim == 1:
            return self._denoise_1d(source)
        elif source.ndim == 2:
            n_times, n_epochs = source.shape
            denoised = np.zeros_like(source)
            for ep in range(n_epochs):
                denoised[:, ep] = self._denoise_1d(source[:, ep])
            return denoised
        else:
            raise ValueError(f"Source must be 1D or 2D, got {source.ndim}D")

    def _denoise_1d(self, source: np.ndarray) -> np.ndarray:
        """Process single 1D source."""
        n_samples = len(source)
        source_sq = source ** 2
        window = min(self.window_samples, n_samples)

        local_mean_sq = ndimage.uniform_filter1d(source_sq, size=window, mode='reflect')
        local_mean = ndimage.uniform_filter1d(source, size=window, mode='reflect')
        local_var = np.maximum(local_mean_sq - local_mean ** 2, 0)

        if self.soft:
            threshold = np.percentile(local_var, self.percentile)
            if threshold < 1e-15:
                threshold = np.max(local_var) * 0.5
            if threshold < 1e-15:
                return source
            weights = 1 / (1 + np.exp(-(local_var - threshold) / (threshold * 0.5)))
            denoised = source * weights
        else:
            threshold = np.percentile(local_var, self.percentile)
            mask = local_var >= threshold
            denoised = source * mask.astype(float)

        return denoised


class KurtosisDenoiser(NonlinearDenoiser):
    """Nonlinear denoiser maximizing kurtosis (super-Gaussianity).

    Similar to FastICA's negentropy maximization. Useful for
    extracting sources with heavy-tailed distributions.

    Parameters
    ----------
    nonlinearity : str
        Nonlinear function to apply: 'tanh', 'cube', or 'gauss'.
        Default 'tanh'.
    alpha : float
        Scaling factor for nonlinearity. Default 1.0.

    Examples
    --------
    >>> denoiser = KurtosisDenoiser(nonlinearity='tanh')
    >>> denoised = denoiser.denoise(source)

    See Also
    --------
    TanhMaskDenoiser : Paper-faithful tanh mask with normalization.
    """

    def __init__(
        self,
        nonlinearity: str = 'tanh',
        alpha: float = 1.0,
    ) -> None:
        if nonlinearity not in ('tanh', 'cube', 'gauss'):
            raise ValueError(f"Unknown nonlinearity: {nonlinearity}")
        self.nonlinearity = nonlinearity
        self.alpha = alpha

    def denoise(self, source: np.ndarray) -> np.ndarray:
        """Apply nonlinear transformation for kurtosis maximization."""
        if self.nonlinearity == 'tanh':
            return np.tanh(self.alpha * source)
        elif self.nonlinearity == 'cube':
            return source ** 3
        elif self.nonlinearity == 'gauss':
            return source * np.exp(-0.5 * (self.alpha * source) ** 2)
        else:
            return source


class TemporalSmoothnessDenoiser(NonlinearDenoiser):
    """Nonlinear denoiser emphasizing temporally smooth sources.

    Promotes sources with high autocorrelation by penalizing
    rapid fluctuations. Useful for slow-wave or DC-shift artifacts.

    Parameters
    ----------
    smoothing_factor : float
        Weight for temporal smoothness penalty. Default 0.1.
    order : int
        Derivative order for smoothness measure. Default 1.

    Examples
    --------
    >>> denoiser = TemporalSmoothnessDenoiser(smoothing_factor=0.2)
    >>> smooth_source = denoiser.denoise(source)
    """

    def __init__(
        self,
        smoothing_factor: float = 0.1,
        order: int = 1,
    ) -> None:
        self.smoothing_factor = smoothing_factor
        self.order = max(1, order)

    def denoise(self, source: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing denoiser."""
        if source.ndim == 1:
            window = max(3, int(len(source) * self.smoothing_factor))
            weights = np.ones(window) / window
            smoothed = np.convolve(source, weights, mode='same')
            return (1 - self.smoothing_factor) * source + self.smoothing_factor * smoothed
        elif source.ndim == 2:
            n_times, n_trials = source.shape
            window = max(3, int(n_times * self.smoothing_factor))
            weights = np.ones(window) / window
            denoised = np.zeros_like(source)
            for t in range(n_trials):
                smoothed = np.convolve(source[:, t], weights, mode='same')
                denoised[:, t] = (1 - self.smoothing_factor) * source[:, t] + self.smoothing_factor * smoothed
            return denoised
        else:
            raise ValueError(f"Source must be 1D or 2D, got {source.ndim}D")

