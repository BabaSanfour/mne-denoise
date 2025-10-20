from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

from .noise_detection import find_next_noisefreq


# ---------------------------------------------------------------------------
# Configuration utilities
# ---------------------------------------------------------------------------

def _build_config(overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a configuration dictionary matching the main-branch defaults."""
    config: Dict[str, Any] = {
        "noisefreqs": overrides.get("noisefreqs", []),
        "minfreq": float(overrides.get("minfreq", 17.0)),
        "maxfreq": float(overrides.get("maxfreq", 99.0)),
        "adaptiveNremove": bool(overrides.get("adaptiveNremove", True)),
        "fixedNremove": int(overrides.get("fixedNremove", 1)),
        "detectionWinsize": float(overrides.get("detectionWinsize", 6.0)),
        "coarseFreqDetectPowerDiff": float(overrides.get("coarseFreqDetectPowerDiff", 4.0)),
        "coarseFreqDetectLowerPowerDiff": float(
            overrides.get("coarseFreqDetectLowerPowerDiff", 1.76091259055681)
        ),
        "searchIndividualNoise": bool(overrides.get("searchIndividualNoise", True)),
        "freqDetectMultFine": float(overrides.get("freqDetectMultFine", 2.0)),
        "detailedFreqBoundsUpper": tuple(overrides.get("detailedFreqBoundsUpper", (-0.05, 0.05))),
        "detailedFreqBoundsLower": tuple(overrides.get("detailedFreqBoundsLower", (-0.4, 0.1))),
        "maxProportionAboveUpper": float(overrides.get("maxProportionAboveUpper", 0.005)),
        "maxProportionBelowLower": float(overrides.get("maxProportionBelowLower", 0.005)),
        "noiseCompDetectSigma": float(overrides.get("noiseCompDetectSigma", 3.0)),
        "adaptiveSigma": bool(overrides.get("adaptiveSigma", True)),
        "minsigma": float(overrides.get("minsigma", 2.5)),
        "maxsigma": float(overrides.get("maxsigma", 5.0)),
        "chunkLength": float(overrides.get("chunkLength", 0.0)),
        "minChunkLength": float(overrides.get("minChunkLength", 30.0)),
        "winSizeCompleteSpectrum": float(overrides.get("winSizeCompleteSpectrum", 300.0)),
        "nkeep": int(overrides.get("nkeep", 0)),
        "plotResults": bool(overrides.get("plotResults", True)),
        "segmentLength": float(overrides.get("segmentLength", 1.0)),
        "prominenceQuantile": float(overrides.get("prominenceQuantile", 0.95)),
        "overwritePlot": bool(overrides.get("overwritePlot", False)),
        "figBase": int(overrides.get("figBase", 100)),
        "figPos": overrides.get("figPos", None),
        "saveSpectra": bool(overrides.get("saveSpectra", False)),
        "maxHarmonics": overrides.get("maxHarmonics", None),
        # Developer/debug flags carried over for API compatibility
        "dss_positive_only": bool(overrides.get("dss_positive_only", False)),
        "dss_divide_by_sumw2": bool(overrides.get("dss_divide_by_sumw2", False)),
        "dss_strict_frames": bool(overrides.get("dss_strict_frames", False)),
        "snapDssToFftBin": bool(overrides.get("snapDssToFftBin", False)),
        "saveDssDebug": bool(overrides.get("saveDssDebug", False)),
        "debugOutDir": overrides.get("debugOutDir", None),
    }

    # Normalise noisefreqs to canonical forms
    raw_freqs = config["noisefreqs"]
    if isinstance(raw_freqs, (float, int)):
        config["noisefreqs"] = [float(raw_freqs)]
    elif isinstance(raw_freqs, str):
        config["noisefreqs"] = raw_freqs
    elif raw_freqs is None:
        config["noisefreqs"] = []
    else:
        config["noisefreqs"] = [float(freq) for freq in raw_freqs]

    if config["fixedNremove"] < 0:
        raise ValueError("fixedNremove must be non-negative.")
    if config["minChunkLength"] <= 0:
        raise ValueError("minChunkLength must be positive.")
    if config["segmentLength"] <= 0:
        raise ValueError("segmentLength must be positive.")
    if config["chunkLength"] < 0:
        raise ValueError("chunkLength must be zero or a positive duration in seconds.")

    return config


# ---------------------------------------------------------------------------
# Dataclasses used for analytics and reporting
# ---------------------------------------------------------------------------

@dataclass
class FrequencyAnalytics:
    frequency: float
    refined_frequencies: List[float] = field(default_factory=list)
    removed_components: List[int] = field(default_factory=list)
    component_scores: List[float] = field(default_factory=list)
    sigma_history: List[float] = field(default_factory=list)
    fixed_history: List[int] = field(default_factory=list)
    cleaning_too_weak: bool = False
    cleaning_too_strong: bool = False
    found_noise: bool = False
    chunk_boundaries: List[Tuple[int, int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frequency": self.frequency,
            "refined_frequencies": self.refined_frequencies,
            "removed_components": self.removed_components,
            "scores": self.component_scores,
            "sigma_history": self.sigma_history,
            "fixed_history": self.fixed_history,
            "cleaning_too_strong": self.cleaning_too_strong,
            "cleaning_too_weak": self.cleaning_too_weak,
            "found_noise": self.found_noise,
            "chunk_boundaries": self.chunk_boundaries,
        }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _as_2d_array(data: Any) -> np.ndarray:
    array = np.asarray(data)
    if array.size == 0:
        raise ValueError("Input data cannot be empty.")
    if array.ndim == 1:
        array = array[:, np.newaxis]
    if array.ndim != 2:
        raise ValueError("Input data must be 1D or 2D.")
    return array.astype(np.float64, copy=False)


def _welch_hamming(
    data: np.ndarray,
    sampling_rate: float,
    nperseg: int,
    axis: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    win = signal.windows.hamming(nperseg, sym=False)
    freq, pxx = signal.welch(
        data,
        fs=sampling_rate,
        window=win,
        noverlap=nperseg // 2,
        nperseg=nperseg,
        axis=axis,
        detrend=False,
    )
    return freq, pxx


def _compute_log_psd(
    data: np.ndarray,
    sampling_rate: float,
    window_seconds: float,
) -> Tuple[np.ndarray, np.ndarray]:
    nperseg = max(8, int(window_seconds * sampling_rate))
    nperseg = min(nperseg, data.shape[0])
    freq, pxx = _welch_hamming(data, sampling_rate, nperseg, axis=0)
    with np.errstate(divide="ignore"):
        pxx_log = 10.0 * np.log10(np.maximum(pxx, np.finfo(np.float64).tiny))
    return freq, pxx_log


def _remove_flat_channels(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    diffs = np.diff(data, axis=0)
    flat = np.where(np.all(np.abs(diffs) < 1e-12, axis=0))[0]
    if flat.size == 0:
        return data, flat, np.empty((1, 0))
    remaining = np.delete(data, flat, axis=1)
    if remaining.shape[1] == 0:
        raise ValueError("All channels appear to be constant; cannot proceed.")
    baseline = data[:1, flat]
    return remaining, flat, baseline


def _reinsert_flat_channels(
    data: np.ndarray,
    flat_indices: np.ndarray,
    baseline: np.ndarray,
) -> np.ndarray:
    if flat_indices.size == 0:
        return data
    total_channels = data.shape[1] + flat_indices.size
    restored = np.zeros((data.shape[0], total_channels), dtype=data.dtype)
    keep_mask = np.ones(total_channels, dtype=bool)
    keep_mask[flat_indices] = False
    restored[:, keep_mask] = data
    if baseline.size:
        restored[:, flat_indices] = np.tile(baseline, (data.shape[0], 1))
    return restored


def _harmonic_basis(
    freq: float,
    sampling_rate: float,
    n_samples: int,
    n_harmonics: int,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    t = np.arange(n_samples) / sampling_rate
    columns: List[np.ndarray] = []
    slices: List[Tuple[int, int]] = []
    for h in range(1, n_harmonics + 1):
        omega = 2.0 * math.pi * h * freq
        sin_col = np.sin(omega * t)
        cos_col = np.cos(omega * t)
        start = len(columns)
        columns.extend([sin_col, cos_col])
        slices.append((start, start + 2))
    basis = np.column_stack(columns) if columns else np.zeros((n_samples, 0))
    return basis, slices


def _select_components(
    scores: np.ndarray,
    sigma: float,
    base_fixed: int,
    adaptive: bool,
) -> List[int]:
    if scores.size == 0:
        return []

    if adaptive:
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        if std < 1e-12:
            std = 1e-12
        z = (scores - mean) / std
        selected = np.where(z > sigma)[0]
        if selected.size < base_fixed:
            order = np.argsort(scores)[::-1]
            selected = order[:max(base_fixed, 0)]
    else:
        order = np.argsort(scores)[::-1]
        selected = order[:max(base_fixed, 0)]

    limit = max(1, scores.size // 5)
    if selected.size > limit:
        selected = np.array(sorted(selected)[-limit:])
    return sorted(int(idx) for idx in selected)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PyZaplinePlus:
    """
    Clean narrow-band artefacts from multichannel data using a Zapline-plus style pipeline.

    The implementation mirrors the feature set of the project's ``main`` branch while keeping
    the code path concise and approachable.
    """

    def __init__(self, data: Any, sampling_rate: float, **kwargs: Any) -> None:
        array = _as_2d_array(data)
        if sampling_rate <= 0 or not np.isfinite(sampling_rate):
            raise ValueError("sampling_rate must be a positive finite number.")

        self.original_shape = array.shape
        self.data = array
        self.sampling_rate = float(sampling_rate)
        self.config = _build_config(kwargs)
        self._prepared = False
        self._transpose_back = False
        self._flat_indices: np.ndarray | None = None
        self._flat_baseline: np.ndarray | None = None
        self._psd_freqs: np.ndarray | None = None
        self._psd_raw_log: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any], List[Any]]:
        if not self._prepared:
            self._prepare_inputs()

        working = self._working_data.copy()
        analytics: List[FrequencyAnalytics] = []
        figures: List[Any] = []

        frequencies = self.config["noisefreqs"]
        if isinstance(frequencies, str) and frequencies == "line":
            frequencies = self._resolve_line_frequency()

        if not frequencies:
            frequencies = self._automatic_noise_detection()
            self.config["automaticFreqDetection"] = bool(frequencies)

        for idx, freq in enumerate(frequencies):
            result = self._clean_frequency(working, float(freq))
            working = result["cleaned"]
            analytics.append(result["analytics"])
            figures.extend(result["figures"])

        restored = self._restore_data(working)
        analytics_dict = {
            f"noise_freq_{idx}": item.to_dict() for idx, item in enumerate(analytics, start=1)
        }
        return restored, dict(self.config), analytics_dict, figures

    # ------------------------------------------------------------------
    # Preparation helpers
    # ------------------------------------------------------------------
    def _prepare_inputs(self) -> None:
        data = self.data
        if data.shape[0] < data.shape[1]:
            data = data.T
            self._transpose_back = True
        self._working_data = data.copy()

        if self.sampling_rate > 500:
            print(
                "WARNING: Zapline-plus performs best on data sampled at 250–500 Hz. "
                f"Current sampling rate is {self.sampling_rate:.1f} Hz."
            )

        # Adjust whole-spectrum window to ensure at least eight segments (MATLAB parity)
        win_seconds = self.config["winSizeCompleteSpectrum"]
        max_window = max(int((self._working_data.shape[0] / self.sampling_rate) / 8), 1)
        if win_seconds * self.sampling_rate > self._working_data.shape[0] / 8:
            self.config["winSizeCompleteSpectrum"] = float(max_window)
            print(
                "Data set is short. Adjusted window size for whole-spectrum calculation to "
                f"{self.config['winSizeCompleteSpectrum']} seconds."
            )

        self._working_data, flat, baseline = _remove_flat_channels(self._working_data)
        self._flat_indices = flat
        self._flat_baseline = baseline

        freq, pxx_log = _compute_log_psd(
            self._working_data,
            self.sampling_rate,
            self.config["winSizeCompleteSpectrum"],
        )
        self._psd_freqs = freq
        self._psd_raw_log = pxx_log

        nkeep = self.config.get("nkeep", 0)
        if nkeep <= 0 or nkeep > self._working_data.shape[1]:
            self.config["nkeep"] = self._working_data.shape[1]

        self.config["baseFixedNremove"] = int(self.config.get("fixedNremove", 1))
        self._prepared = True

    def _resolve_line_frequency(self) -> List[float]:
        if self._psd_freqs is None or self._psd_raw_log is None:
            return []
        candidates = [50.0, 60.0]
        peaks = []
        for cand in candidates:
            mask = (self._psd_freqs >= cand - 1.5) & (self._psd_freqs <= cand + 1.5)
            if not np.any(mask):
                peaks.append(-np.inf)
            else:
                peaks.append(float(np.max(np.mean(self._psd_raw_log[mask], axis=1))))
        dominant = candidates[int(np.argmax(peaks))]
        half_win = self.config["detectionWinsize"] / 2.0
        self.config["minfreq"] = dominant - half_win
        self.config["maxfreq"] = dominant + half_win
        return [dominant]

    def _automatic_noise_detection(self) -> List[float]:
        if self._psd_freqs is None or self._psd_raw_log is None:
            return []
        frequencies: List[float] = []
        start = float(self.config["minfreq"])
        maxfreq = float(self.config["maxfreq"])
        while True:
            freq, _, _, _ = find_next_noisefreq(
                self._psd_raw_log,
                self._psd_freqs,
                minfreq=start,
                threshdiff=self.config["coarseFreqDetectPowerDiff"],
                winsizeHz=self.config["detectionWinsize"],
                maxfreq=maxfreq,
                lower_threshdiff=self.config["coarseFreqDetectLowerPowerDiff"],
                verbose=False,
            )
            if freq is None:
                break
            frequencies.append(float(freq))
            start = float(freq) + self.config["detectionWinsize"] / 2.0
            if start >= maxfreq:
                break
        return frequencies

    # ------------------------------------------------------------------
    # Frequency cleaning pipeline
    # ------------------------------------------------------------------
    def _clean_frequency(self, data: np.ndarray, frequency: float) -> Dict[str, Any]:
        nyquist = self.sampling_rate / 2.0
        if frequency >= nyquist:
            raise ValueError(
                f"Noise frequency {frequency:.2f} Hz is at or above Nyquist ({nyquist:.2f} Hz)."
            )

        sigma = float(self.config["noiseCompDetectSigma"])
        fixed = int(self.config["fixedNremove"])
        base_fixed = int(self.config["baseFixedNremove"])

        analytics = FrequencyAnalytics(frequency=frequency)
        figures: List[Any] = []

        best_clean = data.copy()
        best_scores = np.zeros(1)
        best_removed: List[int] = []
        best_peaks: List[float] = []
        best_chunks: List[Tuple[int, int]] = []
        too_strong_once = False

        for iteration in range(8):
            chunk_info = self._clean_once(
                data,
                frequency,
                sigma=sigma,
                fixed_nremove=fixed,
            )
            cleaned = chunk_info["cleaned"]

            analytics.sigma_history.append(sigma)
            analytics.fixed_history.append(fixed)
            analytics.chunk_boundaries = chunk_info["chunk_boundaries"]

            pxx_clean_log = chunk_info["pxx_clean_log"]
            assessment = self._assess_cleaning(pxx_clean_log, frequency)

            analytics.cleaning_too_weak = assessment["too_weak"]
            analytics.cleaning_too_strong = assessment["too_strong"]
            analytics.refined_frequencies = chunk_info["refined"]
            analytics.found_noise = chunk_info["found_noise"]
            analytics.component_scores = chunk_info["scores"].tolist()
            analytics.removed_components = chunk_info["removed"]

            best_clean = cleaned
            best_scores = chunk_info["scores"]
            best_removed = chunk_info["removed"]
            best_peaks = chunk_info["refined"]
            best_chunks = chunk_info["chunk_boundaries"]

            if not self.config["adaptiveSigma"]:
                break

            if assessment["too_strong"] and sigma < self.config["maxsigma"]:
                sigma = min(self.config["maxsigma"], sigma + 0.25)
                fixed = max(base_fixed, max(0, fixed - 1))
                too_strong_once = True
                continue

            if (
                assessment["too_weak"]
                and not too_strong_once
                and sigma > self.config["minsigma"]
            ):
                sigma = max(self.config["minsigma"], sigma - 0.25)
                fixed += 1
                continue

            break

        self.config["noiseCompDetectSigma"] = sigma
        self.config["fixedNremove"] = max(fixed, base_fixed)

        if self.config["plotResults"]:
            figures.append(
                self._plot_frequency_cleanup(
                    data,
                    best_clean,
                    frequency,
                    best_scores,
                    best_removed,
                    best_peaks,
                )
            )

        self.config["lastCleanedFrequency"] = frequency
        analytics.refined_frequencies = best_peaks
        analytics.removed_components = best_removed
        analytics.component_scores = best_scores.tolist()
        analytics.chunk_boundaries = best_chunks

        self._psd_freqs, self._psd_raw_log = _compute_log_psd(
            best_clean,
            self.sampling_rate,
            self.config["winSizeCompleteSpectrum"],
        )
        self._working_data = best_clean

        return {
            "cleaned": best_clean,
            "analytics": analytics,
            "figures": [fig for fig in figures if fig is not None],
        }

    def _clean_once(
        self,
        data: np.ndarray,
        frequency: float,
        *,
        sigma: float,
        fixed_nremove: int,
    ) -> Dict[str, Any]:
        chunk_boundaries = self._determine_chunks(data, frequency)

        cleaned = data.copy()
        refined: List[float] = []
        removed_all: List[int] = []
        scores_all: List[np.ndarray] = []
        found_noise = False

        for start, end in chunk_boundaries:
            chunk = data[start:end]
            target_freq, detected = self._refine_chunk_frequency(chunk, frequency)
            refined.append(target_freq)

            chunk_clean, scores, removed = self._clean_chunk(
                chunk,
                target_freq,
                sigma=sigma,
                fixed_nremove=fixed_nremove,
            )
            cleaned[start:end] = chunk_clean
            if removed:
                found_noise = True
            removed_all.extend(removed)
            scores_all.append(scores)

        if scores_all:
            scores_stack = np.vstack(scores_all)
            scores_mean = scores_stack.mean(axis=0)
        else:
            scores_mean = np.zeros(1)

        freq, pxx_clean_log = _compute_log_psd(
            cleaned,
            self.sampling_rate,
            self.config["winSizeCompleteSpectrum"],
        )

        return {
            "cleaned": cleaned,
            "scores": scores_mean,
            "removed": sorted(set(removed_all)),
            "refined": refined,
            "found_noise": found_noise,
            "pxx_freqs": freq,
            "pxx_clean_log": pxx_clean_log,
            "chunk_boundaries": chunk_boundaries,
        }

    def _determine_chunks(
        self,
        data: np.ndarray,
        frequency: float,
    ) -> List[Tuple[int, int]]:
        n_samples = data.shape[0]
        if self.config["chunkLength"] > 0:
            chunk_len = max(int(self.config["chunkLength"] * self.sampling_rate), 1)
            bounds = [
                (start, min(n_samples, start + chunk_len))
                for start in range(0, n_samples, chunk_len)
            ]
            if bounds[-1][1] != n_samples:
                bounds[-1] = (bounds[-1][0], n_samples)
            return bounds

        segment_len = max(int(self.config["segmentLength"] * self.sampling_rate), 1)
        min_chunk_samples = max(int(self.config["minChunkLength"] * self.sampling_rate), 1)

        band = self._bandpass(data, frequency, self.config["detectionWinsize"] / 2.0)
        n_segments = max(1, band.shape[0] // segment_len)
        covariances = []
        for idx in range(n_segments):
            start = idx * segment_len
            end = min((idx + 1) * segment_len, band.shape[0])
            segment = band[start:end]
            if segment.shape[0] < 2:
                continue
            cov = np.cov(segment, rowvar=False)
            covariances.append(cov)
        if len(covariances) < 2:
            return [(0, n_samples)]

        distances = []
        for c_prev, c_next in zip(covariances[:-1], covariances[1:]):
            diff = c_next - c_prev
            distances.append(np.linalg.norm(diff, ord="fro"))
        distances = np.asarray(distances)

        if distances.size == 0 or np.allclose(distances, distances[0]):
            return [(0, n_samples)]

        prominence = np.quantile(distances, self.config["prominenceQuantile"])
        peaks, _ = signal.find_peaks(
            distances,
            prominence=max(prominence, 0.0),
            distance=max(1, int(np.ceil(self.config["minChunkLength"] / self.config["segmentLength"]))),
        )

        indices = [0]
        for peak in peaks:
            boundary = min(n_samples, (peak + 1) * segment_len)
            if boundary - indices[-1] >= min_chunk_samples:
                indices.append(boundary)
        if n_samples - indices[-1] < min_chunk_samples:
            indices[-1] = n_samples
        else:
            indices.append(n_samples)
        bounds = [(indices[i], indices[i + 1]) for i in range(len(indices) - 1)]
        if not bounds:
            bounds = [(0, n_samples)]
        return bounds

    def _bandpass(
        self,
        data: np.ndarray,
        center: float,
        half_width: float,
    ) -> np.ndarray:
        if half_width <= 0:
            return data
        nyquist = self.sampling_rate / 2.0
        low = max(center - half_width, 0.5)
        high = min(center + half_width, nyquist * 0.999)
        if high <= low:
            return data
        sos = signal.butter(
            4,
            [low / nyquist, high / nyquist],
            btype="bandpass",
            output="sos",
        )
        return signal.sosfiltfilt(sos, data, axis=0)

    def _refine_chunk_frequency(
        self,
        chunk: np.ndarray,
        frequency: float,
    ) -> Tuple[float, bool]:
        if not self.config["searchIndividualNoise"]:
            return frequency, False

        nperseg = min(len(chunk), max(int(self.config["segmentLength"] * self.sampling_rate), 16))
        freq, pxx = _welch_hamming(chunk, self.sampling_rate, nperseg, axis=0)
        with np.errstate(divide="ignore"):
            pxx_log = 10.0 * np.log10(np.maximum(pxx, np.finfo(np.float64).tiny))

        freq_idx = (
            (freq >= frequency - self.config["detectionWinsize"] / 2.0)
            & (freq <= frequency + self.config["detectionWinsize"] / 2.0)
        )
        if not np.any(freq_idx):
            return frequency, False
        band = np.mean(pxx_log[freq_idx], axis=1)
        if band.size < 3:
            return frequency, False

        third = max(1, band.size // 3)
        reference = np.concatenate([band[:third], band[-third:]])
        center = float(np.mean(reference))
        lower = float(np.mean([np.quantile(band[:third], 0.05), np.quantile(band[-third:], 0.05)]))
        threshold = center + self.config["freqDetectMultFine"] * (center - lower)
        idx = int(np.argmax(band))
        if band[idx] < threshold:
            return frequency, False
        refined = float(freq[freq_idx][idx])
        return refined, True

    def _smooth_chunk(self, chunk: np.ndarray, frequency: float) -> np.ndarray:
        period = max(int(round(self.sampling_rate / max(frequency, 1e-6))), 1)
        kernel = np.ones(period) / period
        padded = np.pad(chunk, ((period, period), (0, 0)), mode="edge")
        smoothed = signal.convolve(padded, kernel[:, None], mode="same")
        smoothed = smoothed[period:-period]
        return smoothed

    def _clean_chunk(
        self,
        chunk: np.ndarray,
        frequency: float,
        *,
        sigma: float,
        fixed_nremove: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        smoothed = self._smooth_chunk(chunk, frequency)
        residual = chunk - smoothed

        nkeep = int(self.config["nkeep"])
        if nkeep < residual.shape[1]:
            u, s, vt = np.linalg.svd(residual, full_matrices=False)
            residual = (u[:, :nkeep] * s[:nkeep]) @ vt[:nkeep, :]

        nyquist = self.sampling_rate / 2.0
        max_harmonics = int(math.floor(nyquist / max(frequency, 1e-6)))
        cap = self.config.get("maxHarmonics")
        if cap is not None:
            max_harmonics = min(max_harmonics, int(cap))
        n_harmonics = max(1, max_harmonics)

        basis, slices = _harmonic_basis(frequency, self.sampling_rate, residual.shape[0], n_harmonics)
        if basis.size == 0:
            return chunk, np.zeros(1), []

        coeffs, *_ = np.linalg.lstsq(basis, residual, rcond=None)
        scores = np.asarray(
            [float(np.linalg.norm(coeffs[start:end])) for start, end in slices],
            dtype=float,
        )

        selected = _select_components(scores, sigma, fixed_nremove, self.config["adaptiveNremove"])
        if not selected:
            return chunk, scores, []

        coeffs_noise = np.zeros_like(coeffs)
        for idx in selected:
            start, end = slices[idx]
            coeffs_noise[start:end, :] = coeffs[start:end, :]
        noise_estimate = basis @ coeffs_noise
        residual_clean = residual - noise_estimate

        cleaned = smoothed + residual_clean
        return cleaned, scores, selected

    def _assess_cleaning(self, pxx_clean_log: np.ndarray, frequency: float) -> Dict[str, bool]:
        if self._psd_raw_log is None or self._psd_freqs is None:
            return {"too_strong": False, "too_weak": False}

        freq = self._psd_freqs
        raw_log = self._psd_raw_log

        fine_mask = (
            (freq >= frequency - self.config["detectionWinsize"] / 2.0)
            & (freq <= frequency + self.config["detectionWinsize"] / 2.0)
        )
        if not np.any(fine_mask):
            return {"too_strong": False, "too_weak": False}

        clean_band = np.mean(pxx_clean_log[fine_mask], axis=1)
        if clean_band.size < 3:
            return {"too_strong": False, "too_weak": False}

        third = max(1, clean_band.size // 3)
        reference = np.concatenate([clean_band[:third], clean_band[-third:]])
        center = float(np.mean(reference))
        lower = float(np.mean([np.quantile(clean_band[:third], 0.05), np.quantile(clean_band[-third:], 0.05)]))

        freq_detect_mult_fine = self.config["freqDetectMultFine"]
        upper_thresh = center + freq_detect_mult_fine * (center - lower)
        lower_thresh = center - freq_detect_mult_fine * (center - lower)

        upper_mask = (
            (freq >= frequency + self.config["detailedFreqBoundsUpper"][0])
            & (freq <= frequency + self.config["detailedFreqBoundsUpper"][1])
        )
        lower_mask = (
            (freq >= frequency + self.config["detailedFreqBoundsLower"][0])
            & (freq <= frequency + self.config["detailedFreqBoundsLower"][1])
        )

        proportion_above = 0.0
        proportion_below = 0.0
        if np.any(upper_mask):
            proportion_above = float(
                np.mean(np.mean(pxx_clean_log[upper_mask], axis=1) > upper_thresh)
            )
        if np.any(lower_mask):
            proportion_below = float(
                np.mean(np.mean(pxx_clean_log[lower_mask], axis=1) < lower_thresh)
            )

        too_weak = proportion_above > self.config["maxProportionAboveUpper"]
        too_strong = proportion_below > self.config["maxProportionBelowLower"]

        return {"too_strong": too_strong, "too_weak": too_weak}

    def _plot_frequency_cleanup(
        self,
        original: np.ndarray,
        cleaned: np.ndarray,
        frequency: float,
        scores: np.ndarray,
        removed: Sequence[int],
        refined: Sequence[float],
    ):
        try:
            nperseg = min(original.shape[0], 4096)
            freq, pxx_original = _welch_hamming(original, self.sampling_rate, nperseg, axis=0)
            _, pxx_clean = _welch_hamming(cleaned, self.sampling_rate, nperseg, axis=0)
            mean_orig = np.mean(pxx_original, axis=1)
            mean_clean = np.mean(pxx_clean, axis=1)

            fig = plt.figure(self.config["figBase"])
            if self.config["overwritePlot"]:
                fig.clf()
            ax = fig.add_subplot(111)
            ax.semilogy(freq, mean_orig, label="Before", color="C0")
            ax.semilogy(freq, mean_clean, label="After", color="C2")
            ax.axvline(frequency, color="C3", linestyle="--", linewidth=1.0, label="Target")
            if refined:
                for ref in refined:
                    ax.axvline(ref, color="C1", linestyle=":", linewidth=0.8)
            ax.set_xlim(
                frequency - self.config["detectionWinsize"],
                frequency + self.config["detectionWinsize"],
            )
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Power")
            ax.set_title(
                f"Zapline-plus cleanup @ {frequency:.2f} Hz "
                f"(removed {len(removed)} components)"
            )
            ax.legend()
            fig.tight_layout()
            return fig
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------
    def _restore_data(self, data: np.ndarray) -> np.ndarray:
        restored = data
        if self._flat_indices is not None and self._flat_baseline is not None:
            restored = _reinsert_flat_channels(restored, self._flat_indices, self._flat_baseline)
        if self._transpose_back:
            restored = restored.T
        return restored.reshape(self.original_shape)


def zapline_plus(
    data: Any,
    sampling_rate: float,
    **kwargs: Any,
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any], List[Any]]:
    runner = PyZaplinePlus(data, sampling_rate, **kwargs)
    return runner.run()
