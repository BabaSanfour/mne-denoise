"""Adaptive ASR variants backed by the AASR MATLAB reference.

This module implements the Hebbian/anti-Hebbian adaptive ASR variants from
the local ``refs/asr/repos/AASR`` reference repository. The adaptive variants
reuse standard ASR burst reconstruction while updating the calibration
subspace between chunks using similarity-matching learning rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal
from scipy.linalg import toeplitz

from ..utils import extract_data_from_mne, reconstruct_mne_object
from .core import (
    ASR,
    ASRState,
    _aggregate_block_covariances,
    _append_clean_rawdata_tail,
    _clean_rawdata_window_starts,
    _clean_windows_grid_diagnostics,
    _covariance_chunk_blocks,
    _covariance_stack_bytes,
    _empty_process_diagnostics,
    _good_raw_sample_mask,
    _max_mem_bytes,
    _moving_average_clean_rawdata,
    _process_memory_info,
    _regularize_spd,
    _resolve_max_dims_clean_rawdata,
    _round_half_up,
    _sample_mask_from_removed_windows,
    _sqrtm_spd,
    _validate_array_2d,
    fit_eeg_distribution,
)

try:
    import mne
    from mne.epochs import BaseEpochs
    from mne.evoked import Evoked
    from mne.io import BaseRaw
except ImportError:  # pragma: no cover - MNE is a required project dependency
    mne = None


_AASR_BETA_GRID = np.arange(1.7, 3.5 + 1e-12, 0.15, dtype=np.float64)


@dataclass
class _AdaptiveSimilarityMatcher:
    """Similarity-matching adaptive subspace learner."""

    variant: str
    tau: float
    learning_rate: float
    M: np.ndarray
    W: np.ndarray
    Minv: np.ndarray

    def fit_next(self, X: np.ndarray) -> np.ndarray:
        """Update the subspace learner on one calibration chunk."""
        y = self.Minv @ (self.W @ X)

        step = float(self.learning_rate)
        self.W = (1.0 - 2.0 * step) * self.W + (2.0 * step * y) @ X.T / X.shape[1]

        lateral_step = step / float(self.tau)
        yy = (y @ y.T) / X.shape[1]
        if self.variant == "psw":
            target = np.eye(yy.shape[0], dtype=np.float64)
        else:
            target = self.M
        self.M = self.M + lateral_step * (yy - target)
        try:
            self.Minv = np.linalg.inv(self.M)
        except np.linalg.LinAlgError:
            self.M = _regularize_spd(self.M, 1e-8)
            self.Minv = np.linalg.inv(self.M)
        return y

    def get_components(self) -> np.ndarray:
        """Return the orthonormalized adaptive basis."""
        components = (self.Minv @ self.W).T
        q, _ = np.linalg.qr(components, mode="reduced")
        return q.astype(np.float64, copy=False)


def _polystab(a: np.ndarray) -> np.ndarray:
    roots = np.roots(a)
    keep = roots != 0
    reflect = 0.5 * (np.sign(np.abs(roots[keep]) - 1.0) + 1.0)
    roots[keep] = (1.0 - reflect) * roots[keep] + reflect / np.conj(roots[keep])
    nz = np.flatnonzero(a != 0)
    b = a[nz[0]] * np.poly(roots)
    if not np.any(np.imag(a)):
        b = np.real(b)
    return np.asarray(b, dtype=np.float64)


def _numf(h: np.ndarray, a: np.ndarray, nb: int) -> np.ndarray:
    nh = int(np.max(h.size))
    impulse = np.zeros(nh, dtype=np.float64)
    impulse[0] = 1.0
    impr = signal.lfilter(np.array([1.0]), a, impulse)
    rhs = np.concatenate(([1.0], np.zeros(nb, dtype=np.float64)))
    return np.linalg.lstsq(toeplitz(impr, rhs), h.T, rcond=None)[0].T


def _denf(R: np.ndarray, na: int) -> np.ndarray:
    nr = int(np.max(np.size(R)))
    Rm = toeplitz(R[na : nr - 1], R[na:0:-1])
    rhs = -R[na + 1 : nr]
    return np.concatenate(([1.0], np.linalg.lstsq(Rm, rhs.T, rcond=None)[0].T))


def _yulewalk(
    order: int, F: np.ndarray, M: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Design the AASR statistics filter."""
    F = np.asarray(F, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    npt = 513
    lap = int(np.fix((npt - 1) / 25))
    Ht = np.zeros(npt, dtype=np.float64)
    Ht[0] = M[0]
    df = np.diff(F)

    nb = 0
    for idx in range(F.size - 1):
        if df[idx] == 0:
            nb = nb - int(lap / 2)
            ne = nb + lap
        else:
            ne = int(np.fix(F[idx + 1] * npt)) - 1
        j = np.arange(nb, ne + 1)
        inc = 0.0 if ne == nb else (j - nb) / (ne - nb)
        Ht[nb : ne + 1] = inc * M[idx + 1] + (1.0 - inc) * M[idx]
        nb = ne + 1

    Ht = np.concatenate([Ht, Ht[-2:0:-1]])
    n = Ht.size
    n2 = int(np.fix((n + 1) / 2))
    nr = 4 * order
    nt = np.arange(nr, dtype=np.float64)

    R = np.real(np.fft.ifft(Ht * Ht))
    R = R[:nr] * (0.54 + 0.46 * np.cos(np.pi * nt / max(nr - 1, 1)))

    Rwindow = np.concatenate(
        (
            np.array([0.5], dtype=np.float64),
            np.ones(max(n2 - 1, 0), dtype=np.float64),
            np.zeros(max(n - n2, 0), dtype=np.float64),
        )
    )
    A = _polystab(_denf(R, order))
    Qh = _numf(np.concatenate(([R[0] / 2.0], R[1:nr])), A, order)

    _, Ss = signal.freqz(Qh, A, worN=n, whole=True)
    Ss = np.maximum(2.0 * np.real(Ss), np.finfo(float).eps)
    hh = np.fft.ifft(np.exp(np.fft.fft(Rwindow * np.fft.ifft(np.log(Ss)))))
    B = np.real(_numf(hh[:nr], A, order))
    return np.asarray(B, dtype=np.float64), np.asarray(A, dtype=np.float64)


def _design_aasr_filter(sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    freqs = (
        np.array(
            [
                0.0,
                2.0,
                3.0,
                13.0,
                16.0,
                40.0,
                min(80.0, (sfreq / 2.0) - 1.0),
                sfreq / 2.0,
            ],
            dtype=np.float64,
        )
        * 2.0
        / sfreq
    )
    mags = np.array([3.0, 0.75, 0.33, 0.33, 1.0, 1.0, 3.0, 3.0], dtype=np.float64)
    return _yulewalk(8, freqs, mags)


def _lfilter_channels(
    X: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
    zi: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a causal IIR filter channel-wise and return final conditions."""
    X = np.asarray(X, dtype=np.float64)
    order = max(len(a), len(b)) - 1
    if order <= 0:
        empty = np.empty((X.shape[0], 0), dtype=np.float64)
        return X.copy(), empty
    if zi is None:
        zi = np.zeros((X.shape[0], order), dtype=np.float64)
    else:
        zi = np.asarray(zi, dtype=np.float64)
        if zi.shape != (X.shape[0], order):
            raise ValueError(
                "Filter state shape does not match data: "
                f"{zi.shape} vs {(X.shape[0], order)}"
            )

    out = np.empty_like(X, dtype=np.float64)
    zf = np.empty_like(zi, dtype=np.float64)
    for ch_idx in range(X.shape[0]):
        out[ch_idx], zf[ch_idx] = signal.lfilter(b, a, X[ch_idx], zi=zi[ch_idx])
    return out, zf


def _copy_asr_state(state: ASRState) -> ASRState:
    return ASRState(
        M=state.M.copy(),
        T=state.T.copy(),
        thresholds=state.thresholds.copy(),
        calibration_patterns=state.calibration_patterns.copy(),
        filter_b=state.filter_b.copy(),
        filter_a=state.filter_a.copy(),
        cov=state.cov.copy(),
        rank=int(state.rank),
        method=state.method,
        riemannian_solver=state.riemannian_solver,
    )


def _copy_process_state(state: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in state.items():
        copied[key] = value.copy() if isinstance(value, np.ndarray) else value
    return copied


def _select_aasr_clean_samples(
    X: np.ndarray,
    sfreq: float,
    *,
    window_length: float,
    window_overlap: float,
    max_bad_channels: float,
    zthresholds: tuple[float, float],
    max_dropout_fraction: float,
    min_clean_fraction: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    win_len = _round_half_up(window_length * sfreq)
    starts = _clean_rawdata_window_starts(X.shape[1], win_len, window_overlap)
    diagnostics = _clean_windows_grid_diagnostics(
        X,
        starts,
        win_len,
        max_bad_channels=max_bad_channels,
        zthresholds=zthresholds,
        max_dropout_fraction=max_dropout_fraction,
        min_clean_fraction=min_clean_fraction,
        fit_quantiles=(0.022, 0.6),
        beta_grid=_AASR_BETA_GRID,
    )
    if not np.any(diagnostics["window_keep_mask"]):
        scores = diagnostics["window_rms_zscores"]
        penalty = np.mean(np.maximum(scores - max(zthresholds), 0.0), axis=1)
        diagnostics["window_keep_mask"][int(np.argmin(penalty))] = True
        diagnostics["window_remove_mask"] = ~diagnostics["window_keep_mask"]

    sample_mask = _sample_mask_from_removed_windows(
        X.shape[1],
        starts,
        win_len,
        diagnostics["window_remove_mask"],
    )
    return X[:, sample_mask], sample_mask, diagnostics


def _adaptive_covariance_sqrt(
    X: np.ndarray,
    *,
    blocksize: int,
    regularization: float,
    max_mem_mb: int | float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    C, memory_info = _aggregate_block_covariances(
        X,
        blocksize,
        "geometric_median",
        covariance_kind="clean_rawdata",
        max_mem_mb=max_mem_mb,
    )
    C = _regularize_spd(C, regularization)
    M = _sqrtm_spd(C, regularization)
    eigvals, V = np.linalg.eigh(M)
    return M, C, eigvals, V, memory_info


class _ChunkedMovingCovariances:
    """Yield moving covariances at update positions without a full time stack."""

    def __init__(
        self,
        X: np.ndarray,
        update_at: np.ndarray,
        window_length: int,
        *,
        chunk_samples: int,
        zi: np.ndarray | None,
    ) -> None:
        self.X = X
        self.update_at = np.asarray(update_at, dtype=int)
        self.window_length = int(window_length)
        self.chunk_samples = int(chunk_samples)
        self.cov_state = zi.copy() if zi is not None else None

    def __iter__(self):
        n_channels, n_times = self.X.shape
        update_idx = 0
        for start in range(0, n_times, self.chunk_samples):
            stop = min(start + self.chunk_samples, n_times)
            chunk = self.X[:, start:stop]
            outer = np.einsum("it,jt->ijt", chunk, chunk, optimize=True)
            flat = outer.reshape(n_channels * n_channels, stop - start, order="F")
            flat, self.cov_state = _moving_average_clean_rawdata(
                self.window_length,
                flat,
                zi=self.cov_state,
            )
            while (
                update_idx < self.update_at.size
                and start < self.update_at[update_idx] <= stop
            ):
                local = int(self.update_at[update_idx] - start - 1)
                yield flat[:, local].reshape(n_channels, n_channels, order="F")
                update_idx += 1
        if update_idx != self.update_at.size:
            raise RuntimeError("Failed to produce all adaptive ASR covariance updates")


def _fit_adaptive_thresholds(
    X: np.ndarray,
    V: np.ndarray,
    *,
    sfreq: float,
    window_length: float,
    window_overlap: float,
    cutoff: float,
    min_clean_fraction: float,
    max_dropout_fraction: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n_times = X.shape[1]
    win_len = _round_half_up(window_length * sfreq)
    starts = _clean_rawdata_window_starts(n_times, win_len, window_overlap)
    projected = np.abs(X.T @ V)

    thresholds = np.empty(projected.shape[1], dtype=np.float64)
    mu_values = np.empty(projected.shape[1], dtype=np.float64)
    sigma_values = np.empty(projected.shape[1], dtype=np.float64)
    beta_values = np.empty(projected.shape[1], dtype=np.float64)
    fit_errors = np.empty(projected.shape[1], dtype=np.float64)
    fit_intervals = np.empty((projected.shape[1], 2), dtype=np.float64)
    for comp_idx in range(projected.shape[1]):
        rms = np.empty(len(starts), dtype=np.float64)
        comp = projected[:, comp_idx]
        for idx, start in enumerate(starts):
            segment = comp[start : start + win_len]
            rms[idx] = np.sqrt(np.mean(segment**2))
        mu, sigma, info = fit_eeg_distribution(
            rms,
            min_clean_fraction=min_clean_fraction,
            max_dropout_fraction=max_dropout_fraction,
            return_info=True,
        )
        mu_values[comp_idx] = mu
        sigma_values[comp_idx] = sigma
        beta_values[comp_idx] = info["beta"]
        fit_errors[comp_idx] = info["fit_error"]
        fit_intervals[comp_idx] = info["fit_interval"]
        thresholds[comp_idx] = mu + cutoff * sigma

    return thresholds, {
        "mu": mu_values,
        "sigma": sigma_values,
        "beta": beta_values,
        "fit_error": fit_errors,
        "fit_interval": fit_intervals,
        "window_starts": starts,
        "window_length_samples": int(win_len),
    }


def _build_adaptive_learner(
    X_filtered: np.ndarray,
    V: np.ndarray,
    *,
    variant: str,
    learning_rate: float,
    tau: float,
    regularization: float,
) -> _AdaptiveSimilarityMatcher:
    Y0 = V.T @ X_filtered
    n_samples = X_filtered.shape[1]
    M0 = (Y0 @ Y0.T) / n_samples
    W0 = (Y0 @ X_filtered.T) / n_samples
    M0 = _regularize_spd(M0, regularization)
    return _AdaptiveSimilarityMatcher(
        variant=variant,
        tau=float(tau),
        learning_rate=float(learning_rate),
        M=M0,
        W=W0.astype(np.float64, copy=False),
        Minv=np.linalg.inv(M0),
    )


def _process_adaptive_chunk(
    X: np.ndarray,
    sfreq: float,
    state: ASRState,
    process_state: dict[str, Any],
    *,
    window_length: float,
    lookahead: float | None,
    stepsize: int | None,
    max_dims: float | int,
    store_reconstruction_matrices: bool,
    adaptive_variant: str,
    max_mem_mb: int | float | None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Mirror the AASR ``reconstruct()`` wrapper around ``asr_process``."""
    X = _validate_array_2d(X)
    n_channels, n_times = X.shape
    lookahead_samples = (
        _round_half_up((window_length / 2.0) * sfreq)
        if lookahead is None
        else _round_half_up(lookahead * sfreq)
    )
    if lookahead_samples >= n_times:
        raise ValueError("lookahead is too long for the data length")
    win_len = max(
        _round_half_up(window_length * sfreq), _round_half_up(1.5 * n_channels)
    )
    stepsize = 32 if stepsize is None else int(stepsize)
    if stepsize < 1:
        raise ValueError("stepsize must be at least 1 sample")

    sig = _append_clean_rawdata_tail(X, lookahead_samples)
    n_stream_input = sig.shape[1]
    max_bad = _resolve_max_dims_clean_rawdata(max_dims, n_channels)
    if max_bad <= 0:
        diagnostics = _empty_process_diagnostics(n_times)
        diagnostics.update(
            _process_memory_info(
                n_channels=n_channels,
                n_stream_input=n_stream_input,
                max_mem_mb=max_mem_mb,
                memory_mode="identity",
                peak_cov_buffer_bytes=0,
                chunk_samples=0,
                used_memory_bound=False,
            )
        )
        return X.copy(), diagnostics, _copy_process_state(process_state)

    carry = process_state.get("carry")
    if carry is None:
        data_stream = np.concatenate(
            [
                2.0 * sig[:, [0]] - sig[:, lookahead_samples:0:-1],
                sig,
            ],
            axis=1,
        )
    else:
        data_stream = np.concatenate([carry, sig], axis=1)
    data_stream = np.asarray(data_stream, dtype=np.float64)
    data_stream[~np.isfinite(data_stream)] = 0.0

    X_stats, iir_state = _lfilter_channels(
        data_stream[:, lookahead_samples : lookahead_samples + n_stream_input],
        state.filter_b,
        state.filter_a,
        zi=process_state.get("iir"),
    )

    update_at = np.minimum(
        np.arange(stepsize, n_stream_input + stepsize, stepsize, dtype=int),
        n_stream_input,
    )
    if update_at.size == 0 or update_at[-1] != n_stream_input:
        update_at = np.append(update_at, n_stream_input)
    update_at = np.unique(update_at)

    last_R = process_state.get("last_R")
    last_trivial = bool(process_state.get("last_trivial", True))
    if last_R is None:
        last_R = np.eye(n_channels, dtype=np.float64)
        last_trivial = True
        update_at = np.concatenate(([1], update_at))

    estimated_cov_bytes = _covariance_stack_bytes(n_stream_input, n_channels)
    max_mem_bytes = _max_mem_bytes(max_mem_mb)
    use_chunked_covariance = (
        max_mem_bytes is not None and estimated_cov_bytes > max_mem_bytes
    )
    cov_state_in = process_state.get("cov")
    cov_source = None
    if use_chunked_covariance:
        chunk_samples = _covariance_chunk_blocks(n_channels, max_mem_bytes)
        cov_source = _ChunkedMovingCovariances(
            X_stats,
            update_at,
            win_len,
            chunk_samples=chunk_samples,
            zi=cov_state_in,
        )
        covariance_iter = iter(cov_source)
        Xcov_flat = None
        cov_state = None
        peak_cov_buffer_bytes = _covariance_stack_bytes(chunk_samples, n_channels)
    else:
        chunk_samples = n_stream_input
        outer = np.einsum("it,jt->ijt", X_stats, X_stats, optimize=True)
        Xcov_flat = outer.reshape(n_channels * n_channels, n_stream_input, order="F")
        Xcov_flat, cov_state = _moving_average_clean_rawdata(
            win_len,
            Xcov_flat,
            zi=cov_state_in,
        )
        covariance_iter = None
        peak_cov_buffer_bytes = estimated_cov_bytes

    sample_mask = np.zeros(n_times, dtype=bool)
    n_reconstructed: list[int] = []
    component_variances: list[np.ndarray] = []
    component_thresholds: list[np.ndarray] = []
    reconstruction_matrices: list[np.ndarray] = []
    window_starts: list[int] = []
    window_stops: list[int] = []

    last_n = 0
    for n in update_at:
        if covariance_iter is None:
            Cw = Xcov_flat[:, n - 1].reshape(n_channels, n_channels, order="F")
        else:
            Cw = next(covariance_iter)
        Cw = (Cw + Cw.T) / 2.0
        D, V = np.linalg.eigh(Cw)
        order = np.argsort(D)
        D = D[order]
        V = V[:, order]
        theta2 = np.sum((state.T @ V) ** 2, axis=0)
        keep = (theta2 > D) | (np.arange(1, n_channels + 1) < (n_channels - max_bad))
        trivial = bool(np.all(keep))
        n_bad = int(n_channels - np.count_nonzero(keep))
        if trivial:
            R = np.eye(n_channels, dtype=np.float64)
        else:
            basis = keep[:, np.newaxis].astype(np.float64) * (V.T @ state.M)
            R = state.M @ np.linalg.pinv(basis) @ V.T
            R = np.real_if_close(R).astype(np.float64)

        applied = (not trivial) or (not last_trivial)
        if applied and n > last_n:
            subrange = slice(last_n, n)
            width = n - last_n
            blend = (1.0 - np.cos(np.pi * np.arange(1, width + 1) / width)) / 2.0
            segment = data_stream[:, subrange]
            data_stream[:, subrange] = (R @ segment) * blend[np.newaxis, :] + (
                last_R @ segment
            ) * (1.0 - blend[np.newaxis, :])

        start_out = max(last_n, lookahead_samples) - lookahead_samples
        stop_out = min(n, lookahead_samples + n_times) - lookahead_samples
        if stop_out > start_out:
            window_starts.append(int(start_out))
            window_stops.append(int(stop_out))
            n_reconstructed.append(n_bad)
            component_variances.append(D.copy())
            component_thresholds.append(theta2.copy())
            if applied:
                sample_mask[start_out:stop_out] = True
            if store_reconstruction_matrices:
                reconstruction_matrices.append(R.copy())

        last_n = int(n)
        last_R = R
        last_trivial = trivial

    carry_out = (
        data_stream[:, -lookahead_samples:].copy() if lookahead_samples > 0 else None
    )
    delayed = data_stream[:, :n_stream_input].copy()
    X_clean = delayed[:, lookahead_samples:]
    if cov_source is not None:
        cov_state = cov_source.cov_state

    n_reconstructed_arr = np.asarray(n_reconstructed, dtype=int)
    diagnostics = {
        "window_starts": np.asarray(window_starts, dtype=int),
        "window_stops": np.asarray(window_stops, dtype=int),
        "sample_mask": sample_mask,
        "n_components_reconstructed": n_reconstructed_arr,
        "component_variances": np.asarray(component_variances, dtype=np.float64),
        "component_thresholds": np.asarray(component_thresholds, dtype=np.float64),
        "n_windows": int(len(n_reconstructed_arr)),
        "fraction_reconstructed_windows": float(
            np.mean(n_reconstructed_arr > 0) if n_reconstructed_arr.size else 0.0
        ),
        "fraction_reconstructed_samples": float(np.mean(sample_mask)),
        "max_components_reconstructed": int(n_reconstructed_arr.max(initial=0)),
        "lookahead_samples": int(lookahead_samples),
        "stepsize_samples": int(stepsize),
        "window_length_samples": int(win_len),
        "covariance_geometry": "standard",
        "adaptive_variant": adaptive_variant,
    }
    diagnostics.update(
        _process_memory_info(
            n_channels=n_channels,
            n_stream_input=n_stream_input,
            max_mem_mb=max_mem_mb,
            memory_mode="chunked" if use_chunked_covariance else "full",
            peak_cov_buffer_bytes=peak_cov_buffer_bytes,
            chunk_samples=chunk_samples,
            used_memory_bound=use_chunked_covariance,
        )
    )
    if store_reconstruction_matrices:
        diagnostics["reconstruction_matrices"] = np.asarray(reconstruction_matrices)

    next_state = {
        "cov": cov_state.copy() if cov_state is not None else None,
        "carry": carry_out,
        "iir": iir_state.copy() if iir_state is not None else None,
        "last_R": last_R.copy(),
        "last_trivial": bool(last_trivial),
    }
    return X_clean, diagnostics, next_state


class AdaptiveASR(ASR):
    """Experimental adaptive ASR with MATLAB-style ``update`` and ``reconstruct``.

    This estimator implements the AASR adaptive update rule from the local
    MATLAB references. Two adaptive variants are exposed:

    - ``variant='psp'``: principal subspace projection updates
    - ``variant='psw'``: principal subspace whitening updates
    """

    def __init__(
        self,
        sfreq: float | None = None,
        *,
        cutoff: float = 20.0,
        variant: str = "psw",
        window_length: float = 0.5,
        update_window_length: float = 0.1,
        calibration_window_length: float = 1.0,
        calibration_window_overlap: float = 0.66,
        ref_max_bad_channels: float = 0.2,
        ref_tolerances: tuple[float, float] = (-3.5, 5.0),
        blocksize: int = 10,
        max_dims: float | int = 0.66,
        max_dropout_fraction: float = 0.1,
        min_clean_fraction: float = 0.25,
        picks: str | list[str] | list[int] | None = "eeg",
        reject_by_annotation: bool = True,
        skip_by_annotation: tuple[str, ...] = ("bad", "bad_acq_skip"),
        regularization: float = 1e-8,
        window_criterion: float | int | str | None = None,
        window_criterion_tolerances: tuple[float, float] = (-np.inf, 7.0),
        lookahead: float | None = None,
        stepsize: int | None = None,
        max_mem_mb: int | None = 512,
        copy: bool = True,
        store_reconstruction_matrices: bool = False,
        learning_rate: float = 0.2,
        tau: float | None = None,
        mw_window_length: float = 20.0,
        mw_mode: str = "final_state",
        random_state: int | None = None,
        n_jobs: int | None = None,
        verbose: bool | str | int | None = None,
    ) -> None:
        super().__init__(
            sfreq=sfreq,
            cutoff=cutoff,
            window_length=window_length,
            window_overlap=calibration_window_overlap,
            max_dropout_fraction=max_dropout_fraction,
            min_clean_fraction=min_clean_fraction,
            method="standard",
            experimental=False,
            calibration="manual",
            picks=picks,
            calibration_window_length=calibration_window_length,
            calibration_window_overlap=calibration_window_overlap,
            ref_max_bad_channels=ref_max_bad_channels,
            ref_tolerances=ref_tolerances,
            blocksize=blocksize,
            max_dims=max_dims,
            reject_by_annotation=reject_by_annotation,
            skip_by_annotation=skip_by_annotation,
            cov_estimator="geometric_median",
            regularization=regularization,
            filter_kind="none",
            window_criterion=window_criterion,
            window_criterion_tolerances=window_criterion_tolerances,
            lookahead=lookahead,
            stepsize=stepsize,
            max_mem_mb=max_mem_mb,
            copy=copy,
            store_reconstruction_matrices=store_reconstruction_matrices,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.variant = variant
        self.update_window_length = update_window_length
        self.calibration_window_length = calibration_window_length
        self.calibration_window_overlap = calibration_window_overlap
        self.ref_max_bad_channels = ref_max_bad_channels
        self.ref_tolerances = ref_tolerances
        self.learning_rate = learning_rate
        self.tau = tau
        self.mw_window_length = mw_window_length
        self.mw_mode = mw_mode

    def fit(
        self,
        X: BaseRaw | BaseEpochs | np.ndarray,
        y=None,
        *,
        calibration: BaseRaw | BaseEpochs | np.ndarray | None = None,
        calibration_mask: np.ndarray | None = None,
    ) -> AdaptiveASR:
        """Fit the initial adaptive ASR state from calibration data."""
        del y
        self._validate_adaptive_params()
        fit_input = X if calibration is None else calibration
        data, sfreq, mne_type, orig_inst, _, _ = extract_data_from_mne(
            fit_input, auto_pick=False
        )
        if mne_type == "evoked":
            raise ValueError(
                "AdaptiveASR.fit() does not support Evoked calibration data"
            )
        sfreq = self._resolve_sfreq(sfreq)
        picks, ch_names = self._resolve_picks(fit_input, data, mne_type)
        data_2d = self._select_fit_data(data, mne_type, picks)
        if calibration_mask is not None:
            calibration_mask = np.asarray(calibration_mask, dtype=bool)
            if calibration_mask.shape != (data_2d.shape[1],):
                raise ValueError(
                    "calibration_mask must have shape (n_times,), got "
                    f"{calibration_mask.shape}"
                )
            data_2d = data_2d[:, calibration_mask]
        if mne_type == "raw" and self.reject_by_annotation:
            good_mask = _good_raw_sample_mask(orig_inst, self.skip_by_annotation)
            data_2d = data_2d[:, good_mask]

        self._warn_preprocessing_state(orig_inst, mne_type)

        if self.variant == "mw":
            # MW-ASR (sliding-window subspace, no Hebbian carry-over).
            # MATLAB AASR_demo.ipynb cell 4 semantics: per-window subspace
            # calibration; final state = last window's calibration; one
            # reconstruction pass over the whole stream uses that state.
            (
                state,
                cal_info,
                learner,
                process_state,
                mw_diagnostics,
            ) = self._fit_mw_state(data_2d, sfreq)
            self.mw_diagnostics_ = mw_diagnostics
        else:
            state, cal_info, learner, process_state = self._fit_adaptive_state(
                data_2d, sfreq
            )
            # Ensure attribute is always defined post-fit for consumer code.
            self.mw_diagnostics_ = []

        self.state_ = state
        self.sfreq_ = float(sfreq)
        self.picks_ = picks
        self.ch_names_ = ch_names
        self.n_channels_ = int(len(picks))
        self.M_ = state.M
        self.mixing_ = state.M
        self.T_ = state.T
        self.threshold_matrix_ = state.T
        self.thresholds_ = state.thresholds
        self.calibration_patterns_ = state.calibration_patterns
        self.patterns_ = state.calibration_patterns
        self.rank_ = state.rank
        self.clean_window_mask_ = cal_info["clean_window_mask"]
        self.calibration_mask_kind_ = "window"
        self.clean_window_scores_ = cal_info["clean_window_scores"]
        self.calibration_info_ = cal_info
        self.adaptive_learner_ = learner
        self._initial_process_state_template_ = _copy_process_state(process_state)
        self.process_state_ = _copy_process_state(process_state)
        self.adaptive_update_history_ = [dict(cal_info)]
        self.history_ = {
            "method": "adaptive",
            "variant": self.variant,
            "source_type": mne_type,
            "n_channels": self.n_channels_,
            "sfreq": self.sfreq_,
        }
        return self

    def partial_fit(
        self,
        X: BaseRaw | BaseEpochs | np.ndarray,
        y=None,
        *,
        calibration_mask: np.ndarray | None = None,
    ) -> AdaptiveASR:
        """Update the adaptive calibration state on a new clean chunk."""
        del y
        if self.variant == "mw":
            raise NotImplementedError(
                "AdaptiveASR(variant='mw') does not support partial_fit. "
                "MW-ASR semantics require a single fit() call over the full "
                "stream; the windowing happens internally. To re-calibrate, "
                "call fit() again."
            )
        if not hasattr(self, "state_"):
            return self.fit(X, calibration_mask=calibration_mask)

        data, sfreq, mne_type, orig_inst, _, _ = extract_data_from_mne(
            X, auto_pick=False
        )
        if mne_type == "evoked":
            raise ValueError("AdaptiveASR.partial_fit() does not support Evoked data")
        sfreq = self._resolve_sfreq(sfreq, fitted=True)
        if not np.isclose(sfreq, self.sfreq_):
            raise ValueError(
                f"Input sfreq {sfreq} does not match fitted sfreq {self.sfreq_}"
            )
        picks, ch_names = self._resolve_picks(X, data, mne_type)
        self._check_transform_channels(picks, ch_names)
        self._warn_preprocessing_state(orig_inst, mne_type)

        data_2d = self._select_fit_data(data, mne_type, picks)
        if calibration_mask is not None:
            calibration_mask = np.asarray(calibration_mask, dtype=bool)
            if calibration_mask.shape != (data_2d.shape[1],):
                raise ValueError(
                    "calibration_mask must have shape (n_times,), got "
                    f"{calibration_mask.shape}"
                )
            data_2d = data_2d[:, calibration_mask]
        if mne_type == "raw" and self.reject_by_annotation:
            good_mask = _good_raw_sample_mask(orig_inst, self.skip_by_annotation)
            data_2d = data_2d[:, good_mask]

        update_info = self._update_adaptive_state(data_2d, sfreq)
        self.adaptive_update_history_.append(update_info)
        self.calibration_info_ = update_info
        self.clean_window_mask_ = update_info["clean_window_mask"]
        self.calibration_mask_kind_ = "window"
        self.clean_window_scores_ = update_info["clean_window_scores"]
        self.M_ = self.state_.M
        self.mixing_ = self.state_.M
        self.T_ = self.state_.T
        self.threshold_matrix_ = self.state_.T
        self.thresholds_ = self.state_.thresholds
        self.calibration_patterns_ = self.state_.calibration_patterns
        self.patterns_ = self.state_.calibration_patterns
        self.rank_ = self.state_.rank
        return self

    def transform(
        self,
        X: BaseRaw | BaseEpochs | Evoked | np.ndarray,
        y=None,
        *,
        copy: bool | None = None,
        return_diagnostics: bool = False,
    ) -> Any:
        """Clean data using the current adaptive ASR state."""
        del y, copy
        self._check_is_fitted()
        data, sfreq, mne_type, orig_inst, _, _ = extract_data_from_mne(
            X, auto_pick=False
        )
        if mne_type == "evoked":
            raise ValueError("AdaptiveASR.transform() does not support Evoked data")
        sfreq = self._resolve_sfreq(sfreq, fitted=True)
        if not np.isclose(sfreq, self.sfreq_):
            raise ValueError(
                f"Input sfreq {sfreq} does not match fitted sfreq {self.sfreq_}"
            )
        picks, ch_names = self._resolve_picks(X, data, mne_type)
        self._check_transform_channels(picks, ch_names)
        self._warn_preprocessing_state(orig_inst, mne_type)

        if mne_type == "epochs":
            cleaned_data, diagnostics = self._transform_epochs_adaptive(
                data, picks, sfreq
            )
        else:
            data_out = np.asarray(data, dtype=np.float64).copy()
            selected = data_out[picks, :]
            selected_clean, diagnostics, next_process_state = _process_adaptive_chunk(
                selected,
                sfreq,
                self.state_,
                self.process_state_,
                window_length=self.window_length,
                lookahead=self.lookahead,
                stepsize=self.stepsize,
                max_dims=self.max_dims,
                store_reconstruction_matrices=self.store_reconstruction_matrices,
                adaptive_variant=self.variant,
                max_mem_mb=self.max_mem_mb,
            )
            if mne_type == "raw" and self.reject_by_annotation:
                good_mask = _good_raw_sample_mask(orig_inst, self.skip_by_annotation)
                selected_clean[:, ~good_mask] = selected[:, ~good_mask]
                diagnostics["sample_mask"] = diagnostics["sample_mask"] & good_mask
            if self._window_criterion_enabled():
                rejection_mask, rejection_diag = self._compute_window_rejection(
                    selected_clean, sfreq
                )
                if mne_type == "raw" and self.reject_by_annotation:
                    rejection_mask = rejection_mask & good_mask
                diagnostics.update(
                    {
                        "rejection_sample_mask": rejection_mask,
                        "rejection_window_starts": rejection_diag["window_starts"],
                        "rejection_window_stops": rejection_diag["window_stops"],
                        "rejection_window_keep_mask": rejection_diag[
                            "window_keep_mask"
                        ],
                        "rejection_window_remove_mask": rejection_diag[
                            "window_remove_mask"
                        ],
                        "fraction_retained_after_window_rejection": float(
                            np.mean(rejection_mask)
                        ),
                        "fraction_rejected_after_window_rejection": float(
                            1.0 - np.mean(rejection_mask)
                        ),
                    }
                )
            data_out[picks, :] = selected_clean
            cleaned_data = data_out
            self.process_state_ = next_process_state

        self._store_transform_diagnostics(diagnostics)
        cleaned = reconstruct_mne_object(
            cleaned_data, orig_inst, mne_type, verbose=False
        )
        if return_diagnostics:
            return cleaned, diagnostics
        return cleaned

    def fit_transform(
        self,
        X: BaseRaw | BaseEpochs | np.ndarray,
        y=None,
        *,
        calibration: BaseRaw | BaseEpochs | np.ndarray | None = None,
        return_diagnostics: bool = False,
    ) -> Any:
        """Fit adaptive ASR and reconstruct ``X`` with the fitted state.

        For ``variant="mw", mw_mode="sliding"`` the work is per-window
        calibrate-AND-transform: each window is calibrated on itself and
        cleaned by that local calibration, then the cleaned slices are
        concatenated. ``fit()`` alone is still legal in sliding mode (it
        records per-window diagnostics) but ``transform()`` afterwards
        applies only the final window's state — semantically equivalent to
        ``mw_mode="final_state"``. The true per-segment behavior requires
        this ``fit_transform`` entry point.
        """
        if (
            self.variant == "mw"
            and getattr(self, "mw_mode", "final_state") == "sliding"
        ):
            return self._fit_transform_mw_sliding(
                X, calibration=calibration, return_diagnostics=return_diagnostics
            )
        self.fit(X, y=y, calibration=calibration)
        return self.transform(X, return_diagnostics=return_diagnostics)

    def _fit_transform_mw_sliding(
        self,
        X: BaseRaw | BaseEpochs | np.ndarray,
        *,
        calibration: BaseRaw | BaseEpochs | np.ndarray | None = None,
        return_diagnostics: bool = False,
    ) -> Any:
        """Per-window calibrate-AND-clean implementation for MW sliding mode.

        For each non-overlapping window of length ``mw_window_length``:

        1. Run the existing AdaptiveASR calibration on that window's data.
        2. Apply the resulting state to clean the same window's data.
        3. Concatenate the cleaned slices and return.

        Windows shorter than ``blocksize`` are skipped (data passes through
        unchanged). Calibration failures are also passed through.
        """
        self._validate_estimator_params()
        self._validate_adaptive_params()
        fit_input = X if calibration is None else calibration
        data, sfreq, mne_type, orig_inst, _, _ = extract_data_from_mne(
            fit_input, auto_pick=False
        )
        if mne_type == "evoked":
            raise ValueError(
                "AdaptiveASR.fit_transform() does not support Evoked input "
                "with variant='mw', mw_mode='sliding'."
            )
        sfreq_val = self._resolve_sfreq(sfreq)
        picks, ch_names = self._resolve_picks(fit_input, data, mne_type)
        data_2d = self._select_fit_data(data, mne_type, picks)
        if mne_type == "raw" and self.reject_by_annotation:
            good_mask = _good_raw_sample_mask(orig_inst, self.skip_by_annotation)
            data_2d_masked = data_2d[:, good_mask]
        else:
            data_2d_masked = data_2d

        self._warn_preprocessing_state(orig_inst, mne_type)

        n_times = data_2d_masked.shape[1]
        win_samples = max(1, int(round(self.mw_window_length * sfreq_val)))
        cleaned = data_2d_masked.copy()
        mw_diagnostics: list[dict[str, Any]] = []

        n_windows = (n_times + win_samples - 1) // win_samples
        last = None  # store the last successful calibration for the public state
        for window_idx in range(n_windows):
            start = window_idx * win_samples
            stop = min(start + win_samples, n_times)
            window = data_2d_masked[:, start:stop]
            entry: dict[str, Any] = {
                "window_idx": int(window_idx),
                "window_start": int(start),
                "window_stop": int(stop),
                "n_samples": int(window.shape[1]),
            }
            if window.shape[1] < self.blocksize:
                entry["status"] = "skipped_too_short"
                mw_diagnostics.append(entry)
                continue
            try:
                state, cal_info, learner, process_state = self._fit_adaptive_state(
                    window, sfreq_val
                )
                window_cleaned, _, _ = _process_adaptive_chunk(
                    window,
                    sfreq_val,
                    state,
                    process_state,
                    window_length=self.window_length,
                    lookahead=self.lookahead,
                    stepsize=self.stepsize,
                    max_dims=self.max_dims,
                    store_reconstruction_matrices=False,
                    adaptive_variant=self.variant,
                    max_mem_mb=self.max_mem_mb,
                )
                cleaned[:, start:stop] = window_cleaned
                entry.update(
                    {
                        "status": "passed",
                        "M": np.asarray(state.M, dtype=np.float64),
                        "T": np.asarray(state.T, dtype=np.float64),
                        "thresholds": np.asarray(state.thresholds, dtype=np.float64),
                        "rank": int(state.rank),
                        "clean_window_fraction": cal_info.get(
                            "calibration_clean_window_fraction"
                        ),
                    }
                )
                last = (state, cal_info, learner, process_state)
            except Exception as exc:  # noqa: BLE001
                entry.update(
                    {
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            mw_diagnostics.append(entry)

        if last is None:
            raise RuntimeError(
                "MW-ASR sliding-mode fit_transform() found no usable window "
                f"(n_windows={n_windows}, mw_window_length={self.mw_window_length})"
            )
        state, cal_info, learner, process_state = last
        cal_info = dict(cal_info)
        cal_info["adaptive_variant"] = "mw"
        cal_info["mw_mode"] = "sliding"
        cal_info["mw_n_windows"] = int(len(mw_diagnostics))
        cal_info["mw_window_length_s"] = float(self.mw_window_length)

        # Populate the standard fitted-state attributes from the FINAL window's
        # calibration so downstream introspection (.M_, .T_, .calibration_info_,
        # ...) behaves the same way as the existing final_state mode.
        self.state_ = state
        self.sfreq_ = float(sfreq_val)
        self.picks_ = picks
        self.ch_names_ = ch_names
        self.n_channels_ = int(len(picks))
        self.M_ = state.M
        self.mixing_ = state.M
        self.T_ = state.T
        self.threshold_matrix_ = state.T
        self.thresholds_ = state.thresholds
        self.calibration_patterns_ = state.calibration_patterns
        self.patterns_ = state.calibration_patterns
        self.rank_ = state.rank
        self.clean_window_mask_ = np.array([], dtype=bool)
        self.calibration_mask_kind_ = "window"
        self.clean_window_scores_ = np.empty((0, len(picks)), dtype=np.float64)
        self.calibration_info_ = cal_info
        self.mw_diagnostics_ = mw_diagnostics
        self.process_state_ = _copy_process_state(process_state)
        self._initial_process_state_template_ = _copy_process_state(process_state)
        self._adaptive_learner_ = learner
        self.history_ = {
            "method": "adaptive",
            "variant": "mw",
            "mw_mode": "sliding",
            "source_type": mne_type,
            "n_channels": self.n_channels_,
            "sfreq": self.sfreq_,
        }
        self.diagnostics_ = {
            "adaptive_variant": "mw",
            "mw_mode": "sliding",
            "covariance_geometry": "adaptive",
            "n_components_reconstructed": np.zeros(len(mw_diagnostics), dtype=int),
            "fraction_reconstructed_samples": 0.0,
            "fraction_reconstructed_windows": 0.0,
            "n_windows": int(len(mw_diagnostics)),
        }

        # Wire the cleaned 2D array back into whatever MNE container the caller
        # provided (or pass through if ndarray input).
        full = np.asarray(data, dtype=np.float64).copy()
        if mne_type == "raw" and self.reject_by_annotation:
            # Two-step indexing: full[picks] gives a (n_picks, n_times) view;
            # then mask the good samples and write cleaned in. Plain
            # full[picks, good_mask] triggers advanced-indexing broadcasting
            # which fails when picks and good_mask have incompatible shapes.
            sub = full[picks].copy()
            sub[:, good_mask] = cleaned
            sub[:, ~good_mask] = data_2d[:, ~good_mask]
            full[picks] = sub
        else:
            full[picks, :] = cleaned
        result = reconstruct_mne_object(full, orig_inst, mne_type, verbose=False)
        if return_diagnostics:
            return result, self.diagnostics_
        return result

    def reset_process_state(self) -> None:
        """Reset the streaming reconstruction state to the fitted baseline."""
        self._check_is_fitted()
        self.process_state_ = _copy_process_state(self._initial_process_state_template_)

    def _validate_adaptive_params(self) -> None:
        if self.variant not in ("psp", "psw", "mw"):
            raise ValueError("variant must be 'psp', 'psw', or 'mw'")
        if self.update_window_length <= 0:
            raise ValueError("update_window_length must be positive")
        if self.calibration_window_length <= 0:
            raise ValueError("clean_window_length must be positive")
        if not (0 <= self.calibration_window_overlap < 1):
            raise ValueError("clean_window_overlap must be in [0, 1)")
        if self.ref_max_bad_channels < 0:
            raise ValueError("clean_max_bad_channels must be non-negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.tau is not None and self.tau <= 0:
            raise ValueError("tau must be positive")
        if self.variant == "mw" and self.mw_window_length <= 0:
            raise ValueError("mw_window_length must be positive for variant='mw'")
        if self.variant == "mw" and self.mw_mode not in ("final_state", "sliding"):
            raise ValueError(
                "mw_mode must be 'final_state' or 'sliding' for variant='mw'"
            )

    def _variant_for_learner(self) -> str:
        """Map the public variant onto the learner's 'psp' / 'psw' vocabulary.

        The underlying Hebbian learner only knows ``'psp'`` / ``'psw'``. MW-ASR
        builds a fresh learner per window but never streams updates through it
        (partial_fit is disabled), so the choice of underlying learner variant
        is cosmetic for MW.
        """
        return "psp" if self.variant == "mw" else self.variant

    def _fit_mw_state(
        self,
        X: np.ndarray,
        sfreq: float,
    ) -> tuple[
        ASRState,
        dict[str, Any],
        _AdaptiveSimilarityMatcher,
        dict[str, Any],
        list[dict[str, Any]],
    ]:
        """MW-ASR: per-window subspace calibration, final-window state.

        Implements the MATLAB ``AASR_demo.ipynb`` Cell 4 pattern: split the
        input into non-overlapping windows of length ``mw_window_length``
        seconds, run the standard subspace calibration on each window,
        record per-window diagnostics, and keep only the final window's
        state for the subsequent reconstruction pass.

        Windows shorter than ``blocksize`` samples are skipped (cannot
        calibrate).
        """
        X = _validate_array_2d(X)
        sfreq = float(sfreq)
        n_times = X.shape[1]
        win_samples = max(1, int(round(self.mw_window_length * sfreq)))

        diagnostics_list: list[dict[str, Any]] = []
        last = None
        n_windows = (n_times + win_samples - 1) // win_samples
        for window_idx in range(n_windows):
            start = window_idx * win_samples
            stop = min(start + win_samples, n_times)
            window = X[:, start:stop]
            if window.shape[1] < self.blocksize:
                continue
            try:
                state, cal_info, learner, process_state = self._fit_adaptive_state(
                    window, sfreq
                )
            except Exception as exc:  # noqa: BLE001
                diagnostics_list.append(
                    {
                        "window_idx": int(window_idx),
                        "window_start": int(start),
                        "window_stop": int(stop),
                        "n_samples": int(window.shape[1]),
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                continue
            diagnostics_list.append(
                {
                    "window_idx": int(window_idx),
                    "window_start": int(start),
                    "window_stop": int(stop),
                    "n_samples": int(window.shape[1]),
                    "status": "passed",
                    "M": np.asarray(state.M, dtype=np.float64),
                    "T": np.asarray(state.T, dtype=np.float64),
                    "thresholds": np.asarray(state.thresholds, dtype=np.float64),
                    "rank": int(state.rank),
                    "clean_window_fraction": cal_info.get(
                        "calibration_clean_window_fraction"
                    ),
                }
            )
            last = (state, cal_info, learner, process_state)

        if last is None:
            raise RuntimeError(
                "MW-ASR fit() found no usable window for calibration "
                f"(n_windows={n_windows}, mw_window_length={self.mw_window_length})"
            )
        state, cal_info, learner, process_state = last
        cal_info = dict(cal_info)
        cal_info["adaptive_variant"] = "mw"
        cal_info["mw_n_windows"] = int(len(diagnostics_list))
        cal_info["mw_window_length_s"] = float(self.mw_window_length)
        return state, cal_info, learner, process_state, diagnostics_list

    def _fit_adaptive_state(
        self,
        X: np.ndarray,
        sfreq: float,
    ) -> tuple[ASRState, dict[str, Any], _AdaptiveSimilarityMatcher, dict[str, Any]]:
        X = _validate_array_2d(X)
        X_clean, clean_sample_mask, clean_diag = _select_aasr_clean_samples(
            X,
            sfreq,
            window_length=self.calibration_window_length,
            window_overlap=self.calibration_window_overlap,
            max_bad_channels=self.ref_max_bad_channels,
            zthresholds=self.ref_tolerances,
            max_dropout_fraction=self.max_dropout_fraction,
            min_clean_fraction=self.min_clean_fraction,
        )
        filter_b, filter_a = _design_aasr_filter(sfreq)
        X_filtered, iir_state = _lfilter_channels(X_clean, filter_b, filter_a)
        M, C, eigvals, V, covariance_memory_info = _adaptive_covariance_sqrt(
            X_filtered,
            blocksize=self.blocksize,
            regularization=self.regularization,
            max_mem_mb=self.max_mem_mb,
        )
        thresholds, threshold_info = _fit_adaptive_thresholds(
            X_filtered,
            V,
            sfreq=sfreq,
            window_length=self.window_length,
            window_overlap=self.calibration_window_overlap,
            cutoff=self.cutoff,
            min_clean_fraction=self.min_clean_fraction,
            max_dropout_fraction=self.max_dropout_fraction,
        )
        state = ASRState(
            M=M,
            T=np.diag(thresholds) @ V.T,
            thresholds=thresholds,
            calibration_patterns=V,
            filter_b=filter_b,
            filter_a=filter_a,
            cov=C,
            rank=int(np.sum(eigvals > self.regularization * np.max(eigvals))),
            method="standard",
            riemannian_solver=None,
        )
        learner = _build_adaptive_learner(
            X_filtered,
            V,
            variant=self._variant_for_learner(),
            learning_rate=self.learning_rate,
            tau=self._resolved_tau(),
            regularization=self.regularization,
        )
        process_state = {
            "cov": None,
            "carry": None,
            "iir": iir_state.copy(),
            "last_R": None,
            "last_trivial": True,
        }
        diagnostics = self._adaptive_calibration_info(
            clean_diag,
            clean_sample_mask,
            thresholds,
            threshold_info,
            event="fit",
        )
        diagnostics.update(covariance_memory_info)
        diagnostics["rank"] = int(state.rank)
        return state, diagnostics, learner, process_state

    def _update_adaptive_state(self, X: np.ndarray, sfreq: float) -> dict[str, Any]:
        X = _validate_array_2d(X)
        X_clean, clean_sample_mask, clean_diag = _select_aasr_clean_samples(
            X,
            sfreq,
            window_length=self.calibration_window_length,
            window_overlap=self.calibration_window_overlap,
            max_bad_channels=self.ref_max_bad_channels,
            zthresholds=self.ref_tolerances,
            max_dropout_fraction=self.max_dropout_fraction,
            min_clean_fraction=self.min_clean_fraction,
        )
        X_filtered, _ = _lfilter_channels(
            X_clean, self.state_.filter_b, self.state_.filter_a
        )
        self.adaptive_learner_.fit_next(X_filtered)
        V = self.adaptive_learner_.get_components()
        M, C, eigvals, _, covariance_memory_info = _adaptive_covariance_sqrt(
            X_filtered,
            blocksize=self.blocksize,
            regularization=self.regularization,
            max_mem_mb=self.max_mem_mb,
        )
        thresholds, threshold_info = _fit_adaptive_thresholds(
            X_filtered,
            V,
            sfreq=sfreq,
            window_length=self.update_window_length,
            window_overlap=self.calibration_window_overlap,
            cutoff=self.cutoff,
            min_clean_fraction=self.min_clean_fraction,
            max_dropout_fraction=self.max_dropout_fraction,
        )

        self.state_.M = M
        self.state_.T = np.diag(thresholds) @ V.T
        self.state_.thresholds = thresholds
        self.state_.calibration_patterns = V
        self.state_.cov = C
        self.state_.rank = int(np.sum(eigvals > self.regularization * np.max(eigvals)))

        diagnostics = self._adaptive_calibration_info(
            clean_diag,
            clean_sample_mask,
            thresholds,
            threshold_info,
            event="update",
        )
        diagnostics.update(covariance_memory_info)
        return diagnostics

    def _adaptive_calibration_info(
        self,
        clean_diag: dict[str, Any],
        clean_sample_mask: np.ndarray,
        thresholds: np.ndarray,
        threshold_info: dict[str, np.ndarray],
        *,
        event: str,
    ) -> dict[str, Any]:
        return {
            "event": event,
            "clean_window_mask": np.asarray(
                clean_diag["window_keep_mask"], dtype=bool
            ).copy(),
            "clean_window_scores": np.asarray(
                clean_diag["window_rms_zscores"], dtype=np.float64
            ).copy(),
            "clean_sample_mask": np.asarray(clean_sample_mask, dtype=bool).copy(),
            "calibration_window_starts": np.asarray(
                clean_diag["window_starts"], dtype=int
            ).copy(),
            "calibration_window_length_samples": int(
                clean_diag["window_stops"][0] - clean_diag["window_starts"][0]
            ),
            "blocksize": int(self.blocksize),
            "n_clean_windows": int(np.sum(clean_diag["window_keep_mask"])),
            "n_calibration_windows": int(clean_diag["n_windows"]),
            "calibration_samples": int(np.sum(clean_sample_mask)),
            "rank": int(self.state_.rank) if hasattr(self, "state_") else 0,
            "thresholds": thresholds.copy(),
            "threshold_mu": threshold_info["mu"].copy(),
            "threshold_sigma": threshold_info["sigma"].copy(),
            "threshold_beta": threshold_info["beta"].copy(),
            "threshold_fit_error": threshold_info["fit_error"].copy(),
            "threshold_fit_interval": threshold_info["fit_interval"].copy(),
            "threshold_window_starts": threshold_info["window_starts"].copy(),
            "threshold_window_length_samples": int(
                threshold_info["window_length_samples"]
            ),
            "covariance_geometry": "standard",
            "adaptive_variant": self.variant,
            "statistics_filter": "yulewalk",
        }

    def _resolved_tau(self) -> float:
        if self.tau is not None:
            return float(self.tau)
        return 1e-5 if self.variant == "psw" else 0.8

    def _transform_epochs_adaptive(
        self,
        data: np.ndarray,
        picks: np.ndarray,
        sfreq: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        cleaned = np.asarray(data, dtype=np.float64).copy()
        epoch_diags = []
        starts_all: list[np.ndarray] = []
        stops_all: list[np.ndarray] = []
        sample_masks: list[np.ndarray] = []
        rejection_masks: list[np.ndarray] = []
        rejection_starts_all: list[np.ndarray] = []
        rejection_stops_all: list[np.ndarray] = []
        rejection_keep_masks: list[np.ndarray] = []
        rejection_remove_masks: list[np.ndarray] = []
        counts: list[np.ndarray] = []
        for epoch_idx in range(cleaned.shape[0]):
            selected = cleaned[epoch_idx, picks, :]
            selected_clean, diag, _ = _process_adaptive_chunk(
                selected,
                sfreq,
                _copy_asr_state(self.state_),
                _copy_process_state(self._initial_process_state_template_),
                window_length=self.window_length,
                lookahead=self.lookahead,
                stepsize=self.stepsize,
                max_dims=self.max_dims,
                store_reconstruction_matrices=self.store_reconstruction_matrices,
                adaptive_variant=self.variant,
                max_mem_mb=self.max_mem_mb,
            )
            cleaned[epoch_idx, picks, :] = selected_clean
            if self._window_criterion_enabled():
                rejection_mask, rejection_diag = self._compute_window_rejection(
                    selected_clean,
                    sfreq,
                )
                diag["rejection_sample_mask"] = rejection_mask
                diag["rejection_window_starts"] = rejection_diag["window_starts"]
                diag["rejection_window_stops"] = rejection_diag["window_stops"]
                diag["rejection_window_keep_mask"] = rejection_diag["window_keep_mask"]
                diag["rejection_window_remove_mask"] = rejection_diag[
                    "window_remove_mask"
                ]
                diag["fraction_retained_after_window_rejection"] = float(
                    np.mean(rejection_mask)
                )
                diag["fraction_rejected_after_window_rejection"] = float(
                    1.0 - np.mean(rejection_mask)
                )
            epoch_diags.append(diag)
            starts_all.append(diag["window_starts"])
            stops_all.append(diag["window_stops"])
            sample_masks.append(diag["sample_mask"])
            if "rejection_sample_mask" in diag:
                rejection_masks.append(diag["rejection_sample_mask"])
                rejection_starts_all.append(diag["rejection_window_starts"])
                rejection_stops_all.append(diag["rejection_window_stops"])
                rejection_keep_masks.append(diag["rejection_window_keep_mask"])
                rejection_remove_masks.append(diag["rejection_window_remove_mask"])
            counts.append(diag["n_components_reconstructed"])

        diagnostics = {
            "epoch_diagnostics": epoch_diags,
            "window_starts": np.concatenate(starts_all)
            if starts_all
            else np.array([], dtype=int),
            "window_stops": np.concatenate(stops_all)
            if stops_all
            else np.array([], dtype=int),
            "sample_mask": np.vstack(sample_masks)
            if sample_masks
            else np.empty((0, 0), dtype=bool),
            "n_components_reconstructed": np.concatenate(counts)
            if counts
            else np.array([], dtype=int),
            "n_windows": int(sum(diag["n_windows"] for diag in epoch_diags)),
            "covariance_geometry": "standard",
            "adaptive_variant": self.variant,
        }
        diagnostics["fraction_reconstructed_windows"] = (
            float(np.mean(diagnostics["n_components_reconstructed"] > 0))
            if diagnostics["n_components_reconstructed"].size
            else 0.0
        )
        diagnostics["fraction_reconstructed_samples"] = (
            float(np.mean(diagnostics["sample_mask"]))
            if diagnostics["sample_mask"].size
            else 0.0
        )
        diagnostics["max_components_reconstructed"] = int(
            diagnostics["n_components_reconstructed"].max(initial=0)
        )
        if rejection_masks:
            diagnostics["rejection_sample_mask"] = np.vstack(rejection_masks)
            diagnostics["rejection_window_starts"] = (
                np.concatenate(rejection_starts_all)
                if rejection_starts_all
                else np.array([], dtype=int)
            )
            diagnostics["rejection_window_stops"] = (
                np.concatenate(rejection_stops_all)
                if rejection_stops_all
                else np.array([], dtype=int)
            )
            diagnostics["rejection_window_keep_mask"] = (
                np.concatenate(rejection_keep_masks)
                if rejection_keep_masks
                else np.array([], dtype=bool)
            )
            diagnostics["rejection_window_remove_mask"] = (
                np.concatenate(rejection_remove_masks)
                if rejection_remove_masks
                else np.array([], dtype=bool)
            )
            diagnostics["fraction_retained_after_window_rejection"] = float(
                np.mean(diagnostics["rejection_sample_mask"])
            )
            diagnostics["fraction_rejected_after_window_rejection"] = float(
                1.0 - np.mean(diagnostics["rejection_sample_mask"])
            )
        return cleaned, diagnostics
