"""ASR processing: reconstruct artifact subspaces in incoming data.

``process_asr`` applies the fitted calibration window by window, rejecting and
reconstructing the principal subspaces whose variance exceeds the
per-direction thresholds (raised-cosine blended, with lookahead). The
Riemannian variants implement the experimental SPD-geometry backends.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._covariance import (
    _covariance_stack_bytes,
    _iter_moving_covariances_at,
    _max_mem_bytes,
    _moving_average_clean_rawdata,
    _process_memory_info,
)
from ._filters import (
    _append_clean_rawdata_tail,
    _apply_statistics_filter_streaming,
    _prepend_clean_rawdata_carry,
)
from ._spd import _regularize_spd, _riemannian_nonlinear_eigenspace
from ._types import ASRState
from ._validation import (
    _resolve_max_dims_clean_rawdata,
    _round_half_up,
    _validate_array_2d,
    _validate_common_params,
)


def process_asr(
    X: np.ndarray,
    sfreq: float,
    state: ASRState,
    *,
    window_length: float = 0.5,
    window_overlap: float = 0.66,
    max_dims: float | int = 0.66,
    regularization: float = 1e-8,
    store_reconstruction_matrices: bool = False,
    max_mem_mb: int | None = 512,
    lookahead: float | None = None,
    stepsize: int | None = None,
    method: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a calibrated ASR model to continuous data.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        Continuous data in the same channel order and units used for
        calibration.
    sfreq : float
        Sampling frequency in Hz.
    state : ASRState
        Fitted calibration state from :func:`calibrate_asr`.
    window_length : float
        Processing window length in seconds.
    window_overlap : float
        Calibration threshold-window overlap. Processing follows the
        ``clean_rawdata`` streaming ASR algorithm and uses ``stepsize`` for
        reconstruction-matrix updates.
    max_dims : float | int
        Maximum number of dimensions reconstructed per window. Floats in
        ``[0, 1]`` are interpreted as a fraction of channels.
    regularization : float
        Relative eigenvalue floor for window covariances.
    store_reconstruction_matrices : bool
        If True, store all window reconstruction matrices in diagnostics.
    max_mem_mb : int | None
        Reserved memory limit for future chunking. Present for API stability.
    lookahead : float | None
        Processing lookahead in seconds. If None, use ``window_length / 2``.
    stepsize : int | None
        Number of samples between reconstruction-matrix updates. If None, use
        ``floor(sfreq * window_length / 2)``, matching ``clean_asr.m``.
    method : {'standard', 'riemannian'} | None
        Covariance geometry for processing. If ``None``, use ``state.method``.

    Returns
    -------
    X_clean : ndarray, shape (n_channels, n_times)
        Cleaned data.
    diagnostics : dict
        Processing diagnostics.
    """
    _validate_common_params(
        sfreq=sfreq,
        cutoff=1.0,
        window_length=window_length,
        window_overlap=window_overlap,
        max_dropout_fraction=0.1,
        min_clean_fraction=0.25,
        regularization=regularization,
    )
    X = _validate_array_2d(X)
    n_channels, n_times = X.shape
    if state.M.shape != (n_channels, n_channels):
        raise ValueError(
            "ASR state channel count does not match data: "
            f"{state.M.shape[0]} vs {n_channels}"
        )
    if method is None:
        method = getattr(state, "method", "standard")
    if method not in ("standard", "riemannian", "riemannian_windowed"):
        raise ValueError(
            "method must be 'standard', 'riemannian', or 'riemannian_windowed'"
        )

    win_len = max(
        _round_half_up(window_length * sfreq), _round_half_up(1.5 * n_channels)
    )
    if n_times < win_len:
        raise ValueError(
            f"Window length ({win_len} samples) exceeds data length ({n_times} samples)"
        )
    lookahead = (win_len / sfreq) / 2.0 if lookahead is None else float(lookahead)
    if lookahead < 0:
        raise ValueError("lookahead must be non-negative")
    lookahead_samples = _round_half_up(lookahead * sfreq)
    if lookahead_samples >= n_times:
        raise ValueError("lookahead is too long for the data length")
    if stepsize is None:
        stepsize = max(1, int(np.floor(sfreq * (win_len / sfreq) / 2.0)))
    else:
        stepsize = int(stepsize)
    if stepsize < 1:
        raise ValueError("stepsize must be at least 1 sample")
    if stepsize > win_len:
        raise ValueError("stepsize must not exceed window_length in samples")

    max_bad = _resolve_max_dims_clean_rawdata(max_dims, n_channels)
    if max_bad <= 0:
        diagnostics = _empty_process_diagnostics(n_times)
        diagnostics.update(
            _process_memory_info(
                n_channels=n_channels,
                n_stream_input=n_times,
                max_mem_mb=max_mem_mb,
                memory_mode="identity",
                peak_cov_buffer_bytes=0,
                chunk_samples=0,
                used_memory_bound=False,
            )
        )
        return X.copy(), diagnostics

    X_proc = _append_clean_rawdata_tail(X, lookahead_samples)
    data_stream = _prepend_clean_rawdata_carry(X_proc, lookahead_samples)
    data_stream[~np.isfinite(data_stream)] = 0.0
    n_stream_input = X_proc.shape[1]
    X_stats = _apply_statistics_filter_streaming(
        data_stream[:, lookahead_samples : lookahead_samples + n_stream_input],
        state.filter_b,
        state.filter_a,
    )
    update_at = np.minimum(
        np.arange(stepsize, n_stream_input + stepsize, stepsize, dtype=int),
        n_stream_input,
    )
    if update_at.size == 0 or update_at[-1] != n_stream_input:
        update_at = np.append(update_at, n_stream_input)
    update_at = np.unique(update_at)
    update_at = np.concatenate(([1], update_at))
    estimated_cov_bytes = _covariance_stack_bytes(n_stream_input, n_channels)
    max_mem_bytes = _max_mem_bytes(max_mem_mb)
    use_rolling_covariance = (
        max_mem_bytes is not None and estimated_cov_bytes > max_mem_bytes
    )

    if method == "riemannian":
        X_clean, diagnostics = _process_asr_riemannian(
            data_stream,
            X_stats,
            state,
            n_times=n_times,
            n_stream_input=n_stream_input,
            lookahead_samples=lookahead_samples,
            update_at=update_at,
            max_bad=max_bad,
            stepsize=stepsize,
            win_len=win_len,
            regularization=regularization,
            store_reconstruction_matrices=store_reconstruction_matrices,
        )
        diagnostics.update(
            _process_memory_info(
                n_channels=n_channels,
                n_stream_input=n_stream_input,
                max_mem_mb=max_mem_mb,
                memory_mode="riemannian",
                peak_cov_buffer_bytes=_covariance_stack_bytes(1, n_channels),
                chunk_samples=n_stream_input,
                used_memory_bound=False,
            )
        )
        return X_clean, diagnostics

    if method == "riemannian_windowed":
        X_clean, diagnostics = _process_asr_riemannian_windowed(
            data_stream,
            X_stats,
            state,
            n_times=n_times,
            n_stream_input=n_stream_input,
            lookahead_samples=lookahead_samples,
            update_at=update_at,
            max_bad=max_bad,
            stepsize=stepsize,
            win_len=win_len,
            regularization=regularization,
            store_reconstruction_matrices=store_reconstruction_matrices,
            use_rolling_covariance=use_rolling_covariance,
        )
        diagnostics.update(
            _process_memory_info(
                n_channels=n_channels,
                n_stream_input=n_stream_input,
                max_mem_mb=max_mem_mb,
                memory_mode=(
                    "riemannian_windowed_rolling"
                    if use_rolling_covariance
                    else "riemannian_windowed"
                ),
                peak_cov_buffer_bytes=_covariance_stack_bytes(1, n_channels),
                chunk_samples=win_len if use_rolling_covariance else n_stream_input,
                used_memory_bound=use_rolling_covariance,
            )
        )
        return X_clean, diagnostics

    if use_rolling_covariance:
        covariance_iter = _iter_moving_covariances_at(X_stats, update_at, win_len)
        Xcov_flat = None
    else:
        outer = np.einsum("it,jt->ijt", X_stats, X_stats, optimize=True)
        Xcov_flat = outer.reshape(n_channels * n_channels, n_stream_input, order="F")
        Xcov_flat, _ = _moving_average_clean_rawdata(win_len, Xcov_flat)
        covariance_iter = None

    sample_mask = np.zeros(n_times, dtype=bool)
    n_reconstructed: list[int] = []
    component_variances: list[np.ndarray] = []
    component_thresholds: list[np.ndarray] = []
    reconstruction_matrices: list[np.ndarray] = []
    window_starts: list[int] = []
    window_stops: list[int] = []

    eye = np.eye(n_channels)
    last_R = eye
    last_trivial = True
    last_n = 0
    for n in update_at:
        if covariance_iter is None:
            assert Xcov_flat is not None
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
            R = eye
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

    X_clean = data_stream[:, lookahead_samples : lookahead_samples + n_times].copy()

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
        "covariance_geometry": method,
    }
    if use_rolling_covariance:
        diagnostics.update(
            _process_memory_info(
                n_channels=n_channels,
                n_stream_input=n_stream_input,
                max_mem_mb=max_mem_mb,
                memory_mode="rolling",
                peak_cov_buffer_bytes=_covariance_stack_bytes(1, n_channels),
                chunk_samples=win_len,
                used_memory_bound=True,
            )
        )
    else:
        diagnostics.update(
            _process_memory_info(
                n_channels=n_channels,
                n_stream_input=n_stream_input,
                max_mem_mb=max_mem_mb,
                memory_mode="full",
                peak_cov_buffer_bytes=estimated_cov_bytes,
                chunk_samples=n_stream_input,
                used_memory_bound=False,
            )
        )
    if store_reconstruction_matrices:
        diagnostics["reconstruction_matrices"] = np.asarray(reconstruction_matrices)
    return X_clean, diagnostics


def _empty_process_diagnostics(n_times: int) -> dict[str, Any]:
    """Return identity-processing diagnostics."""
    return {
        "window_starts": np.array([0], dtype=int),
        "window_stops": np.array([n_times], dtype=int),
        "sample_mask": np.zeros(n_times, dtype=bool),
        "n_components_reconstructed": np.array([0], dtype=int),
        "component_variances": np.empty((1, 0), dtype=np.float64),
        "component_thresholds": np.empty((1, 0), dtype=np.float64),
        "n_windows": 1,
        "fraction_reconstructed_windows": 0.0,
        "fraction_reconstructed_samples": 0.0,
        "max_components_reconstructed": 0,
        "lookahead_samples": 0,
        "stepsize_samples": 0,
        "window_length_samples": 0,
    }


def _process_asr_riemannian(
    data_stream: np.ndarray,
    X_stats: np.ndarray,
    state: ASRState,
    *,
    n_times: int,
    n_stream_input: int,
    lookahead_samples: int,
    update_at: np.ndarray,
    max_bad: int,
    stepsize: int,
    win_len: int,
    regularization: float,
    store_reconstruction_matrices: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the experimental rASRMatlab-style chunk covariance backend."""
    n_channels = data_stream.shape[0]
    eye = np.eye(n_channels)
    Cw = (X_stats @ X_stats.T) / n_stream_input
    Cw = _regularize_spd(Cw, regularization)

    D, V = _riemannian_nonlinear_eigenspace(Cw, regularization)
    theta2 = np.sum((state.T @ V) ** 2, axis=0)
    keep = (theta2 > D) | (np.arange(1, n_channels + 1) < (n_channels - max_bad))
    trivial = bool(np.all(keep))
    n_bad = int(n_channels - np.count_nonzero(keep))

    if trivial:
        R = eye
    else:
        basis = keep[:, np.newaxis].astype(np.float64) * (V.T @ state.M)
        R = state.M @ np.linalg.pinv(basis) @ V.T
        R = np.real_if_close(R).astype(np.float64)

    sample_mask = np.zeros(n_times, dtype=bool)
    n_reconstructed: list[int] = []
    component_variances: list[np.ndarray] = []
    component_thresholds: list[np.ndarray] = []
    reconstruction_matrices: list[np.ndarray] = []
    window_starts: list[int] = []
    window_stops: list[int] = []

    last_R = eye
    last_trivial = True
    last_n = 0
    for n in update_at:
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

    X_clean = data_stream[:, lookahead_samples : lookahead_samples + n_times].copy()
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
        "covariance_geometry": "riemannian",
        "riemannian_solver": "nonlinear_eigenspace",
        "riemannian_mean_iterations": np.zeros(
            len(n_reconstructed_arr),
            dtype=int,
        ),
        "riemannian_mean_converged": np.ones(
            len(n_reconstructed_arr),
            dtype=bool,
        ),
        "riemannian_mean_update_norm": np.zeros(
            len(n_reconstructed_arr),
            dtype=np.float64,
        ),
    }
    if store_reconstruction_matrices:
        diagnostics["reconstruction_matrices"] = np.asarray(reconstruction_matrices)
    return X_clean, diagnostics


def _process_asr_riemannian_windowed(
    data_stream: np.ndarray,
    X_stats: np.ndarray,
    state: ASRState,
    *,
    n_times: int,
    n_stream_input: int,
    lookahead_samples: int,
    update_at: np.ndarray,
    max_bad: int,
    stepsize: int,
    win_len: int,
    regularization: float,
    store_reconstruction_matrices: bool,
    use_rolling_covariance: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Per-window rASR — cutoff-sensitive variant of ``_process_asr_riemannian``.

    The original ``_process_asr_riemannian`` computes one covariance + one
    reconstruction matrix across the entire stream, which makes the keep mask
    ``theta2 > D`` cutoff-insensitive on contaminated data.

    This function follows Blum 2019's actual recipe: **calibration** is
    Riemannian (``state.M`` / ``state.T`` come from
    ``_riemannian_nonlinear_eigenspace`` on the clean reference covariance),
    but per-window **processing** uses the standard SPD eigendecomposition
    of the rolling covariance. Using ``_riemannian_nonlinear_eigenspace`` for
    per-window processing puts ``D`` on the tangent-space scale, which is
    incommensurate with the ``theta2`` threshold and produces a
    cutoff-invariant keep mask. Standard ``np.linalg.eigh`` on the rolling
    covariance restores cutoff sensitivity while keeping the Riemannian
    calibration's robustness benefits in ``state``.
    """
    n_channels = data_stream.shape[0]

    if use_rolling_covariance:
        covariance_iter = _iter_moving_covariances_at(X_stats, update_at, win_len)
        Xcov_flat = None
    else:
        outer = np.einsum("it,jt->ijt", X_stats, X_stats, optimize=True)
        Xcov_flat = outer.reshape(n_channels * n_channels, n_stream_input, order="F")
        Xcov_flat, _ = _moving_average_clean_rawdata(win_len, Xcov_flat)
        covariance_iter = None

    sample_mask = np.zeros(n_times, dtype=bool)
    n_reconstructed: list[int] = []
    component_variances: list[np.ndarray] = []
    component_thresholds: list[np.ndarray] = []
    reconstruction_matrices: list[np.ndarray] = []
    window_starts: list[int] = []
    window_stops: list[int] = []

    eye = np.eye(n_channels)
    last_R = eye
    last_trivial = True
    last_n = 0
    for n in update_at:
        if covariance_iter is None:
            assert Xcov_flat is not None
            Cw = Xcov_flat[:, n - 1].reshape(n_channels, n_channels, order="F")
        else:
            Cw = next(covariance_iter)
        Cw = (Cw + Cw.T) / 2.0
        # Standard SPD eigendecomposition (matches _process_asr_standard).
        # See the docstring for why we do NOT use _riemannian_nonlinear_eigenspace
        # at processing time.
        D, V = np.linalg.eigh(Cw)
        order = np.argsort(D)
        D = D[order]
        V = V[:, order]

        theta2 = np.sum((state.T @ V) ** 2, axis=0)
        keep = (theta2 > D) | (np.arange(1, n_channels + 1) < (n_channels - max_bad))
        trivial = bool(np.all(keep))

        n_bad = int(n_channels - np.count_nonzero(keep))
        if trivial:
            R = eye
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

    X_clean = data_stream[:, lookahead_samples : lookahead_samples + n_times].copy()
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
        "covariance_geometry": "riemannian_windowed",
        "riemannian_solver": "nonlinear_eigenspace",
    }
    if store_reconstruction_matrices:
        diagnostics["reconstruction_matrices"] = np.asarray(reconstruction_matrices)
    return X_clean, diagnostics
