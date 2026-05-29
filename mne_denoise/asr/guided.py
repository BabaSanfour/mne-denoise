"""Guided ASR: DSS-biased soft Artifact Subspace Reconstruction (experimental).

Standard ASR detects *when/where* an EEG subspace is statistically abnormal
(its variance exceeds a clean-calibration threshold) and removes it with a
**binary** keep/reject decision. Its documented weakness, acute in mobile/MoBI
EEG, is over-cleaning: because the decision is variance-only, it can reconstruct
away real high-variance neural activity (task ERPs, SSVEP, gait-locked rhythms).

``GuidedASR`` keeps ASR's abnormality detection but adds two things:

1. **Bias operators** (reused from the DSS machinery) score *what kind* each
   flagged component direction is -- artifact-like vs brain-like -- via the
   quadratic form of the direction against bank covariances ``C_artifact`` and
   ``C_preserve``.
2. **Soft reconstruction** replaces the binary keep/reject with a per-component
   Wiener-style weight ``w in [0, 1]`` (1 = keep, 0 = suppress, intermediate =
   partial attenuation).

The soft weight rescues components ASR would wrongly remove when they are
brain-like, while leaving artifact-like abnormal components suppressed. The
estimator is built on the ``method="riemannian_windowed"`` backbone, so with
``reconstruction="hard"`` and no bias operators it is mathematically identical
to :class:`mne_denoise.asr.ASR` with ``method="riemannian_windowed"``.

This is an **experimental proof-of-concept** and must be opted into with
``experimental=True``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..dss.utils.covariance import compute_covariance
from ..utils import extract_data_from_mne, reconstruct_mne_object
from .core import (
    ASR,
    ASRState,
    _append_clean_rawdata_tail,
    _apply_statistics_filter_streaming,
    _covariance_stack_bytes,
    _empty_process_diagnostics,
    _good_raw_sample_mask,
    _iter_moving_covariances_at,
    _max_mem_bytes,
    _moving_average_clean_rawdata,
    _prepend_clean_rawdata_carry,
    _process_memory_info,
    _resolve_max_dims_clean_rawdata,
    _round_half_up,
    _validate_array_2d,
    _validate_common_params,
)

_EPS = float(np.finfo(np.float64).eps)


# ---------------------------------------------------------------------------
# Bias-operator covariance bank
# ---------------------------------------------------------------------------


def _normalize_cov(cov: np.ndarray) -> np.ndarray:
    """Trace-normalize an SPD covariance to unit mean eigenvalue.

    Normalizing removes the amplitude scale so that *structure* (not raw
    power) drives the component-vs-bias scoring.
    """
    cov = (np.asarray(cov, dtype=np.float64) + cov.T) / 2.0
    trace = float(np.trace(cov))
    if trace > _EPS:
        cov = cov * (cov.shape[0] / trace)
    return cov


def _bias_bank_covariance(
    data_2d: np.ndarray,
    biases: list | tuple | None,
) -> np.ndarray | None:
    """Aggregate a normalized bias covariance from a list of bias operators.

    Each bias is a DSS ``LinearDenoiser`` (or any callable) that maps
    ``(n_channels, n_times)`` data to an equally shaped, structure-emphasized
    copy. The covariance of that biased data is the DSS "biased" covariance
    ``C1``; we trace-normalize and sum across the supplied operators.
    """
    if not biases:
        return None
    n_channels = data_2d.shape[0]
    accum = np.zeros((n_channels, n_channels), dtype=np.float64)
    for bias in biases:
        biased = bias.apply(data_2d) if hasattr(bias, "apply") else bias(data_2d)
        biased = np.asarray(biased, dtype=np.float64)
        if biased.ndim == 3:  # (n_channels, n_times, n_epochs) -> flatten time
            biased = biased.reshape(biased.shape[0], -1)
        accum += _normalize_cov(compute_covariance(biased))
    return _normalize_cov(accum)


def _soft_component_weights(
    D: np.ndarray,
    V: np.ndarray,
    theta2: np.ndarray,
    *,
    forced_keep: np.ndarray,
    artifact_cov: np.ndarray | None,
    preserve_cov: np.ndarray | None,
    soft_weight: str,
    scale: float,
) -> np.ndarray:
    """Per-component soft keep weight ``w in [0, 1]`` for one window.

    ``w_asr`` is a soft version of ASR's ``keep = theta2 > D`` rule: it equals
    1 exactly when ASR would keep (``D <= theta2``) and decays toward 0 as the
    component variance ``D`` exceeds its threshold ``theta2``.

    The bias bank produces a structure vote ``s in [0, 1]`` per component
    (``s>0.5`` brain-like, ``s<0.5`` artifact-like, ``s=0.5`` neutral / aligned
    with neither bias). It then *lifts* the keep weight toward 1 for brain-like
    directions (rescuing high-variance neural activity ASR would wrongly
    remove) and *pushes* it toward 0 for artifact-like directions (including
    ones ASR would have kept). Neutral or no-bias components fall back to
    ``w_asr`` unchanged.
    """
    excess = np.maximum(D - theta2, 0.0)
    w_asr = theta2 / (theta2 + excess + _EPS)

    if artifact_cov is None and preserve_cov is None:
        w = w_asr
    else:
        brain = (
            np.maximum(np.sum(V * (preserve_cov @ V), axis=0), 0.0)
            if preserve_cov is not None
            else np.zeros(V.shape[1])
        )
        artifact = (
            np.maximum(np.sum(V * (artifact_cov @ V), axis=0), 0.0)
            if artifact_cov is not None
            else np.zeros(V.shape[1])
        )
        denom = brain + artifact
        s = np.where(denom > _EPS, brain / (denom + _EPS), 0.5)
        if soft_weight == "sigmoid":
            # Sharpen the structure vote toward a more decisive 0/1.
            s = 1.0 / (1.0 + np.exp(-6.0 * (s - 0.5)))
        gamma = max(float(scale), _EPS)
        lift = (1.0 - w_asr) * np.maximum(2.0 * s - 1.0, 0.0) ** gamma
        push = w_asr * np.maximum(1.0 - 2.0 * s, 0.0) ** gamma
        w = w_asr + lift - push

    w = np.where(forced_keep, 1.0, w)
    return np.clip(w, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Per-window guided processing (soft variant of _process_asr_riemannian_windowed)
# ---------------------------------------------------------------------------


def _process_guided_asr_windowed(
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
    artifact_cov: np.ndarray | None,
    preserve_cov: np.ndarray | None,
    reconstruction: str,
    soft_weight: str,
    soft_weight_scale: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Guided per-window processing.

    Identical to :func:`mne_denoise.asr.core._process_asr_riemannian_windowed`
    except the binary ``keep`` mask is replaced by a continuous per-component
    weight ``w`` (soft reconstruction). With ``reconstruction="hard"`` and no
    bias covariances, ``w`` collapses to the binary ``keep`` mask and the
    output is byte-for-byte identical to the standard windowed backend.
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
    soft_weights: list[np.ndarray] = []
    reconstruction_matrices: list[np.ndarray] = []
    window_starts: list[int] = []
    window_stops: list[int] = []

    hard = reconstruction == "hard"
    index = np.arange(1, n_channels + 1)
    eye = np.eye(n_channels)
    last_R = eye
    last_trivial = True
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
        forced_keep = index < (n_channels - max_bad)
        if hard:
            # Exact standard-ASR path: binary keep -> rank-deficient basis whose
            # pinv drops rejected rows cleanly. Kept byte-identical so that
            # GuidedASR(reconstruction="hard") == ASR(method="riemannian_windowed").
            keep = (theta2 > D) | forced_keep
            w = keep.astype(np.float64)
            trivial = bool(np.all(keep))
            n_bad = int(n_channels - np.count_nonzero(keep))
            if trivial:
                R = eye
            else:
                basis = keep[:, np.newaxis].astype(np.float64) * (V.T @ state.M)
                R = state.M @ np.linalg.pinv(basis) @ V.T
                R = np.real_if_close(R).astype(np.float64)
        else:
            w = _soft_component_weights(
                D,
                V,
                theta2,
                forced_keep=forced_keep,
                artifact_cov=artifact_cov,
                preserve_cov=preserve_cov,
                soft_weight=soft_weight,
                scale=soft_weight_scale,
            )
            trivial = bool(np.all(w >= 1.0 - 1e-12))
            n_bad = int(np.count_nonzero(w < 0.5))
            if trivial:
                R = eye
            else:
                # Numerically stable soft reconstruction. Weighting the basis
                # rows by a small-but-nonzero w and taking a pinv blows up like
                # 1/w; instead blend, in the window eigenbasis, between keeping
                # each component (weight w) and the *binary* hard
                # reconstruction R_hard (weight 1 - w). With w in {0, 1} this is
                # exactly R_hard. R_hard uses a thresholded keep mask, so its
                # pinv is the well-conditioned rank-deficient solve.
                keep_hard = (w >= 0.5) | forced_keep
                if np.all(keep_hard):
                    R_hard = eye
                else:
                    basis = keep_hard[:, np.newaxis].astype(np.float64) * (
                        V.T @ state.M
                    )
                    R_hard = state.M @ np.linalg.pinv(basis) @ V.T
                R = (V * w) @ V.T + (V * (1.0 - w)) @ V.T @ R_hard
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
            soft_weights.append(w.copy())
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
        "soft_weights": np.asarray(soft_weights, dtype=np.float64),
        "n_windows": int(len(n_reconstructed_arr)),
        "fraction_reconstructed_windows": float(
            np.mean(n_reconstructed_arr > 0) if n_reconstructed_arr.size else 0.0
        ),
        "fraction_reconstructed_samples": float(np.mean(sample_mask)),
        "max_components_reconstructed": int(n_reconstructed_arr.max(initial=0)),
        "mean_soft_weight": (
            float(np.mean(np.asarray(soft_weights))) if soft_weights else 1.0
        ),
        "lookahead_samples": int(lookahead_samples),
        "stepsize_samples": int(stepsize),
        "window_length_samples": int(win_len),
        "covariance_geometry": "guided",
        "reconstruction": reconstruction,
    }
    if store_reconstruction_matrices:
        diagnostics["reconstruction_matrices"] = np.asarray(reconstruction_matrices)
    return X_clean, diagnostics


def process_guided_asr(
    X: np.ndarray,
    sfreq: float,
    state: ASRState,
    *,
    artifact_cov: np.ndarray | None = None,
    preserve_cov: np.ndarray | None = None,
    reconstruction: str = "soft",
    soft_weight: str = "wiener",
    soft_weight_scale: float = 1.0,
    window_length: float = 0.5,
    window_overlap: float = 0.66,
    max_dims: float | int = 0.66,
    regularization: float = 1e-8,
    store_reconstruction_matrices: bool = False,
    max_mem_mb: int | None = 512,
    lookahead: float | None = None,
    stepsize: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a calibrated ASR state with guided soft reconstruction.

    Mirrors :func:`mne_denoise.asr.core.process_asr` (the ``riemannian_windowed``
    streaming setup) but routes the per-window step through
    :func:`_process_guided_asr_windowed`.
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
    if reconstruction not in ("soft", "hard"):
        raise ValueError("reconstruction must be 'soft' or 'hard'")

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

    X_clean, diagnostics = _process_guided_asr_windowed(
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
        artifact_cov=artifact_cov,
        preserve_cov=preserve_cov,
        reconstruction=reconstruction,
        soft_weight=soft_weight,
        soft_weight_scale=soft_weight_scale,
    )
    diagnostics.update(
        _process_memory_info(
            n_channels=n_channels,
            n_stream_input=n_stream_input,
            max_mem_mb=max_mem_mb,
            memory_mode=("guided_rolling" if use_rolling_covariance else "guided"),
            peak_cov_buffer_bytes=_covariance_stack_bytes(1, n_channels),
            chunk_samples=win_len if use_rolling_covariance else n_stream_input,
            used_memory_bound=use_rolling_covariance,
        )
    )
    return X_clean, diagnostics


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class GuidedASR(ASR):
    """DSS-biased soft Artifact Subspace Reconstruction (experimental).

    Extends :class:`mne_denoise.asr.ASR` (``method="riemannian_windowed"``
    backbone) with soft, structure-aware reconstruction. See the module
    docstring for the algorithm.

    Parameters
    ----------
    artifact_biases : sequence of DSS bias operators, optional
        Operators (e.g. :class:`mne_denoise.dss.denoisers.LineNoiseBias`,
        ``BandpassBias``) whose biased covariance defines the artifact-like
        subspace ``C_artifact``. Each must accept ``(n_channels, n_times)`` and
        return the same shape (the ``LinearDenoiser`` ``.apply`` contract).
    preserve_biases : sequence of DSS bias operators, optional
        Operators defining the brain-like subspace ``C_preserve`` to protect
        (e.g. ``PeakFilterBias`` for SSVEP, ``BandpassBias`` for a target band).
    reconstruction : {'soft', 'hard'}
        ``'soft'`` (default) uses per-component Wiener weights; ``'hard'``
        reproduces standard ASR's binary keep/reject.
    soft_weight : {'wiener', 'sigmoid'}
        Soft-weight combination rule.
    soft_weight_scale : float
        Sharpness of the soft weighting.
    experimental : bool
        Must be ``True`` to use the guided soft reconstruction.

    All other parameters are forwarded to :class:`mne_denoise.asr.ASR`.

    Notes
    -----
    With ``reconstruction="hard"`` and no bias operators, ``GuidedASR`` is
    identical to ``ASR(method="riemannian_windowed")``.
    """

    def __init__(
        self,
        sfreq: float | None = None,
        *,
        cutoff: float = 20.0,
        window_length: float = 0.5,
        window_overlap: float = 0.66,
        max_dropout_fraction: float = 0.1,
        min_clean_fraction: float = 0.25,
        picks: str | list[str] | list[int] | None = "eeg",
        calibration: str = "auto",
        calibration_window_length: float = 1.0,
        calibration_window_overlap: float = 0.66,
        ref_max_bad_channels: float = 0.075,
        ref_tolerances: tuple[float, float] = (-np.inf, 5.5),
        blocksize: int = 10,
        max_dims: float | int = 0.66,
        reject_by_annotation: bool = True,
        skip_by_annotation: tuple[str, ...] = ("bad", "bad_acq_skip"),
        cov_estimator: str = "geometric_median",
        regularization: float = 1e-8,
        filter_kind: str = "none",
        lookahead: float | None = None,
        stepsize: int | None = None,
        max_mem_mb: int | None = 512,
        copy: bool = True,
        store_reconstruction_matrices: bool = False,
        artifact_biases: list | tuple | None = None,
        preserve_biases: list | tuple | None = None,
        reconstruction: str = "soft",
        soft_weight: str = "wiener",
        soft_weight_scale: float = 1.0,
        experimental: bool = False,
        random_state: int | None = None,
        n_jobs: int | None = None,
        verbose: bool | str | int | None = None,
    ) -> None:
        super().__init__(
            sfreq=sfreq,
            cutoff=cutoff,
            window_length=window_length,
            window_overlap=window_overlap,
            max_dropout_fraction=max_dropout_fraction,
            min_clean_fraction=min_clean_fraction,
            method="riemannian_windowed",
            experimental=experimental,
            calibration=calibration,
            picks=picks,
            calibration_window_length=calibration_window_length,
            calibration_window_overlap=calibration_window_overlap,
            ref_max_bad_channels=ref_max_bad_channels,
            ref_tolerances=ref_tolerances,
            blocksize=blocksize,
            max_dims=max_dims,
            reject_by_annotation=reject_by_annotation,
            skip_by_annotation=skip_by_annotation,
            cov_estimator=cov_estimator,
            regularization=regularization,
            filter_kind=filter_kind,
            lookahead=lookahead,
            stepsize=stepsize,
            max_mem_mb=max_mem_mb,
            copy=copy,
            store_reconstruction_matrices=store_reconstruction_matrices,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.artifact_biases = artifact_biases
        self.preserve_biases = preserve_biases
        self.reconstruction = reconstruction
        self.soft_weight = soft_weight
        self.soft_weight_scale = soft_weight_scale

    # -- fit ---------------------------------------------------------------

    def fit(
        self,
        X,
        y=None,
        *,
        calibration=None,
        calibration_mask=None,
    ) -> GuidedASR:
        """Calibrate ASR, then build the artifact/preserve bias covariances."""
        if self.reconstruction not in ("soft", "hard"):
            raise ValueError("reconstruction must be 'soft' or 'hard'")
        if self.soft_weight not in ("wiener", "sigmoid"):
            raise ValueError("soft_weight must be 'wiener' or 'sigmoid'")
        if self.reconstruction == "soft" and not self.experimental:
            raise ValueError(
                "GuidedASR soft reconstruction is experimental; pass "
                "experimental=True to use it (reconstruction='hard' reproduces "
                "standard ASR and needs no opt-in)."
            )

        super().fit(X, y=y, calibration=calibration, calibration_mask=calibration_mask)

        # Bias operators define artifact / brain *subspaces*, so they are
        # estimated from the primary recording ``X`` (which contains those
        # phenomena), whereas the ASR threshold model above uses ``calibration``
        # when provided.
        data, _, mne_type, _ = extract_data_from_mne(X)
        data_2d = self._select_fit_data(data, mne_type, self.picks_)
        self.artifact_cov_ = _bias_bank_covariance(data_2d, self.artifact_biases)
        self.preserve_cov_ = _bias_bank_covariance(data_2d, self.preserve_biases)
        return self

    # -- transform ---------------------------------------------------------

    def _process(self, selected: np.ndarray, sfreq: float):
        return process_guided_asr(
            selected,
            sfreq,
            self.state_,
            artifact_cov=getattr(self, "artifact_cov_", None),
            preserve_cov=getattr(self, "preserve_cov_", None),
            reconstruction=self.reconstruction,
            soft_weight=self.soft_weight,
            soft_weight_scale=self.soft_weight_scale,
            window_length=self.window_length,
            window_overlap=self.window_overlap,
            max_dims=self.max_dims,
            regularization=self.regularization,
            store_reconstruction_matrices=self.store_reconstruction_matrices,
            max_mem_mb=self.max_mem_mb,
            lookahead=self.lookahead,
            stepsize=self.stepsize,
        )

    def transform(
        self,
        X,
        y=None,
        *,
        copy: bool | None = None,
        return_diagnostics: bool = False,
    ) -> Any:
        """Clean data with the fitted guided soft-reconstruction state."""
        del y, copy
        self._check_is_fitted()
        data, sfreq, mne_type, orig_inst = extract_data_from_mne(X)
        sfreq = self._resolve_sfreq(sfreq, fitted=True)
        if not np.isclose(sfreq, self.sfreq_):
            raise ValueError(
                f"Input sfreq {sfreq} does not match fitted sfreq {self.sfreq_}"
            )
        picks, ch_names = self._resolve_picks(X, data, mne_type)
        self._check_transform_channels(picks, ch_names)
        self._warn_preprocessing_state(orig_inst, mne_type)

        if mne_type == "epochs":
            cleaned_data, diagnostics = self._transform_epochs_guided(
                data, picks, sfreq
            )
        else:
            data_out = np.asarray(data, dtype=np.float64).copy()
            selected = data_out[picks, :]
            selected_clean, diagnostics = self._process(selected, sfreq)
            if mne_type == "raw" and self.reject_by_annotation:
                good_mask = _good_raw_sample_mask(orig_inst, self.skip_by_annotation)
                selected_clean[:, ~good_mask] = selected[:, ~good_mask]
                diagnostics["sample_mask"] = diagnostics["sample_mask"] & good_mask
            data_out[picks, :] = selected_clean
            cleaned_data = data_out

        self._store_transform_diagnostics(diagnostics)
        cleaned = reconstruct_mne_object(
            cleaned_data, orig_inst, mne_type, verbose=False
        )
        if return_diagnostics:
            return cleaned, diagnostics
        return cleaned

    def _transform_epochs_guided(
        self,
        data: np.ndarray,
        picks: np.ndarray,
        sfreq: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        cleaned = np.asarray(data, dtype=np.float64).copy()
        starts_all: list[np.ndarray] = []
        stops_all: list[np.ndarray] = []
        sample_masks: list[np.ndarray] = []
        counts: list[np.ndarray] = []
        soft_all: list[np.ndarray] = []
        n_windows = 0
        for epoch_idx in range(cleaned.shape[0]):
            selected = cleaned[epoch_idx, picks, :]
            selected_clean, diag = self._process(selected, sfreq)
            cleaned[epoch_idx, picks, :] = selected_clean
            starts_all.append(diag["window_starts"])
            stops_all.append(diag["window_stops"])
            sample_masks.append(diag["sample_mask"])
            counts.append(diag["n_components_reconstructed"])
            if diag.get("soft_weights", np.empty((0,))).size:
                soft_all.append(diag["soft_weights"])
            n_windows += int(diag["n_windows"])

        diagnostics = {
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
            "soft_weights": np.concatenate(soft_all, axis=0)
            if soft_all
            else np.empty((0, 0), dtype=np.float64),
            "n_windows": int(n_windows),
            "covariance_geometry": "guided",
            "reconstruction": self.reconstruction,
        }
        counts_arr = diagnostics["n_components_reconstructed"]
        diagnostics["fraction_reconstructed_windows"] = (
            float(np.mean(counts_arr > 0)) if counts_arr.size else 0.0
        )
        diagnostics["fraction_reconstructed_samples"] = (
            float(np.mean(diagnostics["sample_mask"]))
            if diagnostics["sample_mask"].size
            else 0.0
        )
        diagnostics["max_components_reconstructed"] = int(counts_arr.max(initial=0))
        return cleaned, diagnostics
