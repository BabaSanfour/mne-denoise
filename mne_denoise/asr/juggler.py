"""Juggler-style ASR calibration variants.

This module implements the calibration-selection stage described in Kim et al.
(2025) for extreme-motion EEG. The downstream burst reconstruction path
remains the standard ASR implementation already present in ``mne_denoise``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import optimize, stats
from sklearn.cluster import DBSCAN

from ..utils import extract_data_from_mne
from .core import (
    ASR,
    _apply_statistics_filter,
    _design_statistics_filter,
    _good_raw_sample_mask,
    _validate_array_2d,
    calibrate_asr,
)

try:
    from mne.epochs import BaseEpochs
    from mne.io import BaseRaw
except ImportError:  # pragma: no cover - MNE is a required project dependency
    BaseEpochs = Any
    BaseRaw = Any


def select_juggler_reference_samples(
    X: np.ndarray,
    sfreq: float,
    *,
    strategy: str = "dbscan",
    selection_filter_kind: str = "asr",
    dbscan_top_k: int = 5,
    dbscan_eps: float | str = "auto",
    dbscan_min_samples: int | float | str = "auto",
    gev_grid_size: int = 2048,
    min_reference_fraction: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Select calibration samples using Juggler's ASR rules.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        Continuous candidate calibration data.
    sfreq : float
        Sampling frequency in Hz.
    strategy : {'dbscan', 'gev'}
        Juggler reference-selection strategy.
    selection_filter_kind : {'asr', 'highpass', 'none'}
        Statistics-only filter applied before amplitude ranking. The paper
        uses the ASR pre-emphasis filter, so ``'asr'`` is the default.
    dbscan_top_k : int
        Number of largest per-sample channel amplitudes to keep as the DBSCAN
        feature vector. The paper uses five channels.
    dbscan_eps : float | {'auto', 'paper'}
        DBSCAN neighborhood radius. ``'auto'`` and ``'paper'`` use one tenth
        of the modal maximum amplitude, matching the paper description.
    dbscan_min_samples : int | float | {'auto', 'paper'}
        DBSCAN core-neighborhood count. ``'auto'`` and ``'paper'`` use ten
        percent of the mode-derived clean-sample count.
    gev_grid_size : int
        Number of grid points used when locating the fitted GEV mode.
    min_reference_fraction : float
        Minimum acceptable retained fraction. Smaller retained sets are treated
        as calibration failures.

    Returns
    -------
    X_ref : ndarray, shape (n_channels, n_selected_times)
        Selected reference samples from the original input data.
    sample_mask : ndarray, shape (n_times,)
        Boolean mask of the retained reference samples.
    diagnostics : dict
        Selection diagnostics including fitted modes, DBSCAN labels, and the
        retained fraction.
    """
    X = _validate_array_2d(X)
    if strategy not in ("dbscan", "gev"):
        raise ValueError("strategy must be 'dbscan' or 'gev'")
    if dbscan_top_k < 1:
        raise ValueError("dbscan_top_k must be at least 1")
    if gev_grid_size < 32:
        raise ValueError("gev_grid_size must be at least 32")
    if not (0.0 < min_reference_fraction < 1.0):
        raise ValueError("min_reference_fraction must be in (0, 1)")

    filter_b, filter_a = _design_statistics_filter(sfreq, selection_filter_kind)
    X_stats = _apply_statistics_filter(X, filter_b, filter_a)
    X_stats = X_stats - np.median(X_stats, axis=1, keepdims=True)

    amplitude = np.abs(X_stats)
    sorted_amplitude = np.sort(amplitude, axis=0)[::-1]
    top_k = min(int(dbscan_top_k), X.shape[0])
    features = sorted_amplitude[:top_k].T
    leading_amplitude = features[:, 0]

    diagnostics: dict[str, Any] = {
        "reference_selection_strategy": strategy,
        "selection_filter_kind": selection_filter_kind,
        "selection_filter_b": filter_b.copy(),
        "selection_filter_a": filter_a.copy(),
        "leading_amplitude": leading_amplitude.copy(),
        "dbscan_top_k": int(top_k),
    }

    if strategy == "dbscan":
        sample_mask, dbscan_info = _select_dbscan_reference_mask(
            features,
            leading_amplitude,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
        )
        diagnostics.update(dbscan_info)
    else:
        sample_mask, gev_info = _select_gev_reference_mask(
            leading_amplitude,
            grid_size=gev_grid_size,
        )
        diagnostics.update(gev_info)

    keep_fraction = float(np.mean(sample_mask))
    if keep_fraction < min_reference_fraction:
        raise RuntimeError(
            "Juggler reference selection retained too little data: "
            f"{keep_fraction * 100:.1f}% < {min_reference_fraction * 100:.1f}%."
        )

    X_ref = X[:, sample_mask]
    diagnostics.update(
        {
            "reference_sample_mask": sample_mask.copy(),
            "reference_selected_samples": int(X_ref.shape[1]),
            "reference_candidate_samples": int(X.shape[1]),
            "reference_selected_fraction": keep_fraction,
        }
    )
    return X_ref, sample_mask, diagnostics


class JugglerASR(ASR):
    """Juggler's ASR calibration selector on top of standard ASR.

    The burst-repair stage is identical to :class:`mne_denoise.asr.ASR`.
    Only the reference-data selection stage is replaced with the pointwise
    amplitude procedures from Kim et al. (2025):

    - ``strategy='dbscan'``: ASRDBSCAN
    - ``strategy='gev'``: ASRGEV
    """

    def __init__(
        self,
        sfreq: float | None = None,
        *,
        cutoff: float = 20.0,
        strategy: str = "dbscan",
        window_length: float = 0.5,
        window_overlap: float = 0.66,
        max_dropout_fraction: float = 0.1,
        min_clean_fraction: float = 0.25,
        picks: str | list[str] | list[int] | None = "eeg",
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
        filter_kind: str = "asr",
        window_criterion: float | int | str | None = None,
        window_criterion_tolerances: tuple[float, float] = (-np.inf, 7.0),
        lookahead: float | None = None,
        stepsize: int | None = None,
        max_mem_mb: int | None = 512,
        copy: bool = True,
        store_reconstruction_matrices: bool = False,
        selection_filter_kind: str = "asr",
        dbscan_top_k: int = 5,
        dbscan_eps: float | str = "auto",
        dbscan_min_samples: int | float | str = "auto",
        gev_grid_size: int = 2048,
        min_reference_fraction: float = 0.05,
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
            cov_estimator=cov_estimator,
            regularization=regularization,
            filter_kind=filter_kind,
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
        self.strategy = strategy
        self.selection_filter_kind = selection_filter_kind
        self.dbscan_top_k = dbscan_top_k
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.gev_grid_size = gev_grid_size
        self.min_reference_fraction = min_reference_fraction

    def fit(
        self,
        X: BaseRaw | BaseEpochs | np.ndarray,
        y=None,
        *,
        calibration: BaseRaw | BaseEpochs | np.ndarray | None = None,
        calibration_mask: np.ndarray | None = None,
    ) -> JugglerASR:
        """Fit Juggler's ASR from a contaminated or clean calibration stream."""
        del y
        self._validate_estimator_params()
        self._validate_juggler_params()

        fit_input = X if calibration is None else calibration
        data, sfreq, mne_type, orig_inst = extract_data_from_mne(fit_input)
        if mne_type == "evoked":
            raise ValueError(
                "JugglerASR.fit() does not support Evoked calibration data"
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
        reference_data, reference_mask, reference_info = (
            select_juggler_reference_samples(
                data_2d,
                sfreq,
                strategy=self.strategy,
                selection_filter_kind=self.selection_filter_kind,
                dbscan_top_k=self.dbscan_top_k,
                dbscan_eps=self.dbscan_eps,
                dbscan_min_samples=self.dbscan_min_samples,
                gev_grid_size=self.gev_grid_size,
                min_reference_fraction=self.min_reference_fraction,
            )
        )
        state, cal_info = calibrate_asr(
            reference_data,
            sfreq,
            cutoff=self.cutoff,
            window_length=self.window_length,
            window_overlap=self.window_overlap,
            calibration="manual",
            calibration_window_length=self.calibration_window_length,
            calibration_window_overlap=self.calibration_window_overlap,
            ref_max_bad_channels=self.ref_max_bad_channels,
            ref_tolerances=self.ref_tolerances,
            blocksize=self.blocksize,
            max_dropout_fraction=self.max_dropout_fraction,
            min_clean_fraction=self.min_clean_fraction,
            cov_estimator=self.cov_estimator,
            regularization=self.regularization,
            filter_kind=self.filter_kind,
            method="standard",
            max_mem_mb=self.max_mem_mb,
        )
        cal_info.update(reference_info)
        cal_info["clean_window_mask"] = np.array([], dtype=bool)
        cal_info["clean_window_scores"] = np.empty((0, len(picks)), dtype=np.float64)
        cal_info["n_clean_windows"] = 0
        cal_info["n_calibration_windows"] = 0
        cal_info["reference_selection_strategy"] = self.strategy
        cal_info["reference_mask_kind"] = "sample"

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
        self.reference_sample_mask_ = reference_mask
        self.clean_window_mask_ = np.array([], dtype=bool)
        self.clean_window_scores_ = np.empty((0, len(picks)), dtype=np.float64)
        # JugglerASR selects calibration data sample-by-sample, not by windows.
        self.calibration_mask_kind_ = "sample"
        self.calibration_info_ = cal_info
        self.history_ = {
            "method": "juggler",
            "strategy": self.strategy,
            "source_type": mne_type,
            "n_channels": self.n_channels_,
            "sfreq": self.sfreq_,
        }
        return self

    def get_calibration_mask(self) -> np.ndarray:
        """Return the sample-wise reference mask chosen during calibration.

        JugglerASR selects calibration data point-by-point (Kim et al. 2025),
        so the mask is **sample-based** (``calibration_mask_kind_ == "sample"``),
        unlike the window-based mask of the other backends.

        Returns
        -------
        mask : ndarray of bool, shape (n_times,)
            ``True`` where the sample was retained as calibration reference.
        """
        self._check_is_fitted()
        return np.asarray(self.reference_sample_mask_, dtype=bool).copy()

    def _validate_juggler_params(self) -> None:
        if self.strategy not in ("dbscan", "gev"):
            raise ValueError("strategy must be 'dbscan' or 'gev'")
        if self.dbscan_top_k < 1:
            raise ValueError("dbscan_top_k must be at least 1")
        if self.gev_grid_size < 32:
            raise ValueError("gev_grid_size must be at least 32")
        if not (0.0 < self.min_reference_fraction < 1.0):
            raise ValueError("min_reference_fraction must be in (0, 1)")


def _select_dbscan_reference_mask(
    features: np.ndarray,
    leading_amplitude: np.ndarray,
    *,
    dbscan_eps: float | str,
    dbscan_min_samples: int | float | str,
) -> tuple[np.ndarray, dict[str, Any]]:
    del leading_amplitude
    feature_scale = np.linalg.norm(features, axis=1)
    mode = _histogram_mode(feature_scale)
    estimated_clean_count = int(np.sum(feature_scale <= mode))
    eps = _resolve_dbscan_eps(dbscan_eps, mode, feature_scale)
    min_samples = _resolve_dbscan_min_samples(
        dbscan_min_samples,
        estimated_clean_count,
        features.shape[0],
    )
    clusterer = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="chebyshev",
        n_jobs=None,
    )
    labels = clusterer.fit_predict(features)
    candidate_labels = np.unique(labels[labels >= 0])
    if candidate_labels.size == 0:
        raise RuntimeError(
            "DBSCAN found no non-noise cluster. Increase eps or provide a "
            "longer calibration stream."
        )

    cluster_scores = []
    cluster_sizes = []
    for label in candidate_labels:
        label_points = features[labels == label]
        cluster_scores.append(
            float(np.median(np.linalg.norm(label_points, ord=np.inf, axis=1)))
        )
        cluster_sizes.append(int(label_points.shape[0]))

    score_order = np.lexsort(
        (-np.asarray(cluster_sizes, dtype=int), np.asarray(cluster_scores, dtype=float))
    )
    selected_label = int(candidate_labels[score_order[0]])
    sample_mask = labels == selected_label

    diagnostics = {
        "juggler_dbscan_mode": float(mode),
        "juggler_dbscan_scale": "l2_norm",
        "juggler_dbscan_eps": float(eps),
        "juggler_dbscan_min_samples": int(min_samples),
        "juggler_dbscan_labels": labels.copy(),
        "juggler_dbscan_selected_label": selected_label,
        "juggler_dbscan_cluster_sizes": np.asarray(cluster_sizes, dtype=int),
        "juggler_dbscan_cluster_scores": np.asarray(cluster_scores, dtype=np.float64),
        "juggler_dbscan_estimated_clean_count": int(estimated_clean_count),
    }
    return sample_mask, diagnostics


def _select_gev_reference_mask(
    leading_amplitude: np.ndarray,
    *,
    grid_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        shape, loc, scale = stats.genextreme.fit(leading_amplitude)
    except Exception as exc:  # pragma: no cover - SciPy fit failures are rare
        raise RuntimeError(f"GEV fitting failed: {exc}") from exc
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        raise RuntimeError("GEV fitting returned a non-positive scale")

    distribution = stats.genextreme(shape, loc=loc, scale=scale)
    lower = max(float(np.min(leading_amplitude)), float(distribution.ppf(1e-6)))
    upper = min(float(np.max(leading_amplitude)), float(distribution.ppf(1 - 1e-6)))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        mode = _histogram_mode(leading_amplitude)
    else:
        objective = lambda x: -distribution.pdf(x)
        optimum = optimize.minimize_scalar(
            objective, bounds=(lower, upper), method="bounded"
        )
        if optimum.success and np.isfinite(optimum.x):
            mode = float(optimum.x)
        else:
            grid = np.linspace(lower, upper, int(grid_size))
            pdf = distribution.pdf(grid)
            mode = float(grid[int(np.nanargmax(pdf))])
    sample_mask = leading_amplitude <= mode
    diagnostics = {
        "juggler_gev_shape": float(shape),
        "juggler_gev_loc": float(loc),
        "juggler_gev_scale": float(scale),
        "juggler_gev_mode": float(mode),
        "juggler_gev_grid_size": int(grid_size),
    }
    return sample_mask, diagnostics


def _histogram_mode(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot estimate a mode from empty values")
    if np.allclose(values, values[0]):
        return float(values[0])
    edges = np.histogram_bin_edges(values, bins="fd")
    if edges.size < 2:
        return float(np.median(values))
    counts, edges = np.histogram(values, bins=edges)
    idx = int(np.argmax(counts))
    return float(0.5 * (edges[idx] + edges[idx + 1]))


def _resolve_dbscan_eps(
    value: float | str,
    mode: float,
    feature_scale: np.ndarray,
) -> float:
    if isinstance(value, str):
        if value not in ("auto", "paper"):
            raise ValueError("dbscan_eps must be a positive float, 'auto', or 'paper'")
        eps = mode / 10.0
    else:
        eps = float(value)
    if not np.isfinite(eps) or eps <= np.finfo(float).eps:
        positive = feature_scale[feature_scale > 0]
        if positive.size == 0:
            raise RuntimeError(
                "Cannot derive a positive DBSCAN eps from zero-amplitude data"
            )
        eps = max(float(np.median(positive)) / 10.0, np.finfo(float).eps)
    return float(eps)


def _resolve_dbscan_min_samples(
    value: int | float | str,
    estimated_clean_count: int,
    n_times: int,
) -> int:
    if isinstance(value, str):
        if value not in ("auto", "paper"):
            raise ValueError(
                "dbscan_min_samples must be an int, a float fraction, 'auto', or 'paper'"
            )
        min_samples = int(np.ceil(0.10 * max(estimated_clean_count, 1)))
    elif isinstance(value, float) and 0.0 < value <= 1.0:
        min_samples = int(np.ceil(float(value) * max(estimated_clean_count, 1)))
    else:
        min_samples = int(value)
    min_samples = max(2, min_samples)
    min_samples = min(min_samples, n_times)
    return int(min_samples)
