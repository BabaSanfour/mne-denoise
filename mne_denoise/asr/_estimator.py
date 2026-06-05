"""The public ``ASR`` estimator (scikit-learn / MNE interface).

Thin ``BaseEstimator`` / ``TransformerMixin`` wrapper that resolves MNE or
NumPy input, delegates calibration to :mod:`mne_denoise.asr._calibration` and
reconstruction to :mod:`mne_denoise.asr._reconstruction`, and exposes
diagnostics, calibration masks, and annotation export.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ..utils import extract_data_from_mne, reconstruct_mne_object
from ._calibration import calibrate_asr
from ._qa import compute_asr_rejection_mask
from ._reconstruction import process_asr
from ._validation import _validate_common_params
from ._windows import _good_raw_sample_mask, _mask_to_sample_spans, _merge_sample_spans

try:
    import mne
    from mne.epochs import BaseEpochs
    from mne.evoked import Evoked
    from mne.io import BaseRaw
except ImportError:  # pragma: no cover - MNE is a required project dependency
    mne = None


class ASR(BaseEstimator, TransformerMixin):
    """Artifact Subspace Reconstruction transformer.

    Parameters
    ----------
    sfreq : float | None
        Sampling frequency in Hz. Required for NumPy arrays. For MNE objects,
        this may be ``None`` and is inferred from ``info['sfreq']``.
    cutoff : float
        ASR threshold multiplier. Values around 20 are conservative; lower
        values clean more aggressively.
    window_length : float
        Processing/statistics window length in seconds.
    window_overlap : float
        Overlap fraction for processing and threshold-fitting windows.
    max_dropout_fraction : float
        Fraction of lowest RMS values ignored while estimating thresholds.
    min_clean_fraction : float
        Minimum central fraction used to estimate clean RMS statistics.
    method : {'standard', 'riemannian', 'riemannian_windowed'}
        ASR backend.

        - ``'standard'`` — clean_rawdata-faithful Euclidean ASR.
        - ``'riemannian'`` — experimental SPD-manifold covariance backend,
          MATLAB-rASR-faithful. NOTE: this backend computes one covariance +
          one reconstruction matrix for the entire stream, so its cleaned
          output is **cutoff-invariant on real EEG** (the ``cutoff`` knob does
          not meaningfully change the result). Use it for MATLAB parity, not
          for cutoff tuning.
        - ``'riemannian_windowed'`` — per-window Riemannian backend that keeps
          the Riemannian-aggregated (geometric-median) calibration but applies
          a standard per-window eigendecomposition at processing time. Unlike
          ``'riemannian'``, its ``cutoff`` knob works: ``% data modified`` and
          ``% variance reduced`` scale monotonically with ``cutoff`` like
          ``'standard'`` does. This is a **first-class backend** (no
          ``experimental`` flag required): its processing is byte-identical to
          standard ASR and it has a direct MATLAB ``asr_process`` cross-check
          at relerr < 1e-13. Prefer it over ``'riemannian'`` whenever you need
          cutoff control with Riemannian-robust calibration.
    experimental : bool
        Explicit opt-in for the unstable ``method='riemannian'`` research
        backend (cutoff-invariant on real EEG). Not required for
        ``'riemannian_windowed'``.
    calibration : {'auto', 'manual'}
        Calibration mode. ``'auto'`` selects clean windows before fitting;
        ``'manual'`` uses all supplied calibration samples.
    picks : str | list | None
        Channels to clean for MNE objects. Default ``'eeg'`` excludes channels
        in ``info['bads']``. Also accepts ``'mag'``, ``'grad'``, ``'meg'``,
        ``'all'``, a list of channel names, or a list of integer indices (the
        ASR algorithm is unit/scale agnostic, so MEG works the same as EEG).
        For NumPy arrays, ``None`` or a channel-type string uses all rows and a
        list of integers selects rows.
    calibration_window_length : float
        Window length in seconds for automatic clean-window selection.
    calibration_window_overlap : float
        Overlap fraction for automatic clean-window selection.
    ref_max_bad_channels : float
        Maximum fraction of channels exceeding robust tolerances in a clean
        calibration window.
    ref_tolerances : tuple of float
        Lower and upper robust z-score bounds for clean-window selection.
    blocksize : int
        Number of successive samples averaged into each covariance block for
        robust calibration covariance estimation.
    max_dims : float | int
        Maximum number of dimensions reconstructed per processing window.
    reject_by_annotation : bool
        If True, samples under bad annotations are excluded during Raw
        calibration and preserved during Raw transform.
    skip_by_annotation : tuple of str
        Annotation description prefixes treated as bad when
        ``reject_by_annotation=True``.
    cov_estimator : {'geometric_median', 'mean', 'median'}
        Aggregation rule for calibration-window covariance matrices.
    regularization : float
        Relative eigenvalue floor for covariance regularization.
    filter_kind : {'none', 'asr', 'highpass'}
        Statistics-only filter. The cleaned output is reconstructed from the
        original unfiltered data.
    window_criterion : float | int | str | None
        Optional clean_windows-style final rejection criterion. If numeric,
        this is the maximum tolerated number or fraction of bad channels per
        retained window after ASR correction. ``None`` and ``'off'`` disable
        final rejection-mask computation.
    window_criterion_tolerances : tuple of float
        Lower and upper robust z-score thresholds for final clean_windows-style
        retained-sample masking.
    lookahead : float | None
        Processing lookahead in seconds. Defaults to ``window_length / 2``.
    stepsize : int | None
        Number of samples between reconstruction-matrix updates. If ``None``,
        use the clean_rawdata default ``floor(sfreq * window_length / 2)``.
    max_mem_mb : int | None
        Reserved memory limit for future chunking.
    copy : bool
        Reserved API flag. Transform returns a new object/array.
    store_reconstruction_matrices : bool
        Store per-window reconstruction matrices in diagnostics.
    random_state : int | None
        Reserved for future stochastic calibration strategies.
    n_jobs : int | None
        Reserved for future parallel processing.
    verbose : bool | str | int | None
        Verbosity placeholder for API compatibility.

    Attributes
    ----------
    sfreq_ : float
        Sampling frequency used during fitting.
    ch_names_ : list of str | None
        Fitted channel names for MNE inputs.
    picks_ : ndarray
        Row/channel indices cleaned in the fitted data.
    M_ : ndarray
        Calibration covariance square root.
    T_ : ndarray
        Direction-dependent threshold matrix.
    thresholds_ : ndarray
        Per-component RMS thresholds.
    clean_window_mask_ : ndarray
        Calibration windows retained as clean.
    sample_mask_ : ndarray
        Samples reconstructed during the last transform.
    rejection_sample_mask_ : ndarray
        Boolean retained-sample mask from optional clean_windows-style final
        rejection. Present after transforms when ``window_criterion`` is
        enabled.
    n_components_reconstructed_ : ndarray
        Number of reconstructed components per processing window.
    diagnostics_ : dict
        Last-transform diagnostics.
    calibration_info_ : dict
        Calibration diagnostics.
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
        method: str = "standard",
        experimental: bool = False,
        calibration: str = "auto",
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
        filter_kind: str = "none",
        window_criterion: float | int | str | None = None,
        window_criterion_tolerances: tuple[float, float] = (-np.inf, 7.0),
        lookahead: float | None = None,
        stepsize: int | None = None,
        max_mem_mb: int | None = 512,
        copy: bool = True,
        store_reconstruction_matrices: bool = False,
        random_state: int | None = None,
        n_jobs: int | None = None,
        verbose: bool | str | int | None = None,
    ) -> None:
        self.sfreq = sfreq
        self.cutoff = cutoff
        self.window_length = window_length
        self.window_overlap = window_overlap
        self.max_dropout_fraction = max_dropout_fraction
        self.min_clean_fraction = min_clean_fraction
        self.method = method
        self.experimental = experimental
        self.calibration = calibration
        self.picks = picks
        self.calibration_window_length = calibration_window_length
        self.calibration_window_overlap = calibration_window_overlap
        self.ref_max_bad_channels = ref_max_bad_channels
        self.ref_tolerances = ref_tolerances
        self.blocksize = blocksize
        self.max_dims = max_dims
        self.reject_by_annotation = reject_by_annotation
        self.skip_by_annotation = skip_by_annotation
        self.cov_estimator = cov_estimator
        self.regularization = regularization
        self.filter_kind = filter_kind
        self.window_criterion = window_criterion
        self.window_criterion_tolerances = window_criterion_tolerances
        self.lookahead = lookahead
        self.stepsize = stepsize
        self.max_mem_mb = max_mem_mb
        self.copy = copy
        self.store_reconstruction_matrices = store_reconstruction_matrices
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(
        self,
        X: BaseRaw | BaseEpochs | np.ndarray,
        y=None,
        *,
        calibration: BaseRaw | BaseEpochs | np.ndarray | None = None,
        calibration_mask: np.ndarray | None = None,
    ) -> ASR:
        """Fit ASR calibration state.

        Parameters
        ----------
        X : Raw | Epochs | ndarray
            Data used for calibration when ``calibration`` is ``None``.
            NumPy arrays must have shape ``(n_channels, n_times)``.
        y : None
            Ignored.
        calibration : Raw | Epochs | ndarray | None
            Optional separate calibration data with matching channels.
        calibration_mask : ndarray | None
            Optional boolean sample mask for 2D calibration arrays or Raw
            inputs after annotation exclusion.

        Returns
        -------
        self : ASR
            Fitted estimator.
        """
        del y
        self._validate_estimator_params()
        fit_input = X if calibration is None else calibration
        data, sfreq, mne_type, orig_inst, _, _ = extract_data_from_mne(
            fit_input, auto_pick=False
        )
        if mne_type == "evoked":
            raise ValueError("ASR.fit() does not support Evoked calibration data")
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
        state, cal_info = calibrate_asr(
            data_2d,
            sfreq,
            cutoff=self.cutoff,
            window_length=self.window_length,
            window_overlap=self.window_overlap,
            calibration=self.calibration,
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
            method=self.method,
            max_mem_mb=self.max_mem_mb,
        )

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
        self.clean_window_scores_ = cal_info["clean_window_scores"]
        self.calibration_mask_kind_ = "window"
        self.calibration_info_ = cal_info
        self.history_ = {
            "method": self.method,
            "calibration": self.calibration,
            "source_type": mne_type,
            "n_channels": self.n_channels_,
            "sfreq": self.sfreq_,
        }
        return self

    def transform(
        self,
        X: BaseRaw | BaseEpochs | Evoked | np.ndarray,
        y=None,
        *,
        copy: bool | None = None,
        return_diagnostics: bool = False,
    ) -> Any:
        """Apply the fitted ASR model.

        Parameters
        ----------
        X : Raw | Epochs | Evoked | ndarray
            Data to clean.
        y : None
            Ignored.
        copy : bool | None
            Reserved API flag. Transform returns a new object/array.
        return_diagnostics : bool
            If True, return ``(cleaned, diagnostics)``.

        Returns
        -------
        cleaned : Raw | Epochs | Evoked | ndarray
            Cleaned data with the same type/shape as ``X``.
        diagnostics : dict
            Returned only when ``return_diagnostics=True``.
        """
        del y, copy
        self._check_is_fitted()
        data, sfreq, mne_type, orig_inst, _, _ = extract_data_from_mne(
            X, auto_pick=False
        )
        sfreq = self._resolve_sfreq(sfreq, fitted=True)
        if not np.isclose(sfreq, self.sfreq_):
            raise ValueError(
                f"Input sfreq {sfreq} does not match fitted sfreq {self.sfreq_}"
            )
        picks, ch_names = self._resolve_picks(X, data, mne_type)
        self._check_transform_channels(picks, ch_names)
        self._warn_preprocessing_state(orig_inst, mne_type)

        if mne_type == "epochs":
            cleaned_data, diagnostics = self._transform_epochs(data, picks, sfreq)
        else:
            data_out = np.asarray(data, dtype=np.float64).copy()
            selected = data_out[picks, :]
            selected_clean, diagnostics = process_asr(
                selected,
                sfreq,
                self.state_,
                window_length=self.window_length,
                window_overlap=self.window_overlap,
                max_dims=self.max_dims,
                regularization=self.regularization,
                store_reconstruction_matrices=self.store_reconstruction_matrices,
                max_mem_mb=self.max_mem_mb,
                lookahead=self.lookahead,
                stepsize=self.stepsize,
                method=self.method,
            )
            if mne_type == "raw" and self.reject_by_annotation:
                good_mask = _good_raw_sample_mask(orig_inst, self.skip_by_annotation)
                selected_clean[:, ~good_mask] = selected[:, ~good_mask]
                diagnostics["sample_mask"] = diagnostics["sample_mask"] & good_mask
            if self._window_criterion_enabled():
                rejection_mask, rejection_diag = self._compute_window_rejection(
                    selected_clean,
                    sfreq,
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
        """Fit ASR and apply it to ``X``."""
        self.fit(X, y=y, calibration=calibration)
        return self.transform(X, return_diagnostics=return_diagnostics)

    def get_diagnostics(self) -> dict[str, Any]:
        """Return diagnostics from the last transform."""
        self._check_is_fitted()
        if not hasattr(self, "diagnostics_"):
            return {}
        return dict(self.diagnostics_)

    def get_calibration_mask(self) -> np.ndarray:
        """Return the boolean mask of data used for calibration.

        The mask is **window-based** for the standard / Riemannian / adaptive
        backends (one bool per calibration window; see
        :attr:`calibration_mask_kind_` ``== "window"``) and **sample-based**
        for :class:`JugglerASR` (one bool per time sample;
        :attr:`calibration_mask_kind_` ``== "sample"``).

        Returns
        -------
        mask : ndarray of bool
            The calibration clean-window or reference-sample mask.

        See Also
        --------
        get_rejection_mask : retained-sample mask after optional window rejection.
        """
        self._check_is_fitted()
        return np.asarray(self.clean_window_mask_, dtype=bool).copy()

    def get_rejection_mask(self) -> np.ndarray:
        """Return the retained-sample mask from final clean_windows-style rejection.

        Returns
        -------
        mask : ndarray of bool, shape (n_times,)
            ``True`` where samples were kept. Requires ``window_criterion`` to
            have been enabled and ``transform`` to have been run.
        """
        self._check_is_fitted()
        if not hasattr(self, "rejection_sample_mask_"):
            raise RuntimeError(
                "No final rejection mask is available. Enable window_criterion and "
                "run transform first."
            )
        return np.asarray(self.rejection_sample_mask_, dtype=bool).copy()

    def to_annotations(
        self,
        kind: str = "repair",
        *,
        min_components: int = 1,
        description: str | None = None,
    ) -> Any:
        """Convert ASR decisions into MNE annotations.

        One unified entry point for the three annotation kinds. ``"repair"`` and
        ``"rejection"`` are available on every backend that has run
        ``transform``; ``"calibration"`` is available only for sample-based
        reference selection (:class:`JugglerASR`).

        Parameters
        ----------
        kind : {'repair', 'rejection', 'calibration'}
            Which decision to annotate:

            - ``'repair'`` — windows where at least ``min_components`` principal
              components were reconstructed (default).
            - ``'rejection'`` — samples removed by the final
              ``window_criterion`` clean-windows pass.
            - ``'calibration'`` — samples selected as the calibration reference
              (JugglerASR only).
        min_components : int
            Minimum reconstructed component count for ``kind='repair'``.
        description : str | None
            Annotation label. Defaults per kind: ``ASR_REPAIR`` / ``ASR_REJECT``
            / ``ASR_REFERENCE``.

        Returns
        -------
        annotations : mne.Annotations
            Annotation spans for the requested decision.
        """
        self._check_is_fitted()
        if mne is None:
            raise RuntimeError("MNE is required to create annotations")
        if kind == "repair":
            return self._repair_annotations(
                min_components=min_components,
                description=description or "ASR_REPAIR",
            )
        if kind == "rejection":
            return self._rejection_annotations(description=description or "ASR_REJECT")
        if kind == "calibration":
            return self._calibration_annotations(
                description=description or "ASR_REFERENCE"
            )
        raise ValueError(
            f"kind must be 'repair', 'rejection', or 'calibration', got {kind!r}"
        )

    def _repair_annotations(self, *, min_components: int, description: str) -> Any:
        if not hasattr(self, "diagnostics_"):
            raise RuntimeError("No transform diagnostics available")
        starts = self.diagnostics_["window_starts"]
        stops = self.diagnostics_["window_stops"]
        counts = self.diagnostics_["n_components_reconstructed"]
        spans = [
            (int(s), int(e))
            for s, e, c in zip(starts, stops, counts)
            if c >= min_components
        ]
        spans = _merge_sample_spans(spans)
        onsets = [s / self.sfreq_ for s, _ in spans]
        durations = [(e - s) / self.sfreq_ for s, e in spans]
        return mne.Annotations(onsets, durations, [description] * len(spans))

    def _rejection_annotations(self, *, description: str) -> Any:
        if not hasattr(self, "rejection_sample_mask_"):
            raise RuntimeError(
                "No final rejection mask is available. Enable window_criterion and "
                "run transform first."
            )
        mask = np.asarray(self.rejection_sample_mask_, dtype=bool)
        if mask.ndim != 1:
            raise RuntimeError(
                "Rejection annotations require continuous transform diagnostics."
            )
        rejected = ~mask
        if not np.any(rejected):
            return mne.Annotations([], [], [])
        spans = _mask_to_sample_spans(rejected)
        onsets = [s / self.sfreq_ for s, _ in spans]
        durations = [(e - s) / self.sfreq_ for s, e in spans]
        return mne.Annotations(onsets, durations, [description] * len(spans))

    def _calibration_annotations(self, *, description: str) -> Any:
        """Calibration-reference annotations (sample-based backends only)."""
        kind = getattr(self, "calibration_mask_kind_", "window")
        if kind != "sample":
            raise RuntimeError(
                "Calibration annotations are only available for sample-based "
                "reference selection (JugglerASR). This backend uses window-based "
                "calibration; use get_calibration_mask() instead."
            )
        mask = np.asarray(self.reference_sample_mask_, dtype=bool)
        spans = _mask_to_sample_spans(mask)
        onsets = [start / self.sfreq_ for start, _ in spans]
        durations = [(stop - start) / self.sfreq_ for start, stop in spans]
        return mne.Annotations(onsets, durations, [description] * len(spans))

    def _validate_estimator_params(self) -> None:
        if self.method not in ("standard", "riemannian", "riemannian_windowed"):
            raise NotImplementedError(
                "Supported methods are 'standard', 'riemannian_windowed', and "
                "experimental 'riemannian'."
            )
        # 'riemannian_windowed' is promoted to a first-class backend: its
        # processing is byte-identical to standard ASR (MATLAB-parity-tested)
        # and its calibration covariance matches the rASR backend, with a
        # direct MATLAB asr_process cross-check at relerr < 1e-13. See
        # tests/parity/test_riemannian_windowed_parity.py. Only the legacy
        # 'riemannian' backend (cutoff-invariant on real EEG) stays gated.
        if self.method == "riemannian" and not self.experimental:
            raise ValueError(
                "'riemannian' is experimental (cutoff-invariant on real EEG; "
                "see reports/paper_validation/rasr/). Set experimental=True to "
                "enable it, or use method='riemannian_windowed' for a "
                "cutoff-sensitive Riemannian backend."
            )
        if self.lookahead is not None and self.lookahead < 0:
            raise ValueError("lookahead must be non-negative")
        if self.stepsize is not None and self.stepsize < 1:
            raise ValueError("stepsize must be at least 1 sample")
        if (
            self.window_criterion is not None
            and self.window_criterion != "off"
            and isinstance(self.window_criterion, str)
        ):
            raise ValueError("window_criterion must be numeric, None, or 'off'")
        _validate_common_params(
            sfreq=self.sfreq if self.sfreq is not None else 1.0,
            cutoff=self.cutoff,
            window_length=self.window_length,
            window_overlap=self.window_overlap,
            max_dropout_fraction=self.max_dropout_fraction,
            min_clean_fraction=self.min_clean_fraction,
            regularization=self.regularization,
        )

    def _resolve_sfreq(self, sfreq: float | None, fitted: bool = False) -> float:
        if sfreq is None:
            sfreq = self.sfreq_ if fitted and hasattr(self, "sfreq_") else self.sfreq
        if sfreq is None:
            raise ValueError("sfreq must be provided for NumPy array inputs")
        if sfreq <= 0:
            raise ValueError("sfreq must be positive")
        return float(sfreq)

    def _resolve_picks(
        self,
        inst: Any,
        data: np.ndarray,
        mne_type: str,
    ) -> tuple[np.ndarray, list[str] | None]:
        if mne_type == "array":
            n_channels = data.shape[0]
            if self.picks is None or self.picks in (
                "all",
                "data",
                "eeg",
                "meg",
                "mag",
                "grad",
            ):
                picks = np.arange(n_channels, dtype=int)
            else:
                picks = np.asarray(self.picks, dtype=int)
            if picks.size == 0:
                raise ValueError("No channels selected for ASR")
            return picks, None

        if mne is None:
            raise RuntimeError("MNE is required for MNE object inputs")
        info = inst.info
        if self.picks is None or self.picks == "all":
            picks = np.arange(len(info["ch_names"]), dtype=int)
        elif self.picks == "eeg":
            picks = mne.pick_types(
                info,
                meg=False,
                eeg=True,
                eog=False,
                ecg=False,
                stim=False,
                misc=False,
                exclude="bads",
            )
        elif self.picks == "meg":
            picks = mne.pick_types(info, meg=True, eeg=False, exclude="bads")
        elif self.picks in ("mag", "grad"):
            picks = mne.pick_types(info, meg=self.picks, eeg=False, exclude="bads")
        elif isinstance(self.picks, str):
            raise ValueError(
                f"Unsupported picks string {self.picks!r}; use 'eeg', 'mag', "
                "'grad', 'meg', 'all', a list of channel names, or indices."
            )
        elif all(isinstance(pick, str) for pick in self.picks):
            picks = mne.pick_channels(
                info["ch_names"], include=list(self.picks), ordered=True
            )
        else:
            picks = np.asarray(self.picks, dtype=int)
        if picks.size == 0:
            raise ValueError("No channels selected for ASR")
        ch_names = [info["ch_names"][idx] for idx in picks]
        return np.asarray(picks, dtype=int), ch_names

    def _select_fit_data(
        self,
        data: np.ndarray,
        mne_type: str,
        picks: np.ndarray,
    ) -> np.ndarray:
        if mne_type == "epochs":
            # MNE Epochs are (n_epochs, n_channels, n_times). Concatenate epochs
            # for calibration while preserving channel order.
            return np.transpose(data[:, picks, :], (1, 0, 2)).reshape(len(picks), -1)
        return np.asarray(data[picks, :], dtype=np.float64)

    def _check_transform_channels(
        self,
        picks: np.ndarray,
        ch_names: list[str] | None,
    ) -> None:
        if len(picks) != self.n_channels_:
            raise ValueError(
                "Input channel count does not match fitted ASR state: "
                f"{len(picks)} vs {self.n_channels_}"
            )
        if self.ch_names_ is not None and ch_names != self.ch_names_:
            raise ValueError("Input channel names/order do not match fitted ASR state")

    def _transform_epochs(
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
            selected_clean, diag = process_asr(
                selected,
                sfreq,
                self.state_,
                window_length=self.window_length,
                window_overlap=self.window_overlap,
                max_dims=self.max_dims,
                regularization=self.regularization,
                store_reconstruction_matrices=self.store_reconstruction_matrices,
                max_mem_mb=self.max_mem_mb,
                lookahead=self.lookahead,
                stepsize=self.stepsize,
                method=self.method,
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

    def _store_transform_diagnostics(self, diagnostics: dict[str, Any]) -> None:
        self.diagnostics_ = diagnostics
        self.sample_mask_ = diagnostics["sample_mask"]
        self.window_starts_ = diagnostics["window_starts"]
        self.window_stops_ = diagnostics["window_stops"]
        self.n_components_reconstructed_ = diagnostics["n_components_reconstructed"]
        self.n_windows_ = diagnostics["n_windows"]
        self.fraction_reconstructed_windows_ = diagnostics[
            "fraction_reconstructed_windows"
        ]
        self.fraction_reconstructed_samples_ = diagnostics[
            "fraction_reconstructed_samples"
        ]
        self.max_components_reconstructed_ = diagnostics["max_components_reconstructed"]
        if "rejection_sample_mask" in diagnostics:
            self.rejection_sample_mask_ = diagnostics["rejection_sample_mask"]
            self.rejection_window_starts_ = diagnostics["rejection_window_starts"]
            self.rejection_window_stops_ = diagnostics["rejection_window_stops"]
            self.rejection_window_keep_mask_ = diagnostics["rejection_window_keep_mask"]
            self.rejection_window_remove_mask_ = diagnostics[
                "rejection_window_remove_mask"
            ]
            self.fraction_retained_after_window_rejection_ = diagnostics[
                "fraction_retained_after_window_rejection"
            ]
            self.fraction_rejected_after_window_rejection_ = diagnostics[
                "fraction_rejected_after_window_rejection"
            ]
        elif hasattr(self, "rejection_sample_mask_"):
            del self.rejection_sample_mask_
            del self.rejection_window_starts_
            del self.rejection_window_stops_
            del self.rejection_window_keep_mask_
            del self.rejection_window_remove_mask_
            del self.fraction_retained_after_window_rejection_
            del self.fraction_rejected_after_window_rejection_

    def _warn_preprocessing_state(self, inst: Any, mne_type: str) -> None:
        if mne_type == "array" or inst is None:
            return
        highpass = inst.info.get("highpass", None)
        if highpass is not None and highpass < 0.25:
            warnings.warn(
                "ASR assumes high-pass filtered data; input info reports "
                f"highpass={highpass} Hz.",
                UserWarning,
                stacklevel=3,
            )
        if len(inst.info.get("projs", [])) > 0:
            warnings.warn(
                "ASR is sensitive to data rank; active or unapplied projectors "
                "may affect covariance estimates.",
                UserWarning,
                stacklevel=3,
            )

    def _window_criterion_enabled(self) -> bool:
        return self.window_criterion not in (None, "off")

    def _compute_window_rejection(
        self,
        X: np.ndarray,
        sfreq: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        return compute_asr_rejection_mask(
            X,
            sfreq,
            max_bad_channels=self.window_criterion,
            zthresholds=self.window_criterion_tolerances,
            window_length=self.calibration_window_length,
            window_overlap=self.calibration_window_overlap,
            max_dropout_fraction=self.max_dropout_fraction,
            min_clean_fraction=self.min_clean_fraction,
        )

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "state_"):
            raise RuntimeError("ASR is not fitted. Call fit() first.")
