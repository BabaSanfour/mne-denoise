"""Backward-compatible facade for the ASR package.

The implementation now lives in focused submodules (``_types``, ``_validation``,
``_filters``, ``_distribution``, ``_covariance``, ``_windows``, ``_calibration``,
``_reconstruction``, ``_qa``, ``_estimator``). This module re-exports the public
API plus the private helpers that sibling modules and tests historically import
as ``mne_denoise.asr.core.*``. Prefer importing public names from
:mod:`mne_denoise.asr`.
"""

from __future__ import annotations

from ._calibration import (
    _fit_component_thresholds,
    calibrate_asr,
)
from ._covariance import (
    _aggregate_block_covariances,
    _aggregate_covariances,
    _block_covariances_clean_rawdata,
    _block_covariances_rasr,
    _covariance_chunk_blocks,
    _covariance_stack_bytes,
    _iter_block_covariances_clean_rawdata,
    _iter_block_covariances_rasr,
    _iter_moving_covariances_at,
    _max_mem_bytes,
    _moving_average_clean_rawdata,
    _process_memory_info,
    _window_covariances,
)
from ._distribution import (
    _fit_eeg_distribution_clean_rawdata,
    _histc_scaled_bins,
    _robust_location_scale,
    fit_eeg_distribution,
)
from ._estimator import (
    ASR,
)
from ._filters import (
    _append_clean_rawdata_tail,
    _apply_statistics_filter,
    _apply_statistics_filter_streaming,
    _design_statistics_filter,
    _prepend_clean_rawdata_carry,
)
from ._qa import (
    compute_asr_qa_metrics,
    compute_asr_rejection_mask,
)
from ._reconstruction import (
    _empty_process_diagnostics,
    _process_asr_riemannian,
    _process_asr_riemannian_windowed,
    process_asr,
)
from ._types import ASRState
from ._validation import (
    _check_enough_samples,
    _resolve_max_dims,
    _resolve_max_dims_clean_rawdata,
    _round_half_up,
    _validate_array_2d,
    _validate_common_params,
)
from ._windows import (
    _clean_rawdata_window_starts,
    _clean_windows_grid_diagnostics,
    _concatenate_windows,
    _good_raw_sample_mask,
    _mask_to_sample_spans,
    _merge_sample_spans,
    _resolve_max_bad_channels_count,
    _sample_mask_from_removed_windows,
    _select_clean_windows,
    _window_rms,
    _window_starts,
    _window_weights,
)

__all__ = [
    "ASR",
    "ASRState",
    "process_asr",
    "_empty_process_diagnostics",
    "_process_asr_riemannian",
    "_process_asr_riemannian_windowed",
    "compute_asr_qa_metrics",
    "compute_asr_rejection_mask",
    "calibrate_asr",
    "_fit_component_thresholds",
    "_window_starts",
    "_clean_rawdata_window_starts",
    "_window_weights",
    "_window_rms",
    "_resolve_max_bad_channels_count",
    "_select_clean_windows",
    "_clean_windows_grid_diagnostics",
    "_concatenate_windows",
    "_sample_mask_from_removed_windows",
    "_good_raw_sample_mask",
    "_merge_sample_spans",
    "_mask_to_sample_spans",
    "fit_eeg_distribution",
    "_fit_eeg_distribution_clean_rawdata",
    "_histc_scaled_bins",
    "_robust_location_scale",
    "_process_memory_info",
    "_iter_moving_covariances_at",
    "_moving_average_clean_rawdata",
    "_window_covariances",
    "_max_mem_bytes",
    "_covariance_stack_bytes",
    "_covariance_chunk_blocks",
    "_iter_block_covariances_clean_rawdata",
    "_iter_block_covariances_rasr",
    "_aggregate_block_covariances",
    "_block_covariances_clean_rawdata",
    "_block_covariances_rasr",
    "_aggregate_covariances",
    "_validate_common_params",
    "_validate_array_2d",
    "_check_enough_samples",
    "_round_half_up",
    "_resolve_max_dims_clean_rawdata",
    "_resolve_max_dims",
    "_design_statistics_filter",
    "_apply_statistics_filter",
    "_append_clean_rawdata_tail",
    "_prepend_clean_rawdata_carry",
    "_apply_statistics_filter_streaming",
]
