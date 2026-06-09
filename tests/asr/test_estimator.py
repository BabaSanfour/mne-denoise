"""Tests for mne_denoise.asr."""

from __future__ import annotations

import numpy as np
import pytest

from mne_denoise.asr import (
    ASR,
    AdaptiveASR,
    ASRState,
    JugglerASR,
    calibrate_asr,
    compute_asr_qa_metrics,
    compute_asr_rejection_mask,
    fit_eeg_distribution,
    process_asr,
    select_juggler_reference_samples,
)
from mne_denoise.asr._spd import _expm_sym, _karcher_mean_spd, _logm_spd


@pytest.fixture()
def rng():
    """Shared deterministic RNG."""
    return np.random.default_rng(42)


@pytest.fixture()
def synthetic_burst_data(rng):
    """Create multichannel EEG-like data with spatial burst artifacts."""
    sfreq = 250.0
    duration = 12.0
    n_times = int(sfreq * duration)
    n_channels = 8
    t = np.arange(n_times) / sfreq

    brain = np.zeros((n_channels, n_times), dtype=np.float64)
    for ch in range(n_channels):
        phase = rng.uniform(0, 2 * np.pi)
        brain[ch] = (
            0.5 * np.sin(2 * np.pi * 10 * t + phase)
            + 0.2 * np.sin(2 * np.pi * 6 * t + 0.5 * phase)
            + 0.05 * rng.standard_normal(n_times)
        )

    data = brain.copy()
    spatial = rng.standard_normal((n_channels, 2))
    spatial /= np.linalg.norm(spatial, axis=0, keepdims=True)
    burst_mask = np.zeros(n_times, dtype=bool)
    for onset, stop in ((4.0, 4.8), (8.0, 8.6)):
        start_samp = int(onset * sfreq)
        stop_samp = int(stop * sfreq)
        burst_mask[start_samp:stop_samp] = True
        source = rng.standard_normal((2, stop_samp - start_samp)) * 8.0
        data[:, start_samp:stop_samp] += spatial @ source

    return data, brain, burst_mask, sfreq


def test_calibrate_asr_returns_state_and_diagnostics(synthetic_burst_data):
    """Array-level calibration returns a valid ASR state."""
    data, _, _, sfreq = synthetic_burst_data
    state, diagnostics = calibrate_asr(
        data,
        sfreq,
        cutoff=4.0,
        calibration="auto",
        filter_kind="none",
    )

    assert isinstance(state, ASRState)
    assert state.M.shape == (data.shape[0], data.shape[0])
    assert state.T.shape == (data.shape[0], data.shape[0])
    assert state.thresholds.shape == (data.shape[0],)
    assert diagnostics["clean_window_mask"].ndim == 1
    assert diagnostics["n_clean_windows"] > 0
    assert diagnostics["threshold_mu"].shape == (data.shape[0],)
    assert diagnostics["threshold_sigma"].shape == (data.shape[0],)
    assert diagnostics["threshold_beta"].shape == (data.shape[0],)
    assert diagnostics["threshold_fit_interval"].shape == (data.shape[0], 2)


def test_calibrate_asr_low_memory_handles_remainder_two(rng):
    """Low-memory calibration handles the ASRpy block remainder edge case."""
    sfreq = 250.0
    n_channels = 6
    n_times = 1002
    blocksize = 100
    assert n_times % blocksize == 2
    data = 0.05 * rng.standard_normal((n_channels, n_times))

    state, diagnostics = calibrate_asr(
        data,
        sfreq,
        cutoff=5.0,
        calibration="manual",
        blocksize=blocksize,
        filter_kind="none",
        max_mem_mb=0.001,
    )

    assert isinstance(state, ASRState)
    assert state.M.shape == (n_channels, n_channels)
    assert diagnostics["memory_mode"] == "chunked"
    assert diagnostics["used_memory_bound"] is True
    assert (
        diagnostics["estimated_full_cov_bytes"] > diagnostics["peak_cov_buffer_bytes"]
    )
    assert diagnostics["chunk_samples"] == blocksize


def test_fit_eeg_distribution_robust_to_tail_and_dropouts(rng):
    """Clean RMS fitting resists high-tail artifacts and low dropouts."""
    clean = rng.normal(loc=1.0, scale=0.08, size=800)
    high_tail = rng.uniform(4.0, 9.0, size=120)
    dropouts = rng.uniform(0.01, 0.08, size=40)
    values = np.concatenate([clean, high_tail, dropouts])

    mu, sigma, info = fit_eeg_distribution(
        values,
        min_clean_fraction=0.25,
        max_dropout_fraction=0.1,
        return_info=True,
    )

    assert 0.9 < mu < 1.1
    assert 0.02 < sigma < 0.2
    assert sigma < np.std(values) * 0.2
    assert np.isfinite(info["beta"])
    assert info["n_fit_samples"] > 0


def test_fit_eeg_distribution_validation():
    """Clean RMS fitter validates empty and invalid parameter cases."""
    with pytest.raises(ValueError, match="empty RMS"):
        fit_eeg_distribution(np.array([np.nan, np.inf]))
    with pytest.raises(ValueError, match="fit_quantiles"):
        fit_eeg_distribution(np.ones(10), fit_quantiles=(0.7, 0.6))
    with pytest.raises(ValueError, match="beta_grid"):
        fit_eeg_distribution(np.ones(10), beta_grid=np.array([]))


def test_process_asr_reduces_synthetic_bursts(synthetic_burst_data):
    """Standard ASR reduces known burst artifact residual variance."""
    data, brain, burst_mask, sfreq = synthetic_burst_data
    state, _ = calibrate_asr(
        data,
        sfreq,
        cutoff=3.0,
        calibration="auto",
        ref_tolerances=(-np.inf, 3.0),
        filter_kind="none",
    )
    cleaned, diagnostics = process_asr(
        data,
        sfreq,
        state,
        window_length=0.5,
        window_overlap=0.66,
        max_dims=0.5,
    )

    assert cleaned.shape == data.shape
    assert diagnostics["n_windows"] > 0
    assert diagnostics["n_components_reconstructed"].sum() > 0
    assert diagnostics["sample_mask"].any()

    before = np.var(data[:, burst_mask] - brain[:, burst_mask])
    after = np.var(cleaned[:, burst_mask] - brain[:, burst_mask])
    assert after < before


def test_process_asr_low_memory_matches_full_path(synthetic_burst_data):
    """Low-memory rolling covariance processing matches the full path."""
    data, _, _, sfreq = synthetic_burst_data
    state, _ = calibrate_asr(
        data,
        sfreq,
        cutoff=3.0,
        calibration="auto",
        ref_tolerances=(-np.inf, 3.0),
        filter_kind="none",
        max_mem_mb=None,
    )

    full_cleaned, full_diag = process_asr(
        data,
        sfreq,
        state,
        window_length=0.5,
        window_overlap=0.66,
        max_dims=0.5,
        max_mem_mb=None,
    )
    rolling_cleaned, rolling_diag = process_asr(
        data,
        sfreq,
        state,
        window_length=0.5,
        window_overlap=0.66,
        max_dims=0.5,
        max_mem_mb=0.001,
    )

    assert full_diag["memory_mode"] == "full"
    assert rolling_diag["memory_mode"] == "rolling"
    assert rolling_diag["used_memory_bound"] is True
    np.testing.assert_allclose(rolling_cleaned, full_cleaned, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(
        rolling_diag["sample_mask"],
        full_diag["sample_mask"],
    )
    np.testing.assert_array_equal(
        rolling_diag["n_components_reconstructed"],
        full_diag["n_components_reconstructed"],
    )


def test_asr_estimator_numpy_qc_and_no_repair_cap(synthetic_burst_data):
    """Estimator path populates diagnostics and max_dims=0 preserves data."""
    data, _, _, sfreq = synthetic_burst_data
    asr = ASR(
        sfreq=sfreq,
        cutoff=1.0,
        max_dims=0.0,
        filter_kind="none",
        verbose=False,
    )
    cleaned = asr.fit_transform(data)

    np.testing.assert_allclose(cleaned, data, atol=1e-12)
    assert asr.n_windows_ > 0
    assert asr.n_components_reconstructed_.shape == (asr.n_windows_,)
    assert asr.n_components_reconstructed_.sum() == 0
    assert asr.get_calibration_mask().shape == asr.clean_window_mask_.shape


def test_asr_mne_raw_preserves_non_picked_channels(synthetic_burst_data):
    """MNE Raw support cleans EEG picks and preserves non-picked channels."""
    mne = pytest.importorskip("mne")
    data, _, _, sfreq = synthetic_burst_data
    eog = np.vstack(
        [
            np.sin(2 * np.pi * 1.0 * np.arange(data.shape[1]) / sfreq),
            np.cos(2 * np.pi * 1.0 * np.arange(data.shape[1]) / sfreq),
        ]
    )
    raw_data = np.vstack([data, eog])
    ch_names = [f"EEG{idx}" for idx in range(data.shape[0])] + ["EOG1", "EOG2"]
    ch_types = ["eeg"] * data.shape[0] + ["eog", "eog"]
    info = mne.create_info(ch_names, sfreq, ch_types)
    raw = mne.io.RawArray(raw_data, info, verbose=False)

    asr = ASR(cutoff=3.0, picks="eeg", filter_kind="none", verbose=False)
    raw_clean = asr.fit_transform(raw)

    assert isinstance(raw_clean, mne.io.RawArray)
    assert raw_clean.get_data().shape == raw_data.shape
    np.testing.assert_allclose(raw_clean.get_data(picks=["EOG1", "EOG2"]), eog)
    assert asr.ch_names_ == ch_names[: data.shape[0]]


def test_asr_mne_raw_low_memory_preserves_metadata(synthetic_burst_data):
    """Low-memory Raw processing preserves metadata and non-picked channels."""
    mne = pytest.importorskip("mne")
    data, _, _, sfreq = synthetic_burst_data
    eog = np.sin(2 * np.pi * 1.0 * np.arange(data.shape[1]) / sfreq)[None, :]
    raw_data = np.vstack([data, eog])
    ch_names = [f"EEG{idx}" for idx in range(data.shape[0])] + ["EOG1"]
    ch_types = ["eeg"] * data.shape[0] + ["eog"]
    info = mne.create_info(ch_names, sfreq, ch_types)
    raw = mne.io.RawArray(raw_data, info, verbose=False)
    raw.info["bads"] = ["EEG7"]
    raw.set_annotations(mne.Annotations([1.0], [0.25], ["BAD_test"]))

    asr = ASR(
        cutoff=3.0,
        picks="eeg",
        filter_kind="none",
        reject_by_annotation=False,
        max_mem_mb=0.001,
        verbose=False,
    )
    raw_clean = asr.fit_transform(raw)

    assert raw_clean.ch_names == raw.ch_names
    assert raw_clean.info["bads"] == raw.info["bads"]
    assert len(raw_clean.annotations) == len(raw.annotations)
    assert raw_clean.info["sfreq"] == raw.info["sfreq"]
    np.testing.assert_allclose(raw_clean.get_data(picks=["EOG1"]), eog)
    assert asr.diagnostics_["memory_mode"] == "rolling"


def test_asr_raw_bad_annotations_are_preserved(synthetic_burst_data):
    """Bad annotated Raw spans are excluded from final replacement."""
    mne = pytest.importorskip("mne")
    data, _, _, sfreq = synthetic_burst_data
    info = mne.create_info([f"EEG{idx}" for idx in range(data.shape[0])], sfreq, "eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_annotations(
        mne.Annotations(onset=[4.0], duration=[0.5], description=["bad_motion"])
    )

    asr = ASR(cutoff=2.0, filter_kind="none", verbose=False)
    raw_clean = asr.fit_transform(raw)
    bad_start = int(4.0 * sfreq)
    bad_stop = int(4.5 * sfreq)

    np.testing.assert_allclose(
        raw_clean.get_data()[:, bad_start:bad_stop],
        raw.get_data()[:, bad_start:bad_stop],
    )


def test_asr_to_annotations(synthetic_burst_data):
    """Last-transform repaired windows can be converted to annotations."""
    mne = pytest.importorskip("mne")
    data, _, _, sfreq = synthetic_burst_data
    asr = ASR(sfreq=sfreq, cutoff=3.0, filter_kind="none", verbose=False)
    asr.fit_transform(data)
    annotations = asr.to_annotations()

    assert isinstance(annotations, mne.Annotations)
    assert len(annotations) >= 1
    assert set(annotations.description) == {"ASR_REPAIR"}


def test_compute_asr_rejection_mask_flags_burst_samples(synthetic_burst_data):
    """clean_windows-style rejection mask flags high-burst segments."""
    data, _, burst_mask, sfreq = synthetic_burst_data
    sample_mask, diagnostics = compute_asr_rejection_mask(
        data,
        sfreq,
        max_bad_channels=0.25,
        zthresholds=(-np.inf, 3.5),
        window_length=0.5,
        window_overlap=0.66,
    )

    assert sample_mask.shape == (data.shape[1],)
    assert diagnostics["n_rejected_windows"] > 0
    assert diagnostics["window_keep_mask"].shape[0] == diagnostics["n_windows"]
    assert np.mean(sample_mask[burst_mask]) < np.mean(sample_mask[~burst_mask])


def test_asr_window_criterion_mask_and_annotations(synthetic_burst_data):
    """Optional final window rejection exposes retained-sample masks."""
    mne = pytest.importorskip("mne")
    data, _, _, sfreq = synthetic_burst_data
    asr = ASR(
        sfreq=sfreq,
        cutoff=3.0,
        filter_kind="none",
        window_criterion=0.25,
        window_criterion_tolerances=(-np.inf, 2.0),
        verbose=False,
    )
    asr.fit_transform(data)

    rejection_mask = asr.get_rejection_mask()
    assert rejection_mask.shape == (data.shape[1],)
    assert not np.all(rejection_mask)

    annotations = asr.to_annotations("rejection")
    assert isinstance(annotations, mne.Annotations)
    assert len(annotations) >= 1
    assert set(annotations.description) == {"ASR_REJECT"}


def test_compute_asr_qa_metrics(synthetic_burst_data):
    """ASR QA helper summarizes before/after data and estimator diagnostics."""
    data, _, _, sfreq = synthetic_burst_data
    asr = ASR(
        sfreq=sfreq,
        cutoff=3.0,
        filter_kind="none",
        window_criterion=0.25,
        verbose=False,
    )
    cleaned = asr.fit_transform(data)

    metrics = compute_asr_qa_metrics(data, cleaned, asr)

    assert "variance_removed_pct" in metrics
    assert "per_channel_variance_ratio" in metrics
    assert metrics["per_channel_variance_ratio"].shape == (data.shape[0],)
    assert metrics["n_windows"] == asr.n_windows_
    assert metrics["n_clean_calibration_windows"] == asr.clean_window_mask_.sum()
    assert metrics["fraction_reconstructed_samples"] == pytest.approx(
        asr.fraction_reconstructed_samples_
    )
    assert metrics["fraction_retained_after_window_rejection"] == pytest.approx(
        np.mean(asr.rejection_sample_mask_)
    )


def test_compute_asr_qa_metrics_shape_mismatch(synthetic_burst_data):
    """ASR QA helper rejects shape mismatches."""
    data, _, _, _ = synthetic_burst_data
    with pytest.raises(ValueError, match="matching shapes"):
        compute_asr_qa_metrics(data, data[:, :-1])


def test_riemannian_spd_primitives_roundtrip(rng):
    """SPD log/exp and Karcher mean helpers behave consistently."""
    A = rng.standard_normal((5, 5))
    C = A @ A.T + np.eye(5)
    log_C = _logm_spd(C, 1e-8)
    C_roundtrip = _expm_sym(log_C)
    np.testing.assert_allclose(C_roundtrip, C, rtol=1e-8, atol=1e-8)

    covs = np.stack([C, C], axis=0)
    mean_C, info = _karcher_mean_spd(covs, regularization=1e-8)
    np.testing.assert_allclose(mean_C, C, rtol=1e-8, atol=1e-8)
    assert info["riemannian_mean_converged"]


def test_asr_riemannian_experimental_backend(synthetic_burst_data):
    """Experimental Riemannian ASR runs end-to-end and suppresses bursts."""
    data, brain, burst_mask, sfreq = synthetic_burst_data
    asr = ASR(
        sfreq=sfreq,
        cutoff=3.0,
        method="riemannian",
        experimental=True,
        filter_kind="none",
        verbose=False,
    )
    cleaned = asr.fit_transform(data)

    assert cleaned.shape == data.shape
    assert np.all(np.isfinite(cleaned))
    assert asr.calibration_info_["covariance_geometry"] == "riemannian"
    assert asr.diagnostics_["covariance_geometry"] == "riemannian"
    assert asr.diagnostics_["riemannian_mean_iterations"].shape == (asr.n_windows_,)

    before = np.var(data[:, burst_mask] - brain[:, burst_mask])
    after = np.var(cleaned[:, burst_mask] - brain[:, burst_mask])
    assert after < before


@pytest.mark.parametrize("strategy", ["dbscan", "gev"])
def test_select_juggler_reference_samples_rejects_burst_samples(
    synthetic_burst_data,
    strategy,
):
    """Juggler reference selectors prefer low-amplitude samples."""
    data, _, burst_mask, sfreq = synthetic_burst_data
    reference, sample_mask, diagnostics = select_juggler_reference_samples(
        data,
        sfreq,
        strategy=strategy,
    )

    assert reference.shape[0] == data.shape[0]
    assert reference.shape[1] == int(np.sum(sample_mask))
    assert sample_mask.shape == (data.shape[1],)
    assert np.mean(sample_mask[burst_mask]) < np.mean(sample_mask[~burst_mask])
    assert diagnostics["reference_selection_strategy"] == strategy
    assert diagnostics["reference_selected_samples"] == int(np.sum(sample_mask))


@pytest.mark.parametrize("strategy", ["dbscan", "gev"])
def test_juggler_asr_reduces_synthetic_bursts(synthetic_burst_data, strategy):
    """JugglerASR reuses standard ASR repair after sample-wise calibration."""
    data, brain, burst_mask, sfreq = synthetic_burst_data
    asr = JugglerASR(
        sfreq=sfreq,
        cutoff=3.0,
        strategy=strategy,
        verbose=False,
    )
    cleaned = asr.fit_transform(data)

    assert cleaned.shape == data.shape
    assert np.all(np.isfinite(cleaned))
    assert asr.get_calibration_mask().shape == (data.shape[1],)
    assert asr.calibration_info_["reference_selection_strategy"] == strategy
    assert asr.calibration_info_["reference_selected_samples"] == int(
        np.sum(asr.reference_sample_mask_)
    )

    before = np.var(data[:, burst_mask] - brain[:, burst_mask])
    after = np.var(cleaned[:, burst_mask] - brain[:, burst_mask])
    assert after < before


def test_juggler_asr_reference_annotations_and_metrics(synthetic_burst_data):
    """JugglerASR exposes the retained reference spans for QC."""
    mne = pytest.importorskip("mne")
    data, _, _, sfreq = synthetic_burst_data
    asr = JugglerASR(
        sfreq=sfreq,
        cutoff=3.0,
        strategy="dbscan",
        verbose=False,
    )
    cleaned = asr.fit_transform(data)
    annotations = asr.to_annotations("calibration")

    assert isinstance(annotations, mne.Annotations)
    assert len(annotations) >= 1
    assert set(annotations.description) == {"ASR_REFERENCE"}
    assert asr.calibration_mask_kind_ == "sample"

    metrics = compute_asr_qa_metrics(data, cleaned, asr)
    assert metrics["n_clean_calibration_samples"] == int(
        np.sum(asr.reference_sample_mask_)
    )
    assert metrics["n_calibration_candidate_samples"] == data.shape[1]


def test_juggler_asr_mne_raw_preserves_non_picked_channels(synthetic_burst_data):
    """JugglerASR cleans EEG picks while leaving non-picked channels alone."""
    mne = pytest.importorskip("mne")
    data, _, _, sfreq = synthetic_burst_data
    eog = np.vstack(
        [
            np.sin(2 * np.pi * 1.0 * np.arange(data.shape[1]) / sfreq),
            np.cos(2 * np.pi * 1.0 * np.arange(data.shape[1]) / sfreq),
        ]
    )
    raw_data = np.vstack([data, eog])
    ch_names = [f"EEG{idx}" for idx in range(data.shape[0])] + ["EOG1", "EOG2"]
    ch_types = ["eeg"] * data.shape[0] + ["eog", "eog"]
    info = mne.create_info(ch_names, sfreq, ch_types)
    raw = mne.io.RawArray(raw_data, info, verbose=False)

    asr = JugglerASR(cutoff=3.0, picks="eeg", strategy="gev", verbose=False)
    raw_clean = asr.fit_transform(raw)

    assert isinstance(raw_clean, mne.io.RawArray)
    np.testing.assert_allclose(raw_clean.get_data(picks=["EOG1", "EOG2"]), eog)


def test_adaptive_asr_psw_updates_and_reduces_bursts(synthetic_burst_data):
    """Adaptive PSW-ASR updates thresholds and suppresses burst residuals."""
    data, brain, burst_mask, sfreq = synthetic_burst_data
    asr = AdaptiveASR(
        sfreq=sfreq,
        cutoff=5.0,
        variant="psw",
        verbose=False,
    )
    asr.fit(data[:, : int(4 * sfreq)])
    initial_T = asr.T_.copy()
    asr.partial_fit(data[:, int(4 * sfreq) : int(8 * sfreq)])

    assert len(asr.adaptive_update_history_) == 2
    assert asr.calibration_info_["event"] == "update"
    assert asr.calibration_info_["adaptive_variant"] == "psw"
    assert not np.allclose(initial_T, asr.T_)

    asr.reset_process_state()
    cleaned = asr.transform(data)

    assert cleaned.shape == data.shape
    assert np.all(np.isfinite(cleaned))
    assert asr.diagnostics_["adaptive_variant"] == "psw"
    before = np.var(data[:, burst_mask] - brain[:, burst_mask])
    after = np.var(cleaned[:, burst_mask] - brain[:, burst_mask])
    assert after < before


def test_adaptive_asr_reset_process_state_is_reproducible(synthetic_burst_data):
    """Resetting adaptive process state restores deterministic replay."""
    data, _, _, sfreq = synthetic_burst_data
    asr = AdaptiveASR(
        sfreq=sfreq,
        cutoff=5.0,
        variant="psp",
        verbose=False,
    )
    asr.fit(data[:, : int(6 * sfreq)])
    cleaned_first = asr.transform(data)
    asr.reset_process_state()
    cleaned_second = asr.transform(data)
    np.testing.assert_allclose(cleaned_first, cleaned_second, atol=1e-10)


def test_adaptive_asr_low_memory_matches_full_path(synthetic_burst_data):
    """Adaptive ASR honors max_mem_mb without changing reconstruction."""
    data, _, _, sfreq = synthetic_burst_data
    calibration = data[:, : int(6 * sfreq)]

    full = AdaptiveASR(
        sfreq=sfreq,
        cutoff=5.0,
        variant="psw",
        max_mem_mb=None,
        verbose=False,
    )
    low_mem = AdaptiveASR(
        sfreq=sfreq,
        cutoff=5.0,
        variant="psw",
        max_mem_mb=0.001,
        verbose=False,
    )
    full.fit(calibration)
    low_mem.fit(calibration)

    cleaned_full = full.transform(data)
    cleaned_low_mem = low_mem.transform(data)

    assert full.calibration_info_["memory_mode"] == "full"
    assert low_mem.calibration_info_["memory_mode"] == "chunked"
    assert full.diagnostics_["memory_mode"] == "full"
    assert low_mem.diagnostics_["memory_mode"] == "chunked"
    assert low_mem.diagnostics_["used_memory_bound"]
    np.testing.assert_allclose(cleaned_low_mem, cleaned_full, atol=1e-10)


def test_adaptive_asr_mne_raw_preserves_non_picked_channels(synthetic_burst_data):
    """Adaptive ASR cleans EEG picks and preserves non-picked channels."""
    mne = pytest.importorskip("mne")
    data, _, _, sfreq = synthetic_burst_data
    eog = np.vstack(
        [
            np.sin(2 * np.pi * 1.0 * np.arange(data.shape[1]) / sfreq),
            np.cos(2 * np.pi * 1.0 * np.arange(data.shape[1]) / sfreq),
        ]
    )
    raw_data = np.vstack([data, eog])
    ch_names = [f"EEG{idx}" for idx in range(data.shape[0])] + ["EOG1", "EOG2"]
    ch_types = ["eeg"] * data.shape[0] + ["eog", "eog"]
    info = mne.create_info(ch_names, sfreq, ch_types)
    raw = mne.io.RawArray(raw_data, info, verbose=False)

    asr = AdaptiveASR(cutoff=5.0, picks="eeg", variant="psp", verbose=False)
    raw_clean = asr.fit_transform(raw)

    assert isinstance(raw_clean, mne.io.RawArray)
    assert raw_clean.get_data().shape == raw_data.shape
    np.testing.assert_allclose(raw_clean.get_data(picks=["EOG1", "EOG2"]), eog)
    assert asr.ch_names_ == ch_names[: data.shape[0]]


def test_asr_epochs_round_trip(synthetic_burst_data):
    """Epochs can be calibrated by concatenation and transformed per epoch."""
    mne = pytest.importorskip("mne")
    data, _, _, sfreq = synthetic_burst_data
    n_epochs = 3
    epoch_data = np.stack(
        [data[:, idx * 750 : (idx + 1) * 750] for idx in range(n_epochs)]
    )
    info = mne.create_info([f"EEG{idx}" for idx in range(data.shape[0])], sfreq, "eeg")
    epochs = mne.EpochsArray(epoch_data, info, verbose=False)

    asr = ASR(cutoff=4.0, filter_kind="none", verbose=False)
    epochs_clean = asr.fit_transform(epochs)

    assert epochs_clean.get_data().shape == epoch_data.shape
    assert asr.sample_mask_.shape == (n_epochs, epoch_data.shape[-1])


def test_asr_validation_errors(synthetic_burst_data):
    """ASR raises clear errors for unsupported or invalid inputs."""
    data, _, _, sfreq = synthetic_burst_data
    with pytest.raises(ValueError, match="experimental=True"):
        ASR(sfreq=sfreq, method="riemannian").fit(data)
    with pytest.raises(ValueError, match="strategy"):
        JugglerASR(sfreq=sfreq, strategy="unknown").fit(data)
    with pytest.raises(ValueError, match="variant"):
        AdaptiveASR(sfreq=sfreq, variant="unknown").fit(data)
    with pytest.raises(NotImplementedError, match="Supported methods"):
        ASR(sfreq=sfreq, method="unknown").fit(data)
    with pytest.raises(ValueError, match="sfreq"):
        ASR().fit(data)
    with pytest.raises(ValueError, match="at least two channels"):
        ASR(sfreq=sfreq).fit(data[:1])
    with pytest.raises(RuntimeError, match="not fitted"):
        ASR(sfreq=sfreq).transform(data)
