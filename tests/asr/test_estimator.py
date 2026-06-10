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

SFREQ = 250.0


def _epochs(n_epochs=3, n_per=2000):
    mne = pytest.importorskip("mne")
    X = _eeg(n_times=n_epochs * n_per, bursts=6)
    info = mne.create_info([f"EEG{i:02d}" for i in range(8)], SFREQ, "eeg")
    data = X.reshape(8, n_epochs, n_per).transpose(1, 0, 2) * 1e-6
    return mne.EpochsArray(data, info, verbose=False)


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


# ===========================================================================
# Tests relocated from former test_coverage.py / test_robustness.py (PR #36).
# Grouped here by the module they exercise.
# ===========================================================================


def _eeg(n_channels=8, n_times=8000, seed=0, bursts=5):
    rng = np.random.default_rng(seed)
    t = np.arange(n_times) / SFREQ
    X = np.zeros((n_channels, n_times))
    for c in range(n_channels):
        X[c] = 0.6 * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6.28)) + (
            0.05 * rng.standard_normal(n_times)
        )
    for s in np.linspace(800, n_times - 500, bursts).astype(int):
        spatial = rng.standard_normal(n_channels)
        spatial /= np.linalg.norm(spatial)
        X[:, s : s + 150] += 10.0 * np.outer(spatial, rng.standard_normal(150))
    return X


def _inject_bursts(
    data: np.ndarray,
    n_bursts: int = 8,
    burst_duration_s: float = 0.5,
    amplitude: float = 12.0,
    sfreq: float = 250.0,
    seed: int = 97,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    burst_len = int(round(burst_duration_s * sfreq))
    n_times = data.shape[1]
    starts = np.linspace(burst_len, n_times - burst_len, n_bursts).astype(int)
    contaminated = data.copy()
    scale = float(np.median(np.std(data, axis=1)))
    for start in starts:
        stop = min(start + burst_len, n_times)
        spatial = rng.standard_normal(data.shape[0])
        spatial /= max(np.linalg.norm(spatial), 1e-12)
        temporal = rng.standard_normal(stop - start)
        contaminated[:, start:stop] += amplitude * scale * np.outer(spatial, temporal)
    return contaminated


def _make_clean_eeg(
    n_channels: int = 16,
    duration_s: float = 40.0,
    sfreq: float = 250.0,
    seed: int = 11,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sfreq * duration_s)
    t = np.arange(n) / sfreq
    data = np.zeros((n_channels, n), dtype=np.float64)
    for ch in range(n_channels):
        phase = rng.uniform(0, 2 * np.pi)
        data[ch] = 0.6 * np.sin(2 * np.pi * 10.0 * t + phase) + 0.15 * np.sin(
            2 * np.pi * 6.5 * t + phase * 0.8
        )
    data += 0.05 * rng.standard_normal(data.shape)
    return data


def test_calibrate_asr_bad_calibration_raises():
    with pytest.raises(ValueError, match="calibration must be"):
        calibrate_asr(_eeg(), SFREQ, calibration="bogus", filter_kind="none")


def test_calibrate_asr_bad_cov_estimator_raises():
    with pytest.raises(ValueError, match="cov_estimator"):
        calibrate_asr(_eeg(), SFREQ, cov_estimator="bogus", filter_kind="none")


def test_calibrate_asr_bad_method_raises():
    with pytest.raises(ValueError, match="method must be"):
        calibrate_asr(_eeg(), SFREQ, method="bogus", filter_kind="none")


def test_calibrate_asr_bad_blocksize_raises():
    with pytest.raises(ValueError, match="blocksize"):
        calibrate_asr(_eeg(), SFREQ, blocksize=0, filter_kind="none")


def test_asr_unknown_method_raises():
    with pytest.raises(NotImplementedError, match="Supported methods"):
        ASR(sfreq=SFREQ, method="bogus", picks=None, verbose=False).fit(_eeg())


def test_asr_riemannian_requires_experimental():
    with pytest.raises(ValueError, match="experimental"):
        ASR(sfreq=SFREQ, method="riemannian", picks=None, verbose=False).fit(_eeg())


def test_asr_riemannian_windowed_no_experimental_needed():
    cleaned = ASR(
        sfreq=SFREQ, method="riemannian_windowed", picks=None, verbose=False
    ).fit_transform(_eeg())
    assert cleaned.shape == (8, 8000)


def test_get_rejection_mask_without_window_criterion_raises():
    asr = ASR(sfreq=SFREQ, cutoff=20.0, picks=None, verbose=False)
    asr.fit_transform(_eeg())
    with pytest.raises(RuntimeError, match="rejection mask"):
        asr.get_rejection_mask()


def test_to_annotations_bad_kind_raises():
    asr = ASR(sfreq=SFREQ, cutoff=20.0, picks=None, verbose=False)
    asr.fit_transform(_eeg())
    with pytest.raises(ValueError, match="kind must be"):
        asr.to_annotations("bogus")


def test_to_annotations_calibration_on_window_backend_raises():
    asr = ASR(sfreq=SFREQ, cutoff=20.0, picks=None, verbose=False)
    asr.fit_transform(_eeg())
    with pytest.raises(RuntimeError, match="sample-based"):
        asr.to_annotations("calibration")


def test_get_diagnostics_present_on_all_variants():
    for est in (
        ASR(sfreq=SFREQ, picks=None, verbose=False),
        AdaptiveASR(sfreq=SFREQ, variant="psp", picks=None, verbose=False),
        JugglerASR(sfreq=SFREQ, strategy="gev", picks=None, verbose=False),
    ):
        est.fit_transform(_eeg())
        assert isinstance(est.get_diagnostics(), dict)


def test_asr_window_criterion_rejection_and_annotations():
    mne = pytest.importorskip("mne")
    asr = ASR(
        sfreq=SFREQ,
        cutoff=10.0,
        calibration="auto",
        picks=None,
        window_criterion=0.3,
        window_criterion_tolerances=(-np.inf, 5.0),
        verbose=False,
    )
    asr.fit_transform(_eeg(bursts=8))
    mask = asr.get_rejection_mask()
    assert mask.dtype == bool
    ann = asr.to_annotations("rejection")
    assert isinstance(ann, mne.Annotations)


def test_standard_low_memory_matches_full():
    X = _eeg()
    full = ASR(sfreq=SFREQ, cutoff=20.0, picks=None, max_mem_mb=512, verbose=False)
    low = ASR(sfreq=SFREQ, cutoff=20.0, picks=None, max_mem_mb=1, verbose=False)
    c_full = full.fit_transform(X)
    c_low = low.fit_transform(X)
    np.testing.assert_allclose(c_full, c_low, atol=1e-9)


def test_riemannian_windowed_low_memory_runs():
    X = _eeg()
    asr = ASR(
        sfreq=SFREQ,
        cutoff=20.0,
        method="riemannian_windowed",
        picks=None,
        max_mem_mb=1,
        verbose=False,
    )
    cleaned = asr.fit_transform(X)
    assert np.all(np.isfinite(cleaned))


def test_fit_eeg_distribution_returns_params():
    rng = np.random.default_rng(1)
    rms = np.abs(rng.standard_normal(2000)) + 0.5
    mu, sigma, *_ = fit_eeg_distribution(rms)
    assert np.isfinite(mu) and sigma > 0


def test_compute_asr_rejection_mask_standalone():
    X = _eeg(bursts=8)
    mask, info = compute_asr_rejection_mask(X, SFREQ)
    assert mask.dtype == bool
    assert mask.shape == (X.shape[1],)
    assert isinstance(info, dict)


def test_compute_asr_qa_metrics_with_and_without_estimator():
    X = _eeg()
    asr = ASR(sfreq=SFREQ, cutoff=20.0, picks=None, verbose=False)
    cleaned = np.asarray(asr.fit_transform(X))
    m1 = compute_asr_qa_metrics(X, cleaned, asr)
    m2 = compute_asr_qa_metrics(X, cleaned, None)
    assert "variance_removed_pct" in m1
    assert "variance_removed_pct" in m2


def test_process_asr_method_validation():
    state, _ = calibrate_asr(
        _eeg(), SFREQ, method="standard", calibration="manual", filter_kind="none"
    )
    with pytest.raises(ValueError, match="method must be"):
        process_asr(_eeg(), SFREQ, state, method="bogus")


def test_adaptive_and_juggler_accept_epochs():
    mne = pytest.importorskip("mne")
    X = _eeg(n_times=6000)
    info = mne.create_info([f"EEG{i:02d}" for i in range(8)], SFREQ, "eeg")
    # 3 epochs of 2000 samples
    epo = mne.EpochsArray(
        X.reshape(8, 3, 2000).transpose(1, 0, 2) * 1e-6, info, verbose=False
    )
    for est in (
        AdaptiveASR(sfreq=SFREQ, variant="psw", verbose=False),
        JugglerASR(sfreq=SFREQ, strategy="dbscan", verbose=False),
    ):
        out = est.fit_transform(epo)
        assert out.get_data().shape == epo.get_data().shape


def test_asr_window_criterion_on_epochs_populates_rejection():
    epo = _epochs()
    asr = ASR(
        sfreq=SFREQ,
        cutoff=10.0,
        window_criterion=0.3,
        window_criterion_tolerances=(-np.inf, 5.0),
        verbose=False,
    )
    out = asr.fit_transform(epo)
    assert out.get_data().shape == epo.get_data().shape
    diag = asr.get_diagnostics()
    assert "rejection_sample_mask" in diag
    assert "fraction_retained_after_window_rejection" in diag


def test_riemannian_low_memory_runs():
    asr = ASR(
        sfreq=SFREQ,
        cutoff=20.0,
        method="riemannian",
        experimental=True,
        picks=None,
        max_mem_mb=1,
        verbose=False,
    )
    cleaned = asr.fit_transform(_eeg())
    assert np.all(np.isfinite(cleaned))


def test_karcher_mean_spd_validation_guards():
    from mne_denoise.asr._spd import _karcher_mean_spd

    spd = np.stack([np.eye(3), 2.0 * np.eye(3)])
    with pytest.raises(ValueError, match="shape"):
        _karcher_mean_spd(np.eye(3), regularization=1e-8)
    with pytest.raises(ValueError, match="At least one"):
        _karcher_mean_spd(np.empty((0, 3, 3)), regularization=1e-8)
    with pytest.raises(ValueError, match="sample_weight must have shape"):
        _karcher_mean_spd(spd, sample_weight=np.ones(5), regularization=1e-8)
    with pytest.raises(ValueError, match="non-negative"):
        _karcher_mean_spd(spd, sample_weight=np.array([-1.0, 1.0]), regularization=1e-8)
    with pytest.raises(ValueError, match="positive value"):
        _karcher_mean_spd(spd, sample_weight=np.array([0.0, 0.0]), regularization=1e-8)


def test_validate_array_2d_guards():
    from mne_denoise.asr.core import _validate_array_2d

    with pytest.raises(ValueError, match="2D array"):
        _validate_array_2d(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="at least two channels"):
        _validate_array_2d(np.ones((1, 100)))
    bad = np.ones((3, 100))
    bad[1, :] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _validate_array_2d(bad)
    with pytest.raises(ValueError, match="variance"):
        _validate_array_2d(np.ones((3, 100)))  # zero variance


def test_window_starts_guards():
    from mne_denoise.asr.core import _window_starts

    with pytest.raises(ValueError, match="at least 2 samples"):
        _window_starts(100, 1, 0.5)
    with pytest.raises(ValueError, match="exceeds data length"):
        _window_starts(50, 100, 0.5)


def test_calibrate_asr_common_param_guards():
    X = _eeg()
    with pytest.raises(ValueError, match="cutoff must be positive"):
        calibrate_asr(X, SFREQ, cutoff=-1.0, filter_kind="none")
    with pytest.raises(ValueError, match="window_length must be positive"):
        calibrate_asr(X, SFREQ, window_length=-1.0, filter_kind="none")
    with pytest.raises(ValueError, match="window_overlap"):
        calibrate_asr(X, SFREQ, window_overlap=1.5, filter_kind="none")
    with pytest.raises(ValueError, match="max_dropout_fraction"):
        calibrate_asr(X, SFREQ, max_dropout_fraction=1.5, filter_kind="none")
    with pytest.raises(ValueError, match="min_clean_fraction"):
        calibrate_asr(X, SFREQ, min_clean_fraction=0.0, filter_kind="none")
    with pytest.raises(ValueError, match="regularization must be positive"):
        calibrate_asr(X, SFREQ, regularization=0.0, filter_kind="none")


def test_riemannian_chunked_covariance_paths():
    X = _eeg()  # 8000 samples -> rASR cov stack exceeds a 0.05 MB budget
    c1 = ASR(
        sfreq=SFREQ,
        cutoff=20.0,
        method="riemannian",
        experimental=True,
        cov_estimator="geometric_median",
        picks=None,
        max_mem_mb=0.05,
        verbose=False,
    ).fit_transform(X)
    assert np.all(np.isfinite(c1))
    c2 = ASR(
        sfreq=SFREQ,
        cutoff=20.0,
        method="riemannian",
        experimental=True,
        cov_estimator="mean",
        picks=None,
        max_mem_mb=0.05,
        verbose=False,
    ).fit_transform(X)
    assert np.all(np.isfinite(c2))
    with pytest.raises(ValueError, match="median"):
        ASR(
            sfreq=SFREQ,
            cutoff=20.0,
            method="riemannian",
            experimental=True,
            cov_estimator="median",
            picks=None,
            max_mem_mb=0.05,
            verbose=False,
        ).fit_transform(X)


def test_asr_picks_by_channel_names():
    mne = pytest.importorskip("mne")
    ch = [f"EEG{i:02d}" for i in range(8)]
    info = mne.create_info(ch, SFREQ, "eeg")
    raw = mne.io.RawArray(_eeg() * 1e-6, info, verbose=False)
    asr = ASR(
        sfreq=SFREQ,
        cutoff=20.0,
        picks=["EEG00", "EEG01", "EEG02", "EEG03"],
        verbose=False,
    )
    out = asr.fit_transform(raw)
    assert out.get_data().shape == raw.get_data().shape


def test_asr_transform_channel_count_mismatch_raises():
    mne = pytest.importorskip("mne")
    info8 = mne.create_info([f"EEG{i:02d}" for i in range(8)], SFREQ, "eeg")
    raw8 = mne.io.RawArray(_eeg() * 1e-6, info8, verbose=False)
    asr = ASR(sfreq=SFREQ, cutoff=20.0, verbose=False)
    asr.fit(raw8)
    info6 = mne.create_info([f"EEG{i:02d}" for i in range(6)], SFREQ, "eeg")
    raw6 = mne.io.RawArray(_eeg(n_channels=6) * 1e-6, info6, verbose=False)
    with pytest.raises(ValueError, match="channel count does not match"):
        asr.transform(raw6)


@pytest.mark.parametrize("method", ["standard", "riemannian_windowed"])
def test_store_reconstruction_matrices(method):
    asr = ASR(
        sfreq=SFREQ,
        cutoff=5.0,
        method=method,
        picks=None,
        store_reconstruction_matrices=True,
        verbose=False,
    )
    asr.fit_transform(_eeg(bursts=8))
    assert "reconstruction_matrices" in asr.get_diagnostics()


def test_resolve_max_dims_and_robust_scale_and_mem_bytes():
    from mne_denoise.asr.core import (
        _max_mem_bytes,
        _resolve_max_dims,
        _robust_location_scale,
    )

    assert _resolve_max_dims(0.5, 10) == 5
    assert _resolve_max_dims(3, 10) == 3
    with pytest.raises(ValueError, match="float max_dims"):
        _resolve_max_dims(1.5, 10)
    with pytest.raises(ValueError, match="integer max_dims"):
        _resolve_max_dims(20, 10)

    mu, sigma = _robust_location_scale(np.ones(50))  # mad=0 -> tiny positive fallback
    assert sigma > 0 and np.isfinite(mu)

    with pytest.raises(ValueError, match="max_mem_mb must be"):
        _max_mem_bytes(-1.0)
    assert _max_mem_bytes(None) is None


def test_core_window_and_resolve_helpers():
    from mne_denoise.asr.core import (
        _append_clean_rawdata_tail,
        _clean_rawdata_window_starts,
        _prepend_clean_rawdata_carry,
        _resolve_max_bad_channels_count,
        _resolve_max_dims_clean_rawdata,
        _window_weights,
    )

    assert _window_weights(2).shape == (2,)  # small-window flat weights
    assert _window_weights(16).shape == (16,)  # hanning taper
    assert _clean_rawdata_window_starts(1000, 100, 0.5).size > 0

    assert _resolve_max_bad_channels_count(0.1, 20) >= 0  # float fraction
    assert _resolve_max_bad_channels_count(3, 20) == 3  # int branch
    with pytest.raises(ValueError, match="non-negative"):
        _resolve_max_bad_channels_count(-1, 20)

    assert _resolve_max_dims_clean_rawdata(0.5, 20) == 10  # float fraction
    with pytest.raises(ValueError, match="non-negative"):
        _resolve_max_dims_clean_rawdata(-0.5, 20)
    with pytest.raises(ValueError, match="non-negative"):
        _resolve_max_dims_clean_rawdata(-3, 20)

    X = np.ones((4, 5))
    with pytest.raises(ValueError, match="lookahead tail"):
        _append_clean_rawdata_tail(X, 10)
    with pytest.raises(ValueError, match="lookahead carry"):
        _prepend_clean_rawdata_carry(X, 10)
    assert _append_clean_rawdata_tail(X, 0).shape == X.shape  # zero -> copy
    assert _prepend_clean_rawdata_carry(X, 0).shape == X.shape


def test_asr_supports_meg_picks():
    mne = pytest.importorskip("mne")
    rng = np.random.default_rng(0)
    n = 6000
    t = np.arange(n) / SFREQ
    X = np.zeros((12, n))
    for c in range(12):
        X[c] = (
            0.6 * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6.28))
            + 0.05 * rng.standard_normal(n)
        ) * 1e-13  # magnetometer (Tesla) scale
    for s in (1200, 3000, 4500):
        sp = rng.standard_normal(12)
        sp /= np.linalg.norm(sp)
        X[:, s : s + 150] += 10e-13 * np.outer(sp, rng.standard_normal(150))
    info = mne.create_info([f"MEG{i:02d}" for i in range(12)], SFREQ, "mag")
    raw = mne.io.RawArray(X, info, verbose=False)
    out = ASR(sfreq=SFREQ, cutoff=20.0, picks="mag", verbose=False).fit_transform(raw)
    assert out.get_data().shape == raw.get_data().shape
    assert np.all(np.isfinite(out.get_data()))


def test_asr_unknown_picks_string_raises():
    mne = pytest.importorskip("mne")
    info = mne.create_info([f"EEG{i:02d}" for i in range(8)], SFREQ, "eeg")
    raw = mne.io.RawArray(_eeg() * 1e-6, info, verbose=False)
    with pytest.raises(ValueError, match="Unsupported picks"):
        ASR(sfreq=SFREQ, picks="bogus", verbose=False).fit(raw)


@pytest.mark.parametrize("cutoff", [5.0, 20.0, 50.0])
def test_standard_cutoff_parametric(cutoff):
    clean = _make_clean_eeg()
    dirty = _inject_bursts(clean)
    asr = ASR(sfreq=250.0, cutoff=cutoff, picks=None, verbose=False)
    cleaned = asr.fit_transform(dirty)
    assert cleaned.shape == dirty.shape
    assert np.all(np.isfinite(cleaned))


@pytest.mark.parametrize("cutoff", [5.0, 20.0, 50.0])
def test_rasr_windowed_cutoff_parametric(cutoff):
    clean = _make_clean_eeg()
    dirty = _inject_bursts(clean)
    asr = ASR(
        sfreq=250.0,
        cutoff=cutoff,
        method="riemannian_windowed",
        experimental=True,
        picks=None,
        verbose=False,
    )
    cleaned = asr.fit_transform(dirty)
    assert cleaned.shape == dirty.shape


def test_rasr_windowed_cutoff_monotonicity():
    """riemannian_windowed % windows reconstructed must decrease as k rises."""
    clean = _make_clean_eeg()
    dirty = _inject_bursts(clean)
    fractions = {}
    for k in (5.0, 20.0, 50.0):
        asr = ASR(
            sfreq=250.0,
            cutoff=k,
            method="riemannian_windowed",
            experimental=True,
            picks=None,
            verbose=False,
        )
        asr.fit_transform(dirty)
        fractions[k] = float(asr.diagnostics_["fraction_reconstructed_windows"])
    # Monotone non-increasing (allow ties)
    keys = sorted(fractions)
    for a, b in zip(keys[:-1], keys[1:]):
        assert fractions[a] >= fractions[b] - 1e-6, (
            f"non-monotone: k={a}->{fractions[a]}, k={b}->{fractions[b]}"
        )


def test_standard_4_channel_substrate():
    """Standard ASR should at minimum NOT crash on 4-channel data."""
    clean = _make_clean_eeg(n_channels=4, duration_s=60.0)
    dirty = _inject_bursts(clean)
    asr = ASR(sfreq=250.0, cutoff=20.0, picks=None, verbose=False)
    cleaned = asr.fit_transform(dirty)
    assert cleaned.shape == dirty.shape


def test_standard_64_channel_substrate():
    """Standard ASR should not crash on 64-channel data."""
    clean = _make_clean_eeg(n_channels=64, duration_s=30.0)
    dirty = _inject_bursts(clean, n_bursts=6)
    asr = ASR(sfreq=250.0, cutoff=20.0, picks=None, verbose=False)
    cleaned = asr.fit_transform(dirty)
    assert cleaned.shape == dirty.shape


def test_rasr_windowed_64_channel_substrate():
    """rASR-windowed should handle 64 channels."""
    clean = _make_clean_eeg(n_channels=64, duration_s=30.0)
    dirty = _inject_bursts(clean, n_bursts=6)
    asr = ASR(
        sfreq=250.0,
        cutoff=20.0,
        method="riemannian_windowed",
        experimental=True,
        picks=None,
        verbose=False,
    )
    cleaned = asr.fit_transform(dirty)
    assert cleaned.shape == dirty.shape


@pytest.mark.parametrize("sfreq", [250.0, 1000.0])
def test_standard_sfreq_parametric(sfreq):
    """Standard ASR should not crash at typical to high sample rates.

    Uses a long-enough substrate (60 s) so the clean-windows selection still
    leaves enough samples for the threshold-fit step even after aggressive
    burst contamination.
    """
    clean = _make_clean_eeg(duration_s=60.0, sfreq=sfreq)
    dirty = _inject_bursts(clean, sfreq=sfreq, n_bursts=4)
    asr = ASR(sfreq=sfreq, cutoff=20.0, picks=None, verbose=False)
    cleaned = asr.fit_transform(dirty)
    assert cleaned.shape == dirty.shape


def test_short_recording_too_short_raises_clearly():
    """A sub-window recording should raise a clear error, not silently misbehave.

    Use 0.3 s at 250 Hz = 75 samples — below the default 0.5 s = 125 sample
    window length. ASR's calibration cannot run with fewer samples than one
    window.
    """
    clean = _make_clean_eeg(duration_s=0.3, n_channels=8)
    asr = ASR(sfreq=250.0, cutoff=20.0, picks=None, verbose=False)
    with pytest.raises((ValueError, RuntimeError)):
        asr.fit(clean)


def test_minimal_30s_recording_succeeds():
    """30 s is the documented minimum that should generally succeed."""
    clean = _make_clean_eeg(duration_s=30.0, n_channels=8)
    dirty = _inject_bursts(clean, n_bursts=4)
    asr = ASR(sfreq=250.0, cutoff=20.0, picks=None, verbose=False)
    cleaned = asr.fit_transform(dirty)
    assert cleaned.shape == dirty.shape


def test_all_variants_preserve_clean_substrate():
    """On a fully clean substrate, every variant should produce output that
    is essentially the input (correlation > 0.95, no over-cleaning).
    """
    clean = _make_clean_eeg(n_channels=16, duration_s=60.0)
    variants = []
    # Standard
    asr1 = ASR(sfreq=250.0, cutoff=20.0, picks=None, verbose=False)
    variants.append(("standard", asr1.fit_transform(clean)))
    # riemannian
    asr2 = ASR(
        sfreq=250.0,
        cutoff=20.0,
        method="riemannian",
        experimental=True,
        picks=None,
        verbose=False,
    )
    variants.append(("riemannian", asr2.fit_transform(clean)))
    # riemannian_windowed
    asr3 = ASR(
        sfreq=250.0,
        cutoff=20.0,
        method="riemannian_windowed",
        experimental=True,
        picks=None,
        verbose=False,
    )
    variants.append(("riemannian_windowed", asr3.fit_transform(clean)))
    # PSP
    asr4 = AdaptiveASR(
        sfreq=250.0,
        cutoff=20.0,
        variant="psp",
        picks=None,
        verbose=False,
    )
    variants.append(("psp", asr4.fit_transform(clean)))
    # Juggler DBSCAN
    asr5 = JugglerASR(
        sfreq=250.0,
        cutoff=20.0,
        strategy="dbscan",
        picks=None,
        verbose=False,
    )
    variants.append(("juggler_dbscan", asr5.fit_transform(clean)))

    for name, cleaned in variants:
        corr = float(np.corrcoef(cleaned.ravel(), clean.ravel())[0, 1])
        assert corr > 0.95, f"{name} over-cleaned a clean substrate; corr={corr:.4f}"


def test_evoked_fit_raises():
    """ASR.fit() should refuse Evoked input across variants."""
    import mne

    clean = _make_clean_eeg(n_channels=8, duration_s=30.0)
    info = mne.create_info(
        ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"],
        sfreq=250.0,
        ch_types="eeg",
    )
    # Build a fake Evoked
    data = clean[:, :3000].copy() * 1e-6
    evoked = mne.EvokedArray(data, info, tmin=0.0, verbose=False)
    asr = ASR(sfreq=250.0, cutoff=20.0, picks=None, verbose=False)
    with pytest.raises(ValueError, match="Evoked"):
        asr.fit(evoked)
