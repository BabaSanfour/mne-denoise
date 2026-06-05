"""Targeted coverage tests for ASR validation, accessors, and standalone helpers.

These complement the parity / behavior suites by exercising the error branches,
the unified accessor API, the standalone module functions, and the less-common
input / memory paths that the happy-path tests do not reach.
"""

from __future__ import annotations

import numpy as np
import pytest

from mne_denoise.asr import (
    ASR,
    AdaptiveASR,
    JugglerASR,
    calibrate_asr,
    compute_asr_qa_metrics,
    compute_asr_rejection_mask,
    fit_eeg_distribution,
    process_asr,
)

SFREQ = 250.0


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


# ---------------------------------------------------------------------------
# calibrate_asr / ASR parameter validation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Unified accessor error paths
# ---------------------------------------------------------------------------


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


def test_juggler_to_annotations_calibration_ok():
    mne = pytest.importorskip("mne")
    j = JugglerASR(
        sfreq=SFREQ, cutoff=20.0, strategy="dbscan", picks=None, verbose=False
    )
    j.fit_transform(_eeg())
    ann = j.to_annotations("calibration")
    assert isinstance(ann, mne.Annotations)


def test_get_diagnostics_present_on_all_variants():
    for est in (
        ASR(sfreq=SFREQ, picks=None, verbose=False),
        AdaptiveASR(sfreq=SFREQ, variant="psp", picks=None, verbose=False),
        JugglerASR(sfreq=SFREQ, strategy="gev", picks=None, verbose=False),
    ):
        est.fit_transform(_eeg())
        assert isinstance(est.get_diagnostics(), dict)


# ---------------------------------------------------------------------------
# window_criterion rejection path (ASR)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Memory-bounded (rolling covariance) paths
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Standalone module functions
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Alternate input types (Epochs) for the subclasses
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Juggler error guards
# ---------------------------------------------------------------------------


def test_juggler_min_reference_fraction_floor_raises():
    from mne_denoise.asr import select_juggler_reference_samples

    X = _eeg(bursts=40)  # heavily contaminated
    with pytest.raises(RuntimeError, match="retained too little"):
        select_juggler_reference_samples(
            X, SFREQ, strategy="gev", min_reference_fraction=0.99
        )


def test_juggler_invalid_strategy_raises():
    with pytest.raises(ValueError, match="strategy must be"):
        JugglerASR(sfreq=SFREQ, strategy="bogus", picks=None, verbose=False).fit(_eeg())


# ---------------------------------------------------------------------------
# Epochs + window_criterion (rejection path inside the per-epoch transform)
# ---------------------------------------------------------------------------


def _epochs(n_epochs=3, n_per=2000):
    mne = pytest.importorskip("mne")
    X = _eeg(n_times=n_epochs * n_per, bursts=6)
    info = mne.create_info([f"EEG{i:02d}" for i in range(8)], SFREQ, "eeg")
    data = X.reshape(8, n_epochs, n_per).transpose(1, 0, 2) * 1e-6
    return mne.EpochsArray(data, info, verbose=False)


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


def test_adaptive_window_criterion_on_epochs_populates_rejection():
    epo = _epochs()
    aasr = AdaptiveASR(
        sfreq=SFREQ,
        cutoff=10.0,
        variant="psp",
        window_criterion=0.3,
        window_criterion_tolerances=(-np.inf, 5.0),
        verbose=False,
    )
    out = aasr.fit_transform(epo)
    assert out.get_data().shape == epo.get_data().shape
    diag = aasr.get_diagnostics()
    assert "rejection_sample_mask" in diag


# ---------------------------------------------------------------------------
# AdaptiveASR MW parameter validation
# ---------------------------------------------------------------------------


def test_adaptive_mw_bad_window_length_raises():
    with pytest.raises(ValueError, match="mw_window_length"):
        AdaptiveASR(
            sfreq=SFREQ,
            variant="mw",
            mw_window_length=-1.0,
            picks=None,
            verbose=False,
        ).fit(_eeg())


def test_adaptive_mw_bad_mode_raises():
    with pytest.raises(ValueError, match="mw_mode"):
        AdaptiveASR(
            sfreq=SFREQ,
            variant="mw",
            mw_mode="bogus",
            picks=None,
            verbose=False,
        ).fit(_eeg())


# ---------------------------------------------------------------------------
# JugglerASR fit guards + DBSCAN parameter resolution
# ---------------------------------------------------------------------------


def test_juggler_evoked_calibration_raises():
    mne = pytest.importorskip("mne")
    info = mne.create_info([f"EEG{i:02d}" for i in range(8)], SFREQ, "eeg")
    evoked = mne.EvokedArray(_eeg(n_times=2000) * 1e-6, info, tmin=0.0, verbose=False)
    with pytest.raises(ValueError, match="Evoked"):
        JugglerASR(sfreq=SFREQ, picks=None, verbose=False).fit(evoked)


def test_juggler_calibration_mask_shape_raises():
    with pytest.raises(ValueError, match="calibration_mask must have shape"):
        JugglerASR(sfreq=SFREQ, picks=None, verbose=False).fit(
            _eeg(), calibration_mask=np.ones(10, dtype=bool)
        )


def test_juggler_dbscan_bad_eps_string_raises():
    from mne_denoise.asr import select_juggler_reference_samples

    with pytest.raises(ValueError, match="dbscan_eps"):
        select_juggler_reference_samples(
            _eeg(), SFREQ, strategy="dbscan", dbscan_eps="bogus"
        )


def test_juggler_dbscan_bad_min_samples_string_raises():
    from mne_denoise.asr import select_juggler_reference_samples

    with pytest.raises(ValueError, match="dbscan_min_samples"):
        select_juggler_reference_samples(
            _eeg(), SFREQ, strategy="dbscan", dbscan_min_samples="bogus"
        )


# ---------------------------------------------------------------------------
# Riemannian (non-windowed) low-memory block-covariance path
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _karcher_mean_spd direct validation guards
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Viz branches not reached by the smoke suite (montage topomap, auto-picks)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# core.py low-level validation guards
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# AdaptiveASR max_dims=0 -> identity (no-reconstruction) process path
# ---------------------------------------------------------------------------


def test_adaptive_max_dims_zero_is_identity():
    X = _eeg()
    aasr = AdaptiveASR(
        sfreq=SFREQ, variant="psp", max_dims=0.0, picks=None, verbose=False
    )
    cleaned = np.asarray(aasr.fit_transform(X))
    np.testing.assert_allclose(cleaned, X, atol=1e-9)


# ---------------------------------------------------------------------------
# JugglerASR DBSCAN numeric-eps fallback (non-positive eps -> derived eps)
# ---------------------------------------------------------------------------


def test_juggler_dbscan_numeric_eps_fallback():
    from mne_denoise.asr import select_juggler_reference_samples

    _, mask, _ = select_juggler_reference_samples(
        _eeg(bursts=8), SFREQ, strategy="dbscan", dbscan_eps=0.0
    )
    assert mask.dtype == bool


# ---------------------------------------------------------------------------
# Viz: ax-reuse (else: fig = ax.figure) + error / single-estimator branches
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Riemannian chunked covariance aggregation (memory-bounded rASR calibration)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# AdaptiveASR partial_fit calibration_mask + raw-transform window rejection
# ---------------------------------------------------------------------------


def test_adaptive_partial_fit_calibration_mask():
    X = _eeg(n_times=8000)
    aasr = AdaptiveASR(sfreq=SFREQ, variant="psp", picks=None, verbose=False)
    aasr.fit(X[:, :4000])
    with pytest.raises(ValueError, match="calibration_mask must have shape"):
        aasr.partial_fit(X[:, 4000:], calibration_mask=np.ones(10, dtype=bool))
    mask = np.ones(4000, dtype=bool)
    mask[:200] = False
    aasr.partial_fit(X[:, 4000:], calibration_mask=mask)
    assert aasr.calibration_mask_kind_ == "window"


def test_adaptive_transform_raw_with_window_criterion():
    mne = pytest.importorskip("mne")
    X = _eeg(n_times=8000, bursts=8)
    info = mne.create_info([f"EEG{i:02d}" for i in range(8)], SFREQ, "eeg")
    raw = mne.io.RawArray(X * 1e-6, info, verbose=False)
    aasr = AdaptiveASR(
        sfreq=SFREQ,
        cutoff=10.0,
        variant="psp",
        window_criterion=0.3,
        window_criterion_tolerances=(-np.inf, 5.0),
        verbose=False,
    )
    aasr.fit_transform(raw)
    diag = aasr.get_diagnostics()
    assert "rejection_sample_mask" in diag


# ---------------------------------------------------------------------------
# JugglerASR DBSCAN min_samples numeric (float-fraction + int) resolution
# ---------------------------------------------------------------------------


def test_juggler_dbscan_min_samples_numeric():
    from mne_denoise.asr import select_juggler_reference_samples

    X = _eeg(bursts=8)
    _, m1, _ = select_juggler_reference_samples(
        X, SFREQ, strategy="dbscan", dbscan_min_samples=0.05
    )
    _, m2, _ = select_juggler_reference_samples(
        X, SFREQ, strategy="dbscan", dbscan_min_samples=5
    )
    assert m1.dtype == bool and m2.dtype == bool


# ---------------------------------------------------------------------------
# Viz: fname save + overlay channel-name pick (MNE input)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# AdaptiveASR _validate_adaptive_params guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,msg",
    [
        ({"update_window_length": -1.0}, "update_window_length"),
        ({"calibration_window_length": -1.0}, "clean_window_length"),
        ({"calibration_window_overlap": 1.5}, "clean_window_overlap"),
        ({"ref_max_bad_channels": -0.1}, "clean_max_bad_channels"),
        ({"learning_rate": -1.0}, "learning_rate"),
        ({"tau": -1.0}, "tau must be positive"),
    ],
)
def test_adaptive_validate_param_guards(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        AdaptiveASR(
            sfreq=SFREQ, variant="psp", picks=None, verbose=False, **kwargs
        ).fit(_eeg())


# ---------------------------------------------------------------------------
# core picks resolution + transform channel-mismatch guards
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# store_reconstruction_matrices path (both standard and riemannian_windowed)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Small numeric helpers: _resolve_max_dims, _robust_location_scale, _max_mem_bytes
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# core window / resolve helpers (pure, direct-tested guards)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# select_juggler_reference_samples parameter guards
# ---------------------------------------------------------------------------


def test_select_juggler_reference_samples_param_guards():
    from mne_denoise.asr import select_juggler_reference_samples

    X = _eeg()
    with pytest.raises(ValueError, match="strategy must be"):
        select_juggler_reference_samples(X, SFREQ, strategy="bogus")
    with pytest.raises(ValueError, match="dbscan_top_k"):
        select_juggler_reference_samples(X, SFREQ, dbscan_top_k=0)
    with pytest.raises(ValueError, match="gev_grid_size"):
        select_juggler_reference_samples(X, SFREQ, strategy="gev", gev_grid_size=8)
    with pytest.raises(ValueError, match="min_reference_fraction"):
        select_juggler_reference_samples(X, SFREQ, min_reference_fraction=1.5)


# ---------------------------------------------------------------------------
# JugglerASR constructor-level parameter guards (_validate_juggler_params)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,msg",
    [
        ({"dbscan_top_k": 0}, "dbscan_top_k"),
        ({"gev_grid_size": 8}, "gev_grid_size"),
        ({"min_reference_fraction": 1.5}, "min_reference_fraction"),
    ],
)
def test_juggler_constructor_param_guards(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        JugglerASR(sfreq=SFREQ, picks=None, verbose=False, **kwargs).fit(_eeg())


# ---------------------------------------------------------------------------
# MEG support: ASR is unit/scale agnostic, so picks="mag"/"grad"/"meg" work
# ---------------------------------------------------------------------------


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
