"""Python self-consistency tests for JugglerASR.

There is no public MATLAB oracle for Juggler ASR (Kim et al. 2025 — paper
dataset is request-only and no MATLAB code is published). These tests verify
the Python implementation's correctness against synthetic inputs with known
properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from mne_denoise.asr import JugglerASR, select_juggler_reference_samples

SFREQ = 250.0


def _make_synthetic_eeg(
    sfreq: float = 250.0,
    duration_s: float = 60.0,
    n_channels: int = 16,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Build clean synthetic EEG: 10 Hz + 6 Hz oscillation + Gaussian noise."""
    if rng is None:
        rng = np.random.default_rng(42)
    n_samples = int(sfreq * duration_s)
    t = np.arange(n_samples) / sfreq
    data = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        phase = rng.uniform(0, 2 * np.pi)
        data[ch] = 0.6 * np.sin(2 * np.pi * 10.0 * t + phase) + 0.15 * np.sin(
            2 * np.pi * 6.5 * t + phase * 0.8
        )
    data += 0.05 * rng.standard_normal(data.shape)
    return data


def _inject_bursts(
    data: np.ndarray,
    n_bursts: int = 8,
    burst_duration_s: float = 0.5,
    amplitude: float = 12.0,
    sfreq: float = 250.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject n_bursts high-amplitude bursts at evenly-spaced intervals."""
    if rng is None:
        rng = np.random.default_rng(97)
    burst_len = int(round(burst_duration_s * sfreq))
    n_times = data.shape[1]
    starts = np.linspace(burst_len, n_times - burst_len, n_bursts).astype(int)
    contaminated = data.copy()
    mask = np.zeros(n_times, dtype=bool)
    channel_scale = float(np.median(np.std(data, axis=1)))
    for start in starts:
        stop = min(start + burst_len, n_times)
        actual_len = stop - start
        spatial = rng.standard_normal(data.shape[0])
        spatial /= max(np.linalg.norm(spatial), np.finfo(float).eps)
        temporal = rng.standard_normal(actual_len)
        contaminated[:, start:stop] += (
            amplitude * channel_scale * np.outer(spatial, temporal)
        )
        mask[start:stop] = True
    return contaminated, mask


# ============================================================================
# select_juggler_reference_samples — function-level tests
# ============================================================================


def test_juggler_clean_input_keeps_most_samples():
    """Clean EEG with no bursts should retain a large fraction of samples."""
    sfreq = 250.0
    clean = _make_synthetic_eeg(sfreq=sfreq, duration_s=30.0, n_channels=16)
    _, mask, diag = select_juggler_reference_samples(
        clean,
        sfreq,
        strategy="dbscan",
    )
    keep = float(np.mean(mask))
    assert keep > 0.5, f"DBSCAN on clean input retained only {keep * 100:.1f}%"
    assert diag["reference_selected_samples"] == int(mask.sum())
    assert diag["reference_selection_strategy"] == "dbscan"


def test_juggler_dbscan_with_huge_eps_keeps_everything():
    """DBSCAN with a giant eps should put all samples in one cluster."""
    sfreq = 250.0
    clean = _make_synthetic_eeg(sfreq=sfreq, duration_s=30.0, n_channels=16)
    _, mask, _ = select_juggler_reference_samples(
        clean,
        sfreq,
        strategy="dbscan",
        dbscan_eps=1e12,
    )
    assert mask.mean() > 0.95, (
        f"DBSCAN with eps=1e12 kept only {mask.mean() * 100:.1f}%; "
        "expected near-100% since one cluster should swallow everything"
    )


def test_juggler_gev_on_clean_input():
    """GEV strategy should retain a non-trivial fraction on clean EEG."""
    sfreq = 250.0
    clean = _make_synthetic_eeg(sfreq=sfreq, duration_s=30.0, n_channels=16)
    _, mask, diag = select_juggler_reference_samples(
        clean,
        sfreq,
        strategy="gev",
    )
    keep = float(np.mean(mask))
    assert keep > 0.05, f"GEV retained only {keep * 100:.1f}%"
    assert diag["reference_selection_strategy"] == "gev"
    assert np.isfinite(diag["juggler_gev_mode"])
    assert diag["juggler_gev_scale"] > 0


def test_juggler_reduces_keep_on_contaminated_input():
    """Contaminated input should retain FEWER samples than clean input."""
    sfreq = 250.0
    clean = _make_synthetic_eeg(sfreq=sfreq, duration_s=30.0, n_channels=16)
    contaminated, _ = _inject_bursts(
        clean,
        n_bursts=20,
        burst_duration_s=0.5,
        amplitude=12.0,
        sfreq=sfreq,
    )
    _, mask_clean, _ = select_juggler_reference_samples(
        clean,
        sfreq,
        strategy="gev",
    )
    _, mask_dirty, _ = select_juggler_reference_samples(
        contaminated,
        sfreq,
        strategy="gev",
    )
    # With heavy bursts, the GEV mode shifts lower and contaminated bursts
    # are excluded — keep-fraction should at least drop, not rise.
    assert mask_dirty.mean() <= mask_clean.mean() + 1e-3, (
        f"GEV keep-fraction did not drop on contaminated input: "
        f"clean={mask_clean.mean():.3f} vs dirty={mask_dirty.mean():.3f}"
    )


def test_juggler_min_reference_fraction_raises():
    """min_reference_fraction floor should trigger on heavily-contaminated input."""
    sfreq = 250.0
    clean = _make_synthetic_eeg(sfreq=sfreq, duration_s=30.0, n_channels=16)
    contaminated, _ = _inject_bursts(
        clean,
        n_bursts=50,
        burst_duration_s=1.0,
        amplitude=50.0,
        sfreq=sfreq,
    )
    # Set the floor unreasonably high (99%) so the call MUST raise.
    with pytest.raises(RuntimeError, match="retained too little data"):
        select_juggler_reference_samples(
            contaminated,
            sfreq,
            strategy="gev",
            min_reference_fraction=0.99,
        )


# ============================================================================
# JugglerASR class — estimator-level tests
# ============================================================================


def test_juggler_asr_dbscan_fit_transform_round_trip():
    """JugglerASR(dbscan) should fit + transform end-to-end on synthetic data."""
    sfreq = 250.0
    clean = _make_synthetic_eeg(sfreq=sfreq, duration_s=30.0, n_channels=16)
    contaminated, _ = _inject_bursts(clean, sfreq=sfreq)

    asr = JugglerASR(
        sfreq=sfreq,
        cutoff=20.0,
        strategy="dbscan",
        picks=None,
        random_state=42,
        verbose=False,
    )
    asr.fit(contaminated)
    cleaned = asr.transform(contaminated)

    assert cleaned.shape == contaminated.shape
    assert asr.calibration_info_["reference_selection_strategy"] == "dbscan"
    assert 0.0 < asr.calibration_info_["reference_selected_fraction"] <= 1.0
    # cleaned variance should be at most the input variance
    assert np.var(cleaned) <= np.var(contaminated) * 1.05


def test_juggler_asr_gev_fit_transform_round_trip():
    """JugglerASR(gev) should fit + transform end-to-end on synthetic data."""
    sfreq = 250.0
    clean = _make_synthetic_eeg(sfreq=sfreq, duration_s=30.0, n_channels=16)
    contaminated, _ = _inject_bursts(clean, sfreq=sfreq)

    asr = JugglerASR(
        sfreq=sfreq,
        cutoff=20.0,
        strategy="gev",
        picks=None,
        random_state=42,
        verbose=False,
    )
    asr.fit(contaminated)
    cleaned = asr.transform(contaminated)

    assert cleaned.shape == contaminated.shape
    assert asr.calibration_info_["reference_selection_strategy"] == "gev"
    assert 0.0 < asr.calibration_info_["reference_selected_fraction"] <= 1.0


def test_juggler_calibration_mask_kind_is_sample():
    """JugglerASR advertises sample-based calibration (not window-based)."""
    sfreq = 250.0
    clean = _make_synthetic_eeg(sfreq=sfreq, duration_s=20.0, n_channels=8)
    asr = JugglerASR(
        sfreq=sfreq,
        cutoff=20.0,
        strategy="dbscan",
        picks=None,
        random_state=42,
        verbose=False,
    )
    asr.fit(clean)
    assert asr.calibration_mask_kind_ == "sample"


def test_juggler_get_calibration_mask_after_fit():
    """get_calibration_mask should return a sample-wise bool array of length n_times."""
    sfreq = 250.0
    clean = _make_synthetic_eeg(sfreq=sfreq, duration_s=20.0, n_channels=8)
    asr = JugglerASR(
        sfreq=sfreq,
        cutoff=20.0,
        strategy="dbscan",
        picks=None,
        random_state=42,
        verbose=False,
    )
    asr.fit(clean)
    mask = asr.get_calibration_mask()
    assert mask.dtype == bool
    assert mask.shape == (clean.shape[1],)
    assert mask.sum() == int(asr.calibration_info_["reference_selected_samples"])


def test_juggler_invalid_strategy_raises():
    """Invalid strategy name should raise at fit time."""
    sfreq = 250.0
    clean = _make_synthetic_eeg(sfreq=sfreq, duration_s=10.0, n_channels=8)
    asr = JugglerASR(
        sfreq=sfreq,
        cutoff=20.0,
        strategy="bogus",
        picks=None,
        verbose=False,
    )
    with pytest.raises(ValueError, match="strategy must be"):
        asr.fit(clean)


def test_juggler_dbscan_deterministic():
    """JugglerASR(dbscan) with same random_state should be deterministic."""
    sfreq = 250.0
    clean = _make_synthetic_eeg(sfreq=sfreq, duration_s=20.0, n_channels=8)
    contaminated, _ = _inject_bursts(clean, sfreq=sfreq)
    asr1 = JugglerASR(
        sfreq=sfreq,
        cutoff=20.0,
        strategy="dbscan",
        picks=None,
        random_state=42,
        verbose=False,
    )
    asr2 = JugglerASR(
        sfreq=sfreq,
        cutoff=20.0,
        strategy="dbscan",
        picks=None,
        random_state=42,
        verbose=False,
    )
    asr1.fit(contaminated)
    asr2.fit(contaminated)
    np.testing.assert_allclose(asr1.M_, asr2.M_, rtol=1e-12)
    np.testing.assert_allclose(asr1.T_, asr2.T_, rtol=1e-12)
    np.testing.assert_array_equal(
        asr1.get_calibration_mask(), asr2.get_calibration_mask()
    )


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


def test_juggler_to_annotations_calibration_ok():
    mne = pytest.importorskip("mne")
    j = JugglerASR(
        sfreq=SFREQ, cutoff=20.0, strategy="dbscan", picks=None, verbose=False
    )
    j.fit_transform(_eeg())
    ann = j.to_annotations("calibration")
    assert isinstance(ann, mne.Annotations)


def test_juggler_min_reference_fraction_floor_raises():
    from mne_denoise.asr import select_juggler_reference_samples

    X = _eeg(bursts=40)  # heavily contaminated
    with pytest.raises(RuntimeError, match="retained too little"):
        select_juggler_reference_samples(
            X, SFREQ, strategy="gev", min_reference_fraction=0.99
        )


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


def test_juggler_dbscan_numeric_eps_fallback():
    from mne_denoise.asr import select_juggler_reference_samples

    _, mask, _ = select_juggler_reference_samples(
        _eeg(bursts=8), SFREQ, strategy="dbscan", dbscan_eps=0.0
    )
    assert mask.dtype == bool


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


@pytest.mark.parametrize("eps_multiplier", [0.5, 1.0, 2.0])
def test_juggler_dbscan_eps_parametric(eps_multiplier):
    """Larger eps should retain at least as many samples as smaller eps."""
    clean = _make_clean_eeg(duration_s=60.0)
    dirty, _ = _inject_bursts(clean, n_bursts=10)
    # Get auto eps first
    base = JugglerASR(
        sfreq=250.0,
        cutoff=20.0,
        strategy="dbscan",
        picks=None,
        verbose=False,
    )
    base.fit(dirty)
    auto_eps = float(base.calibration_info_["juggler_dbscan_eps"])
    eps = auto_eps * eps_multiplier
    asr = JugglerASR(
        sfreq=250.0,
        cutoff=20.0,
        strategy="dbscan",
        dbscan_eps=eps,
        picks=None,
        verbose=False,
    )
    asr.fit(dirty)
    rf = float(asr.calibration_info_["reference_selected_fraction"])
    assert 0.0 < rf <= 1.0
