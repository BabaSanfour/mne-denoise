"""Unit tests for the MW-ASR variant of :class:`AdaptiveASR`.

MW-ASR (multi-window ASR, the AASR demo's Cell 4 pipeline) has no MATLAB
oracle fixtures, so these tests pin its documented Python semantics instead of
comparing against a reference ``.mat`` file:

* ``mw_mode="final_state"`` (default) -- each ``mw_window_length`` window is
  calibrated independently and recorded in ``mw_diagnostics_``; the final
  estimator state is the calibration of the *last* window. A single covering
  window therefore collapses to a plain PSP calibration.
* ``mw_mode="sliding"`` -- each window is calibrated on itself and cleaned with
  that local calibration; a single covering window equals
  ``AdaptiveASR(variant="psp").fit_transform``.

Because PSP is MATLAB-parity-tested in ``test_aasr_parity.py``, the
equivalences asserted below transitively validate the MW machinery.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne_denoise.asr import AdaptiveASR

SFREQ = 250.0


def _epochs(n_epochs=3, n_per=2000):
    mne = pytest.importorskip("mne")
    X = _eeg(n_times=n_epochs * n_per, bursts=6)
    info = mne.create_info([f"EEG{i:02d}" for i in range(8)], SFREQ, "eeg")
    data = X.reshape(8, n_epochs, n_per).transpose(1, 0, 2) * 1e-6
    return mne.EpochsArray(data, info, verbose=False)


# =============================================================================
# Synthetic data
# =============================================================================


def _make_synthetic(
    n_channels: int = 8,
    n_samples: int = 6000,
    sfreq: float = 250.0,
    seed: int = 0,
) -> np.ndarray:
    """Generate a deterministic AASR-style synthetic stream.

    Mirrors the lighter end of ``generate_aasr_input.py``: a small 10 Hz +
    6.5 Hz brain background, 5% Gaussian sensor noise, and up to two short
    spatial bursts so the clean-window calibration has contrast to find.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / sfreq
    base = 0.6 * np.sin(2 * np.pi * 10 * t) + 0.15 * np.sin(2 * np.pi * 6.5 * t)

    data = np.empty((n_channels, n_samples), dtype=np.float64)
    for c in range(n_channels):
        phase = rng.uniform(0, 2 * np.pi)
        data[c] = (
            base
            + 0.6 * np.sin(2 * np.pi * 10 * t + phase)
            + 0.05 * rng.standard_normal(n_samples)
        )

    # Inject up to two short bursts; skip any that do not fit the trial length.
    burst_samples = max(1, int(0.6 * sfreq))
    for start in (int(4 * sfreq), int(15 * sfreq)):
        if start + burst_samples > n_samples:
            continue
        stop = min(start + burst_samples, n_samples)
        spatial = rng.standard_normal(n_channels)
        spatial /= max(np.linalg.norm(spatial), 1e-12)
        temporal = rng.standard_normal(stop - start)
        data[:, start:stop] += 7.0 * np.outer(spatial, temporal)
    return data


# =============================================================================
# MW final-state mode (default)
# =============================================================================


def test_mw_single_window_equals_psp():
    """One window covering all data should reproduce PSP exactly."""
    sfreq = 250.0
    data = _make_synthetic(n_samples=4000, sfreq=sfreq, seed=11)

    # mw_window_length longer than the recording -> a single calibration window.
    mw = AdaptiveASR(
        sfreq=sfreq,
        cutoff=20.0,
        variant="mw",
        mw_window_length=data.shape[1] / sfreq + 1.0,
        verbose=False,
    )
    mw.fit(data)

    psp = AdaptiveASR(sfreq=sfreq, cutoff=20.0, variant="psp", verbose=False)
    psp.fit(data)

    assert len(mw.mw_diagnostics_) == 1
    assert mw.mw_diagnostics_[0]["status"] == "passed"
    assert mw.calibration_info_["adaptive_variant"] == "mw"
    # State is byte-for-byte identical: same data, same single calibration.
    assert_allclose(mw.M_, psp.M_, rtol=0.0, atol=1e-12)
    assert_allclose(mw.T_, psp.T_, rtol=0.0, atol=1e-12)
    assert_allclose(mw.thresholds_, psp.thresholds_, rtol=0.0, atol=1e-12)


def test_mw_three_windows_final_state_equals_psp_on_last_window():
    """Final state equals PSP on the last window only (demo Cell 4)."""
    sfreq = 250.0
    data = _make_synthetic(n_samples=6000, sfreq=sfreq, seed=22)
    win_s = (data.shape[1] / sfreq) / 3.0  # ~8 s -> 3 windows over 24 s

    mw = AdaptiveASR(
        sfreq=sfreq,
        cutoff=20.0,
        variant="mw",
        mw_window_length=win_s,
        verbose=False,
    )
    mw.fit(data)

    assert len(mw.mw_diagnostics_) == 3
    last_diag = mw.mw_diagnostics_[-1]
    assert last_diag["status"] == "passed"

    # Re-calibrate PSP on the last window alone and compare against MW state.
    last_window = data[:, last_diag["window_start"] : last_diag["window_stop"]]
    psp_last = AdaptiveASR(sfreq=sfreq, cutoff=20.0, variant="psp", verbose=False)
    psp_last.fit(last_window)

    assert_allclose(mw.M_, psp_last.M_, rtol=0.0, atol=1e-12)
    assert_allclose(mw.T_, psp_last.T_, rtol=0.0, atol=1e-12)
    assert mw.calibration_info_["mw_n_windows"] == 3
    assert mw.calibration_info_["mw_window_length_s"] == pytest.approx(win_s)


def test_mw_partial_fit_raises_not_implemented():
    """partial_fit is disabled for MW, which re-calibrates per fit() call."""
    sfreq = 250.0
    data = _make_synthetic(n_samples=4000, sfreq=sfreq, seed=33)
    mw = AdaptiveASR(
        sfreq=sfreq,
        cutoff=20.0,
        variant="mw",
        mw_window_length=8.0,
        verbose=False,
    )
    mw.fit(data)

    extra = _make_synthetic(n_samples=2000, sfreq=sfreq, seed=44)
    with pytest.raises(NotImplementedError, match="variant='mw'"):
        mw.partial_fit(extra)


def test_mw_diagnostics_empty_for_psp_psw():
    """mw_diagnostics_ is always defined (empty list) for non-MW variants."""
    sfreq = 250.0
    data = _make_synthetic(n_samples=4000, sfreq=sfreq, seed=55)
    for variant in ("psp", "psw"):
        asr = AdaptiveASR(sfreq=sfreq, cutoff=20.0, variant=variant, verbose=False)
        asr.fit(data)
        assert asr.mw_diagnostics_ == []


# =============================================================================
# MW sliding mode (robustness sprint addition)
# =============================================================================


def test_mw_sliding_invalid_mw_mode_raises():
    """mw_mode must be 'final_state' or 'sliding' when variant='mw'."""
    sfreq = 250.0
    data = _make_synthetic(n_samples=4000, sfreq=sfreq, seed=66)
    asr = AdaptiveASR(
        sfreq=sfreq,
        cutoff=20.0,
        variant="mw",
        mw_window_length=8.0,
        mw_mode="bogus",
        verbose=False,
    )
    with pytest.raises(ValueError, match="mw_mode must be"):
        asr.fit(data)


def test_mw_sliding_fit_transform_returns_cleaned_shape():
    """Sliding fit_transform returns a finite cleaned array of input shape."""
    sfreq = 250.0
    data = _make_synthetic(n_samples=6000, sfreq=sfreq, seed=77)
    asr = AdaptiveASR(
        sfreq=sfreq,
        cutoff=20.0,
        variant="mw",
        mw_window_length=8.0,
        mw_mode="sliding",
        verbose=False,
    )
    cleaned = asr.fit_transform(data)

    assert cleaned.shape == data.shape
    assert np.all(np.isfinite(cleaned))
    assert asr.calibration_info_["mw_mode"] == "sliding"
    assert asr.calibration_info_["adaptive_variant"] == "mw"


def test_mw_sliding_diagnostics_one_per_window():
    """Sliding mode records one mw_diagnostics_ entry per processing window."""
    sfreq = 250.0
    data = _make_synthetic(n_samples=6000, sfreq=sfreq, seed=88)
    win_s = (data.shape[1] / sfreq) / 3.0  # 3 windows
    asr = AdaptiveASR(
        sfreq=sfreq,
        cutoff=20.0,
        variant="mw",
        mw_window_length=win_s,
        mw_mode="sliding",
        verbose=False,
    )
    asr.fit_transform(data)

    assert len(asr.mw_diagnostics_) == 3
    statuses = [d["status"] for d in asr.mw_diagnostics_]
    assert all(s in ("passed", "skipped_too_short", "failed") for s in statuses)


def test_mw_sliding_single_window_equals_psp_fit_transform():
    """One covering window in sliding mode equals PSP fit_transform.

    Sliding mode calibrates each window on itself and cleans that window with
    the local calibration. With a single window covering all the data this is
    exactly ``AdaptiveASR(variant="psp").fit_transform(data)`` -- calibrate on
    all data, reconstruct all data. PSP is MATLAB-parity-tested, so this
    equality transitively validates the per-window calibrate-and-clean path.
    """
    sfreq = 250.0
    data = _make_synthetic(n_samples=5000, sfreq=sfreq, seed=123)

    mw = AdaptiveASR(
        sfreq=sfreq,
        cutoff=20.0,
        variant="mw",
        mw_window_length=data.shape[1] / sfreq + 1.0,  # one covering window
        mw_mode="sliding",
        verbose=False,
    )
    cleaned_mw = mw.fit_transform(data)

    psp = AdaptiveASR(sfreq=sfreq, cutoff=20.0, variant="psp", verbose=False)
    cleaned_psp = psp.fit_transform(data)

    relerr = np.linalg.norm(cleaned_mw - cleaned_psp) / max(
        np.linalg.norm(cleaned_psp), np.finfo(float).eps
    )
    assert relerr < 1e-10, (
        f"sliding single-window output diverged from PSP fit_transform: "
        f"relerr={relerr:.3e}"
    )


def test_mw_sliding_default_final_state_unchanged():
    """Default mw_mode stays 'final_state' (sliding is opt-in).

    Regression guard so the sliding implementation never silently becomes the
    default behavior.
    """
    sfreq = 250.0
    data = _make_synthetic(n_samples=4000, sfreq=sfreq, seed=99)

    # No mw_mode specified -> default final_state.
    asr = AdaptiveASR(
        sfreq=sfreq,
        cutoff=20.0,
        variant="mw",
        mw_window_length=8.0,
        verbose=False,
    )
    asr.fit(data)

    assert asr.calibration_info_.get("mw_mode", "final_state") != "sliding"
    assert "mw_n_windows" in asr.calibration_info_


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


def test_adaptive_max_dims_zero_is_identity():
    X = _eeg()
    aasr = AdaptiveASR(
        sfreq=SFREQ, variant="psp", max_dims=0.0, picks=None, verbose=False
    )
    cleaned = np.asarray(aasr.fit_transform(X))
    np.testing.assert_allclose(cleaned, X, atol=1e-9)


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


@pytest.mark.parametrize("length_s", [5.0, 20.0, 40.0])
@pytest.mark.parametrize("mode", ["final_state", "sliding"])
def test_mw_window_length_parametric(length_s, mode):
    clean = _make_clean_eeg(duration_s=120.0)
    dirty = _inject_bursts(clean, n_bursts=20, sfreq=250.0)
    asr = AdaptiveASR(
        sfreq=250.0,
        cutoff=20.0,
        variant="mw",
        mw_window_length=length_s,
        mw_mode=mode,
        picks=None,
        verbose=False,
    )
    if mode == "sliding":
        cleaned = asr.fit_transform(dirty)
    else:
        asr.fit(dirty)
        cleaned = asr.transform(dirty)
    assert cleaned.shape == dirty.shape
    assert np.all(np.isfinite(cleaned))
    # Number of windows should be > 0
    assert len(asr.mw_diagnostics_) > 0
