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
