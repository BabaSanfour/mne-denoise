"""Python-side tests for the MW-ASR variant added in the AASR sprint.

These tests do NOT have MATLAB oracle fixtures (would require generating
``aasr_case_*_mw_*.mat`` against the MATLAB ``AASR_demo.ipynb`` Cell 4
pipeline). Instead they verify the documented MW semantics:

1. When ``mw_window_length`` covers all the data, MW collapses to a single
   PSP-style calibration on that data: state.M / T match PSP exactly.
2. When the data spans multiple MW windows, ``mw_diagnostics_`` has one
   entry per window and the final state matches a PSP calibration on the
   *last* window only.
3. ``partial_fit`` / ``update`` are not supported for ``variant="mw"``.
4. ``mw_diagnostics_`` is empty for ``variant="psp"`` / ``"psw"`` so consumer
   code can assume the attribute is always defined.
"""

# ruff: noqa: I001

from __future__ import annotations

import numpy as np
import pytest

from mne_denoise.asr import AdaptiveASR


def _make_synthetic(
    n_channels: int = 8,
    n_samples: int = 6000,
    sfreq: float = 250.0,
    seed: int = 0,
) -> np.ndarray:
    """Generate a deterministic AASR-style synthetic stream.

    Mirrors the lighter end of ``generate_aasr_input.py``: small 10 Hz +
    6.5 Hz brain + 5% Gaussian noise + a couple of bursts so the
    calibration step actually has something to find.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / sfreq
    base = 0.6 * np.sin(2 * np.pi * 10 * t) + 0.15 * np.sin(2 * np.pi * 6.5 * t)
    X = np.empty((n_channels, n_samples), dtype=np.float64)
    for c in range(n_channels):
        phase = rng.uniform(0, 2 * np.pi)
        X[c] = (
            base
            + 0.6 * np.sin(2 * np.pi * 10 * t + phase)
            + 0.05 * rng.standard_normal(n_samples)
        )
    # Inject up to 2 short bursts so the clean-window selection has contrast.
    burst_samples = max(1, int(0.6 * sfreq))
    candidate_starts = [int(4 * sfreq), int(15 * sfreq)]
    for start in candidate_starts:
        if start + burst_samples > n_samples:
            continue  # burst window doesn't fit in this trial length
        stop = min(start + burst_samples, n_samples)
        spatial = rng.standard_normal(n_channels)
        spatial /= max(np.linalg.norm(spatial), 1e-12)
        temporal = rng.standard_normal(stop - start)
        X[:, start:stop] += 7.0 * np.outer(spatial, temporal)
    return X


def test_mw_single_window_equals_psp() -> None:
    """MW with one window covering all data ≡ PSP on that data."""
    sfreq = 250.0
    X = _make_synthetic(n_samples=4000, sfreq=sfreq, seed=11)

    mw = AdaptiveASR(
        sfreq=sfreq,
        cutoff=20.0,
        variant="mw",
        mw_window_length=X.shape[1] / sfreq + 1.0,  # one window covers all
        verbose=False,
    )
    mw.fit(X)
    psp = AdaptiveASR(sfreq=sfreq, cutoff=20.0, variant="psp", verbose=False)
    psp.fit(X)

    assert len(mw.mw_diagnostics_) == 1
    assert mw.mw_diagnostics_[0]["status"] == "passed"
    assert mw.calibration_info_["adaptive_variant"] == "mw"
    np.testing.assert_allclose(mw.M_, psp.M_, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(mw.T_, psp.T_, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(mw.thresholds_, psp.thresholds_, rtol=0.0, atol=1e-12)


def test_mw_three_windows_final_state_equals_psp_on_last_window() -> None:
    """MW final state = PSP on the *last* window only (demo Cell 4)."""
    sfreq = 250.0
    X = _make_synthetic(n_samples=6000, sfreq=sfreq, seed=22)
    win_s = (X.shape[1] / sfreq) / 3.0  # ≈ 8 s → 3 windows over 24 s

    mw = AdaptiveASR(
        sfreq=sfreq,
        cutoff=20.0,
        variant="mw",
        mw_window_length=win_s,
        verbose=False,
    )
    mw.fit(X)
    assert len(mw.mw_diagnostics_) == 3
    last_diag = mw.mw_diagnostics_[-1]
    assert last_diag["status"] == "passed"
    last_window = X[:, last_diag["window_start"] : last_diag["window_stop"]]
    psp_last = AdaptiveASR(sfreq=sfreq, cutoff=20.0, variant="psp", verbose=False)
    psp_last.fit(last_window)
    np.testing.assert_allclose(mw.M_, psp_last.M_, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(mw.T_, psp_last.T_, rtol=0.0, atol=1e-12)
    assert mw.calibration_info_["mw_n_windows"] == 3
    assert mw.calibration_info_["mw_window_length_s"] == pytest.approx(win_s)


def test_mw_partial_fit_raises_not_implemented() -> None:
    """Streaming partial_fit is disabled for MW (it re-calibrates per fit())."""
    sfreq = 250.0
    X = _make_synthetic(n_samples=4000, sfreq=sfreq, seed=33)
    mw = AdaptiveASR(
        sfreq=sfreq,
        cutoff=20.0,
        variant="mw",
        mw_window_length=8.0,
        verbose=False,
    )
    mw.fit(X)
    extra = _make_synthetic(n_samples=2000, sfreq=sfreq, seed=44)
    with pytest.raises(NotImplementedError, match="variant='mw'"):
        mw.partial_fit(extra)


def test_mw_diagnostics_empty_for_psp_psw() -> None:
    """Consumer code can rely on mw_diagnostics_ being defined as a list."""
    sfreq = 250.0
    X = _make_synthetic(n_samples=4000, sfreq=sfreq, seed=55)
    for variant in ("psp", "psw"):
        asr = AdaptiveASR(sfreq=sfreq, cutoff=20.0, variant=variant, verbose=False)
        asr.fit(X)
        assert asr.mw_diagnostics_ == []


# ============================================================================
# MW-sliding mode tests (added in the robustness sprint)
# ============================================================================


def test_mw_sliding_invalid_mw_mode_raises() -> None:
    """mw_mode must be 'final_state' or 'sliding' when variant='mw'."""
    sfreq = 250.0
    X = _make_synthetic(n_samples=4000, sfreq=sfreq, seed=66)
    asr = AdaptiveASR(
        sfreq=sfreq,
        cutoff=20.0,
        variant="mw",
        mw_window_length=8.0,
        mw_mode="bogus",
        verbose=False,
    )
    with pytest.raises(ValueError, match="mw_mode must be"):
        asr.fit(X)


def test_mw_sliding_fit_transform_returns_cleaned_shape() -> None:
    """MW-sliding fit_transform returns a cleaned array of the right shape."""
    sfreq = 250.0
    X = _make_synthetic(n_samples=6000, sfreq=sfreq, seed=77)
    asr = AdaptiveASR(
        sfreq=sfreq,
        cutoff=20.0,
        variant="mw",
        mw_window_length=8.0,
        mw_mode="sliding",
        verbose=False,
    )
    cleaned = asr.fit_transform(X)
    assert cleaned.shape == X.shape
    assert np.all(np.isfinite(cleaned))
    assert asr.calibration_info_["mw_mode"] == "sliding"
    assert asr.calibration_info_["adaptive_variant"] == "mw"


def test_mw_sliding_diagnostics_one_per_window() -> None:
    """MW-sliding mw_diagnostics_ should have one entry per processing window."""
    sfreq = 250.0
    X = _make_synthetic(n_samples=6000, sfreq=sfreq, seed=88)
    win_s = (X.shape[1] / sfreq) / 3.0  # 3 windows
    asr = AdaptiveASR(
        sfreq=sfreq,
        cutoff=20.0,
        variant="mw",
        mw_window_length=win_s,
        mw_mode="sliding",
        verbose=False,
    )
    asr.fit_transform(X)
    assert len(asr.mw_diagnostics_) == 3
    statuses = [d["status"] for d in asr.mw_diagnostics_]
    assert all(s in ("passed", "skipped_too_short", "failed") for s in statuses)


def test_mw_sliding_single_window_equals_psp_fit_transform() -> None:
    """PROOF: MW-sliding with one covering window == PSP fit_transform.

    MW-sliding calibrates each window on itself and cleans that window with
    the local calibration. With a single window covering all the data, this
    is exactly ``AdaptiveASR(variant="psp").fit_transform(data)`` — calibrate
    on all data, reconstruct all data. Since PSP is MATLAB-parity-tested
    (``test_aasr_parity.py``), this equality transitively validates the
    MW-sliding per-window calibrate-and-clean machinery.
    """
    sfreq = 250.0
    X = _make_synthetic(n_samples=5000, sfreq=sfreq, seed=123)

    mw = AdaptiveASR(
        sfreq=sfreq,
        cutoff=20.0,
        variant="mw",
        mw_window_length=X.shape[1] / sfreq + 1.0,  # one window covers all
        mw_mode="sliding",
        verbose=False,
    )
    cleaned_mw = mw.fit_transform(X)

    psp = AdaptiveASR(sfreq=sfreq, cutoff=20.0, variant="psp", verbose=False)
    cleaned_psp = psp.fit_transform(X)

    relerr = np.linalg.norm(cleaned_mw - cleaned_psp) / max(
        np.linalg.norm(cleaned_psp), np.finfo(float).eps
    )
    assert relerr < 1e-10, (
        f"MW-sliding single-window output diverged from PSP fit_transform: "
        f"relerr={relerr:.3e}"
    )


def test_mw_sliding_default_final_state_unchanged() -> None:
    """Default mw_mode='final_state' must NOT trigger sliding behavior.

    Regression guard: the sliding implementation should be opt-in only.
    """
    sfreq = 250.0
    X = _make_synthetic(n_samples=4000, sfreq=sfreq, seed=99)
    # No mw_mode specified -> default final_state
    asr = AdaptiveASR(
        sfreq=sfreq,
        cutoff=20.0,
        variant="mw",
        mw_window_length=8.0,
        verbose=False,
    )
    asr.fit(X)
    # final_state mode keeps the existing semantics (state from last window only)
    assert asr.calibration_info_.get("mw_mode", "final_state") != "sliding"
    assert "mw_n_windows" in asr.calibration_info_
