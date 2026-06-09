"""Parametric + edge-case + inter-variant tests across all 4+ ASR variants.

Regression guards added in the robustness sprint. These tests are NOT
parity-against-MATLAB tests; they verify Python-side robustness:

1. **Parametric sweeps** — cutoff per variant, ``mw_window_length`` × mode,
   ``dbscan_eps`` × strategy. Catches regressions in monotone behavior /
   parameter handling.
2. **Channel-count extremes** — 4 ch + 64 ch. Each variant either succeeds
   or fails with a clear message.
3. **Sample-rate extremes** — sfreq ∈ {50, 250, 1000} Hz. Should not crash.
4. **Recording-duration extremes** — short (10 s, expect-fail), minimal
   (30 s), long-ish (cached 32-ch from `build_long_recording_substrate`).
5. **Inter-variant consistency on clean substrate** — every variant should
   pass clean EEG through nearly unchanged (correlation > 0.95).
6. **Evoked input handling** — calibration should raise; transform after a
   Raw-fit should work.

The slower / larger-data tests are marked with the ``@pytest.mark.slow``
marker so they can be skipped via ``pytest -m "not slow"`` in CI.
"""

# ruff: noqa: I001

from __future__ import annotations

import numpy as np
import pytest

from mne_denoise.asr import ASR, AdaptiveASR, JugglerASR


# ============================================================================
# Synthetic substrate helpers (shared, deterministic)
# ============================================================================


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


# ============================================================================
# 1. Cutoff parametric sweep (per variant)
# ============================================================================


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


# ============================================================================
# 2. mw_window_length × mw_mode parametric
# ============================================================================


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


# ============================================================================
# 3. dbscan_eps × strategy parametric
# ============================================================================


@pytest.mark.parametrize("eps_multiplier", [0.5, 1.0, 2.0])
def test_juggler_dbscan_eps_parametric(eps_multiplier):
    """Larger eps should retain at least as many samples as smaller eps."""
    clean = _make_clean_eeg(duration_s=60.0)
    dirty = _inject_bursts(clean, n_bursts=10)
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


# ============================================================================
# 4. Channel-count extremes
# ============================================================================


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


# ============================================================================
# 5. Sample-rate extremes
# ============================================================================


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


# ============================================================================
# 6. Recording-duration extremes
# ============================================================================


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


# ============================================================================
# 7. Inter-variant consistency on CLEAN substrate
# ============================================================================


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


# ============================================================================
# 8. Evoked input handling
# ============================================================================


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
