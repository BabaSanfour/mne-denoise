from __future__ import annotations

import numpy as np

from mne_denoise import zapline_plus


def test_zapline_plus_preserves_shape() -> None:
    """Calling zapline_plus on synthetic data should preserve array shape."""
    rng = np.random.default_rng(42)
    fs = 1000.0
    duration = 1.0
    t = np.arange(0, duration, 1 / fs)
    base = rng.standard_normal((t.size, 4))
    line = 0.3 * np.sin(2 * np.pi * 50 * t)[:, None]
    data = base + line

    cleaned, *_ = zapline_plus(data, fs, noisefreqs="line", plotResults=False)

    assert cleaned.shape == data.shape
    assert np.isfinite(cleaned).all()
