"""Minimal example demonstrating zapline_plus on synthetic data."""

from __future__ import annotations

import numpy as np

from mne_denoise import zapline_plus


def main() -> None:
    rng = np.random.default_rng(0)
    fs = 1000.0
    duration = 2.0
    t = np.arange(0, duration, 1 / fs)
    noise = 0.2 * np.sin(2 * np.pi * 50 * t)[:, None]
    data = rng.standard_normal((t.size, 8)) + noise

    cleaned, config, analytics, _figs = zapline_plus(
        data,
        fs,
        noisefreqs="line",
        adaptiveNremove=True,
        plotResults=False,
    )

    print("Input shape:", data.shape)
    print("Output shape:", cleaned.shape)
    print("Frequencies processed:", [info["freq"] for info in analytics.values()])
    print("Configuration snippet:", {k: config[k] for k in ("adaptiveNremove", "fixedNremove")})


if __name__ == "__main__":
    main()
