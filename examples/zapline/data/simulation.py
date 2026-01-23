"""
ZapLine Paper Simulation Generator.

Implements the exact simulation procedure described in:
    de Cheveigné, A. (2020). ZapLine: A simple and effective method to remove
    power line artifacts. NeuroImage, 207, 116356.

Paper Description (verbatim from Section 2.4):
    - "signal": 10000×100 matrix of normally distributed random samples
    - "line noise": sinusoid, half-wave rectified, raised to power 3,
      multiplied by a 1×100 random matrix (spatial topography)
    - Signal and noise scaled to approximately the same power

Note: The paper does NOT specify sampling rate or line frequency explicitly
for the simulation. We provide reasonable defaults (500 Hz, 50 Hz) that
can be overridden.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ZapLinePaperSimulation:
    """Container for paper-faithful simulation data.

    Attributes
    ----------
    signal : np.ndarray
        Pure signal component (n_times, n_channels).
    line_noise : np.ndarray
        Pure line noise component (n_times, n_channels).
    data : np.ndarray
        Combined signal + line_noise, matched in power (n_times, n_channels).
    signal_topography : np.ndarray
        Spatial pattern of signal (n_channels,).
    noise_topography : np.ndarray
        Spatial pattern of line noise (n_channels,).
    sfreq : float
        Sampling frequency.
    line_freq : float
        Line frequency.
    n_times : int
        Number of time samples.
    n_channels : int
        Number of channels.
    snr_db : float
        Signal-to-noise ratio in dB (0 dB = equal power).
    """

    signal: np.ndarray
    line_noise: np.ndarray
    data: np.ndarray
    signal_topography: np.ndarray
    noise_topography: np.ndarray
    sfreq: float
    line_freq: float
    n_times: int
    n_channels: int
    snr_db: float


def generate_zapline_paper_simulation(
    n_times: int = 10000,
    n_channels: int = 100,
    sfreq: float = 500.0,
    line_freq: float = 50.0,
    snr_db: float = 0.0,
    random_state: int | None = None,
) -> ZapLinePaperSimulation:
    """Generate synthetic data matching the ZapLine paper simulation.

    This implements the exact procedure from de Cheveigné (2020), Section 2.4:

    1. Signal: Gaussian random noise (broadband) with random spatial topography
    2. Line noise: sinusoid → half-wave rectify → power of 3 → random topography
    3. Scale both to specified SNR (paper used ~0 dB, i.e., equal power)

    The half-wave rectified + cubic sinusoid creates a waveform that is:
    - Periodic at the line frequency
    - Rich in harmonics (due to nonlinear distortion)
    - Representative of real line noise with harmonic structure

    Parameters
    ----------
    n_times : int
        Number of time samples. Paper uses 10000.
    n_channels : int
        Number of channels. Paper uses 100.
    sfreq : float
        Sampling frequency in Hz. Paper does not specify; 500 Hz is reasonable.
    line_freq : float
        Power line frequency in Hz. 50 Hz (Europe) or 60 Hz (Americas).
    snr_db : float
        Signal-to-noise ratio in dB. 0 dB means equal power (paper specification).
        Positive values = more signal, negative values = more noise.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ZapLinePaperSimulation
        Dataclass containing signal, noise, combined data, and metadata.

    Examples
    --------
    >>> from mne_denoise.examples.zapline.data.simulation import (
    ...     generate_zapline_paper_simulation,
    ... )
    >>> sim = generate_zapline_paper_simulation(random_state=42)
    >>> print(f"Data shape: {sim.data.shape}")
    Data shape: (10000, 100)
    >>> print(f"SNR: {sim.snr_db} dB")
    SNR: 0.0 dB

    Notes
    -----
    The paper demonstrates ZapLine with d=1 (remove 1 component) on this data,
    reflecting the single spatial source of line noise.

    References
    ----------
    de Cheveigné, A. (2020). ZapLine: A simple and effective method to remove
    power line artifacts. NeuroImage, 207, 116356.
    """
    rng = np.random.RandomState(random_state)

    # Time vector
    t = np.arange(n_times) / sfreq

    # === 1. Create signal: Gaussian random noise (broadband) ===
    # Paper: "10000×100 matrix of normally distributed random samples"
    signal_source = rng.randn(n_times)
    signal_topography = rng.randn(n_channels)
    signal = np.outer(signal_source, signal_topography)  # (n_times, n_channels)

    # === 2. Create line noise: sinusoid → half-wave rectify → cube ===
    # Paper: "sinusoid, half-wave rectified, raised to power 3"
    sinusoid = np.sin(2 * np.pi * line_freq * t)
    half_wave_rectified = np.maximum(sinusoid, 0)  # Keep only positive half
    line_source = half_wave_rectified**3  # Cubic nonlinearity

    # Paper: "multiplied by a 1×100 random matrix" (spatial topography)
    noise_topography = rng.randn(n_channels)
    line_noise = np.outer(line_source, noise_topography)  # (n_times, n_channels)

    # === 3. Scale to specified SNR ===
    # Paper: "scaled to approximately the same power" (i.e., 0 dB SNR)

    # Compute RMS power
    signal_power = np.sqrt(np.mean(signal**2))
    noise_power = np.sqrt(np.mean(line_noise**2))

    # Scale noise to achieve target SNR
    # SNR_dB = 10 * log10(P_signal / P_noise)
    # P_noise_target = P_signal / 10^(SNR_dB / 10)
    target_noise_power = signal_power / (10 ** (snr_db / 20))
    line_noise_scaled = line_noise * (target_noise_power / noise_power)

    # Combine
    data = signal + line_noise_scaled

    return ZapLinePaperSimulation(
        signal=signal,
        line_noise=line_noise_scaled,
        data=data,
        signal_topography=signal_topography,
        noise_topography=noise_topography,
        sfreq=sfreq,
        line_freq=line_freq,
        n_times=n_times,
        n_channels=n_channels,
        snr_db=snr_db,
    )


def compute_snr(
    cleaned: np.ndarray,
    ground_truth: np.ndarray,
    noise_removed: np.ndarray,
) -> tuple[float, float, float]:
    """Compute signal quality metrics for cleaned data.

    Parameters
    ----------
    cleaned : np.ndarray
        ZapLine-cleaned data.
    ground_truth : np.ndarray
        Original signal (without noise).
    noise_removed : np.ndarray
        Difference: original_noisy - cleaned.

    Returns
    -------
    correlation : float
        Correlation between cleaned and ground truth.
    snr_improvement_db : float
        Improvement in SNR (dB).
    rmse : float
        RMS error between cleaned and ground truth.
    """
    # Flatten for global metrics
    cleaned_flat = cleaned.ravel()
    gt_flat = ground_truth.ravel()
    noise_flat = noise_removed.ravel()

    # Correlation
    correlation = np.corrcoef(cleaned_flat, gt_flat)[0, 1]

    # RMSE
    rmse = np.sqrt(np.mean((cleaned_flat - gt_flat) ** 2))

    # SNR improvement: how much noise power was removed
    signal_power = np.mean(gt_flat**2)
    noise_power = np.mean(noise_flat**2)
    if noise_power > 0:
        snr_improvement_db = 10 * np.log10(signal_power / noise_power)
    else:
        snr_improvement_db = np.inf

    return correlation, snr_improvement_db, rmse


if __name__ == "__main__":
    # Demo
    import matplotlib.pyplot as plt
    from scipy import signal as sig

    print("ZapLine Paper Simulation Demo")
    print("=" * 50)

    sim = generate_zapline_paper_simulation(random_state=42)

    print(f"Data shape: {sim.data.shape}")
    print(f"Sampling rate: {sim.sfreq} Hz")
    print(f"Line frequency: {sim.line_freq} Hz")
    print(f"SNR: {sim.snr_db} dB")

    # Plot PSD to show harmonics in line noise
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Signal PSD (should be flat - white noise)
    freqs, psd_signal = sig.welch(sim.signal[:, 0], sim.sfreq, nperseg=512)
    axes[0].semilogy(freqs, psd_signal)
    axes[0].set_title("Signal PSD (Broadband Noise)")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("PSD")

    # Line noise PSD (should show line freq + harmonics)
    freqs, psd_noise = sig.welch(sim.line_noise[:, 0], sim.sfreq, nperseg=512)
    axes[1].semilogy(freqs, psd_noise)
    axes[1].set_title("Line Noise PSD (Harmonic Rich)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].axvline(sim.line_freq, color="r", linestyle="--", alpha=0.7, label="f0")
    axes[1].axvline(2 * sim.line_freq, color="r", linestyle=":", alpha=0.5, label="2f0")
    axes[1].axvline(3 * sim.line_freq, color="r", linestyle=":", alpha=0.3, label="3f0")
    axes[1].legend()

    # Combined PSD
    freqs, psd_data = sig.welch(sim.data[:, 0], sim.sfreq, nperseg=512)
    axes[2].semilogy(freqs, psd_data)
    axes[2].set_title("Combined Data PSD")
    axes[2].set_xlabel("Frequency (Hz)")

    plt.tight_layout()
    plt.savefig("zapline_paper_simulation.png", dpi=150)
    print("\nSaved: zapline_paper_simulation.png")
    plt.show()
