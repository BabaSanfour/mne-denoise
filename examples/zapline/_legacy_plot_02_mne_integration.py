"""
ZapLine: MNE Sample EEG/MEG Data Demo.
=====================================

This example demonstrates ZapLine on real EEG/MEG data from MNE's sample dataset.
We show how to integrate ZapLine with MNE-Python workflows.

The algorithm removes power line artifacts while preserving:
- Full frequency bandwidth (unlike lowpass filtering)
- Full data rank (unlike ICA-based methods)
- Minimal distortion at non-artifact frequencies

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

Reference
---------
de Cheveigné, A. (2020). ZapLine: A simple and effective method to remove
power line artifacts. NeuroImage, 207, 116356.
"""

# %%
# Imports
# -------
# Add parent to path for development
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, r"D:\PhD\mne-denoise")

# Check if MNE is available
try:
    import mne
    from mne.datasets import sample

    HAS_MNE = True
    print(f"MNE version: {mne.__version__}")
except ImportError:
    HAS_MNE = False
    print("MNE not installed. This example requires MNE-Python.")
    print("Install with: pip install mne")

from mne_denoise import compute_psd_reduction, dss_zapline, zapline_plus

# %%
# Load MNE Sample Data
# --------------------

if HAS_MNE:
    # Load sample data
    data_path = sample.data_path()
    raw_fname = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"

    # Read raw data (just first 60 seconds for speed)
    raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
    raw.crop(tmax=60)

    print("\nData info:")
    print(f"  Channels: {len(raw.ch_names)}")
    print(f"  Sample rate: {raw.info['sfreq']} Hz")
    print(f"  Duration: {raw.times[-1]:.1f} s")
    print(f"  Line frequency: {raw.info['line_freq']} Hz")

# %%
# Apply ZapLine to MEG Gradiometers
# ---------------------------------

if HAS_MNE:
    # Pick MEG gradiometers (more sensitive to line noise)
    raw_meg = raw.copy().pick_types(meg="grad", eeg=False, exclude="bads")

    # Get data array (channels x times)
    data = raw_meg.get_data()
    sfreq = raw_meg.info["sfreq"]
    line_freq = raw.info["line_freq"] or 60.0  # Default to 60 Hz if not set

    print("\nMEG Gradiometers:")
    print(f"  Shape: {data.shape}")
    print(f"  Removing {line_freq} Hz line noise")

    # Apply ZapLine
    result = dss_zapline(
        data,
        line_freq=line_freq,
        sfreq=sfreq,
        n_remove="auto",  # Automatic component selection
        threshold=3.0,  # Z-score threshold for outlier detection
    )

    print(f"  Components removed: {result.n_removed}")

# %%
# Compare PSD Before and After
# ----------------------------

if HAS_MNE:
    from scipy import signal as sig

    # Compute PSDs
    nperseg = int(4 * sfreq)
    freqs, psd_before = sig.welch(data, sfreq, nperseg=nperseg, axis=1)
    _, psd_after = sig.welch(result.cleaned, sfreq, nperseg=nperseg, axis=1)
    _, psd_removed = sig.welch(result.removed, sfreq, nperseg=nperseg, axis=1)

    # Average across channels
    psd_before_mean = np.mean(psd_before, axis=0)
    psd_after_mean = np.mean(psd_after, axis=0)
    psd_removed_mean = np.mean(psd_removed, axis=0)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Before
    ax = axes[0]
    ax.semilogy(freqs, psd_before_mean, "k-", linewidth=1.5, label="Original")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (T/m)²/Hz")
    ax.set_title("MEG Gradiometers - Original")
    ax.set_xlim([0, 150])
    ax.grid(True, alpha=0.3)

    # Mark line frequencies
    for h in range(1, 4):
        ax.axvline(
            line_freq * h,
            color="r",
            alpha=0.3,
            linestyle="--",
            label=f"{line_freq * h} Hz" if h == 1 else None,
        )
    ax.legend()

    # After
    ax = axes[1]
    ax.semilogy(freqs, psd_after_mean, "g-", linewidth=1.5, label="Cleaned")
    ax.semilogy(
        freqs, psd_removed_mean, "r-", linewidth=1.5, alpha=0.7, label="Removed"
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (T/m)²/Hz")
    ax.set_title(f"After ZapLine (d={result.n_removed})")
    ax.set_xlim([0, 150])
    ax.grid(True, alpha=0.3)

    # Mark line frequencies
    for h in range(1, 4):
        ax.axvline(line_freq * h, color="r", alpha=0.3, linestyle="--")
    ax.legend()

    plt.tight_layout()
    plt.savefig("zapline_mne_meg.png", dpi=150, bbox_inches="tight")
    plt.show()

# %%
# Apply ZapLine to EEG Channels
# -----------------------------

if HAS_MNE:
    # Pick EEG channels
    raw_eeg = raw.copy().pick_types(meg=False, eeg=True, exclude="bads")

    if len(raw_eeg.ch_names) > 0:
        data_eeg = raw_eeg.get_data()

        print("\nEEG Channels:")
        print(f"  Shape: {data_eeg.shape}")

        # Apply ZapLine
        result_eeg = dss_zapline(
            data_eeg,
            line_freq=line_freq,
            sfreq=sfreq,
            n_remove="auto",
        )

        print(f"  Components removed: {result_eeg.n_removed}")

        # Compute PSDs
        freqs, psd_before = sig.welch(data_eeg, sfreq, nperseg=nperseg, axis=1)
        _, psd_after = sig.welch(result_eeg.cleaned, sfreq, nperseg=nperseg, axis=1)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Before
        ax = axes[0]
        ax.semilogy(freqs, np.mean(psd_before, axis=0), "k-", linewidth=1.5)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (V²/Hz)")
        ax.set_title("EEG - Original")
        ax.set_xlim([0, 150])
        ax.grid(True, alpha=0.3)
        for h in range(1, 4):
            ax.axvline(line_freq * h, color="r", alpha=0.3, linestyle="--")

        # After
        ax = axes[1]
        ax.semilogy(freqs, np.mean(psd_after, axis=0), "g-", linewidth=1.5)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (V²/Hz)")
        ax.set_title(f"EEG - After ZapLine (d={result_eeg.n_removed})")
        ax.set_xlim([0, 150])
        ax.grid(True, alpha=0.3)
        for h in range(1, 4):
            ax.axvline(line_freq * h, color="r", alpha=0.3, linestyle="--")

        plt.tight_layout()
        plt.savefig("zapline_mne_eeg.png", dpi=150, bbox_inches="tight")
        plt.show()
    else:
        print("No EEG channels in sample data")

# %%
# Using ZapLine-Plus for Adaptive Cleaning
# ----------------------------------------
# ZapLine-Plus (Klug & Kloosterman, 2022) adds automatic chunking
# and adaptive parameter selection.

if HAS_MNE:
    print("\nApplying ZapLine-Plus (adaptive)...")

    result_plus = zapline_plus(
        data,
        sfreq,
        noisefreqs="line",  # Auto-detect 50 or 60 Hz
        adaptive_sigma=True,
        verbose=True,
    )

    print("\nZapLine-Plus Results:")
    print(f"  Detected frequency: {result_plus.config['noisefreqs']}")
    print(f"  Final sigma: {result_plus.config['noise_comp_detect_sigma']:.2f}")
    print(f"  Chunks processed: {len(result_plus.chunk_results)}")
    print(f"  Power removed: {result_plus.analytics['proportion_removed']:.2%}")

# %%
# Create MNE Raw with Cleaned Data
# --------------------------------
# Integrate the cleaned data back into MNE

if HAS_MNE:
    # Create a copy and replace data
    raw_clean = raw_meg.copy()
    raw_clean._data = result.cleaned

    print("\nCreated cleaned Raw object:")
    print(f"  {raw_clean}")

    # Compare evoked responses (if we had events)
    # This would show artifact-free averaging

# %%
# Quantitative Metrics
# --------------------

if HAS_MNE:
    metrics = compute_psd_reduction(data, result.cleaned, sfreq, line_freq)

    print("\nPower Reduction Metrics (MEG):")
    print(f"  Power at {line_freq} Hz (before): {metrics['power_original']:.2e}")
    print(f"  Power at {line_freq} Hz (after):  {metrics['power_cleaned']:.2e}")
    print(
        f"  Reduction: {metrics['reduction_db']:.1f} dB ({metrics['reduction_ratio']:.1f}x)"
    )

# %%
# Summary
# -------

print("\n" + "=" * 60)
print("ZapLine MNE Integration Demo Complete!")
print("=" * 60)

if HAS_MNE:
    print(f"""
Key Results:
  - MEG gradiometers: {result.n_removed} components removed
  - EEG channels: {result_eeg.n_removed if "result_eeg" in dir() else "N/A"} components removed
  - Power at line frequency reduced by {metrics["reduction_db"]:.1f} dB
  - ZapLine-Plus adaptive: {len(result_plus.chunk_results)} chunks processed

Integration with MNE:
  - Apply to raw.get_data() array
  - Replace raw._data with cleaned result
  - Full compatibility with MNE pipelines
    """)
else:
    print("\nInstall MNE-Python to run this example:")
    print("  pip install mne")

print("\n[OK] Demo completed!")
