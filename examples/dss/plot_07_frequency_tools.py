"""
DSS Frequency Domain Tools
==========================

This example demonstrates frequency-domain tools in the DSS module:

- `narrowband_scan` - Scan frequencies to find optimal peak
- `narrowband_dss` - Extract component at specific frequency
- `ssvep_dss` - SSVEP-optimized extraction with harmonics
- `time_shift_dss` - Extract temporally structured components

Reference
---------
de Cheveigné & Simon (2008). Denoising based on spatial filtering.
"""

# %%
# Imports
# -------
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal

import sys
sys.path.insert(0, r'D:\PhD\mne-denoise')

from mne_denoise.dss import (
    narrowband_scan,
    narrowband_dss,
    ssvep_dss,
    time_shift_dss,
    NarrowbandScanResult,
)

np.random.seed(42)

# %%
# Generate Test Data with Multiple Frequency Peaks
# --------------------------------------------------

print("Generating test data with multiple frequency peaks...")

n_channels = 30
n_times = 10000  # 20 seconds at 500 Hz
sfreq = 500
t = np.arange(n_times) / sfreq

# Create spatial patterns
patterns = {
    'alpha': np.sin(np.linspace(0, 2*np.pi, n_channels)),
    'ssvep': np.exp(-((np.arange(n_channels) - 15)**2) / 50),
    'beta': np.cos(np.linspace(0, 3*np.pi, n_channels)),
}
for k in patterns:
    patterns[k] /= np.linalg.norm(patterns[k])

# Time courses
# Alpha with amplitude modulation
alpha_source = np.sin(2*np.pi*10*t) * (1 + 0.5*np.sin(2*np.pi*0.3*t))

# SSVEP at 12 Hz with harmonics (typical visual stimulation)
ssvep_source = (np.sin(2*np.pi*12*t) + 
                0.5*np.sin(2*np.pi*24*t) + 
                0.25*np.sin(2*np.pi*36*t))

# Beta at 22 Hz
beta_source = np.sin(2*np.pi*22*t) * 0.7

# Mix sources
data = np.zeros((n_channels, n_times))
data += 3 * np.outer(patterns['alpha'], alpha_source)
data += 4 * np.outer(patterns['ssvep'], ssvep_source)
data += 2 * np.outer(patterns['beta'], beta_source)
data += 2 * np.random.randn(n_channels, n_times)  # Noise

print(f"Data: {n_channels} channels × {n_times} samples")
print(f"Embedded peaks: 10 Hz (alpha), 12+24+36 Hz (SSVEP), 22 Hz (beta)")

# %%
# 1. Narrowband Scan - Find Frequency Peaks
# -------------------------------------------
# 
# Scans across frequencies to find optimal DSS eigenvalue peaks.

print("\n1. Running narrowband_scan...")

result = narrowband_scan(
    data=data,
    sfreq=sfreq,
    freq_range=(5, 40),
    freq_step=0.5,
    bandwidth=2.0,
    n_components=3
)

print(f"  Scanned {len(result.frequencies)} frequencies")
print(f"  Peak at {result.frequencies[np.argmax(result.eigenvalues[:, 0])]:.1f} Hz")

# %%
# 2. Narrowband DSS - Extract at Specific Frequency
# ---------------------------------------------------

print("\n2. Running narrowband_dss at 12 Hz...")

filters_12hz, patterns_12hz, eigenvalues_12hz = narrowband_dss(
    data=data,
    sfreq=sfreq,
    freq=12.0,
    bandwidth=2.0,
    n_components=3
)
# Compute sources
sources_12hz = filters_12hz @ data

print(f"  Eigenvalues: {eigenvalues_12hz[:3].round(3)}")

# %%
# 3. SSVEP DSS - Multi-Harmonic Extraction
# -----------------------------------------

print("\n3. Running ssvep_dss at 12 Hz with 3 harmonics...")

filters_ssvep, patterns_ssvep, eigenvalues_ssvep = ssvep_dss(
    data=data,
    sfreq=sfreq,
    stim_freq=12.0,
    n_harmonics=3
)

# Compute sources
ssvep_sources = filters_ssvep @ data

# Create result dict for compatibility
ssvep_result = {
    'sources': ssvep_sources,
    'eigenvalues': eigenvalues_ssvep,
    'filters': filters_ssvep,
    'patterns': patterns_ssvep,
}

print(f"  Eigenvalues: {eigenvalues_ssvep[:3].round(3)}")
print(f"  Captures harmonics at: 12, 24, 36 Hz")

# %%
# 4. Time-Shift DSS - Temporal Structure
# ----------------------------------------

print("\n4. Running time_shift_dss...")

ts_result = time_shift_dss(
    data=data,
    shifts=[1, 2, 3, 4, 5],  # Lag samples
    n_components=5
)

print(f"  Eigenvalues: {ts_result.eigenvalues[:5].round(3)}")

# %%
# Visualization
# -------------

fig = plt.figure(figsize=(16, 12))

# 1. Narrowband scan results
ax1 = fig.add_subplot(2, 2, 1)
for i in range(min(3, result.eigenvalues.shape[1])):
    ax1.plot(result.frequencies, result.eigenvalues[:, i], 
             linewidth=2, label=f'Component {i+1}')
ax1.axvline(10, color='red', linestyle='--', alpha=0.5, label='Alpha (10 Hz)')
ax1.axvline(12, color='green', linestyle='--', alpha=0.5, label='SSVEP (12 Hz)')
ax1.axvline(22, color='blue', linestyle='--', alpha=0.5, label='Beta (22 Hz)')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('DSS Eigenvalue')
ax1.set_title('1. Narrowband Scan: Eigenvalues across frequencies')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# 2. Narrowband DSS at 12 Hz - time course
ax2 = fig.add_subplot(2, 2, 2)
time_show = slice(0, 1000)  # First 2 seconds
ax2.plot(t[time_show], sources_12hz[0, time_show], 'b-', linewidth=1.5, label='DSS 12 Hz')
ax2.plot(t[time_show], ssvep_source[time_show] * 2, 'r--', alpha=0.5, label='True SSVEP')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Amplitude')
ax2.set_title(f'2. Narrowband DSS at 12 Hz (λ={eigenvalues_12hz[0]:.3f})')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. SSVEP DSS - spectrum
ax3 = fig.add_subplot(2, 2, 3)
f, psd_ssvep = signal.welch(ssvep_result['sources'][0], sfreq, nperseg=1024)
ax3.semilogy(f, psd_ssvep, 'b-', linewidth=2, label='SSVEP DSS component')
for h in [12, 24, 36]:
    ax3.axvline(h, color='red', linestyle='--', alpha=0.7, label=f'{h} Hz' if h==12 else '')
ax3.set_xlim([0, 50])
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('PSD')
ax3.set_title(f'3. SSVEP DSS: 12 Hz + harmonics (λ={ssvep_result["eigenvalues"][0]:.3f})')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Time-shift DSS - autocorrelation
ax4 = fig.add_subplot(2, 2, 4)
# Show eigenvalue spectrum
ax4.bar(range(1, len(ts_result.eigenvalues)+1), ts_result.eigenvalues, 
        color='steelblue', alpha=0.7)
ax4.set_xlabel('Component')
ax4.set_ylabel('Eigenvalue')
ax4.set_title('4. Time-Shift DSS: Components with temporal structure')
ax4.grid(True, alpha=0.3)

fig.suptitle('DSS Frequency Domain Tools', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../output/dss_07_frequency_tools.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Additional: Compare extracted vs true sources
# ----------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Alpha extraction at 10 Hz
filters_10hz, patterns_10hz, eigenvalues_10hz = narrowband_dss(
    data=data, sfreq=sfreq, freq=10.0, bandwidth=2.0, n_components=3
)
sources_10hz = filters_10hz @ data

ax = axes[0, 0]
ax.plot(t[time_show], sources_10hz[0, time_show], 'b-', linewidth=1.5, label='Extracted')
ax.plot(t[time_show], alpha_source[time_show] * 2, 'r--', alpha=0.5, label='True')
ax.set_title(f'Alpha (10 Hz) extraction, λ={eigenvalues_10hz[0]:.3f}')
ax.legend()
ax.grid(True, alpha=0.3)

# Beta extraction at 22 Hz
filters_22hz, patterns_22hz, eigenvalues_22hz = narrowband_dss(
    data=data, sfreq=sfreq, freq=22.0, bandwidth=2.0, n_components=3
)
sources_22hz = filters_22hz @ data

ax = axes[0, 1]
ax.plot(t[time_show], sources_22hz[0, time_show], 'b-', linewidth=1.5, label='Extracted')
ax.plot(t[time_show], beta_source[time_show] * 2, 'r--', alpha=0.5, label='True')
ax.set_title(f'Beta (22 Hz) extraction, λ={eigenvalues_22hz[0]:.3f}')
ax.legend()
ax.grid(True, alpha=0.3)

# SSVEP full spectrum comparison
ax = axes[1, 0]
f_orig, psd_orig = signal.welch(data.mean(axis=0), sfreq, nperseg=1024)
ax.semilogy(f_orig, psd_orig, 'gray', alpha=0.5, label='Original (mean)')
ax.semilogy(f, psd_ssvep, 'b-', linewidth=2, label='SSVEP DSS')
ax.set_xlim([0, 50])
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD')
ax.set_title('SSVEP extraction: Original vs DSS')
ax.legend()
ax.grid(True, alpha=0.3)

# Narrowband scan peak detection
ax = axes[1, 1]
# Find peaks
peaks, _ = signal.find_peaks(result.eigenvalues[:, 0], height=0.1, distance=5)
ax.plot(result.frequencies, result.eigenvalues[:, 0], 'b-', linewidth=2)
ax.scatter(result.frequencies[peaks], result.eigenvalues[peaks, 0], 
           color='red', s=100, zorder=5, label='Detected peaks')
for p in peaks:
    ax.annotate(f'{result.frequencies[p]:.0f} Hz', 
                xy=(result.frequencies[p], result.eigenvalues[p, 0]),
                xytext=(5, 10), textcoords='offset points')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Eigenvalue')
ax.set_title('Narrowband scan peak detection')
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle('Frequency Extraction Validation', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('../output/dss_07_frequency_validation.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Summary
# -------

print("\n" + "="*60)
print("FREQUENCY TOOLS COMPLETE")
print("="*60)
print(f"""
Demonstrated 4 frequency-domain tools:

1. narrowband_scan
   - Scanned 5-40 Hz in 0.5 Hz steps
   - Found peaks at: {', '.join([f'{result.frequencies[p]:.0f}' for p in peaks[:5]])} Hz

2. narrowband_dss
   - Extracted 10 Hz: ev={eigenvalues_10hz[0]:.3f}
   - Extracted 12 Hz: ev={eigenvalues_12hz[0]:.3f}
   - Extracted 22 Hz: ev={eigenvalues_22hz[0]:.3f}

3. ssvep_dss
   - Multi-harmonic extraction at 12+24+36 Hz
   - ev={ssvep_result['eigenvalues'][0]:.3f}

4. time_shift_dss
   - Components with temporal autocorrelation
   - Top eigenvalue: {ts_result.eigenvalues[0]:.3f}

Figures saved:
  - dss_07_frequency_tools.png
  - dss_07_frequency_validation.png
""")
print("[OK] Frequency tools complete!")
