"""
DSS Bias Functions Gallery
==========================

This example demonstrates all linear bias functions (denoisers) available
in the DSS module. Each bias function extracts different signal characteristics.

Bias Functions Covered:
- `TrialAverageBias` - Evoked responses (stimulus-locked)
- `BandpassBias` - Frequency-band specific activity (e.g., alpha rhythm)
- `NotchBias` - Suppress specific frequency
- `PeakFilterBias` - Isolate narrow frequency peak
- `CombFilterBias` - Periodic signals (SSVEP, harmonics)
- `CycleAverageBias` - Quasi-periodic signals (cardiac)

Reference
---------
Särelä & Valpola (2005). Denoising Source Separation. JMLR.
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
    DSS, compute_dss,
    TrialAverageBias,
    BandpassBias,
    NotchBias,
    PeakFilterBias,
    CombFilterBias,
    CycleAverageBias,
)

np.random.seed(42)

# %%
# Generate Multi-Component Test Signal
# -------------------------------------
# 
# We create data with multiple embedded signals that each bias function can extract.

print("Generating multi-component test signal...")

n_channels = 30
n_times = 5000
sfreq = 500
t = np.arange(n_times) / sfreq

# Create spatial patterns for different sources
patterns = {}
patterns['evoked'] = np.sin(np.linspace(0, np.pi, n_channels))
patterns['alpha'] = np.cos(np.linspace(0, 2*np.pi, n_channels))
patterns['beta'] = np.sin(np.linspace(0, 3*np.pi, n_channels))
patterns['line'] = np.ones(n_channels) + 0.3 * np.random.randn(n_channels)
patterns['ssvep'] = np.exp(-((np.arange(n_channels) - 15)**2) / 50)
patterns['cardiac'] = np.sin(np.linspace(0, 4*np.pi, n_channels))

# Normalize patterns
for k in patterns:
    patterns[k] /= np.linalg.norm(patterns[k])

# Create time courses
sources = {}

# 1. Evoked response (epoched)
n_epochs = 50
epoch_len = 100  # samples
evoked_template = np.exp(-(np.arange(epoch_len) - 30)**2 / 200) * np.sin(2*np.pi*8*np.arange(epoch_len)/sfreq)
sources['evoked'] = np.zeros(n_times)
for i in range(n_epochs):
    start = i * n_times // n_epochs
    if start + epoch_len < n_times:
        sources['evoked'][start:start+epoch_len] += evoked_template

# 2. Alpha rhythm (8-12 Hz)
sources['alpha'] = np.sin(2 * np.pi * 10 * t) * (1 + 0.3 * np.sin(2*np.pi*0.5*t))

# 3. Beta rhythm (15-30 Hz)  
sources['beta'] = np.sin(2 * np.pi * 20 * t) * 0.5

# 4. Line noise (50 Hz)
sources['line'] = np.sin(2 * np.pi * 50 * t)

# 5. SSVEP (15 Hz + harmonics)
sources['ssvep'] = (np.sin(2*np.pi*15*t) + 
                    0.5*np.sin(2*np.pi*30*t) + 
                    0.25*np.sin(2*np.pi*45*t))

# 6. Cardiac (~1 Hz quasi-periodic)
heart_rate = 1.0 + 0.1 * np.random.randn()  # ~1 Hz with jitter
cardiac_phase = np.cumsum(np.ones(n_times) * heart_rate / sfreq * 2 * np.pi)
sources['cardiac'] = np.sin(cardiac_phase) + 0.3 * np.sin(3*cardiac_phase)

# Mix all sources into multichannel data
amplitudes = {'evoked': 3, 'alpha': 2, 'beta': 1.5, 'line': 1.5, 'ssvep': 2, 'cardiac': 2}
data = np.zeros((n_channels, n_times))
for k in sources:
    data += amplitudes[k] * np.outer(patterns[k], sources[k])

# Add noise
data += np.random.randn(n_channels, n_times) * 2.0

print(f"Data shape: {data.shape}")
print(f"Sources embedded: {list(sources.keys())}")

# %%
# 1. TrialAverageBias - Evoked Responses
# ---------------------------------------
# 
# Extracts components that are reproducible across trials.

print("\n1. TrialAverageBias (evoked responses)...")

# Reshape to epochs for trial-average bias
epoch_len = n_times // n_epochs
data_epochs = data[:, :epoch_len * n_epochs].reshape(n_channels, epoch_len, n_epochs)

bias_trial = TrialAverageBias()
dss_trial = DSS(bias=bias_trial, n_components=5)
dss_trial.fit(data_epochs)

sources_trial = dss_trial.transform(data_epochs)
print(f"  Eigenvalues: {dss_trial.eigenvalues_[:5].round(3)}")

# %%
# 2. BandpassBias - Alpha Rhythm
# -------------------------------
# 
# Enhances components in a specific frequency band.

print("\n2. BandpassBias (alpha 8-12 Hz)...")

bias_alpha = BandpassBias(freq_band=(8, 12), sfreq=sfreq)
dss_alpha = DSS(bias=bias_alpha, n_components=5)
dss_alpha.fit(data)

sources_alpha = dss_alpha.transform(data)
print(f"  Eigenvalues: {dss_alpha.eigenvalues_[:5].round(3)}")

# %%
# 3. NotchBias - Line Noise Suppression
# --------------------------------------
# 
# Suppresses components at a specific frequency (by finding and inverting).

print("\n3. NotchBias (suppress 50 Hz)...")

# NotchBias finds components AT the notch frequency (to remove them)
bias_notch = NotchBias(freq=50, sfreq=sfreq, bandwidth=2.0)
dss_notch = DSS(bias=bias_notch, n_components=5)
dss_notch.fit(data)

sources_notch = dss_notch.transform(data)
print(f"  Eigenvalues: {dss_notch.eigenvalues_[:5].round(3)}")

# %%
# 4. PeakFilterBias - Narrow Peak
# --------------------------------
# 
# Isolates a narrow frequency peak.

print("\n4. PeakFilterBias (15 Hz SSVEP fundamental)...")

# Use NotchBias as a narrow peak filter
bias_peak = NotchBias(freq=15, sfreq=sfreq, bandwidth=2.0)
dss_peak = DSS(bias=bias_peak, n_components=5)
dss_peak.fit(data)

sources_peak = dss_peak.transform(data)
print(f"  Eigenvalues: {dss_peak.eigenvalues_[:5].round(3)}")

# %%
# 5. CombFilterBias - Periodic Harmonics (SSVEP)
# -----------------------------------------------
# 
# Enhances periodic signals with harmonics.

print("\n5. CombFilterBias (15 Hz + harmonics)...")

bias_comb = CombFilterBias(fundamental_freq=15, sfreq=sfreq, n_harmonics=3)
dss_comb = DSS(bias=bias_comb, n_components=5)
dss_comb.fit(data)

sources_comb = dss_comb.transform(data)
print(f"  Eigenvalues: {dss_comb.eigenvalues_[:5].round(3)}")

# %%
# 6. CycleAverageBias - Quasi-Periodic (Cardiac)
# -----------------------------------------------
# 
# Extracts quasi-periodic signals with known events.

print("\n6. CycleAverageBias (cardiac artifact)...")

# Create cardiac event indices
cardiac_events = np.where(np.diff((sources['cardiac'] > 0.9).astype(int)) == 1)[0]

if len(cardiac_events) > 5:
    bias_cycle = CycleAverageBias(event_samples=cardiac_events, window=(-20, 50))
    dss_cycle = DSS(bias=bias_cycle, n_components=5)
    dss_cycle.fit(data)
    
    sources_cycle = dss_cycle.transform(data)
    print(f"  Eigenvalues: {dss_cycle.eigenvalues_[:5].round(3)}")
else:
    print("  Not enough cardiac events detected")
    sources_cycle = np.zeros((5, n_times))
    dss_cycle = None

# %%
# Visualization - All Bias Functions
# ------------------------------------

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Helper to compute PSD
def compute_psd(x, fs):
    f, psd = signal.welch(x, fs, nperseg=min(512, len(x)))
    return f, psd

# 1. Trial Average - show evoked template
ax = axes[0, 0]
# Reshape sources for plotting
if sources_trial.ndim == 2:
    comp1 = sources_trial[0]
else:
    comp1 = sources_trial[0].flatten()
ax.plot(comp1[:epoch_len], 'b-', linewidth=2, label='DSS Component 1')
ax.plot(evoked_template * 5, 'r--', linewidth=1.5, alpha=0.7, label='True evoked')
ax.set_title(f'1. TrialAverageBias (λ={dss_trial.eigenvalues_[0]:.3f})')
ax.set_xlabel('Time (samples)')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Bandpass (Alpha)
ax = axes[0, 1]
f, psd = compute_psd(sources_alpha[0], sfreq)
ax.semilogy(f, psd, 'b-', linewidth=2)
ax.axvspan(8, 12, alpha=0.3, color='green', label='Alpha band')
ax.axvline(10, color='r', linestyle='--', alpha=0.5, label='True alpha')
ax.set_xlim([0, 50])
ax.set_title(f'2. BandpassBias 8-12 Hz (λ={dss_alpha.eigenvalues_[0]:.3f})')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Notch (50 Hz)
ax = axes[1, 0]
f, psd = compute_psd(sources_notch[0], sfreq)
ax.semilogy(f, psd, 'b-', linewidth=2)
ax.axvline(50, color='r', linestyle='--', label='Line noise', linewidth=2)
ax.set_xlim([0, 100])
ax.set_title(f'3. NotchBias 50 Hz (λ={dss_notch.eigenvalues_[0]:.3f})')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Peak (15 Hz)
ax = axes[1, 1]
f, psd = compute_psd(sources_peak[0], sfreq)
ax.semilogy(f, psd, 'b-', linewidth=2)
ax.axvline(15, color='r', linestyle='--', label='SSVEP fundamental', linewidth=2)
ax.set_xlim([0, 50])
ax.set_title(f'4. PeakFilterBias 15 Hz (λ={dss_peak.eigenvalues_[0]:.3f})')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD')
ax.legend()
ax.grid(True, alpha=0.3)

# 5. Comb (15 Hz + harmonics)
ax = axes[2, 0]
f, psd = compute_psd(sources_comb[0], sfreq)
ax.semilogy(f, psd, 'b-', linewidth=2)
for h in [15, 30, 45]:
    ax.axvline(h, color='r', linestyle='--', alpha=0.7)
ax.set_xlim([0, 60])
ax.set_title(f'5. CombFilterBias 15+30+45 Hz (λ={dss_comb.eigenvalues_[0]:.3f})')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD')
ax.grid(True, alpha=0.3)

# 6. Cycle Average (Cardiac)
ax = axes[2, 1]
if sources_cycle.shape[0] > 0 and np.any(sources_cycle != 0):
    ax.plot(sources_cycle[0, :500], 'b-', linewidth=2)
    # Mark cardiac events
    for ev in cardiac_events[cardiac_events < 500]:
        ax.axvline(ev, color='r', linestyle=':', alpha=0.5)
    ax.set_title(f'6. CycleAverageBias (cardiac)')
else:
    ax.text(0.5, 0.5, 'Insufficient cardiac events', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('6. CycleAverageBias (cardiac) - N/A')
ax.set_xlabel('Time (samples)')
ax.grid(True, alpha=0.3)

fig.suptitle('DSS Bias Functions Gallery: Different Denoisers for Different Signals', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../output/dss_06_bias_functions.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Summary
# -------

print("\n" + "="*60)
print("BIAS FUNCTIONS GALLERY COMPLETE")  
print("="*60)
print(f"""
Demonstrated 6 linear bias functions:

1. TrialAverageBias    - Evoked responses (ev={dss_trial.eigenvalues_[0]:.3f})
2. BandpassBias        - Alpha rhythm 8-12 Hz (ev={dss_alpha.eigenvalues_[0]:.3f})
3. NotchBias           - Line noise 50 Hz (ev={dss_notch.eigenvalues_[0]:.3f})
4. PeakFilterBias      - SSVEP 15 Hz (ev={dss_peak.eigenvalues_[0]:.3f})
5. CombFilterBias      - Harmonics 15+30+45 Hz (ev={dss_comb.eigenvalues_[0]:.3f})
6. CycleAverageBias    - Cardiac artifact

Each bias function finds components maximizing different signal properties.
Higher eigenvalue = stronger match to the bias criterion.

Figure saved: dss_06_bias_functions.png
""")
print("[OK] Bias functions gallery complete!")
