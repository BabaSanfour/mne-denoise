"""
Evoked Response Enhancement with DSS
=====================================

This example replicates key figures from de Cheveigné & Simon (2008):
"Denoising based on spatial filtering" - J Neurosci Methods

Demonstrates:
- **Figure 1**: Power distribution across DSS components
- **Figure 2**: RMS evoked response before/after denoising  
- **Figure 3**: Component reproducibility and selection

Reference
---------
de Cheveigné, A., & Simon, J. Z. (2008). Denoising based on spatial filtering.
Journal of Neuroscience Methods, 171(2), 331-339.
"""

# %%
# Imports
# -------
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, r'D:\PhD\mne-denoise')

from mne_denoise.dss import DSS, TrialAverageBias, compute_dss

np.random.seed(42)

# %%
# Generate Simulated Auditory Evoked Data
# ----------------------------------------
# 
# We simulate MEG-like data with:
# - Multiple spatial sources (auditory cortex bilateral)
# - Evoked M100 response around 100ms
# - Background noise

print("Generating simulated MEG evoked data...")

# Parameters matching paper
n_epochs = 100      # Number of trials
n_channels = 60     # Number of sensors
n_times = 350       # Samples per epoch (-200 to 500 ms)
sfreq = 500         # Sampling rate
times = np.linspace(-200, 500, n_times)  # Time in ms

# Create evoked response template: M100 peak around 100ms
def create_evoked_template(peak_time=100, width=30):
    """Create Gaussian-windowed oscillatory template."""
    envelope = np.exp(-((times - peak_time)**2) / (2 * width**2))
    oscillation = np.sin(2 * np.pi * 10 * times / 1000)  # 10 Hz
    return envelope * oscillation

# Main M100 template
m100_template = create_evoked_template(100, 30)
m100_template /= np.std(m100_template)  # Normalize

# Secondary component (M200)
m200_template = create_evoked_template(200, 40)
m200_template /= np.std(m200_template)

# Spatial patterns for bilateral auditory cortex
# Main pattern: symmetric bilateral (like auditory cortex)
pattern_main = np.sin(np.linspace(0, 2*np.pi, n_channels))
pattern_main /= np.linalg.norm(pattern_main)

# Secondary pattern: asymmetric
pattern_second = np.cos(np.linspace(0, 2*np.pi, n_channels)) * np.linspace(0.5, 1.5, n_channels)
pattern_second /= np.linalg.norm(pattern_second)

# Third pattern: frontal
pattern_third = np.exp(-np.linspace(-2, 2, n_channels)**2)
pattern_third /= np.linalg.norm(pattern_third)

# Generate epoched data: (n_channels, n_times, n_epochs)
data = np.zeros((n_channels, n_times, n_epochs))

for ep in range(n_epochs):
    # Evoked components (same across trials)
    signal = np.outer(pattern_main, m100_template) * 5.0
    signal += np.outer(pattern_second, m200_template) * 2.5
    signal += np.outer(pattern_third, create_evoked_template(150, 25)) * 1.5
    
    # Noise (different each trial) - this should be suppressed by DSS
    noise = np.random.randn(n_channels, n_times) * 3.0
    
    data[:, :, ep] = signal + noise

print(f"Data shape: {data.shape} (channels, times, epochs)")
print(f"SNR (signal variance / noise variance): {np.var(signal) / np.var(noise):.2f}")

# %%
# Apply DSS with Trial-Average Bias
# ----------------------------------
# 
# DSS finds spatial filters that maximize reproducibility across trials.
# The bias function (trial averaging) enhances stimulus-evoked components.

print("\nApplying DSS with Trial-Average Bias...")

bias = TrialAverageBias()
dss = DSS(bias=bias, n_components=n_channels)
dss.fit(data)

# Get all components
sources = dss.transform(data)  # (n_components, n_times*n_epochs)
eigenvalues = dss.eigenvalues_

print(f"DSS Components: {sources.shape[0]}")
print(f"Top 5 eigenvalues (reproducibility): {eigenvalues[:5].round(3)}")

# %%
# Figure 1: Power Distribution Across Components
# -----------------------------------------------
# 
# Replicating Figure 1 from the paper showing:
# - (a) Power per component before (black) and after (red) averaging
# - (b) Cumulative power retained vs component cutoff
# - (c) Reproducibility (evoked/total power ratio)

# Reshape sources: (n_components, n_times, n_epochs)
sources_3d = sources.reshape(sources.shape[0], n_times, n_epochs)

# Compute power per component
power_total = np.var(sources_3d, axis=(1, 2))
power_total_pct = 100 * power_total / power_total.sum()

# After averaging: evoked power
sources_avg = sources_3d.mean(axis=2)  # Average across epochs
power_evoked = np.var(sources_avg, axis=1)
power_evoked_pct = 100 * power_evoked / power_evoked.sum()

# Reproducibility: ratio of evoked to total power
reproducibility = power_evoked / (power_total + 1e-12)

# Create Figure 1
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# (a) Power per component
ax = axes[0]
n_show = min(20, len(power_total_pct))
x = np.arange(1, n_show + 1)
ax.bar(x - 0.2, power_total_pct[:n_show], 0.4, label='Total (all trials)', color='black', alpha=0.7)
ax.bar(x + 0.2, power_evoked_pct[:n_show], 0.4, label='Evoked (averaged)', color='red', alpha=0.7)
ax.set_xlabel('Component')
ax.set_ylabel('% Power')
ax.set_title('(a) Power per component')
ax.legend()
ax.set_xlim([0, n_show + 1])

# (b) Cumulative power
ax = axes[1]
cum_total = np.cumsum(power_total_pct)
cum_evoked = np.cumsum(power_evoked_pct)
ax.plot(np.arange(1, len(cum_total)+1), cum_total, 'k-', linewidth=2, label='Total power')
ax.plot(np.arange(1, len(cum_evoked)+1), cum_evoked, 'r-', linewidth=2, label='Evoked power')
ax.axhline(96, color='gray', linestyle='--', alpha=0.5, label='96% threshold')
ax.axvline(10, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Components retained')
ax.set_ylabel('% Power retained')
ax.set_title('(b) Cumulative power')
ax.legend(loc='lower right')
ax.set_xlim([0, 21])
ax.set_ylim([0, 105])

# (c) Reproducibility per component
ax = axes[2]
colors = ['blue' if r > 0.1 else 'lightblue' for r in reproducibility[:n_show]]
ax.bar(x, 100 * reproducibility[:n_show], color=colors, alpha=0.7)
ax.axhline(10, color='red', linestyle='--', alpha=0.5, label='10% cutoff')
ax.set_xlabel('Component')
ax.set_ylabel('% Reproducible')
ax.set_title('(c) Reproducibility (evoked/total)')
ax.set_xlim([0, n_show + 1])
ax.legend()

fig.suptitle('Figure 1: Power distribution across DSS components\n(de Cheveigné & Simon 2008)', 
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('../output/dss_05_figure1_power.png', dpi=150, bbox_inches='tight')
plt.show()

# Find optimal cutoff
n_keep = np.where(reproducibility > 0.05)[0]
if len(n_keep) > 0:
    n_keep = n_keep[-1] + 1
else:
    n_keep = 10
    
print(f"\nFigure 1 Results:")
print(f"  Top {n_keep} components capture {cum_evoked[n_keep-1]:.1f}% of evoked power")
print(f"  Top {n_keep} components capture {cum_total[n_keep-1]:.1f}% of total power")
print(f"  Component 1 reproducibility: {100*reproducibility[0]:.1f}%")

# %%
# Figure 2: RMS Evoked Response Before/After Denoising
# -----------------------------------------------------
# 
# Replicating Figure 2 showing noise reduction from DSS.

def compute_rms_with_bootstrap(data_3d, n_bootstrap=100):
    """Compute RMS and bootstrap confidence interval."""
    n_ch, n_times, n_ep = data_3d.shape
    
    # Average over epochs, then RMS over channels
    avg = data_3d.mean(axis=2)
    rms = np.sqrt(np.mean(avg**2, axis=0))
    
    # Bootstrap for error bars
    rng = np.random.default_rng(42)
    rms_boots = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_ep, n_ep, replace=True)
        avg_boot = data_3d[:, :, idx].mean(axis=2)
        rms_boot = np.sqrt(np.mean(avg_boot**2, axis=0))
        rms_boots.append(rms_boot)
    
    rms_std = np.std(rms_boots, axis=0)
    return rms, rms_std

# Original data RMS
rms_original, rms_std_original = compute_rms_with_bootstrap(data)

# DSS denoised data (keep top components)
n_keep = 10
cleaned = dss.inverse_transform(sources[:n_keep])
cleaned_3d = cleaned.reshape(n_channels, n_times, n_epochs)
rms_denoised, rms_std_denoised = compute_rms_with_bootstrap(cleaned_3d)

# Plot Figure 2
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# (a) RMS before vs after
ax = axes[0]
ax.plot(times, rms_original, 'r-', linewidth=1.5, label='Before DSS', alpha=0.8)
ax.fill_between(times, rms_original - 2*rms_std_original, 
                rms_original + 2*rms_std_original, alpha=0.15, color='red')
ax.plot(times, rms_denoised, 'b-', linewidth=1.5, label=f'After DSS ({n_keep} comp.)')
ax.fill_between(times, rms_denoised - 2*rms_std_denoised,
                rms_denoised + 2*rms_std_denoised, alpha=0.15, color='blue')
ax.axvline(0, color='k', linestyle='--', alpha=0.5, label='Stimulus onset')
ax.axvline(100, color='green', linestyle=':', alpha=0.5, label='M100 peak')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('RMS Field')
ax.set_title('(a) RMS evoked response')
ax.legend()

# (b) First DSS component time course
ax = axes[1]
comp1 = sources_3d[0]  # (n_times, n_epochs)
comp1_avg = comp1.mean(axis=1)

# Bootstrap for component 1
rng = np.random.default_rng(42)
boots = [comp1[:, rng.choice(n_epochs, n_epochs, replace=True)].mean(axis=1) 
         for _ in range(100)]
comp1_std = np.std(boots, axis=0)

ax.plot(times, comp1_avg, 'b-', linewidth=2)
ax.fill_between(times, comp1_avg - 2*comp1_std, comp1_avg + 2*comp1_std, 
                alpha=0.2, color='blue')
ax.axvline(0, color='k', linestyle='--', alpha=0.5)
ax.axvline(100, color='green', linestyle=':', alpha=0.5)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Amplitude')
ax.set_title(f'(b) First DSS component (λ={eigenvalues[0]:.3f})')

fig.suptitle('Figure 2: RMS evoked response before/after DSS denoising\n(de Cheveigné & Simon 2008)', 
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('../output/dss_05_figure2_rms.png', dpi=150, bbox_inches='tight')
plt.show()

# Compute SNR improvement
snr_before = np.max(rms_original[times > 0]) / np.std(rms_original[times < 0])
snr_after = np.max(rms_denoised[times > 0]) / np.std(rms_denoised[times < 0])
print(f"\nFigure 2 Results:")
print(f"  SNR before DSS: {snr_before:.1f}")
print(f"  SNR after DSS: {snr_after:.1f}")
print(f"  Improvement: {snr_after/snr_before:.1f}x")

# %%
# Figure 3: Component Time Courses and Selection
# -----------------------------------------------
# 
# Shows time courses of multiple components and identifies stimulus-driven ones.

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# (a) Component time courses
ax = axes[0]
n_comps_show = 15
for i in range(n_comps_show):
    comp_avg = sources_3d[i].mean(axis=1)
    # Normalize for display
    comp_avg = comp_avg / (np.std(comp_avg) + 1e-10) - i * 3
    ax.plot(times, comp_avg, linewidth=1.5, 
            color='blue' if reproducibility[i] > 0.1 else 'gray',
            alpha=0.8 if reproducibility[i] > 0.1 else 0.4)
    ax.text(times[-1] + 10, -i * 3, f'λ={eigenvalues[i]:.2f}', fontsize=8)

ax.axvline(0, color='k', linestyle='--', alpha=0.5)
ax.axvline(100, color='green', linestyle=':', alpha=0.5)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Components (offset for clarity)')
ax.set_title(f'(a) First {n_comps_show} DSS components (blue = reproducible, gray = noise)')
ax.set_xlim([times[0], times[-1] + 50])

# (b) Eigenvalue spectrum
ax = axes[1]
ax.bar(np.arange(1, len(eigenvalues)+1), eigenvalues, color='steelblue', alpha=0.7)
ax.axhline(0.1, color='red', linestyle='--', label='Selection threshold')
ax.set_xlabel('Component')
ax.set_ylabel('Eigenvalue (reproducibility)')
ax.set_title('(b) DSS eigenvalue spectrum - higher = more stimulus-driven')
ax.legend()
ax.set_xlim([0, min(30, len(eigenvalues)) + 1])

fig.suptitle('Figure 3: Component analysis and selection\n(de Cheveigné & Simon 2008)', 
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('../output/dss_05_figure3_components.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Summary
# -------

print("\n" + "="*60)
print("EVOKED RESPONSE ENHANCEMENT COMPLETE")
print("="*60)
print(f"""
Replicating de Cheveigné & Simon (2008):

Data: {n_epochs} epochs, {n_channels} channels, {n_times} times

Key Results:
  • Top 10 components capture {cum_evoked[9]:.1f}% of evoked power
  • Component 1 reproducibility: {100*reproducibility[0]:.1f}%
  • SNR improvement: {snr_after/snr_before:.1f}x

Paper comparison (their data):
  • Top 10 components: ~96% evoked power
  • Component 1: ~60% reproducible
  
Figures saved:
  • dss_05_figure1_power.png
  • dss_05_figure2_rms.png  
  • dss_05_figure3_components.png
""")
print("[OK] Evoked response enhancement complete!")
