"""
04: Nonlinear Denoisers Showcase
================================
Demonstrates all nonlinear denoisers from Särelä & Valpola (2005) paper.

This script showcases:
1. WienerMaskDenoiser (Eq. 7) - Adaptive variance masking
2. TanhMaskDenoiser - ICA tanh nonlinearity
3. RobustTanhDenoiser - MATLAB-style s - tanh(s)
4. GaussDenoiser - FastICA Gaussian
5. SkewDenoiser - Skewness nonlinearity
6. QuasiPeriodicDenoiser (Sec 3.4) - Cycle averaging
7. DCTDenoiser - Frequency-domain masking
8. Spectrogram2DDenoiser - Time-frequency masking
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mne_denoise.dss import (
    IterativeDSS,
    WienerMaskDenoiser,
    TanhMaskDenoiser,
    RobustTanhDenoiser,
    GaussDenoiser,
    SkewDenoiser,
    KurtosisDenoiser,
    QuasiPeriodicDenoiser,
)
from mne_denoise.dss.denoisers import DCTDenoiser, Spectrogram2DDenoiser

# Setup
plt.style.use('seaborn-v0_8-whitegrid')
OUT_DIR = "results_04_nonlinear_denoisers"
os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(42)

print("=" * 60)
print("04: Nonlinear Denoisers Showcase")
print("=" * 60)

# =============================================================================
# 1. Wiener Mask Denoiser (Paper Eq. 7)
# =============================================================================
print("\n--- 1. Wiener Mask Denoiser (Eq. 7) ---")
print("   For bursty/non-stationary signals")

# Create bursty signal (amplitude modulated)
sfreq = 500
n_samples = 10000
t = np.arange(n_samples) / sfreq

# AM source (bursty target)
am_envelope = 1 + 0.8 * np.sin(2 * np.pi * 0.5 * t)
s_am = am_envelope * np.sin(2 * np.pi * 10 * t)

# Constant source (distractor)
s_const = np.sin(2 * np.pi * 15 * t)

# Mix
A = np.random.randn(8, 2)
X_burst = A @ np.vstack([s_am, s_const]) + 0.3 * np.random.randn(8, n_samples)

# Apply Wiener mask DSS
wiener = WienerMaskDenoiser(window_samples=int(0.2*sfreq), noise_percentile=25)
idss = IterativeDSS(denoiser=wiener.denoise, n_components=2, max_iter=30)
idss.fit(X_burst)
sources = idss.transform(X_burst)

# Find best match to AM source
corrs = [np.abs(np.corrcoef(sources[i], s_am)[0,1]) for i in range(2)]
best = np.argmax(corrs)

fig, axes = plt.subplots(3, 1, figsize=(12, 8))
axes[0].plot(t[:2000], s_am[:2000], 'g-', label='True AM source')
axes[0].plot(t[:2000], am_envelope[:2000], 'k--', linewidth=2, label='Envelope')
axes[0].legend()
axes[0].set_title('True Bursty Source (Amplitude Modulated)')

axes[1].plot(t[:2000], sources[best, :2000], 'b-')
axes[1].set_title(f'Wiener Mask DSS (correlation = {corrs[best]:.3f})')

# Show the mask effect
src_test = np.sin(2 * np.pi * 10 * t[:1000]) * (1 + 0.8 * np.sin(2 * np.pi * 0.5 * t[:1000]))
denoised = wiener.denoise(src_test)
axes[2].plot(t[:1000], src_test, alpha=0.7, label='Original')
axes[2].plot(t[:1000], denoised, alpha=0.7, label='Wiener masked')
axes[2].legend()
axes[2].set_title('Wiener Mask Effect on Single Signal')
axes[2].set_xlabel('Time (s)')

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/wiener_mask.png", dpi=150)
print(f"Saved: {OUT_DIR}/wiener_mask.png")

# =============================================================================
# 2. Tanh Denoisers (Paper Sec 3.2)
# =============================================================================
print("\n--- 2. Tanh Denoisers ---")
print("   TanhMask: tanh(s) - super-Gaussian")
print("   RobustTanh: s - tanh(s) - MATLAB style")

# Create sources with different kurtosis
s_super = np.random.laplace(0, 1, n_samples)  # Super-Gaussian (kurtosis > 3)
s_gauss = np.random.randn(n_samples)  # Gaussian (kurtosis = 3)
s_sub = np.random.uniform(-1.7, 1.7, n_samples)  # Sub-Gaussian (kurtosis < 3)

print(f"True Kurtosis: Super={stats.kurtosis(s_super, fisher=False):.2f}, "
      f"Gauss={stats.kurtosis(s_gauss, fisher=False):.2f}, "
      f"Sub={stats.kurtosis(s_sub, fisher=False):.2f}")

A = np.random.randn(8, 3)
X_kurt = A @ np.vstack([s_super, s_gauss, s_sub])

# TanhMask
tanh_mask = TanhMaskDenoiser(alpha=1.0)
idss_tanh = IterativeDSS(denoiser=tanh_mask.denoise, n_components=3, max_iter=50)
idss_tanh.fit(X_kurt)
src_tanh = idss_tanh.transform(X_kurt)

# RobustTanh
robust_tanh = RobustTanhDenoiser(alpha=1.0)
idss_robust = IterativeDSS(denoiser=robust_tanh.denoise, n_components=3, max_iter=50)
idss_robust.fit(X_kurt)
src_robust = idss_robust.transform(X_kurt)

# Compare kurtosis
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

for col in range(3):
    axes[0, col].hist(src_tanh[col], bins=50, density=True, alpha=0.7)
    k = stats.kurtosis(src_tanh[col], fisher=False)
    axes[0, col].set_title(f'TanhMask Comp {col+1}\nKurtosis = {k:.2f}')
    
    axes[1, col].hist(src_robust[col], bins=50, density=True, alpha=0.7, color='orange')
    k = stats.kurtosis(src_robust[col], fisher=False)
    axes[1, col].set_title(f'RobustTanh Comp {col+1}\nKurtosis = {k:.2f}')

axes[0, 0].set_ylabel('TanhMask: tanh(s)')
axes[1, 0].set_ylabel('RobustTanh: s-tanh(s)')

plt.suptitle('Tanh Denoisers: Super-Gaussian vs Sub-Gaussian Extraction')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/tanh_denoisers.png", dpi=150)
print(f"Saved: {OUT_DIR}/tanh_denoisers.png")

# =============================================================================
# 3. Gauss and Skew Denoisers
# =============================================================================
print("\n--- 3. Gauss and Skew Denoisers ---")

gauss = GaussDenoiser(a=1.0)
skew = SkewDenoiser()

# Demo on single signal
s_demo = np.linspace(-3, 3, 1000)
gauss_out = gauss.denoise(s_demo)
skew_out = skew.denoise(s_demo)
tanh_out = tanh_mask.denoise(s_demo)
robust_out = robust_tanh.denoise(s_demo)
kurt_out = KurtosisDenoiser().denoise(s_demo)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(s_demo, s_demo, 'k--', label='Identity (s)', linewidth=2)
ax.plot(s_demo, tanh_out, label='TanhMask: tanh(s)')
ax.plot(s_demo, robust_out, label='RobustTanh: s - tanh(s)')
ax.plot(s_demo, gauss_out, label='Gauss: s·exp(-s²/2)')
ax.plot(s_demo, skew_out, label='Skew: s²')
ax.plot(s_demo, kurt_out, label='Kurtosis: s³')
ax.set_xlabel('Input s')
ax.set_ylabel('Output f(s)')
ax.set_title('Nonlinear Denoising Functions')
ax.legend()
ax.grid(True)
ax.set_xlim([-3, 3])
ax.set_ylim([-5, 5])

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/nonlinearity_functions.png", dpi=150)
print(f"Saved: {OUT_DIR}/nonlinearity_functions.png")

# =============================================================================
# 4. Quasi-Periodic Denoiser (Paper Sec 3.4)
# =============================================================================
print("\n--- 4. Quasi-Periodic Denoiser (Sec 3.4) ---")
print("   For ECG-like quasi-periodic artifacts")

# Create synthetic ECG-like signal
def make_ecg(n_samples, sfreq, heart_rate=72, jitter=0.05):
    ecg = np.zeros(n_samples)
    beat_interval = int(sfreq * 60 / heart_rate)
    qrs_len = int(0.1 * sfreq)
    qrs_t = np.linspace(-2, 2, qrs_len)
    qrs_template = np.exp(-qrs_t**2) * 2
    
    beat_pos = 0
    while beat_pos < n_samples - qrs_len:
        jitter_samples = int(np.random.randn() * jitter * beat_interval)
        start = beat_pos + jitter_samples
        if 0 <= start < n_samples - qrs_len:
            ecg[start:start+qrs_len] += qrs_template
        beat_pos += beat_interval
    return ecg

s_ecg = make_ecg(n_samples, sfreq, heart_rate=72, jitter=0.05)
s_alpha = np.sin(2 * np.pi * 10 * t)
s_noise = np.random.randn(n_samples) * 0.3

A = np.random.randn(8, 3)
X_ecg = A @ np.vstack([s_ecg, s_alpha, s_noise]) + 0.2 * np.random.randn(8, n_samples)

# Quasi-periodic DSS
beat_interval = int(sfreq * 60 / 72)
qp = QuasiPeriodicDenoiser(peak_distance=int(beat_interval * 0.8), peak_height_percentile=75)
idss_qp = IterativeDSS(denoiser=qp.denoise, n_components=3, max_iter=20)
idss_qp.fit(X_ecg)
sources_qp = idss_qp.transform(X_ecg)

# Find ECG component
corrs = [np.abs(np.corrcoef(sources_qp[i], s_ecg)[0,1]) for i in range(3)]
best_ecg = np.argmax(corrs)

fig, axes = plt.subplots(2, 1, figsize=(12, 6))
axes[0].plot(t[:3000], s_ecg[:3000], 'g-', label='True ECG')
axes[0].legend()
axes[0].set_title('True ECG Source (Quasi-Periodic)')

src_norm = sources_qp[best_ecg] / np.std(sources_qp[best_ecg]) * np.std(s_ecg)
axes[1].plot(t[:3000], src_norm[:3000], 'b-')
axes[1].set_title(f'Quasi-Periodic DSS (correlation = {corrs[best_ecg]:.3f})')
axes[1].set_xlabel('Time (s)')

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/quasi_periodic.png", dpi=150)
print(f"Saved: {OUT_DIR}/quasi_periodic.png")

# =============================================================================
# 5. DCT Denoiser
# =============================================================================
print("\n--- 5. DCT Denoiser ---")
print("   Frequency-domain masking via DCT")

# Create signal with high and low frequency components
s_low = np.sin(2 * np.pi * 2 * t[:2000])  # 2 Hz
s_high = np.sin(2 * np.pi * 50 * t[:2000])  # 50 Hz
s_mixed = s_low + 0.5 * s_high

# Apply DCT denoiser (lowpass)
dct_lp = DCTDenoiser(cutoff_fraction=0.2)  # Keep first 20% of coeffs
s_filtered = dct_lp.denoise(s_mixed)

fig, axes = plt.subplots(3, 1, figsize=(12, 8))
axes[0].plot(t[:2000], s_mixed, label='Mixed (2 Hz + 50 Hz)')
axes[0].legend()
axes[0].set_title('Original Signal')

axes[1].plot(t[:2000], s_filtered, label='DCT Lowpass (20% kept)', color='orange')
axes[1].legend()
axes[1].set_title('DCT Denoised')

axes[2].plot(t[:2000], s_low, 'g-', label='True 2 Hz')
axes[2].plot(t[:2000], s_filtered, 'b--', label='DCT recovered', alpha=0.7)
axes[2].legend()
axes[2].set_title(f'Comparison (correlation = {np.corrcoef(s_low, s_filtered)[0,1]:.3f})')
axes[2].set_xlabel('Time (s)')

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/dct_denoiser.png", dpi=150)
print(f"Saved: {OUT_DIR}/dct_denoiser.png")

# =============================================================================
# 6. Spectrogram 2D Denoiser
# =============================================================================
print("\n--- 6. Spectrogram 2D Denoiser ---")
print("   Time-frequency masking via STFT")

# Create chirp signal (frequency changes over time)
s_chirp = signal.chirp(t[:4000], f0=5, f1=40, t1=t[3999], method='linear')
s_noise = np.random.randn(4000) * 0.5
s_noisy = s_chirp + s_noise

# Apply spectrogram denoiser
spec_denoiser = Spectrogram2DDenoiser(nperseg=256, threshold_percentile=60)
s_clean = spec_denoiser.denoise(s_noisy)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Original spectrogram
f, t_spec, Sxx = signal.spectrogram(s_noisy, sfreq, nperseg=256)
axes[0, 0].pcolormesh(t_spec, f, 10*np.log10(Sxx+1e-10), shading='gouraud')
axes[0, 0].set_ylabel('Frequency (Hz)')
axes[0, 0].set_title('Noisy Spectrogram')
axes[0, 0].set_ylim([0, 60])

# Cleaned spectrogram
f, t_spec, Sxx_clean = signal.spectrogram(s_clean, sfreq, nperseg=256)
axes[0, 1].pcolormesh(t_spec, f, 10*np.log10(Sxx_clean+1e-10), shading='gouraud')
axes[0, 1].set_ylabel('Frequency (Hz)')
axes[0, 1].set_title('Denoised Spectrogram')
axes[0, 1].set_ylim([0, 60])

# Time domain
t_short = np.arange(4000) / sfreq
axes[1, 0].plot(t_short, s_noisy, alpha=0.7, label='Noisy')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_title('Noisy Signal')

axes[1, 1].plot(t_short, s_clean, alpha=0.7, label='Denoised', color='orange')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_title('Spectrogram 2D Denoised')

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/spectrogram_2d.png", dpi=150)
print(f"Saved: {OUT_DIR}/spectrogram_2d.png")

print(f"\n[OK] All results saved to {OUT_DIR}/")
