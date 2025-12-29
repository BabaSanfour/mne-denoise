"""Visualization for comparing original vs denoised data."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mne

def plot_psd_comparison(inst_orig, inst_denoised, fmin=0, fmax=np.inf, 
                       show=True, average=True, ax=None):
    """Plot PSD comparison (Original vs Denoised).
    
    Parameters
    ----------
    inst_orig : MNE Object
        Original Raw/Epochs/Evoked.
    inst_denoised : MNE Object
        Denoised Raw/Epochs/Evoked.
    fmin, fmax : float
        Frequency range.
    show : bool
        Show figure.
    average : bool
        If True, plot average across channels.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
        
    # Plot comparison
    for inst, label, color in [(inst_orig, 'Original', 'k'), (inst_denoised, 'Denoised', 'r')]:
        spectrum = inst.compute_psd(fmin=fmin, fmax=fmax)
        psd = spectrum.get_data(return_freqs=False)
        freqs = spectrum.freqs
        
        if average:
            # Average over all dimensions except frequency (last one)
            # This handles (n_epochs, n_ch, n_freqs) and (n_ch, n_freqs) automatically
            axis = tuple(range(psd.ndim - 1))
            psd_mean = np.mean(psd, axis=axis)
            ax.semilogy(freqs, psd_mean, label=label, color=color)
        else:
            # Plot all channels (collapsed over epochs if present)
            if psd.ndim == 3:
                psd = np.mean(psd, axis=0) # (n_ch, n_freqs)
            ax.semilogy(freqs, psd.T, color=color, alpha=0.2)
            # Add a dummy line for legend
            ax.plot([], [], color=color, label=label)
        
    ax.legend()
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density")
    ax.set_title("PSD Comparison")
    ax.grid(True)
    
    if show:
        plt.show()
    return fig


def plot_spectral_psd_comparison(inst_orig, components, sfreq, peak_freq=None,
                                 fmin=1, fmax=40, show=True):
    """Plot side-by-side PSD comparison for spectral/narrowband DSS.
    
    Creates a two-panel figure showing original data PSD and DSS component PSDs,
    with optional peak frequency marking. Designed for spectral analysis workflows.
    
    Parameters
    ----------
    inst_orig : Raw | Epochs
        Original MNE data object.
    components : ndarray
        DSS components array from transform(), shape (n_components, n_times) or
        (n_components, n_times, n_epochs).
    sfreq : float
        Sampling frequency in Hz.
    peak_freq : float, optional
        Peak frequency to mark with vertical line (e.g., detected alpha peak).
    fmin, fmax : float
        Frequency range for PSD computation. Default 1-40 Hz.
    show : bool
        If True, show the figure.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
        
    Examples
    --------
    >>> from mne_denoise.dss.variants import narrowband_dss
    >>> from mne_denoise.viz import plot_spectral_psd_comparison
    >>> dss = narrowband_dss(sfreq=250, freq=10, bandwidth=3)
    >>> dss.fit(raw)
    >>> components = dss.transform(raw)
    >>> plot_spectral_psd_comparison(raw, components, sfreq=250, peak_freq=10)
    """
    import mne
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Left: Original data PSD
    psd_orig = inst_orig.compute_psd(fmin=fmin, fmax=fmax)
    psd_orig.plot(axes=axes[0], show=False, spatial_colors=False, average=True)
    axes[0].set_title('Original Data PSD (Average)')
    
    if peak_freq is not None:
        axes[0].axvline(peak_freq, color='red', linestyle='--', alpha=0.7,
                       label=f'Peak: {peak_freq:.1f} Hz')
        axes[0].legend()
    
    # Right: DSS Component PSD
    # Handle different component types
    import mne
    
    # Convert components to numpy array if needed
    if isinstance(components, (mne.io.BaseRaw, mne.BaseEpochs)):
        comp_data_raw = components.get_data()
    else:
        comp_data_raw = np.asarray(components)
    
    # Handle 2D (Raw) or 3D (Epochs) components
    if comp_data_raw.ndim == 2:
        n_comp = min(3, comp_data_raw.shape[0])
        comp_data = comp_data_raw[:n_comp]
    else:  # 3D: (n_comp, n_times, n_epochs) or (n_epochs, n_ch, n_times)
        # Check if it's (n_epochs, n_ch, n_times) from Epochs.get_data()
        if comp_data_raw.shape[0] < comp_data_raw.shape[2]:  # n_epochs < n_times
            # Transpose to (n_ch, n_times, n_epochs)
            comp_data_raw = comp_data_raw.transpose(1, 2, 0)
        n_comp = min(3, comp_data_raw.shape[0])
        # Average over epochs for cleaner visualization
        comp_data = comp_data_raw[:n_comp].mean(axis=2)
    
    # Create RawArray for PSD computation
    comp_info = mne.create_info(n_comp, sfreq, 'misc')
    comp_raw = mne.io.RawArray(comp_data, comp_info)
    
    psd_comp = comp_raw.compute_psd(fmin=fmin, fmax=fmax, picks='all')
    psd_comp.plot(axes=axes[1], show=False, picks='all')
    axes[1].set_title('DSS Components PSD')
    
    if peak_freq is not None:
        axes[1].axvline(peak_freq, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if show:
        plt.show()
    return fig


def plot_evoked_comparison(inst_orig, inst_denoised, ci=0.95, n_boot=1000, 
                         colors=('r', 'b'), linestyles=('-', '-'), 
                         labels=('Original', 'Denoised'),
                         show=True, ax=None):
    """Plot Global Field Power (GFP) comparison with optional Bootstrap CI.

    Parameters
    ----------
    inst_orig : Epochs | Evoked
        Original data. If Epochs, CI can be computed.
    inst_denoised : Epochs | Evoked
        Denoised data.
    ci : float | None
        Confidence interval (e.g. 0.95). If None, no CI is plotted.
        Requires `inst_orig` and `inst_denoised` to be Epochs.
    n_boot : int
        Number of bootstrap resamples. Default 1000.
    colors : tuple
        Colors for original and denoised lines. Default ('r', 'b').
    linestyles : tuple
        Linestyles for original and denoised lines. Default ('-', '-').
    labels : tuple
        Labels for original and denoised lines.
    show : bool
        If True, show the figure.
    ax : matplotlib.axes.Axes | None
        Target axes.
        
    Returns
    -------
    fig : Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    from scipy import stats

    def _compute_gfp(inst):
        """Compute GFP (RMS across channels)."""
        # (n_channels, n_times) or (n_epochs, n_channels, n_times)
        data = inst.get_data()
        if data.ndim == 3:
            # For Epochs, we want the GFP of the *Evoked* (mean over epochs)
            # But for CI, we need dispersion.
            # Standard GFP of Evoked:
             evoked_data = data.mean(axis=0)
             return np.sqrt(np.mean(evoked_data**2, axis=0))
        else:
            return np.sqrt(np.mean(data**2, axis=0))

    def _bootstrap_gfp(inst):
        """Bootstrap GFP of the Mean (Evoked)."""
        data = inst.get_data() # (n_epochs, n_channels, n_times)
        n_epochs = data.shape[0]
        rng = np.random.default_rng(42)
        
        boots = []
        for _ in range(n_boot):
             idx = rng.choice(n_epochs, n_epochs, replace=True)
             # Bootstrapped evoked
             evoked_boot = data[idx].mean(axis=0) 
             # GFP of bootstrapped evoked
             gfp_boot = np.sqrt(np.mean(evoked_boot**2, axis=0))
             boots.append(gfp_boot)
        
        boots = np.array(boots)
        alpha = (1 - ci) / 2
        return np.percentile(boots, [100*alpha, 100*(1-alpha)], axis=0)

    # Plot Original
    times = inst_orig.times
    gfp_orig = _compute_gfp(inst_orig)
    ax.plot(times, gfp_orig, color=colors[0], linestyle=linestyles[0], 
            label=labels[0], linewidth=1.5)
    
    if ci is not None and isinstance(inst_orig, mne.BaseEpochs):
        ci_low, ci_high = _bootstrap_gfp(inst_orig)
        ax.fill_between(times, ci_low, ci_high, color=colors[0], alpha=0.2, linewidth=0)

    # Plot Denoised
    gfp_denoised = _compute_gfp(inst_denoised)
    ax.plot(times, gfp_denoised, color=colors[1], linestyle=linestyles[1], 
            label=labels[1], linewidth=1.5)

    if ci is not None and isinstance(inst_denoised, mne.BaseEpochs):
         ci_low, ci_high = _bootstrap_gfp(inst_denoised)
         ax.fill_between(times, ci_low, ci_high, color=colors[1], alpha=0.2, linewidth=0)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Global Field Power (Amplitude)")
    ax.set_title("Evoked Response Comparison (GFP)")
    ax.legend(loc='best')
    ax.grid(True, linestyle=':')
    
    if show:
        plt.show()
    return fig

def plot_time_course_comparison(inst_orig, inst_denoised, picks=None, start=0, stop=None, show=True):
    """Butterfly plot of time courses."""

    if picks is None:
        picks = list(range(min(5, len(inst_orig.ch_names))))
        
    times = inst_orig.times

    if isinstance(inst_orig, (mne.io.BaseRaw, mne.io.RawArray)):
        data1 = inst_orig.get_data(picks=picks, start=start, stop=stop)
        data2 = inst_denoised.get_data(picks=picks, start=start, stop=stop)
        if start is not None:
             times = times[start:stop] if stop else times[start:]
    elif isinstance(inst_orig, mne.BaseEpochs):
        data1 = inst_orig.get_data(picks=picks)
        data2 = inst_denoised.get_data(picks=picks)
    else: # Evoked or Array
        data1 = inst_orig.get_data(picks=picks)
        data2 = inst_denoised.get_data(picks=picks)
    
    fig, axes = plt.subplots(len(picks), 1, sharex=True, figsize=(10, 2*len(picks)))
    if len(picks) == 1: 
        axes = [axes]
        
    for i, idx in enumerate(picks):
        ax = axes[i]
        # data (n_ch, n_times) or (n_epochs, n_ch, n_times)
        if data1.ndim == 3:
             d1 = data1[:, i, :].mean(axis=0)
             d2 = data2[:, i, :].mean(axis=0)
        else:
             d1 = data1[i]
             d2 = data2[i]
             
        ax.plot(times, d1, label='Original', alpha=0.6)
        ax.plot(times, d2, label='Denoised', alpha=0.6)
        ax.set_ylabel(inst_orig.ch_names[idx])
        if i == 0:
            ax.legend()
            
    axes[-1].set_xlabel("Time (s)")
    if show:
        plt.show()
    return fig

def plot_power_map(inst_orig, inst_denoised, info=None, show=True, ax=None):
    """
    Plot topomap of the ratio of variance (Denoised / Original).
    
    This plots the spatial distribution of the signal power that is preserved 
    after denoising. A value close to 1.0 means no power was removed. 
    Lower values indicate power reduction (noise removal).
    
    Parameters
    ----------
    inst_orig : instance of Raw | Epochs | Evoked
        The original data instance.
    inst_denoised : instance of Raw | Epochs | Evoked
        The denoised data instance.
    info : mne.Info | None
        Measurement info. If None, it is obtained from ``inst_orig``.
    show : bool
        If True, show the figure.
    ax : matplotlib.axes.Axes | None
        The axes to plot to. If None, a new figure is created.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure handle.
    """
    if info is None:
        if hasattr(inst_orig, 'info'):
            info = inst_orig.info
        else:
            raise ValueError("info is required")
            
    def _get_var(inst):
        if isinstance(inst, (mne.io.BaseRaw, mne.io.RawArray, mne.Evoked)):
            d = inst.get_data() # (n_ch, n_times)
            return np.var(d, axis=1)
        elif isinstance(inst, mne.BaseEpochs):
            d = inst.get_data() # (n_epochs, n_ch, n_times)
            return np.mean(np.var(d, axis=2), axis=0)
        else:
             raise ValueError("Unknown data type")
             
    var1 = _get_var(inst_orig)
    var2 = _get_var(inst_denoised)
    
    ratio = var2 / var1
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
        
    im, _ = mne.viz.plot_topomap(ratio, info, axes=ax, show=False, names=inst_orig.ch_names,
                                 vlim=(0, 1), cmap='viridis')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Power Ratio (Denoised/Original)')
    ax.set_title("Preserved Power Fraction")
    
    if show:
        plt.show()
    return fig

def plot_spectrogram_comparison(inst_orig, inst_denoised, fmin=1, fmax=40, n_freqs=20, picks=None, show=True):
    """
    Compare Time-Frequency spectrograms (averaged over channels).
    
    Displays three panels: Original, Denoised, and Difference (Original - Denoised).
    Result is averaged over the specified `picks`.
    
    Parameters
    ----------
    inst_orig : instance of Raw | Epochs | Evoked
        The original data instance.
    inst_denoised : instance of Raw | Epochs | Evoked
        The denoised data instance.
    fmin : float
        Lower frequency bound.
    fmax : float
        Upper frequency bound.
    n_freqs : int
        Number of frequencies to compute.
    picks : list | str | None
        Channels to average over. If None, averages all data channels.
    show : bool
        If True, show the figure.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure handle.
    """
    freqs = np.linspace(fmin, fmax, n_freqs)
    
    # Helper to compute TFR
    def _compute_tfr_safe(inst, label):
        tfr = inst.compute_tfr(method='multitaper', freqs=freqs, n_cycles=freqs/2., picks=picks)
        return tfr.data.mean(axis=0)  # (n_freqs, n_times)
             
    data1 = _compute_tfr_safe(inst_orig, "Original")
    data2 = _compute_tfr_safe(inst_denoised, "Denoised")
    
    # Difference
    diff = data1 - data2
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True, sharey=True)
    
    # Common scaling
    vmax = max(abs(data1).max(), abs(data2).max())
    vmin = 0
    # Times need to be extracted from one instance
    times = inst_orig.times
    
    # Helper
    def _plot_im(ax, d, title, cmap='RdBu_r', vlims=None):
        if vlims:
            _vmin, _vmax = vlims
        else:
            _vmin, _vmax = None, None
            
        im = ax.imshow(d, origin='lower', aspect='auto', cmap=cmap, 
                       extent=[times[0], times[-1], freqs[0], freqs[-1]],
                       vmin=_vmin, vmax=_vmax)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Freq (Hz)")
        plt.colorbar(im, ax=ax)
        
    _plot_im(axes[0], data1, "Original", cmap='viridis', vlims=(vmin, vmax))
    _plot_im(axes[1], data2, "Denoised", cmap='viridis', vlims=(vmin, vmax))
    _plot_im(axes[2], diff, "Original - Denoised", cmap='RdBu_r') # Diverging for diff
    
    if show:
        plt.show()
    return fig

def plot_denoising_summary(inst_orig, inst_denoised, info=None, show=True):
    """
    Plot comprehensive denoising summary.
    
    Layout:
    - Top Left: Power Map (Preserved Power Ratio)
    - Top Right: PSD Comparison (Log scale)
    - Bottom: Global Field Power (GFP) comparison over time
    
    Parameters
    ----------
    inst_orig : instance of Raw | Epochs | Evoked
        Original data.
    inst_denoised : instance of Raw | Epochs | Evoked
        Denoised data.
    info : mne.Info | None
        Measurement info.
    show : bool
        If True, show the figure.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1])
    
    # Ax1: Power Map (Top Left)
    ax_map = fig.add_subplot(gs[0, 0])
    plot_power_map(inst_orig, inst_denoised, info=info, ax=ax_map, show=False)
    
    # Ax2: PSD (Top Right)
    ax_psd = fig.add_subplot(gs[0, 1])
    plot_psd_comparison(inst_orig, inst_denoised, ax=ax_psd, show=False)
    
    # Ax3: GFP (Bottom)
    ax_gfp = fig.add_subplot(gs[1, :])
    
    def _get_gfp(inst):
        if isinstance(inst, (mne.io.BaseRaw, mne.io.RawArray, mne.Evoked)):
            d = inst.get_data()
            return np.std(d, axis=0)
        elif isinstance(inst, mne.BaseEpochs):
            # Mean of single-trial GFPs
            d = inst.get_data() # (n_epochs, n_ch, n_times)
            gfp_trials = np.std(d, axis=1) # (n_epochs, n_times)
            return np.mean(gfp_trials, axis=0)
        return None
        
    gfp1 = _get_gfp(inst_orig)
    gfp2 = _get_gfp(inst_denoised)
    times = inst_orig.times
    
    if gfp1 is not None:
        ax_gfp.plot(times, gfp1, label='Original GFP', color='k', alpha=0.7)
        ax_gfp.plot(times, gfp2, label='Denoised GFP', color='r', alpha=0.7)
        ax_gfp.fill_between(times, gfp1, gfp2, color='gray', alpha=0.2, label='Difference')
        ax_gfp.legend()
        ax_gfp.set_xlabel("Time (s)")
        ax_gfp.set_ylabel("Global Field Power")
        ax_gfp.set_title("Temporal Signal Comparison (GFP)")
        ax_gfp.grid(True, linestyle=':')
        
    fig.suptitle("Denoising Summary", fontsize=14, fontweight='bold')
    
    if show:
        plt.show()
    return fig
