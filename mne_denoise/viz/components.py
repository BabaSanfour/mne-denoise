"""Visualization for DSS components."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mne
from ._utils import _get_info, _get_patterns, _get_scores, _get_components


def plot_narrowband_scan(frequencies, eigenvalues, peak_freq=None, 
                         true_freqs=None, ax=None, show=True):
    """Plot narrowband DSS frequency scan results.
    
    Visualizes the eigenvalue spectrum across frequencies from a narrowband scan,
    helping identify dominant oscillatory components.
    
    Parameters
    ----------
    frequencies : array-like
        Frequencies that were scanned (Hz).
    eigenvalues : array-like
        Eigenvalues at each frequency (higher = stronger oscillatory component).
    peak_freq : float, optional
        Detected peak frequency to highlight.
    true_freqs : list of float, optional
        Known true frequencies to mark (e.g., for synthetic data validation).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    show : bool
        If True, show the figure.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
        
    Examples
    --------
    >>> from mne_denoise.dss.variants import narrowband_scan
    >>> from mne_denoise.viz import plot_narrowband_scan
    >>> _, freqs, eigs = narrowband_scan(data, sfreq=250, freq_range=(5, 30))
    >>> plot_narrowband_scan(freqs, eigs, peak_freq=freqs[np.argmax(eigs)])
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure
        
    # Plot eigenvalue spectrum
    ax.plot(frequencies, eigenvalues, 'b-o', markersize=4, linewidth=2)
    
    # Highlight peak if provided
    if peak_freq is not None:
        peak_idx = np.argmin(np.abs(frequencies - peak_freq))
        ax.plot(peak_freq, eigenvalues[peak_idx], 'r*', markersize=15, 
                label=f'Peak: {peak_freq:.1f} Hz')
        ax.axvline(peak_freq, color='red', linestyle='--', alpha=0.5)
    
    # Mark true frequencies if provided (for synthetic data)
    if true_freqs is not None:
        for i, freq in enumerate(true_freqs):
            color = ['red', 'green', 'orange', 'purple'][i % 4]
            ax.axvline(freq, color=color, linestyle='--', alpha=0.5, 
                      label=f'True: {freq} Hz')
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('DSS Score (Eigenvalue)')
    ax.set_title('Narrowband Scan: Oscillatory Component Detection')
    ax.grid(True, alpha=0.3)
    
    # Only show legend if there are labeled elements
    if peak_freq is not None or true_freqs is not None:
        ax.legend()
    
    if show:
        plt.show()
    return fig


def plot_score_curve(estimator, mode='raw', ax=None, show=True):
    """
    Plot component scores (eigenvalues or power ratios).
    
    Parameters
    ----------
    estimator : instance of DSS | IterativeDSS
        The fitted estimator.
    mode : {'raw', 'cumulative', 'ratio'}
        The plotting mode.
        - 'raw': Plot pure eigenvalues/scores.
        - 'cumulative': Plot cumulative normalized score.
        - 'ratio': Plot score as a ratio (biased/baseline).
    ax : matplotlib.axes.Axes | None
        The target axes. If None, a new figure is created.
    show : bool
        If True, show the figure.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure handle.
    """
    scores = _get_scores(estimator)
    if scores is None:
        print("No scores (eigenvalues) found in estimator.")
        return None
        
    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)
    else:
        fig = ax.figure
        
    n_comp = len(scores)
    x = np.arange(1, n_comp + 1)
    
    if mode == 'cumulative':
        y = np.cumsum(scores)
        y = y / y[-1]
        ylabel = "Cumulative Score (Normalized)"
    elif mode == 'ratio':
        y = scores
        ylabel = "Power Ratio (Biased / Baseline)"
    else: # raw
        y = scores
        ylabel = "Score / Eigenvalue"
        
    ax.plot(x, y, '.-', color='black')
    ax.set_xlabel("Component")
    ax.set_ylabel(ylabel)
    ax.set_title("Component Scores")
    ax.grid(True, linestyle=':')
    
    if show:
        plt.show()
    return fig

def plot_spatial_patterns(estimator, info=None, n_components=None, show=True):
    """
    Plot spatial patterns (topomaps) for components.
    
    Parameters
    ----------
    estimator : instance of DSS | IterativeDSS
        The fitted estimator.
    info : mne.Info | None
        Measurement info. If None, it is obtained from the estimator.
    n_components : int | None
        Number of components to plot. If None, plots all available pattern components.
    show : bool
        If True, show the figure.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure handle.
    """
    info = _get_info(estimator, info)
    patterns = _get_patterns(estimator) # (n_ch, n_comp)
    
    if info is None:
        raise ValueError("MNE Info is required for plotting topomaps.")
        
    if n_components is None:
        n_components = patterns.shape[1]
    
    # Create temp info with sfreq=1.0 so 'times' correspond to indices (0, 1, 2...)
    # This ensures "Comp %d" formats correctly as "Comp 0", "Comp 1", etc.
    temp_info = info.copy()
    with temp_info._unlock():
        temp_info['sfreq'] = 1.0
        
    evoked = mne.EvokedArray(patterns[:, :n_components], temp_info, tmin=0)
    
    # Times are now [0.0, 1.0, 2.0, ...]
    times = evoked.times
    
    fig = evoked.plot_topomap(times=times, ch_type=None, show=show, 
                              time_format="Comp %d", colorbar=True)
    return fig

def plot_component_summary(estimator, data=None, info=None, n_components=None, 
                         show=True, plot_ci=True):
    """
    Comprehensive dashboard for checking DSS components.
    
    Plots Topomap | Time Course | Power Spectrum for selected components side-by-side.
    
    Parameters
    ----------
    estimator : instance of DSS | IterativeDSS
        The fitted estimator.
    data : instance of Raw | Epochs | array | None
        The data to transform to component sources. 
        If None, sources must be cached in the estimator (e.g. IterativeDSS).
    info : mne.Info | None
        Measurement info. If None, it is obtained from the estimator.
    n_components : int | list of int | None
        Components to plot. If int, the first n components are successfully plotted.
        If list, specific component indices are plotted.
    show : bool
        If True, show the figure.
    plot_ci : bool
        If True and data is Epochs/3D, plot 95% confidence intervals (SEM).

    Returns
    -------
    fig : matplotlib.figure.Figure | None
        The figure handle. Returns None if no components are plotted.
    """
    info = _get_info(estimator, info)
    patterns = _get_patterns(estimator)
    sources = _get_components(estimator, data)
    
    if sources is None:
        raise ValueError("Component sources missing. Provide `data` or ensure estimator has `sources_`.")
        
    if isinstance(n_components, int):
        cols = list(range(n_components))
    elif n_components is not None:
        cols = n_components
    else:
        # Default first 5
        cols = list(range(min(5, patterns.shape[1])))
        
    n_plots = len(cols)
    if n_plots == 0:
        return
        
    # Layout: Rows = Components
    # Cols = Topo, Time, PSD
    fig = plt.figure(figsize=(12, 3 * n_plots), constrained_layout=True)
    gs = GridSpec(n_plots, 3, figure=fig, width_ratios=[1, 2, 1])
    
    # Pre-compute PSD variables if needed
    sfreq = info['sfreq'] if info else 1000.0
    
    for idx, comp_idx in enumerate(cols):
        # 1. Topomap
        ax_topo = fig.add_subplot(gs[idx, 0])
        if info:
            # Handle mixed channel types (plot_topomap requires single type)
            # We prioritize: Grad > Mag > EEG > SEEG > ECoG
            picks = None
            ch_types_dict = mne.channel_indices_by_type(info)
            for ch_type in ['grad', 'mag', 'eeg', 'seeg', 'ecog']:
                if ch_type in ch_types_dict and len(ch_types_dict[ch_type]) > 0:
                     picks = ch_types_dict[ch_type]
                     break
            
            # If no prioritized type found, take whatever is available (e.g. 'hbo')
            if picks is None:
                 # Take the first non-empty list from the dict
                 for idxs in ch_types_dict.values():
                     if len(idxs) > 0:
                         picks = idxs
                         break
            
            if picks is not None:
                topo_info = mne.pick_info(info, picks)
                topo_data = patterns[picks, comp_idx]
                mne.viz.plot_topomap(topo_data, topo_info, axes=ax_topo, show=False)
                ax_topo.set_title(f"Comp {comp_idx} Pattern")
            else:
                 ax_topo.text(0.5, 0.5, "No Channels", ha='center')
        else:
            ax_topo.text(0.5, 0.5, "No Info for Topo", ha='center')
            
        # 2. Time Course
        ax_time = fig.add_subplot(gs[idx, 1])
        
        # Sources: (n_comp, n_times) or (n_comp, n_times, n_epochs)
        if sources.ndim == 3:
            # (n_comp, n_times, n_epochs)
            comp_data = sources[comp_idx] # (n_times, n_epochs)
            
            # Mean
            mean_tc = comp_data.mean(axis=1)
            times = np.arange(len(mean_tc)) / sfreq
            ax_time.plot(times, mean_tc, label='Mean', color='k')
            
            if plot_ci:
                # Simple std err or bootstrap
                std_tc = comp_data.std(axis=1) / np.sqrt(comp_data.shape[1])
                ax_time.fill_between(times, mean_tc - 2*std_tc, mean_tc + 2*std_tc, 
                                     color='gray', alpha=0.3, label='95% CI (SEM)')
        else:
            # (n_comp, n_times)
            comp_data = sources[comp_idx]
            times = np.arange(len(comp_data)) / sfreq
            ax_time.plot(times, comp_data, color='k')
            
        ax_time.set_title(f"Comp {comp_idx} Time Course")
        ax_time.set_xlabel("Time (s)")
        
        # 3. PSD
        ax_psd = fig.add_subplot(gs[idx, 2])
        if sources.ndim == 3:
            d_flat = sources[comp_idx].T # (n_epochs, n_times)
        else:
            d_flat = sources[comp_idx][np.newaxis, :]
            
        psd_spec, freqs = mne.time_frequency.psd_array_welch(
            d_flat, sfreq=sfreq, fmin=0, fmax=np.inf, n_fft=min(2048, d_flat.shape[-1]), verbose=False
        )
        # psd_spec: (n_epochs, n_freqs)
        mean_psd = np.mean(psd_spec, axis=0) # (n_freqs,)
        
        ax_psd.semilogy(freqs, mean_psd)
        ax_psd.set_title("PSD")
        ax_psd.set_xlabel("Freq (Hz)")
        ax_psd.set_xlim(0, min(100, sfreq/2)) # Zoom to relevant freq
        
    if show:
        plt.show()
    return fig

def plot_component_image(estimator, data=None, n_components=None, show=True):
    """
    Plot raster image of components (epochs/trials view).
    
    Parameters
    ----------
    estimator : instance of DSS | IterativeDSS
        The fitted estimator.
    data : instance of Raw | Epochs | array | None
        The data to transform.
    n_components : int | None
        Number of components to plot. If None, plots first 5.
    show : bool
        If True, show the figure.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure handle.
    """
    sources = _get_components(estimator, data)
    if sources is None:
        raise ValueError("Sources not available. Provide data.")

    # Handle 2D: (n_comp, n_times) -> (n_comp, n_times, 1) to treat as 1 epoch
    if sources.ndim == 2:
        sources = sources[:, :, np.newaxis]
        
    if sources.ndim != 3:
        raise ValueError("Requires component sources.")
        
    if n_components is None:
        n_components = min(5, sources.shape[0])
        
    fig, axes = plt.subplots(n_components, 1, figsize=(8, 2*n_components), sharex=True)
    if n_components == 1:
        axes = [axes]
        
    for i in range(n_components):
        # Image: x=time, y=epochs
        # sources: (n_comp, n_times, n_epochs) -> (n_epochs, n_times)
        img = sources[i].T 
        im = axes[i].imshow(img, aspect='auto', origin='lower', cmap='RdBu_r')
        axes[i].set_title(f"Comp {i}")
        axes[i].set_ylabel("Epochs")
        
    axes[-1].set_xlabel("Time (samples)")
    
    if show:
        plt.show()
    return fig

def plot_component_time_series(estimator, data=None, n_components=None, 
                             show=True, ax=None):
    """
    Plot stacked component time series (vertical offsets).
    
    Replicates the "Figure 3a" style from de Cheveigné & Simon (2008),
    showing multiple components simultaneously to identify stimulus-driven ones.
    
    Parameters
    ----------
    estimator : instance of DSS | IterativeDSS
        The fitted estimator.
    data : instance of Raw | Epochs | array | None
        The data to transform.
    n_components : int | None
        Number of components to plot. If None, plots first 20.
    show : bool
        If True, show the figure.
    ax : matplotlib.axes.Axes | None
        Target axes.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    sources = _get_components(estimator, data)
    scores = _get_scores(estimator)
    
    if sources is None:
        raise ValueError("Sources not available. Provide data.")
        
    # Handle sources shape
    if sources.ndim == 3:
        # If epochs, plot the Trial Average (Evoked)
        # (n_comps, n_times, n_epochs) -> (n_comps, n_times)
        sources = sources.mean(axis=2)
    
    if n_components is None:
        n_components = min(20, sources.shape[0])
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(6, n_components * 0.4)))
    else:
        fig = ax.figure
        
    times = np.arange(sources.shape[1])
    # Try to get real times from info if possible, but estimator doesn't always handle it easily
    # If data was passed, we could check attributes, but `_get_components` consumes it.
    # For now, samples is safe.
    
    offset_step = 3.0
    
    for i in range(n_components):
        comp = sources[i]
        # Z-score normalize for standardized display
        std = np.std(comp)
        if std < 1e-15:
            std = 1.0
        comp_norm = comp / std
        
        # Plot with offset
        offset = -i * offset_step
        
        # Color logic: Highlight first few? Or just blue?
        # Legacy script highlights reproducible ones. We don't have reproducibility metric here easily unless calculated.
        # Just use blue/gray.
        color = 'steelblue' if i < 5 else 'gray'
        alpha = 0.8 if i < 5 else 0.5
        
        ax.plot(times, comp_norm + offset, color=color, alpha=alpha, linewidth=1.5)
        
        # Label
        label = f"Comp {i}"
        if scores is not None and i < len(scores):
            label += f" (λ={scores[i]:.2f})"
        ax.text(times[-1], offset, label, va='center', fontsize=8, color=color)
        
    ax.set_yticks([])
    ax.set_xlabel("Time (samples)")
    ax.set_title(f"First {n_components} Component Time Series")
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    if show:
        plt.show()
    return fig
