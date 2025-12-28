"""Internal utilities for MNE-Denoise visualization."""

import numpy as np
import mne

def _get_info(estimator, info=None):
    """Resolve MNE info from estimator or argument."""
    if info is not None:
        return info
    if hasattr(estimator, 'info_') and estimator.info_ is not None:
        return estimator.info_
    if hasattr(estimator, '_mne_info') and estimator._mne_info is not None:
        return estimator._mne_info
    return None

def _get_patterns(estimator):
    """Extract spatial patterns from estimator."""
    if hasattr(estimator, 'patterns_') and estimator.patterns_ is not None:
        return estimator.patterns_
    raise ValueError("Estimator does not have a 'patterns_' attribute or it is None. "
                     "Make sure the estimator is fitted.")

def _get_filters(estimator):
    """Extract spatial filters from estimator."""
    if hasattr(estimator, 'filters_') and estimator.filters_ is not None:
        return estimator.filters_
    raise ValueError("Estimator does not have a 'filters_' attribute or it is None.")

def _get_scores(estimator):
    """Extract scores (eigenvalues or similar) from estimator."""
    if hasattr(estimator, 'eigenvalues_'):
        return estimator.eigenvalues_
    if hasattr(estimator, 'convergence_info_'):
        # For iterative DSS, we might not have a simple "score"
        # Return None or handle differently downstream
        return None
    return None

def _get_components(estimator, data=None):
    """
    Get component time series.
    
    If 'sources_' (cached) is present (e.g. IterativeDSS), use it.
    Otherwise, compute from data using 'transform'.
    """
    # 1. Try cached sources if no new data provided
    if data is None:
        if hasattr(estimator, 'sources_') and estimator.sources_ is not None:
             return estimator.sources_
        raise ValueError("Data must be provided if sources are not cached in the estimator.")
        
    # 2. Compute sources for the provided data    
    # Check if we need to force 'return_type' for LinearDSS
    if hasattr(estimator, 'return_type'):
        old_return_type = estimator.return_type
        try:
            estimator.return_type = 'sources'
            sources = estimator.transform(data)
        finally:
            estimator.return_type = old_return_type
    else:
        # Assumed to be IterativeDSS or similar which transforms to sources
        sources = estimator.transform(data)
        
    # 3. Standardize shape to (n_components, n_times, [n_epochs])
    # LinearDSS with MNE Epochs input returns (n_epochs, n_components, n_times)
    # We want dimension 0 to be components for easier plotting.
    
    if isinstance(data, (mne.BaseEpochs, mne.epochs.BaseEpochs)): # Explicit check
        # Expected shape from transform: (n_epochs, n_comp, n_times)
        if sources.ndim == 3 and sources.shape[1] == _get_filters(estimator).shape[0]:
            # Transpose to (n_comp, n_times, n_epochs)
            sources = np.transpose(sources, (1, 2, 0))
            
    return sources

def _handle_picks(info, picks=None):
    """Wraps mne.pick_types/pick_channels."""
    if picks is None:
        return mne.pick_types(info, meg=True, eeg=True, seeg=True, ecog=True, fnirs=True, exclude='bads')
    return mne.io.pick._picks_to_idx(info, picks)
