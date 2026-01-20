"""ZapLine-plus pipeline implementation.

Implements the "ZapLine-plus" algorithm (Klug & Kloosterman, 2022).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union, Dict

import numpy as np
from scipy.signal import welch

from .core import dss_zapline, ZapLineResult
from .adaptive import (
    find_noise_freqs,
    segment_data,
    find_fine_peak,
    check_artifact_presence,
    detect_harmonics,
)
from .cleanline import apply_hybrid_cleanup
from ..dss.utils import iterative_outlier_removal

logger = logging.getLogger(__name__)


def dss_zapline_plus(
    data: np.ndarray,
    sfreq: float,
    line_freqs: Optional[Union[float, List[float]]] = None,
    fmin: float = 17.0,
    fmax: float = 99.0,
    n_remove_params: Optional[Dict] = None,
    qa_params: Optional[Dict] = None,
    process_harmonics: bool = False,
    max_harmonics: Optional[int] = None,
    hybrid_fallback: bool = False,
    min_chunk_len: float = 30.0,
) -> ZapLineResult:
    """Apply Zapline-plus to data.
    
    Parameters
    ----------
    data : ndarray
        Input data (n_channels, n_times).
    sfreq : float
        Sampling frequency.
    line_freqs : float or list or None
        Target line frequencies. If None, automatic detection is used.
    fmin, fmax : float
        Frequency range for automatic detection.
    n_remove_params : dict
        Parameters for component selection (sigma, min_remove, max_prop).
    qa_params : dict
        Parameters for QA loop (max_sigma, min_sigma).
    process_harmonics : bool
        If True, detect and process harmonics of detected frequencies.
    max_harmonics : int, optional
        Maximum number of harmonics to process.
    hybrid_fallback : bool
        If True, apply hybrid notch cleanup when QA fails.
    min_chunk_len : float
        Minimum length of adaptive segments in seconds (default 30.0).
        
    Returns
    -------
    result : ZapLineResult
    """
    data = np.asarray(data)
    n_channels, n_times = data.shape
    
    # Defaults
    if n_remove_params is None:
        n_remove_params = {}
    sigma_init = n_remove_params.get('sigma', 3.0)
    min_remove = n_remove_params.get('min_remove', 1)
    max_prop_remove = n_remove_params.get('max_prop', 0.2)
    
    if qa_params is None:
        qa_params = {}
    max_sigma = qa_params.get('max_sigma', 4.0)
    min_sigma = qa_params.get('min_sigma', 2.5)
    
    # 1. Automatic frequency detection
    if line_freqs is None:
        logger.info("Detecting line noise frequencies...")
        line_freqs = find_noise_freqs(data, sfreq, fmin=fmin, fmax=fmax)
        logger.info(f"Detected: {line_freqs}")
    elif isinstance(line_freqs, (int, float)):
        line_freqs = [float(line_freqs)]
        
    if not line_freqs:
        return ZapLineResult(
            cleaned=data.copy(),
            removed=np.zeros_like(data),
            n_removed=0,
            dss_filters=np.array([]),
            dss_patterns=np.array([]),
            dss_eigenvalues=np.array([]),
            line_freq=0.0,
            removed_topographies=[],
            chunk_info=[],
        )
    
    current_data = data.copy()
    all_chunk_topographies = []
    all_chunk_metadata = []
    
    # Build list of all frequencies to process
    all_freqs_to_process = []
    for lfreq in line_freqs:
        all_freqs_to_process.append(lfreq)
        if process_harmonics:
            harmonics = detect_harmonics(current_data, sfreq, lfreq, max_harmonics)
            all_freqs_to_process.extend(harmonics)
    
    logger.info(f"Will process frequencies: {all_freqs_to_process}")
    
    # Process each frequency
    for freq_i, target_freq in enumerate(all_freqs_to_process):
        logger.info(f"Processing frequency {freq_i+1}/{len(all_freqs_to_process)}: {target_freq:.2f} Hz")
        
        # Adaptive segmentation
        segments = segment_data(current_data, sfreq, target_freq=target_freq, min_chunk_len=min_chunk_len)

        logger.info(f"  Segments: {len(segments)}")
        
        cleaned_chunks = []
        
        # Process each segment
        for seg_idx, (start, end) in enumerate(segments):
            chunk = current_data[:, start:end]
            
            # Fine-tune frequency
            fine_freq = find_fine_peak(chunk, sfreq, target_freq)
            
            # Check artifact presence
            present = check_artifact_presence(chunk, sfreq, fine_freq)
            
            # Set parameters
            current_sigma = sigma_init
            current_min_remove = min_remove if present else 0
            
            # QA loop
            max_retries = 5
            best_chunk_clean = None
            is_too_strong = False
            
            for retry in range(max_retries):
                res = dss_zapline(
                    chunk, 
                    line_freq=fine_freq, 
                    sfreq=sfreq, 
                    n_remove='auto', 
                    threshold=current_sigma
                )
                
                n_rem = res.n_removed
                max_rem_cap = int(n_channels * max_prop_remove)
                n_rem = min(n_rem, max_rem_cap)
                n_rem = max(n_rem, current_min_remove)
                
                if n_rem != res.n_removed:
                    res = dss_zapline(
                        chunk, 
                        line_freq=fine_freq, 
                        sfreq=sfreq, 
                        n_remove=n_rem
                    )
                
                status = _check_spectral_qa(res.cleaned, sfreq, fine_freq)
                
                if status == "ok":
                    best_chunk_clean = res.cleaned
                    break
                elif status == "weak":
                    if is_too_strong:
                        best_chunk_clean = res.cleaned
                        break
                    else:
                        current_sigma = max(current_sigma - 0.25, min_sigma)
                        current_min_remove = current_min_remove + 1
                elif status == "strong":
                    is_too_strong = True
                    current_sigma = min(current_sigma + 0.25, max_sigma)
                    current_min_remove = max(current_min_remove - 1, 0)
            
            if best_chunk_clean is None:
                best_chunk_clean = res.cleaned
            
            # Apply hybrid fallback if enabled and still weak
            if hybrid_fallback and status == "weak":
                best_chunk_clean = apply_hybrid_cleanup(
                    best_chunk_clean, sfreq, fine_freq
                )
            
            cleaned_chunks.append(best_chunk_clean)
            
            # Collect topographies
            if res.n_removed > 0 and res.dss_patterns.size > 0:
                removed_patterns = res.dss_patterns[:, :res.n_removed]
                all_chunk_topographies.append(removed_patterns)
            else:
                all_chunk_topographies.append(None)
            
            all_chunk_metadata.append({
                'frequency': target_freq,
                'fine_freq': fine_freq,
                'start': start,
                'end': end,
                'n_removed': res.n_removed,
                'artifact_present': present,
            })
        
        # Concatenate chunks for this frequency
        if cleaned_chunks:
            current_data = np.concatenate(cleaned_chunks, axis=1)
    
    return ZapLineResult(
        cleaned=current_data,
        removed=data - current_data,
        n_removed=-1,  # Mixed
        dss_filters=np.array([]),
        dss_patterns=np.array([]),
        dss_eigenvalues=np.array([]),
        line_freq=line_freqs[0] if line_freqs else 0,
        removed_topographies=all_chunk_topographies,
        chunk_info=all_chunk_metadata,
    )


def _check_spectral_qa(data: np.ndarray, sfreq: float, target_freq: float) -> str:
    """Check if cleaning was too weak, too strong, or ok."""
    n_times = data.shape[1]
    n_fft = min(n_times, int(sfreq * 4))
    freqs, psd = welch(data, fs=sfreq, nperseg=n_fft, axis=-1)
    
    psd_log = 10 * np.log10(np.clip(psd, 1e-20, None))
    mean_log_psd = np.mean(psd_log, axis=0)
    
    f_win_low = target_freq - 3
    f_win_high = target_freq + 3
    
    mask_win = (freqs >= f_win_low) & (freqs <= f_win_high)
    win_psd = mean_log_psd[mask_win]
    if len(win_psd) < 3:
        return "ok"
    
    n_third = len(win_psd) // 3
    left = win_psd[:n_third]
    right = win_psd[-n_third:]
    center_power = np.mean(np.concatenate([left, right]))
    
    q_left = np.percentile(left, 5)
    q_right = np.percentile(right, 5)
    deviation = center_power - np.mean([q_left, q_right])
    
    # Check "Too weak"
    f_tight_low = target_freq - 0.05
    f_tight_high = target_freq + 0.05
    mask_tight = (freqs >= f_tight_low) & (freqs <= f_tight_high)
    tight_psd = mean_log_psd[mask_tight]
    thresh_weak = center_power + 2 * deviation
    
    if np.any(tight_psd > thresh_weak):
        return "weak"
    
    # Check "Too strong"
    f_notch_low = target_freq - 0.4
    f_notch_high = target_freq + 0.1
    mask_notch = (freqs >= f_notch_low) & (freqs <= f_notch_high)
    notch_psd = mean_log_psd[mask_notch]
    thresh_strong = center_power - 2 * deviation
    
    if np.any(notch_psd < thresh_strong):
        return "strong"
    
    return "ok"
