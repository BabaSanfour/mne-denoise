"""Utility scaffolding shared by DSS and ZapLine-Plus."""

from __future__ import annotations

from typing import Any, Iterable, Sequence


def compute_psd(data: Sequence[Sequence[float]], sfreq: float) -> Any:
    """Future wrapper around the Welch helper in :mod:`mne_denoise.core`."""
    raise NotImplementedError(
        "compute_psd will likely call the existing _welch_hamming function "
        "from mne_denoise.core and format the results for the new API."
    )


def identify_peaks(psd: Any, sfreq: float) -> Iterable[float]:
    """Future convenience wrapper around noise_detection.find_next_noisefreq."""
    raise NotImplementedError(
        "identify_peaks will orchestrate successive calls to find_next_noisefreq "
        "to build a ranked list of candidate artefact frequencies."
    )


def chunk_signal(data: Sequence[Sequence[float]], max_duration: float) -> Iterable[Any]:
    """Future extraction of the adaptive chunking logic in PyZaplinePlus."""
    raise NotImplementedError(
        "chunk_signal will reuse the '_chunk_data' method from PyZaplinePlus "
        "to expose chunk boundaries to other pipelines."
    )
