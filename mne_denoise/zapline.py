"""ZapLine-Plus orchestration scaffolding.

While :mod:`mne_denoise.core` currently provides the full implementation,
this module sketches the architecture we plan to migrate to.  The functions
defined here wrap the legacy API so downstream code can experiment with the
future layout without losing functionality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .core import PyZaplinePlus, zapline_plus as _legacy_zapline_plus


@dataclass
class ZapLineConfig:
    """Structured view of the configuration dictionary used today.

    When the refactor lands, the dictionary produced by
    :class:`~mne_denoise.core.PyZaplinePlus` will be converted into this
    dataclass (and back again) to keep the public API stable.
    """

    noisefreqs: Any = "line"
    adaptive_n_remove: bool = True
    fixed_n_remove: int = 1
    legacy_payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ZapLineSummary:
    """Container for analytics emitted by the ZapLine-Plus pipeline.

    TODO:
        * Surface per-frequency attenuation metrics.
        * Expose DSS eigenvalues / component scores for inspection.
        * Track warnings or recovery actions triggered during processing.
    """

    frequencies_processed: List[float] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def zapline_plus(
    data: Sequence[Sequence[float]],
    sfreq: float,
    *,
    noisefreqs: Any = "line",
    adaptiveNremove: bool = True,
    fixedNremove: int = 1,
    **kwargs: Any,
) -> Tuple[Any, ZapLineConfig, ZapLineSummary, List[Any]]:
    """Delegate to :func:`mne_denoise.core.zapline_plus` while returning stubs."""

    cleaned, config_dict, analytics_dict, figures = _legacy_zapline_plus(
        data,
        sampling_rate=sfreq,
        noisefreqs=noisefreqs,
        adaptiveNremove=adaptiveNremove,
        fixedNremove=fixedNremove,
        **kwargs,
    )

    config = ZapLineConfig(
        noisefreqs=config_dict.get("noisefreqs", noisefreqs),
        adaptive_n_remove=config_dict.get("adaptiveNremove", adaptiveNremove),
        fixed_n_remove=config_dict.get("fixedNremove", fixedNremove),
        legacy_payload=config_dict,
    )
    summary = ZapLineSummary(
        frequencies_processed=_extract_frequencies(analytics_dict),
        notes=["Analytics captured in legacy dictionary; will be formalised later."],
    )

    return cleaned, config, summary, figures


def plan_frequency_detection(*, config: ZapLineConfig) -> Iterable[float]:
    """Placeholder for the future frequency-detection strategy."""
    raise NotImplementedError(
        "plan_frequency_detection will build on noise_detection.find_next_noisefreq."
    )


def plan_chunking_strategy(*, config: ZapLineConfig) -> Iterable[slice]:
    """Placeholder for automatic data chunking heuristics."""
    raise NotImplementedError("plan_chunking_strategy will reuse PyZaplinePlus logic.")


def _extract_frequencies(analytics: Dict[str, Any]) -> List[float]:
    """Pull frequency metadata from the legacy analytics dictionary."""
    frequencies: List[float] = []
    for value in analytics.values():
        freq = value.get("frequency") if isinstance(value, dict) else None
        if isinstance(freq, (int, float)):
            frequencies.append(float(freq))
    return frequencies
