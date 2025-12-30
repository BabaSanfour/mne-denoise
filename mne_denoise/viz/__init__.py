"""Visualization functions for MNE-Denoise."""

from .comparison import (
    plot_denoising_summary,
    plot_evoked_comparison,
    plot_overlay_comparison,
    plot_power_map,
    plot_psd_comparison,
    plot_spectral_psd_comparison,
    plot_spectrogram_comparison,
    plot_time_course_comparison,
)
from .components import (
    plot_component_image,
    plot_component_spectrogram,
    plot_component_summary,
    plot_component_time_series,
    plot_narrowband_scan,
    plot_score_curve,
    plot_spatial_patterns,
    plot_tf_mask,
)
from .zapline import plot_zapline_analytics

__all__ = [
    "plot_component_summary",
    "plot_spatial_patterns",
    "plot_score_curve",
    "plot_component_image",
    "plot_component_time_series",
    "plot_narrowband_scan",
    "plot_tf_mask",
    "plot_component_spectrogram",
    "plot_psd_comparison",
    "plot_time_course_comparison",
    "plot_evoked_comparison",
    "plot_spectrogram_comparison",
    "plot_power_map",
    "plot_denoising_summary",
    "plot_spectral_psd_comparison",
    "plot_overlay_comparison",
    "plot_zapline_analytics",
]
