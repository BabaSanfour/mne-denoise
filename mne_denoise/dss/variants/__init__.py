"""DSS Variants and Applications.

Contains specialized implementations of DSS for specific tasks:
- TSR: Time-Shift DSS for temporal structure
- SSVEP: SSVEP enhancement using comb filters
- Narrowband: Frequency scanning and band-specific extraction
"""

from .tsr import time_shift_dss, smooth_dss
from .ssvep import ssvep_dss
from .narrowband import narrowband_scan, narrowband_dss

__all__ = [
    "time_shift_dss",
    "smooth_dss",
    "ssvep_dss",
    "narrowband_scan",
    "narrowband_dss",
]
