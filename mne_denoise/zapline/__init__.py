"""ZapLine: Line noise removal using DSS.

References
----------
.. [1] de Cheveigné, A. (2020). ZapLine: A simple and effective method to remove power line artifacts. NeuroImage, 207, 116356.
.. [2] Klug, M., & Kloosterman, N. A. (2022). Zapline-plus: A Zapline extension for adaptive removal of frequency-specific noise artifacts. Human Brain Mapping, 43(9), 2743–2758.
"""

from .core import (
    dss_zapline,
    dss_zapline_adaptive,
    zapline_plus,
    ZapLineResult,
    ZapLinePlusResult,
    compute_psd_reduction
)
from .api import ZapLine

__all__ = [
    "dss_zapline",
    "dss_zapline_adaptive",
    "zapline_plus",
    "ZapLineResult",
    "ZapLinePlusResult",
    "compute_psd_reduction",
    "ZapLine",
]
