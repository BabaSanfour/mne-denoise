"""ZapLine: Line noise removal using DSS.

References
----------
de Cheveigné, A. (2020). ZapLine: A simple and effective method to remove
    power line artifacts. NeuroImage, 207, 116356.
"""

from .estimator import ZapLine
from .core import ZapLineResult, apply_zapline, dss_zapline
from .plus import dss_zapline_plus

__all__ = ["dss_zapline", "apply_zapline", "dss_zapline_plus", "ZapLine", "ZapLineResult"]
