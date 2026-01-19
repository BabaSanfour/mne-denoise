"""ZapLine: Line noise removal using DSS.

References
----------
de Cheveigné, A. (2020). ZapLine: A simple and effective method to remove
    power line artifacts. NeuroImage, 207, 116356.
"""

from .estimator import ZapLine
from .linear import ZapLineResult, apply_zapline, dss_zapline

__all__ = ["dss_zapline", "apply_zapline", "ZapLine", "ZapLineResult"]
