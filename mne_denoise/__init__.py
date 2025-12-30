"""MNE-Denoise: Denoising tools for MNE-Python.

Modules
-------
- `dss`: Denoising Source Separation (Linear, Nonlinear, Variants).
- `zapline`: ZapLine line noise removal.
"""

from . import dss, zapline

__version__ = "0.0.1"

__all__ = [
    "dss",
    "zapline",
]
