"""MNE-Denoise: Denoising tools for MNE-Python.

Modules
-------
- `asr`: Artifact Subspace Reconstruction.
- `dss`: Denoising Source Separation (Linear, Nonlinear, Variants).
- `zapline`: ZapLine line noise removal.
- `icanclean`: Reference-based artifact removal using CCA.
"""

from . import asr, dss, icanclean, zapline

__version__ = "0.0.1"

__all__ = [
    "asr",
    "dss",
    "icanclean",
    "zapline",
]
