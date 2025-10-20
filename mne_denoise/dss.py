"""Future-facing Denoising Source Separation (DSS) scaffolding.

The current mne-denoise codebase implements ZapLine-Plus in
:mod:`mne_denoise.core`.  That implementation already performs DSS-like
operations internally; the goal of this module is to extract that logic into a
reusable component so other artefact-removal pipelines can share it.

This file intentionally contains *structure only*.  Concrete numerical routines
will be ported from :class:`mne_denoise.core.PyZaplinePlus` and related helper
functions once the public API stabilises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Tuple


@dataclass
class DSSConfig:
    """Container for DSS hyperparameters and diagnostic switches.

    Planned fields mirror the internals of :class:`~mne_denoise.core.PyZaplinePlus`:
    number of components to retain, regularisation strength, reference metrics,
    and flags controlling harmonics weighting.  The defaults listed here are
    placeholders and will be reconciled with the legacy implementation during
    the refactor.
    """

    n_components: Optional[int] = None
    regularization: Optional[float] = None
    reference_metric: str = "variance"
    store_filters: bool = True
    store_patterns: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


class DSS:
    """Skeleton for the future DSS estimator.

    The final version will follow a ``fit``/``transform``/``inverse_transform``
    pattern similar to scikit-learn.  For the moment, the methods only describe
    the intended behaviour while delegating actual denoising to the existing
    ZapLine-Plus machinery.
    """

    def __init__(self, config: Optional[DSSConfig] = None, **kwargs: Any) -> None:
        self.config = config or DSSConfig(extra=dict(kwargs))
        # TODO: validate kwargs and merge with config in a consistent manner.

    def fit(self, data: Iterable[Iterable[float]], *, sfreq: float) -> "DSS":
        """Estimate DSS filters from data."""
        raise NotImplementedError(
            "DSS.fit will be implemented by extracting the covariance / "
            "eigendecomposition steps from PyZaplinePlus."
        )

    def transform(
        self,
        data: Iterable[Iterable[float]],
        *,
        return_mixing: bool = False,
    ) -> Tuple[Any, Optional[Any]]:
        """Apply previously fitted DSS filters to new data."""
        raise NotImplementedError(
            "DSS.transform will project data onto the DSS components derived "
            "during fit and optionally return mixing matrices."
        )

    def fit_transform(
        self,
        data: Iterable[Iterable[float]],
        *,
        sfreq: float,
        return_mixing: bool = False,
    ) -> Tuple[Any, Optional[Any]]:
        """Convenience wrapper calling :meth:`fit` followed by :meth:`transform`."""
        raise NotImplementedError(
            "DSS.fit_transform will reuse intermediate results from fit to "
            "avoid redundant decompositions."
        )
