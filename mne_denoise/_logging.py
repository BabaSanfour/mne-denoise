"""Centralized MNE-style logging for :mod:`mne_denoise`.

``verbose`` is deliberately a logging control, rather than an algorithm
parameter.  ``None`` leaves the configured logger alone, booleans map to the
usual MNE levels, and strings/integers are standard :mod:`logging` levels.
Public operations use :func:`verbose` or :func:`use_log_level` so a per-call
override is always restored, including when the operation raises.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from numbers import Integral
from typing import Any, TypeVar

logger = logging.getLogger("mne_denoise")

_F = TypeVar("_F", bound=Callable[..., Any])


def _level_from_verbose(verbose: bool | str | int | None) -> int | None:
    """Resolve one MNE-style verbosity value to a logging level."""
    if verbose is None:
        return None
    if isinstance(verbose, bool):
        return logging.INFO if verbose else logging.WARNING
    if isinstance(verbose, str):
        name = verbose.upper()
        try:
            return int(logging._nameToLevel[name])  # noqa: SLF001
        except KeyError as err:
            raise ValueError(
                f"Unknown logging level {verbose!r}; use a standard logging "
                "level name or integer."
            ) from err
    if isinstance(verbose, Integral):
        return int(verbose)
    raise TypeError(
        "verbose must be None, a bool, a standard logging level name, or an "
        f"integer; got {type(verbose).__name__}."
    )


def set_log_level_from_verbose(verbose: bool | str | int | None) -> None:
    """Set the package logger level explicitly from an MNE-style value.

    Parameters
    ----------
    verbose : bool | str | int | None
        ``True`` maps to ``INFO`` and ``False`` to ``WARNING``; a level name
        (e.g. ``"DEBUG"``) or integer is used directly; ``None`` leaves the
        current level unchanged so external logging configuration is respected.
    """
    level = _level_from_verbose(verbose)
    if level is not None:
        logger.setLevel(level)


@contextmanager
def use_log_level(verbose: bool | str | int | None) -> Iterator[None]:
    """Temporarily apply an MNE-style verbosity value.

    ``verbose=None`` inherits the existing logger configuration.  A concrete
    value is restored in a ``finally`` block, which makes nested algorithm
    calls and exceptions safe.
    """
    level = _level_from_verbose(verbose)
    if level is None:
        yield
        return

    previous = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(previous)


def verbose(function: _F) -> _F:
    """Decorate a public operation with a temporary ``verbose`` override.

    The decorator accepts the same forms as MNE-Python's ``@verbose``.  For a
    bound estimator method it uses ``self.verbose`` when no keyword was
    supplied; for a function it uses its explicit ``verbose`` argument.
    """
    signature = inspect.signature(function)

    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        call_kwargs = kwargs
        if "verbose" in kwargs:
            level = kwargs["verbose"]
            if "verbose" not in signature.parameters:
                # Keep the decorator useful for legacy/inherited methods that
                # do not expose a per-call keyword in their original
                # signature. Public methods in this package declare it
                # explicitly; this fallback prevents an accidental TypeError
                # while preserving the scoped override.
                call_kwargs = dict(kwargs)
                call_kwargs.pop("verbose")
        else:
            try:
                bound = signature.bind_partial(*args, **kwargs)
            except TypeError:
                bound = None
            if bound is not None and "verbose" in bound.arguments:
                level = bound.arguments["verbose"]
            elif args and hasattr(args[0], "verbose"):
                level = args[0].verbose
            else:
                level = None
        with use_log_level(level):
            return function(*args, **call_kwargs)

    return wrapper  # type: ignore[return-value]
