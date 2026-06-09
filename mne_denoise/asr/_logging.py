"""Logging helpers for the ASR package.

Follows the mne-denoise convention of a per-package :class:`logging.Logger`
(``mne_denoise.asr``). The estimators expose an MNE-style ``verbose`` parameter;
:func:`set_log_level_from_verbose` maps it onto this logger so progress messages
are emitted at the requested level. Child module loggers
(``logging.getLogger(__name__)``) inherit this level unless configured
otherwise.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("mne_denoise.asr")


def set_log_level_from_verbose(verbose: bool | str | int | None) -> None:
    """Set the ASR logger level from an MNE-style ``verbose`` value.

    Parameters
    ----------
    verbose : bool | str | int | None
        ``True`` maps to ``INFO`` and ``False`` to ``WARNING``; a level name
        (e.g. ``"DEBUG"``) or integer is used directly; ``None`` leaves the
        current level unchanged so external logging configuration is respected.
    """
    if verbose is None:
        return
    if isinstance(verbose, bool):
        level: int | str = logging.INFO if verbose else logging.WARNING
    elif isinstance(verbose, str):
        level = verbose.upper()
    else:
        level = int(verbose)
    logger.setLevel(level)
