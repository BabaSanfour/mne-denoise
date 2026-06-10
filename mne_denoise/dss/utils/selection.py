"""Component selection utilities for DSS.

Provides automatic component selection using outlier detection and
eigenvalue ratio analysis.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

import numpy as np


def iterative_outlier_removal(scores: np.ndarray, sigma: float = 3.0) -> int:
    """Detect outliers iteratively using mean + sigma threshold.

    This algorithm iteratively identifies values that exceed `mean + sigma * std`,
    removes them from consideration, and repeats until no more outliers are found.
    It follows NoiseTools' ``nt_dss``-style outlier rule, which ZapLine-plus
    (Klug & Kloosterman, 2022, §2.4 "Detection of noise components") adopts to
    automatically choose how many spatial components to remove: outliers in the
    component scores are flagged with a ``mean + sigma * SD`` threshold and
    removed, the mean/SD are recomputed across the remaining components, and the
    loop repeats until none are left; the count of removed outliers is taken as
    the number of components to reject. ZapLine-plus uses a default of
    ``sigma=3`` and reports this iterative mean/SD rule to be more robust than a
    median-absolute-deviation rule in this setting. Callers cap the resulting
    count via ``max_prop_remove`` (ZapLine-plus caps at one-fifth of the
    channels) and floor it via ``min_select``; see
    :class:`~mne_denoise.dss.linear.DSS`.

    Useful for automatic component selection in DSS applications, such as:
    - ZapLine: Selecting how many line-noise components to remove
    - Narrowband scan: Identifying significant frequency peaks
    - Any DSS where components need automatic thresholding

    Parameters
    ----------
    scores : ndarray
        Component scores (e.g., eigenvalues, power ratios).
        Higher values are considered more significant.
    sigma : float
        Sigma threshold for outlier detection. Default 3.0.
        Components with `score > mean + sigma * std` are outliers.

    Returns
    -------
    n_outliers : int
        Number of outliers (significant components) detected.

    Examples
    --------
    >>> from mne_denoise.dss.utils import iterative_outlier_removal
    >>> scores = np.array([0.9, 0.8, 0.2, 0.15, 0.1, 0.08])
    >>> n_significant = iterative_outlier_removal(scores, sigma=2.0)
    >>> print(f"Found {n_significant} significant components")

    References
    ----------
    .. [1] Klug, M., & Kloosterman, N. A. (2022). Zapline-plus: A Zapline
           extension for automatic and adaptive removal of frequency-specific
           noise artifacts in M/EEG. Human Brain Mapping, 43(9), 2743-2758.
           (§2.4, "Detection of noise components".)
    .. [2] NoiseTools: http://audition.ens.fr/adc/NoiseTools/
    """
    scores = np.asarray(scores)
    n_outliers = 0
    remaining = scores.copy()

    while len(remaining) > 2:
        mean_val = np.mean(remaining)
        std_val = np.std(remaining)

        if std_val < 1e-12:
            break

        threshold = mean_val + sigma * std_val
        outliers = remaining > threshold

        if not np.any(outliers):
            break

        n_outliers += np.sum(outliers)
        remaining = remaining[~outliers]

    return n_outliers


def auto_select_components(eigenvalues: np.ndarray, threshold: float = 3.0) -> int:
    """Select number of components using outlier detection.

    Convenience wrapper around `iterative_outlier_removal` for DSS eigenvalues.

    Parameters
    ----------
    eigenvalues : ndarray
        DSS eigenvalues (component scores).
    threshold : float
        Sigma threshold for outlier detection. Default 3.0.

    Returns
    -------
    n_components : int
        Number of significant components to keep/remove.
    """
    return iterative_outlier_removal(eigenvalues, threshold)


def detect_eigenvalue_knee(
    scores: np.ndarray,
    rel_floor: float = 0.01,
    min_ratio: float = 3.0,
) -> int:
    """Locate the knee/elbow in a sorted-descending eigenvalue spectrum.

    Finds the index ``k`` such that the largest log-space drop between
    consecutive eigenvalues occurs between ``scores[k-1]`` and ``scores[k]``,
    interpreting that drop as the boundary between a cluster of significant
    components and the noise floor. Returns ``k`` (the count of components
    above the knee).

    The drop must (a) occur above a relative floor — to avoid picking knees
    that sit entirely in the noise tail — and (b) exceed ``log10(min_ratio)``
    decades, so a smoothly decaying spectrum without a clear bimodal split
    correctly returns 0.

    Parameters
    ----------
    scores : ndarray
        Component scores in descending order (e.g., DSS eigenvalues).
    rel_floor : float, default=0.01
        Relative cutoff for valid knee anchors. Only drop positions ``i`` with
        ``scores[i] > scores[0] * rel_floor`` are considered. Default 0.01
        means anchors must be at least 1% of the maximum eigenvalue.
    min_ratio : float, default=3.0
        Minimum drop ratio (linear scale) between consecutive eigenvalues
        required to qualify as a knee. A drop of ``min_ratio=3.0`` corresponds
        to a 0.477-decade gap in log10 space.

    Returns
    -------
    n_above_knee : int
        Number of eigenvalues above the detected knee. Returns 0 when no
        qualifying knee exists (monotonic decay or all-noise spectra).

    Notes
    -----
    Complementary to :func:`iterative_outlier_removal`, which is robust when
    a small number of components stand out as statistical outliers but fails
    when many co-equal strong components are present (e.g., high-channel-count
    MEG with coherent line noise spread across many sensors). Use
    :func:`auto_select_components_robust` to combine both.

    Examples
    --------
    >>> import numpy as np
    >>> # User MEG case from Issue #34: 7 strong + 8 near-zero
    >>> evs = np.array(
    ...     [
    ...         0.989,
    ...         0.969,
    ...         0.715,
    ...         0.677,
    ...         0.584,
    ...         0.422,
    ...         0.164,
    ...         0.005,
    ...         0.0012,
    ...         1.5e-4,
    ...         8.6e-5,
    ...         7.6e-5,
    ...         4.7e-5,
    ...         3.8e-5,
    ...         3.2e-7,
    ...     ]
    ... )
    >>> detect_eigenvalue_knee(evs)
    7
    """
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    if n < 2:
        return int(n)

    max_score = scores[0]
    if max_score <= 0:
        return 0

    # log-space drops between consecutive eigenvalues (drops[i] = log10(s[i] / s[i+1]))
    floor = max_score * 1e-12
    log_scores = np.log10(np.clip(scores, floor, None))
    drops = -np.diff(log_scores)

    # Only consider drop positions anchored on a "still meaningful" eigenvalue
    # (avoids picking the very last gap deep in the numerical noise tail).
    anchor_threshold = max_score * rel_floor
    valid = scores[:-1] > anchor_threshold
    if not np.any(valid):
        return 0

    valid_drops = np.where(valid, drops, -np.inf)
    knee_idx = int(np.argmax(valid_drops))

    # Sanity gate: require a meaningful gap to avoid false positives on
    # smoothly-decaying spectra.
    if valid_drops[knee_idx] < np.log10(min_ratio):
        return 0

    return knee_idx + 1


def auto_select_components_robust(
    eigenvalues: np.ndarray,
    sigma: float = 3.0,
    knee_rel_floor: float = 0.01,
    knee_min_ratio: float = 3.0,
) -> int:
    """Auto-select component count using outlier + knee detection.

    Layers :func:`iterative_outlier_removal` and :func:`detect_eigenvalue_knee`
    and returns the maximum of the two counts. This handles both regimes:

    - **Few outliers** (typical EEG): :func:`iterative_outlier_removal` finds
      the 1-2 dominant components.
    - **Many co-equal strong components** (typical high-channel-count MEG with
      coherent line noise): the outlier path returns 0 because the strong
      components don't stand out from each other; :func:`detect_eigenvalue_knee`
      catches the boundary to the noise floor.
    - **Clean spectrum** (monotonic decay, no clear bimodal split): both
      return 0 — no false removals.

    Parameters
    ----------
    eigenvalues : ndarray
        DSS eigenvalues in descending order.
    sigma : float, default=3.0
        Sigma threshold for :func:`iterative_outlier_removal`.
    knee_rel_floor : float, default=0.01
        ``rel_floor`` for :func:`detect_eigenvalue_knee`.
    knee_min_ratio : float, default=3.0
        ``min_ratio`` for :func:`detect_eigenvalue_knee`.

    Returns
    -------
    n_components : int
        Suggested number of significant components.
    """
    n_outlier = iterative_outlier_removal(eigenvalues, sigma=sigma)
    n_knee = detect_eigenvalue_knee(
        eigenvalues, rel_floor=knee_rel_floor, min_ratio=knee_min_ratio
    )
    return int(max(n_outlier, n_knee))


def eigenvalue_ratio_selection(
    eigenvalues: np.ndarray, ratio_threshold: float = 2.0
) -> int:
    """Select components using eigenvalue ratio (scree test).

    Identifies the first significant "drop" in the eigenvalue spectrum.
    A component is considered significant if the ratio
    ``eigenvalue[i] / eigenvalue[i+1]`` exceeds ``ratio_threshold``.

    This method works well when eigenvalue contrast is moderate (e.g., direct
    DSS without smoothing), where the iterative outlier removal may fail to
    detect any significant components.

    .. note::
       Unlike :func:`iterative_outlier_removal` (the ZapLine-plus component
       detector), this scree-style ratio rule is a convenience heuristic
       specific to ``mne-denoise``: it is not part of ZapLine-plus or the DSS
       literature, and ``ratio_threshold=2.0`` is an empirical default, not a
       theory-derived constant.

    Parameters
    ----------
    eigenvalues : ndarray
        DSS eigenvalues, sorted in descending order.
    ratio_threshold : float
        Minimum ratio between consecutive eigenvalues to indicate a
        significant drop. Default 2.0 (i.e., a 2× drop).

    Returns
    -------
    n_components : int
        Number of significant components (before the first big drop).
        Returns 0 if no drop exceeds the threshold.

    Examples
    --------
    >>> eigenvalues = np.array([0.012, 0.005, 0.004, 0.0015, 0.0008])
    >>> n = eigenvalue_ratio_selection(eigenvalues, ratio_threshold=2.0)
    >>> print(f"{n} component(s) before the first big drop")
    1 component(s) before the first big drop
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)

    if len(eigenvalues) < 2:
        return len(eigenvalues)

    # Guard against zero/negative eigenvalues
    eigenvalues = np.maximum(eigenvalues, 0.0)

    for i in range(len(eigenvalues) - 1):
        if eigenvalues[i + 1] < 1e-15:
            # Denominator effectively zero → everything before this is signal
            return i + 1
        ratio = eigenvalues[i] / eigenvalues[i + 1]
        if ratio >= ratio_threshold:
            return i + 1

    return 0


def max_gap_selection(
    eigenvalues: np.ndarray, min_ratio: float = 1.2
) -> int:
    """Select components by finding the largest gap in the eigenvalue spectrum.

    Instead of requiring a fixed ratio threshold, this method finds the
    position of the maximum consecutive eigenvalue ratio (biggest "drop")
    and uses it as the cutpoint — provided the drop exceeds ``min_ratio``.

    This is the most lenient automatic method and works even when eigenvalue
    contrast is weak (e.g., direct DSS without smoothing on weak artifacts).

    .. note::
       Like :func:`eigenvalue_ratio_selection`, this largest-gap rule is a
       convenience heuristic specific to ``mne-denoise`` (not part of
       ZapLine-plus or the DSS literature); ``min_ratio=1.2`` is empirical.
       Prefer :func:`iterative_outlier_removal` for line-noise / ZapLine-style
       selection.

    Parameters
    ----------
    eigenvalues : ndarray
        DSS eigenvalues, sorted in descending order.
    min_ratio : float
        Minimum ratio at the largest gap to be considered meaningful.
        Default 1.2 (i.e., at least a 20%% drop). Set lower for noisier
        data, higher for stricter selection.

    Returns
    -------
    n_components : int
        Number of significant components (before the biggest gap).
        Returns 0 if no gap exceeds ``min_ratio``.

    Examples
    --------
    >>> eigenvalues = np.array([0.0066, 0.0052, 0.0042, 0.0035, 0.003])
    >>> n = max_gap_selection(eigenvalues, min_ratio=1.2)
    >>> print(f"{n} component(s)")  # Finds the 1.26× drop at position 0
    1 component(s)
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)

    if len(eigenvalues) < 2:
        return len(eigenvalues)

    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Compute consecutive ratios
    denominators = np.maximum(eigenvalues[1:], 1e-15)
    ratios = eigenvalues[:-1] / denominators

    # Find the position of the largest gap
    max_idx = np.argmax(ratios)

    if ratios[max_idx] >= min_ratio:
        return int(max_idx + 1)

    return 0
