"""Internal spatial operations shared by denoising algorithms."""

from __future__ import annotations

import numpy as np

from ._validation import check_chunk_size


def apply_spatial_transform(
    matrix: np.ndarray,
    data: np.ndarray,
    *,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Apply a spatial matrix along the first axis of 2D or 3D data.

    Parameters
    ----------
    matrix : ndarray, shape (n_output_channels, n_input_channels)
        Spatial transformation matrix.
    data : ndarray, shape (n_input_channels, ...)
        Channel-first continuous or multidimensional data.
    chunk_size : int | None, default=None
        Number of flattened samples transformed at a time. None applies the
        matrix in one operation.

    Returns
    -------
    transformed : ndarray
        Transformed data with the trailing dimensions of ``data`` preserved.
    """
    matrix = np.asarray(matrix)
    data = np.asarray(data)
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got {matrix.ndim}D")
    if data.ndim not in (2, 3):
        raise ValueError(f"data must be 2D or 3D, got {data.ndim}D")
    if matrix.shape[1] != data.shape[0]:
        raise ValueError(
            "matrix and data channel dimensions do not match "
            f"({matrix.shape[1]} != {data.shape[0]})"
        )
    chunk_size = check_chunk_size(chunk_size)

    flat = data.reshape(data.shape[0], -1)
    if chunk_size is None:
        transformed = matrix @ flat
    else:
        transformed = np.empty((matrix.shape[0], flat.shape[1]), dtype=np.float64)
        for start in range(0, flat.shape[1], chunk_size):
            stop = min(start + chunk_size, flat.shape[1])
            transformed[:, start:stop] = matrix @ flat[:, start:stop]
    return transformed.reshape((matrix.shape[0], *data.shape[1:]))


def fit_mixing_matrix(
    data: np.ndarray,
    sources: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Fit a least-squares projection from components to channel data.

    Parameters
    ----------
    data : ndarray, shape (n_channels, ...)
        Channel-first target data.
    sources : ndarray, shape (n_components, ...)
        Component time courses with the same trailing observation dimensions.
    sample_weight : ndarray | None
        Optional weights shaped like the trailing observation dimensions, or
        flattened in their C-order layout.

    Returns
    -------
    mixing : ndarray, shape (n_channels, n_components)
        Least-squares sensor projection.
    """
    data = np.asarray(data, dtype=np.float64)
    sources = np.asarray(sources, dtype=np.float64)
    if data.ndim < 2 or sources.ndim != data.ndim:
        raise ValueError("data and sources must have matching dimensions of at least 2")
    if data.shape[1:] != sources.shape[1:]:
        raise ValueError("data and sources must have matching observation dimensions")

    data_flat = data.reshape(data.shape[0], -1)
    sources_flat = sources.reshape(sources.shape[0], -1)
    if sample_weight is None:
        weighted_sources = sources_flat
    else:
        weights = np.asarray(sample_weight, dtype=np.float64)
        if weights.shape == data.shape[1:]:
            weights = weights.reshape(-1)
        elif weights.shape != (data_flat.shape[1],):
            raise ValueError(
                "sample_weight must match the observation dimensions or have "
                f"shape ({data_flat.shape[1]},); got {weights.shape}"
            )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("sample_weight must be finite and non-negative")
        if weights.sum() <= 0:
            raise ValueError("sample_weight must have a positive sum")
        weighted_sources = sources_flat * weights
    return (data_flat @ weighted_sources.T) @ np.linalg.pinv(
        sources_flat @ weighted_sources.T, hermitian=True
    )
