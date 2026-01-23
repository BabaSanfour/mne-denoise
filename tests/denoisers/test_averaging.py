"""Unit tests for averaging denoisers (AverageBias)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne_denoise.dss.denoisers.averaging import AverageBias


def test_average_bias_epochs():
    """Test AverageBias(axis='epochs') on simple 3D data."""
    # Create data: (n_channels, n_times, n_epochs)
    n_epochs = 10
    data = np.zeros((1, 5, n_epochs))

    # Trials have a constant component (1) + noise
    # We make trial 0 have value 1, trial 1 have value 3 -> mean 2
    data[0, :, :] = 2

    bias = AverageBias(axis="epochs")
    biased = bias.apply(data)

    # Result should be the mean repeated
    expected = np.ones((1, 5, n_epochs)) * 2
    assert_allclose(biased, expected)


def test_average_bias_epochs_weighted():
    """Test AverageBias(axis='epochs') with weights."""
    # 2 epochs
    data = np.zeros((1, 1, 2))
    data[0, 0, 0] = 10
    data[0, 0, 1] = 20

    # Weighted average: 0.8 * 10 + 0.2 * 20 = 8 + 4 = 12
    weights = [0.8, 0.2]
    bias = AverageBias(axis="epochs", weights=weights)
    biased = bias.apply(data)

    assert_allclose(biased[0, 0, :], 12)


def test_average_bias_epochs_errors():
    """Test error handling for epochs axis."""
    bias = AverageBias(axis="epochs")
    # 2D input should fail (expects epoched)
    data = np.zeros((2, 10))
    # Note: Error message changed to "AverageBias(axis='epochs') requires 3D data" in source
    with pytest.raises(
        ValueError, match="AverageBias.*axis='epochs'.*requires 3D data"
    ):
        bias.apply(data)


def test_average_bias_weight_mismatch():
    """Test error when weights length doesn't match epochs."""
    data = np.zeros((1, 5, 10))  # 10 epochs
    weights = [1, 2, 3]  # Only 3 weights

    bias = AverageBias(axis="epochs", weights=weights)
    with pytest.raises(ValueError, match="weights length.*must match"):
        bias.apply(data)
