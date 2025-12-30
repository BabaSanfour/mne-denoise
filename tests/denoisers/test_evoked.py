"""Unit tests for evoked denoisers (TrialAverageBias)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne_denoise.dss.denoisers.evoked import TrialAverageBias


def test_trial_average_bias():
    """Test TrialAverageBias on simple 3D data."""
    # Create data: (n_channels, n_times, n_epochs)
    n_epochs = 10
    data = np.zeros((1, 5, n_epochs))

    # Trials have a constant component (1) + noise
    # We make trial 0 have value 1, trial 1 have value 3 -> mean 2
    data[0, :, :] = 2

    bias = TrialAverageBias()
    biased = bias.apply(data)

    # Result should be the mean repeated
    expected = np.ones((1, 5, n_epochs)) * 2
    assert_allclose(biased, expected)


def test_trial_average_bias_weighted():
    """Test TrialAverageBias with weights."""
    # 2 epochs
    data = np.zeros((1, 1, 2))
    data[0, 0, 0] = 10
    data[0, 0, 1] = 20

    # Weighted average: 0.8 * 10 + 0.2 * 20 = 8 + 4 = 12
    weights = [0.8, 0.2]
    bias = TrialAverageBias(weights=weights)
    biased = bias.apply(data)

    assert_allclose(biased[0, 0, :], 12)


def test_trial_average_bias_errors():
    """Test error handling."""
    bias = TrialAverageBias()
    # 2D input should fail (expects epoched)
    data = np.zeros((2, 10))
    with pytest.raises(ValueError, match="requires 3D epoched data"):
        bias.apply(data)


def test_trial_average_bias_weight_mismatch():
    """Test error when weights length doesn't match epochs."""
    data = np.zeros((1, 5, 10))  # 10 epochs
    weights = [1, 2, 3]  # Only 3 weights

    bias = TrialAverageBias(weights=weights)
    with pytest.raises(ValueError, match="weights length.*must match"):
        bias.apply(data)
