"""Unit tests for denoiser base classes."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

def test_base_linear_denoiser():
    """Test LinearDenoiser abstract base class."""
    from mne_denoise.dss.denoisers.base import LinearDenoiser
    
    # Check that we cannot instantiate ABC
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        LinearDenoiser()
        
    # Check robust implementation
    class MockLinear(LinearDenoiser):
        def apply(self, data):
            return data * 2
            
    denoiser = MockLinear()
    data = np.ones((2, 2))
    assert_allclose(denoiser.apply(data), data * 2)
    # Check __call__ alias
    assert_allclose(denoiser(data), data * 2)


def test_base_nonlinear_denoiser():
    """Test NonlinearDenoiser abstract base class."""
    from mne_denoise.dss.denoisers.base import NonlinearDenoiser
    
    # Check that we cannot instantiate ABC
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        NonlinearDenoiser()
        
    # Check robust implementation
    class MockNonlinear(NonlinearDenoiser):
        def denoise(self, source):
            return source ** 2
            
    denoiser = MockNonlinear()
    source = np.array([1, 2, 3])
    assert_allclose(denoiser.denoise(source), [1, 4, 9])
    # Check __call__ alias
    assert_allclose(denoiser(source), [1, 4, 9])
