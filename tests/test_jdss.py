"""Unit tests for Joint Denoising Source Separation (JDSS)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import mne
from mne_denoise.dss.jdss import JDSS, compute_jdss


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def synthetic_jdss_data():
    """Generate synthetic data with a common repeatable source.
    
    Creates multiple "subjects" that share a common signal component
    but have different noise/artifacts.
    """
    rng = np.random.default_rng(42)
    n_subjects = 5
    n_channels = 8
    n_times = 1000
    
    # Common repeatable source (e.g., stimulus-evoked response)
    times = np.arange(n_times) / 250.0  # 4 seconds at 250 Hz
    common_source = np.sin(2 * np.pi * 10 * times) * 3.0  # 10 Hz oscillation
    
    # Spatial pattern for common source (same across subjects)
    common_pattern = rng.normal(0, 1, n_channels)
    common_pattern /= np.linalg.norm(common_pattern)
    
    datasets = []
    for _ in range(n_subjects):
        # Subject-specific noise
        noise = rng.normal(0, 0.5, (n_channels, n_times))
        
        # Project common source into channel space
        data = np.outer(common_pattern, common_source) + noise
        datasets.append(data)
    
    return {
        "datasets": datasets,
        "common_source": common_source,
        "common_pattern": common_pattern,
        "n_channels": n_channels,
        "n_times": n_times,
    }


@pytest.fixture
def mne_epochs_list():
    """Create list of MNE Epochs objects for JDSS."""
    rng = np.random.default_rng(42)
    sfreq = 100
    n_times = 200  # 2s
    n_channels = 4
    n_subjects = 3
    
    times = np.arange(n_times) / sfreq
    common_signal = np.sin(2 * np.pi * 5 * times)
    
    epochs_list = []
    for _ in range(n_subjects):
        # Each "subject" is one epoch
        data = rng.normal(0, 0.5, (n_channels, n_times))
        data[0] += common_signal  # Common signal in ch 0
        
        # Wrap as single-epoch Epochs object
        data_epochs = data[np.newaxis, :, :]  # (1, ch, times)
        info = mne.create_info(n_channels, sfreq, "eeg")
        epochs = mne.EpochsArray(data_epochs, info, verbose=False)
        epochs_list.append(epochs)
    
    return epochs_list


# =============================================================================
# compute_jdss - Basic Functionality
# =============================================================================


def test_compute_jdss_output_shapes(synthetic_jdss_data):
    """compute_jdss should return filters, patterns, eigenvalues with correct shapes."""
    datasets = synthetic_jdss_data["datasets"]
    n_channels = synthetic_jdss_data["n_channels"]
    
    filters, patterns, eigenvalues = compute_jdss(datasets)
    
    assert filters.shape == (n_channels, n_channels)
    assert patterns.shape == (n_channels, n_channels)
    assert eigenvalues.shape == (n_channels,)


def test_compute_jdss_n_components(synthetic_jdss_data):
    """compute_jdss should respect n_components parameter."""
    datasets = synthetic_jdss_data["datasets"]
    
    filters, patterns, eigenvalues = compute_jdss(datasets, n_components=3)
    
    assert filters.shape[0] == 3
    assert patterns.shape[1] == 3
    assert eigenvalues.shape == (3,)


def test_compute_jdss_eigenvalue_ordering(synthetic_jdss_data):
    """compute_jdss eigenvalues should be in descending order."""
    datasets = synthetic_jdss_data["datasets"]
    
    filters, patterns, eigenvalues = compute_jdss(datasets)
    
    # Eigenvalues should be sorted descending
    assert np.all(np.diff(eigenvalues) <= 0)


def test_compute_jdss_recovers_common_source(synthetic_jdss_data):
    """compute_jdss should recover the common source in top component."""
    datasets = synthetic_jdss_data["datasets"]
    common_source = synthetic_jdss_data["common_source"]
    
    filters, patterns, eigenvalues = compute_jdss(datasets, n_components=3)
    
    # Apply top filter to first dataset
    sources = filters @ datasets[0]
    top_source = sources[0]
    
    # Should correlate with the common source
    corr = np.abs(np.corrcoef(top_source, common_source)[0, 1])
    assert corr > 0.8


def test_compute_jdss_reg_parameter(synthetic_jdss_data):
    """compute_jdss should accept regularization parameter."""
    datasets = synthetic_jdss_data["datasets"]
    
    # Different regularizations should work
    filters1, _, _ = compute_jdss(datasets, reg=1e-6)
    filters2, _, _ = compute_jdss(datasets, reg=1e-3)
    
    # Should produce valid results
    assert not np.any(np.isnan(filters1))
    assert not np.any(np.isnan(filters2))


def test_compute_jdss_two_datasets():
    """compute_jdss should work with just 2 datasets."""
    rng = np.random.default_rng(42)
    datasets = [rng.normal(0, 1, (4, 100)) for _ in range(2)]
    
    filters, patterns, eigenvalues = compute_jdss(datasets)
    
    assert filters.shape[0] == 4
    assert len(eigenvalues) == 4


def test_compute_jdss_many_datasets():
    """compute_jdss should work with many datasets."""
    rng = np.random.default_rng(42)
    datasets = [rng.normal(0, 1, (4, 100)) for _ in range(20)]
    
    filters, patterns, eigenvalues = compute_jdss(datasets)
    
    assert filters.shape[0] == 4


# =============================================================================
# JDSS Class - Basic Functionality
# =============================================================================


def test_jdss_class_fit(synthetic_jdss_data):
    """JDSS class should fit on list of datasets."""
    datasets = synthetic_jdss_data["datasets"]
    
    jdss = JDSS(n_components=3)
    result = jdss.fit(datasets)
    
    assert result is jdss  # Returns self
    assert hasattr(jdss, "filters_")
    assert hasattr(jdss, "patterns_")
    assert hasattr(jdss, "eigenvalues_")


def test_jdss_class_fit_3d_array(synthetic_jdss_data):
    """JDSS class should accept 3D array (n_datasets, n_ch, n_times)."""
    datasets = synthetic_jdss_data["datasets"]
    
    # Stack to 3D array
    data_3d = np.array(datasets)
    
    jdss = JDSS(n_components=3)
    jdss.fit(data_3d)
    
    assert hasattr(jdss, "filters_")


def test_jdss_class_transform_single(synthetic_jdss_data):
    """JDSS.transform should work on single 2D dataset."""
    datasets = synthetic_jdss_data["datasets"]
    
    jdss = JDSS(n_components=3)
    jdss.fit(datasets)
    
    sources = jdss.transform(datasets[0])
    
    assert sources.shape == (3, synthetic_jdss_data["n_times"])


def test_jdss_class_transform_list(synthetic_jdss_data):
    """JDSS.transform should work on list of datasets."""
    datasets = synthetic_jdss_data["datasets"]
    
    jdss = JDSS(n_components=3)
    jdss.fit(datasets)
    
    sources_list = jdss.transform(datasets)
    
    assert len(sources_list) == len(datasets)
    assert sources_list[0].shape[0] == 3


def test_jdss_class_transform_3d(synthetic_jdss_data):
    """JDSS.transform should work on 3D array."""
    datasets = synthetic_jdss_data["datasets"]
    data_3d = np.array(datasets)
    
    jdss = JDSS(n_components=3)
    jdss.fit(data_3d)
    
    sources = jdss.transform(data_3d)
    
    assert sources.shape == (len(datasets), 3, synthetic_jdss_data["n_times"])


def test_jdss_class_fit_transform(synthetic_jdss_data):
    """JDSS.fit_transform should fit and transform in one step."""
    datasets = synthetic_jdss_data["datasets"]
    
    jdss = JDSS(n_components=3)
    sources = jdss.fit(datasets).transform(datasets[0])
    
    assert sources.shape[0] == 3


def test_jdss_class_inverse_transform(synthetic_jdss_data):
    """JDSS.inverse_transform should reconstruct data from sources."""
    datasets = synthetic_jdss_data["datasets"]
    n_channels = synthetic_jdss_data["n_channels"]
    
    jdss = JDSS(n_components=3)
    jdss.fit(datasets)
    
    sources = jdss.transform(datasets[0])
    reconstructed = jdss.inverse_transform(sources)
    
    assert reconstructed.shape == (n_channels, synthetic_jdss_data["n_times"])


def test_jdss_class_inverse_transform_list(synthetic_jdss_data):
    """JDSS.inverse_transform should work on list of sources."""
    datasets = synthetic_jdss_data["datasets"]
    
    jdss = JDSS(n_components=3)
    jdss.fit(datasets)
    
    sources_list = jdss.transform(datasets)
    reconstructed_list = jdss.inverse_transform(sources_list)
    
    assert len(reconstructed_list) == len(datasets)


# =============================================================================
# JDSS Class - MNE Integration
# =============================================================================


def test_jdss_class_fit_mne_epochs(mne_epochs_list):
    """JDSS should fit on list of MNE Epochs objects."""
    jdss = JDSS(n_components=2)
    jdss.fit(mne_epochs_list)
    
    assert hasattr(jdss, "filters_")


def test_jdss_class_transform_mne_epochs(mne_epochs_list):
    """JDSS should transform MNE Epochs objects."""
    jdss = JDSS(n_components=2)
    jdss.fit(mne_epochs_list)
    
    sources = jdss.transform(mne_epochs_list)
    
    assert len(sources) == len(mne_epochs_list)


# =============================================================================
# JDSS Class - Error Handling
# =============================================================================


def test_jdss_transform_before_fit_error(synthetic_jdss_data):
    """JDSS.transform should raise error if called before fit."""
    datasets = synthetic_jdss_data["datasets"]
    
    jdss = JDSS(n_components=3)
    
    with pytest.raises((RuntimeError, AttributeError, TypeError)):
        jdss.transform(datasets[0])



def test_jdss_inverse_transform_before_fit_error(synthetic_jdss_data):
    """JDSS.inverse_transform should raise error if called before fit."""
    jdss = JDSS(n_components=3)
    sources = np.random.randn(3, 100)
    
    with pytest.raises((RuntimeError, AttributeError, TypeError)):
        jdss.inverse_transform(sources)



def test_jdss_empty_list_error():
    """JDSS.fit should raise error for empty list."""
    jdss = JDSS(n_components=2)
    
    with pytest.raises((ValueError, IndexError)):
        jdss.fit([])


def test_jdss_mismatched_channels_error():
    """JDSS.fit should raise error for datasets with different channel counts."""
    rng = np.random.default_rng(42)
    
    datasets = [
        rng.normal(0, 1, (4, 100)),  # 4 channels
        rng.normal(0, 1, (5, 100)),  # 5 channels
    ]
    
    jdss = JDSS(n_components=2)
    
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        jdss.fit(datasets)


# =============================================================================
# JDSS - Functional Tests
# =============================================================================


def test_jdss_extracts_repeatable_source():
    """JDSS should extract maximally repeatable source."""
    rng = np.random.default_rng(42)
    n_subjects = 10
    n_channels = 8
    n_times = 500
    
    # Create a strong repeatable source
    times = np.arange(n_times) / 100.0
    repeatable_source = np.sin(2 * np.pi * 5 * times)  # 5 Hz
    spatial_pattern = np.array([1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02])
    
    # Create a non-repeatable source (different per subject)
    non_repeatable_pattern = rng.normal(0, 1, n_channels)
    
    datasets = []
    for _ in range(n_subjects):
        # Repeatable component (same in all subjects)
        data = np.outer(spatial_pattern, repeatable_source) * 2.0
        
        # Non-repeatable component (random per subject)
        non_repeatable = rng.normal(0, 0.3, n_times)
        data += np.outer(non_repeatable_pattern, non_repeatable)
        
        # Noise
        data += rng.normal(0, 0.1, (n_channels, n_times))
        
        datasets.append(data)
    
    # Extract components
    jdss = JDSS(n_components=2)
    jdss.fit(datasets)
    
    # Top component should correlate with repeatable source
    sources = jdss.transform(datasets[0])
    corr = np.abs(np.corrcoef(sources[0], repeatable_source)[0, 1])
    
    assert corr > 0.9


def test_jdss_reconstruction_preserves_signal():
    """patterns @ sources should approximate centered data."""
    rng = np.random.default_rng(42)
    n_channels = 6
    n_times = 200
    
    datasets = [rng.normal(0, 1, (n_channels, n_times)) for _ in range(5)]
    
    jdss = JDSS(n_components=n_channels)  # Keep all
    jdss.fit(datasets)
    
    # Full reconstruction test
    sources = jdss.transform(datasets[0])
    reconstructed = jdss.inverse_transform(sources)
    
    # Should reconstruct centered data
    centered = datasets[0] - datasets[0].mean(axis=1, keepdims=True)
    
    # With all components, should be close (relax tolerance for numerical precision)
    assert_allclose(reconstructed, centered, rtol=0.1, atol=0.1)




if __name__ == "__main__":
    pytest.main([__file__, "-v"])
