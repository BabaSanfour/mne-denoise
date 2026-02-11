"""Unit tests for DSS module - Linear DSS (compute_dss and DSS class)."""

from __future__ import annotations

import mne
import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne_denoise.dss import DSS, compute_dss

# =============================================================================
# compute_dss - Core Algorithm Tests
# =============================================================================


def test_compute_dss_shape():
    """compute_dss should return correct shapes."""
    rng = np.random.default_rng(42)
    n_channels = 16

    # Create random covariance matrices
    A = rng.standard_normal((n_channels, n_channels))
    cov0 = A @ A.T  # Baseline covariance
    cov1 = cov0.copy()  # Biased covariance (identity bias)

    filters, patterns, eigenvalues = compute_dss(cov0, cov1, n_components=5)

    assert filters.shape == (5, n_channels)
    assert patterns.shape == (n_channels, 5)
    assert eigenvalues.shape == (5,)


def test_compute_dss_identity_bias():
    """With identity bias (cov0 == cov1), all eigenvalues should be ~1."""
    rng = np.random.default_rng(42)
    n_channels = 8

    A = rng.standard_normal((n_channels, n_channels))
    cov = A @ A.T

    filters, patterns, eigenvalues = compute_dss(cov, cov)

    # All eigenvalues should be approximately 1
    assert_allclose(eigenvalues, np.ones(n_channels), atol=0.1)


def test_compute_dss_known_signal():
    """compute_dss should maximize biased/baseline variance ratio."""
    np.random.default_rng(42)
    n_channels = 4

    # Create baseline covariance: isotropic (identity-like)
    cov0 = np.eye(n_channels)

    # Create biased covariance: high variance in first direction
    cov1 = np.diag([10.0, 1.0, 1.0, 1.0])  # First component 10x stronger

    filters, patterns, eigenvalues = compute_dss(cov0, cov1, n_components=1)

    # Top filter should align with first basis vector (highest bias)
    top_filter = filters[0]
    expected_direction = np.array([1, 0, 0, 0])
    alignment = np.abs(
        np.dot(top_filter / np.linalg.norm(top_filter), expected_direction)
    )
    assert alignment > 0.95, f"Alignment {alignment} too low"

    # Top eigenvalue should be ~10 (ratio of biased to baseline variance)
    assert eigenvalues[0] > 5, f"Eigenvalue {eigenvalues[0]} too low"


def test_compute_dss_eigenvalue_ordering():
    """Eigenvalues should be in descending order."""
    rng = np.random.default_rng(42)
    n_channels = 6

    A = rng.standard_normal((n_channels, n_channels))
    B = rng.standard_normal((n_channels, n_channels))
    cov0 = A @ A.T
    cov1 = B @ B.T

    _, _, eigenvalues = compute_dss(cov0, cov1)

    # Check descending order
    assert np.all(eigenvalues[:-1] >= eigenvalues[1:])


def test_compute_dss_orthogonal_filters():
    """DSS filters should be orthogonal in whitened space."""
    rng = np.random.default_rng(42)
    n_channels = 8

    A = rng.standard_normal((n_channels, n_channels))
    cov0 = A @ A.T
    cov1 = cov0 * 1.1  # Slightly different

    filters, _, _ = compute_dss(cov0, cov1)

    # Filters should be approximately orthogonal when projected through cov0
    gram = filters @ cov0 @ filters.T
    off_diag = gram - np.diag(np.diag(gram))
    assert np.max(np.abs(off_diag)) < 0.1


def test_compute_dss_n_components():
    """n_components should limit the output size."""
    rng = np.random.default_rng(42)
    n_channels = 10

    A = rng.standard_normal((n_channels, n_channels))
    cov = A @ A.T

    for n_comp in [1, 3, 5]:
        filters, patterns, eigenvalues = compute_dss(cov, cov, n_components=n_comp)
        assert filters.shape[0] == n_comp
        assert patterns.shape[1] == n_comp
        assert len(eigenvalues) == n_comp


def test_compute_dss_rank():
    """Rank parameter should limit whitening dimensionality."""
    rng = np.random.default_rng(42)
    n_channels = 10

    A = rng.standard_normal((n_channels, n_channels))
    cov = A @ A.T

    filters, _, _ = compute_dss(cov, cov, rank=5)

    # Output should be at most rank dimensions
    assert filters.shape[0] <= 5


def test_compute_dss_reconstruction():
    """Patterns @ (filters @ data) should reconstruct centered data."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 6, 500

    data = rng.standard_normal((n_channels, n_samples))
    data_c = data - data.mean(axis=1, keepdims=True)

    cov = data_c @ data_c.T / n_samples

    filters, patterns, _ = compute_dss(cov, cov)

    sources = filters @ data_c
    reconstructed = patterns @ sources

    # Should reconstruct (approximately - some numerical error expected)
    assert_allclose(reconstructed, data_c, atol=0.5)


# =============================================================================
# compute_dss - Error Handling
# =============================================================================


def test_compute_dss_error_shape_mismatch():
    """compute_dss should raise error when covariance shapes mismatch."""
    c1 = np.eye(5)
    c2 = np.eye(6)

    with pytest.raises(ValueError, match="shapes mismatch"):
        compute_dss(c1, c2)


def test_compute_dss_error_not_square():
    """compute_dss should raise error for non-square covariance."""
    c = np.ones((5, 6))

    with pytest.raises(ValueError, match="must be square"):
        compute_dss(c, c)


def test_compute_dss_error_no_variance():
    """compute_dss should raise error when covariance has no variance."""
    c = np.zeros((5, 5))

    with pytest.raises(ValueError, match="no significant variance"):
        compute_dss(c, c)


def test_compute_dss_error_tiny_eigenvalues():
    """compute_dss should raise error when all eigenvalues are tiny."""
    c = np.eye(5) * 1e-20

    with pytest.raises(ValueError, match="no significant variance|No components"):
        compute_dss(c, c)


# =============================================================================
# DSS Class - Basic Functionality
# =============================================================================


def test_dss_fit_transform():
    """DSS class should support fit_transform workflow."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 8, 1000
    data = rng.standard_normal((n_channels, n_samples))

    dss = DSS(bias=lambda x: x, n_components=3)
    sources = dss.fit_transform(data)

    assert sources.shape == (3, n_samples)
    assert dss.filters_ is not None
    assert dss.filters_.shape == (3, n_channels)
    assert dss.patterns_ is not None
    assert dss.mixing_ is not None
    assert dss.eigenvalues_ is not None


def test_dss_custom_bias_callable():
    """DSS should accept custom callable bias."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((8, 1000))

    def my_bias(x):
        return x * 2

    dss = DSS(bias=my_bias, n_components=3)
    sources = dss.fit_transform(data)

    assert sources.shape == (3, 1000)


def test_dss_denoiser_bias():
    """DSS should accept LinearDenoiser bias."""
    from mne_denoise.dss.denoisers import BandpassBias

    rng = np.random.default_rng(42)
    data = rng.standard_normal((8, 2000))

    bias = BandpassBias(freq_band=(8, 12), sfreq=250)
    dss = DSS(bias=bias, n_components=3)
    sources = dss.fit_transform(data)

    assert sources.shape == (3, 2000)


def test_dss_3d_data():
    """DSS should handle 3D epoched data."""
    rng = np.random.default_rng(42)
    n_ch, n_times, n_epochs = 8, 100, 5
    data = rng.standard_normal((n_ch, n_times, n_epochs))

    dss = DSS(bias=lambda x: x, n_components=3)
    dss.fit(data)
    sources = dss.transform(data)

    assert sources.shape == (3, n_times, n_epochs)


def test_dss_without_normalization():
    """DSS should work without input normalization."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((5, 200))

    dss = DSS(bias=lambda x: x, n_components=3, normalize_input=False)
    sources = dss.fit_transform(data)

    assert sources.shape == (3, 200)


def test_dss_return_type_raw():
    """DSS transform with return_type='raw' should reconstruct data."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((5, 200))

    dss = DSS(bias=lambda x: x, n_components=3, return_type="raw")
    dss.fit(data)
    rec = dss.transform(data)

    assert rec.shape == data.shape


def test_dss_cov_method():
    """DSS should accept different covariance methods."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((5, 200))

    dss = DSS(bias=lambda x: x, n_components=3, cov_method="shrinkage")
    sources = dss.fit_transform(data)

    assert sources.shape == (3, 200)


def test_dss_cov_kws():
    """DSS should pass cov_kws to covariance computation."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((5, 200))

    dss = DSS(bias=lambda x: x, n_components=3, cov_kws={"shrinkage": 0.5})
    sources = dss.fit_transform(data)

    assert sources.shape == (3, 200)


# =============================================================================
# DSS Class - inverse_transform
# =============================================================================


def test_dss_inverse_transform_2d():
    """inverse_transform should work with 2D sources."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((8, 500))

    dss = DSS(bias=lambda x: x, n_components=5)
    sources = dss.fit_transform(data)
    rec = dss.inverse_transform(sources)

    assert rec.shape == data.shape


def test_dss_inverse_transform_3d():
    """inverse_transform should handle 3D sources."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((8, 100, 4))

    dss = DSS(bias=lambda x: x, n_components=5)
    sources = dss.fit_transform(data)
    rec = dss.inverse_transform(sources)

    assert rec.ndim == 3


def test_dss_inverse_transform_boolean_mask():
    """inverse_transform should work with boolean component mask."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((8, 500))

    dss = DSS(bias=lambda x: x, n_components=5)
    sources = dss.fit_transform(data)

    mask = np.array([True, True, True, False, False])
    rec = dss.inverse_transform(sources, component_indices=mask)

    assert rec.shape == data.shape


def test_dss_inverse_transform_integer_indices():
    """inverse_transform should work with integer component indices."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((8, 500))

    dss = DSS(bias=lambda x: x, n_components=5)
    sources = dss.fit_transform(data)

    indices = np.array([0, 2])
    rec = dss.inverse_transform(sources, component_indices=indices)

    assert rec.shape == data.shape


# =============================================================================
# DSS Class - Error Handling
# =============================================================================


def test_dss_error_unsupported_type():
    """DSS should raise error for unsupported input types."""
    dss = DSS(bias=lambda x: x, normalize_input=False)

    with pytest.raises(TypeError, match="Unsupported input type"):
        dss.fit("not an array")


def test_dss_error_transform_before_fit():
    """DSS should raise error when transform called before fit."""
    dss = DSS(bias=lambda x: x)
    data = np.random.randn(5, 100)

    with pytest.raises(RuntimeError, match="not fitted"):
        dss.transform(data)


def test_dss_error_inverse_transform_before_fit():
    """DSS should raise error when inverse_transform called before fit."""
    dss = DSS(bias=lambda x: x)
    sources = np.random.randn(3, 100)

    with pytest.raises(RuntimeError, match="not fitted"):
        dss.inverse_transform(sources)


def test_dss_error_mask_length_mismatch():
    """inverse_transform should raise error for wrong mask length."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((8, 500))

    dss = DSS(bias=lambda x: x, n_components=5)
    sources = dss.fit_transform(data)

    wrong_mask = np.array([True, True, True])
    with pytest.raises(ValueError, match="Mask length"):
        dss.inverse_transform(sources, component_indices=wrong_mask)


def test_dss_supports_rank_numpy():
    """DSS should support rank parameter with numpy arrays (no warning)."""
    import warnings

    rng = np.random.default_rng(42)
    data = rng.standard_normal((8, 500))

    # Should not warn
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dss = DSS(bias=lambda x: x, n_components=3, rank=5)
        dss.fit(data)

        # Filter out unrelated warnings if any (e.g. from MNE)
        rank_warnings = [
            warning for warning in w if "rank" in str(warning.message).lower()
        ]
        assert len(rank_warnings) == 0


# =============================================================================
# Integration / Functional Tests
# =============================================================================


def test_dss_recovers_narrowband_signal():
    """DSS with bandpass bias should recover narrowband signal."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 8, 2000
    sfreq = 250

    noise = rng.standard_normal((n_channels, n_samples))

    t = np.arange(n_samples) / sfreq
    source = np.sin(2 * np.pi * 10 * t)  # 10 Hz

    mixing = rng.standard_normal(n_channels)
    mixing = mixing / np.linalg.norm(mixing)
    data = noise + 5 * np.outer(mixing, source)

    from mne_denoise.dss.denoisers import BandpassBias

    bias = BandpassBias(freq_band=(8, 12), sfreq=sfreq)

    dss = DSS(bias=bias, n_components=3)
    sources = dss.fit_transform(data)

    top_source = sources[0]
    correlation = np.abs(np.corrcoef(top_source, source)[0, 1])
    assert correlation > 0.8


def test_dss_evoked_workflow():
    """DSS with trial average bias should recover evoked response."""
    rng = np.random.default_rng(42)
    n_channels, n_times, n_epochs = 8, 100, 50

    noise = rng.standard_normal((n_channels, n_times, n_epochs))

    evoked = np.zeros(n_times)
    evoked[40:60] = np.hanning(20)

    mixing = rng.standard_normal(n_channels)
    mixing = mixing / np.linalg.norm(mixing)
    signal = np.outer(mixing, evoked)[:, :, np.newaxis]

    data = noise + signal

    from mne_denoise.dss.denoisers import AverageBias

    bias = AverageBias(axis="epochs")
    dss = DSS(bias=bias, n_components=3)
    sources = dss.fit_transform(data)

    top_source_avg = sources[0].mean(axis=1)
    correlation = np.abs(np.corrcoef(top_source_avg, evoked)[0, 1])
    assert correlation > 0.7


# =============================================================================
# MNE Integration Tests
# =============================================================================


def test_dss_with_mne_raw():
    """DSS should work with MNE Raw objects."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 8, 5000
    sfreq = 500.0

    data = rng.standard_normal((n_channels, n_samples))

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    raw = mne.io.RawArray(data, info, verbose=False)

    dss = DSS(bias=lambda x: x, n_components=3)
    sources = dss.fit_transform(raw)

    assert sources.shape == (3, n_samples)
    assert dss.info_ is not None


def test_dss_with_mne_epochs():
    """DSS should work with MNE Epochs objects."""
    rng = np.random.default_rng(42)
    n_channels, n_times, n_epochs = 8, 100, 20
    sfreq = 250.0

    # Create epoch data: MNE expects (n_epochs, n_channels, n_times)
    data = rng.standard_normal((n_epochs, n_channels, n_times))

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    epochs = mne.EpochsArray(data, info, verbose=False)

    from mne_denoise.dss.denoisers import AverageBias

    dss = DSS(bias=AverageBias(axis="epochs"), n_components=3)
    sources = dss.fit_transform(epochs)

    # Sources should be (n_epochs, n_components, n_times)
    assert sources.shape == (n_epochs, 3, n_times)


def test_dss_with_mne_evoked():
    """DSS should work with MNE Evoked objects."""
    rng = np.random.default_rng(42)
    n_channels, n_times = 8, 200
    sfreq = 250.0

    data = rng.standard_normal((n_channels, n_times))

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    evoked = mne.EvokedArray(data, info, verbose=False)

    dss = DSS(bias=lambda x: x, n_components=3)
    sources = dss.fit_transform(evoked)

    assert sources.shape == (3, n_times)


def test_dss_mne_return_type_epochs():
    """DSS transform with return_type should return MNE object."""
    rng = np.random.default_rng(42)
    n_channels, n_times, n_epochs = 4, 50, 10
    sfreq = 100.0

    data = rng.standard_normal((n_epochs, n_channels, n_times))

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    epochs = mne.EpochsArray(data, info, verbose=False)

    dss = DSS(bias=lambda x: x, n_components=3, return_type="epochs")
    dss.fit(epochs)
    result = dss.transform(epochs)

    # Should return MNE Epochs object
    assert isinstance(result, mne.epochs.BaseEpochs)


def test_dss_mne_return_type_raw():
    """DSS transform with return_type='raw' should return Raw object."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 4, 2000
    sfreq = 500.0

    data = rng.standard_normal((n_channels, n_samples))

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    raw = mne.io.RawArray(data, info, verbose=False)

    dss = DSS(bias=lambda x: x, n_components=3, return_type="raw")
    dss.fit(raw)
    result = dss.transform(raw)

    # Should return MNE Raw object
    assert isinstance(result, mne.io.BaseRaw)


def test_dss_mne_with_weights():
    """DSS should handle MNE objects with weights (falls back to numpy path)."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 4, 1000
    sfreq = 250.0

    data = rng.standard_normal((n_channels, n_samples))

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    raw = mne.io.RawArray(data, info, verbose=False)

    weights = np.ones(n_samples)
    weights[500:] = 0.5  # Lower weight for second half

    dss = DSS(bias=lambda x: x, n_components=2)
    dss.fit(raw, weights=weights)

    assert dss.filters_.shape == (2, n_channels)


def test_dss_mne_normalization():
    """DSS normalization should work with MNE objects."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 4, 1000
    sfreq = 250.0

    # Create data with different scales
    data = rng.standard_normal((n_channels, n_samples))
    data[0] *= 1e-6  # Simulate gradiometer scale
    data[1] *= 1e-12  # Simulate magnetometer scale

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    raw = mne.io.RawArray(data, info, verbose=False)

    dss = DSS(bias=lambda x: x, n_components=3, normalize_input=True)
    sources = dss.fit_transform(raw)

    assert sources.shape == (3, n_samples)
    assert dss.channel_norms_ is not None


# =============================================================================
# More Tests with Known Expected Outputs for DSS class
# =============================================================================


def test_dss_array_extracts_known_signal():
    """DSS should extract a known sinusoidal signal from noise (numpy array)."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 8, 2000
    sfreq = 250

    # Create known signal: 10 Hz sinusoid in a known direction
    t = np.arange(n_samples) / sfreq
    signal = np.sin(2 * np.pi * 10 * t)  # Pure 10 Hz

    # Known mixing: signal goes to first 4 channels with weights [1, 0.8, 0.5, 0.2]
    mixing_weights = np.array([1, 0.8, 0.5, 0.2, 0, 0, 0, 0])
    mixing_weights = mixing_weights / np.linalg.norm(mixing_weights)

    # Add noise
    noise = rng.standard_normal((n_channels, n_samples)) * 0.5
    data = noise + 3 * np.outer(mixing_weights, signal)

    # Use bandpass bias around 10 Hz
    from mne_denoise.dss.denoisers import BandpassBias

    bias = BandpassBias(freq_band=(8, 12), sfreq=sfreq)

    dss = DSS(bias=bias, n_components=1, normalize_input=False)
    sources = dss.fit_transform(data)

    # Top source should correlate highly with original signal
    correlation = np.abs(np.corrcoef(sources[0], signal)[0, 1])
    assert correlation > 0.9, f"Correlation {correlation} too low, expected > 0.9"

    # Top filter should align with mixing weights direction
    top_filter = dss.filters_[0]
    alignment = np.abs(np.dot(top_filter / np.linalg.norm(top_filter), mixing_weights))
    assert alignment > 0.8, f"Filter alignment {alignment} too low, expected > 0.8"


def test_dss_array_evoked_extracts_known_erp():
    """DSS with trial average should extract known ERP from epoched data."""
    rng = np.random.default_rng(42)
    n_channels, n_times, n_epochs = 8, 100, 50

    # Create known ERP: Gaussian peak at sample 50
    erp = np.zeros(n_times)
    erp[40:60] = np.hanning(20) * 2  # Peak amplitude 2

    # Known mixing: ERP in first 3 channels
    mixing_weights = np.array([1, 0.7, 0.3, 0, 0, 0, 0, 0])
    mixing_weights = mixing_weights / np.linalg.norm(mixing_weights)

    # Add noise
    noise = rng.standard_normal((n_channels, n_times, n_epochs)) * 0.5
    signal = np.outer(mixing_weights, erp)[:, :, np.newaxis]  # (n_ch, n_times, 1)
    data = noise + signal  # Signal replicated across epochs

    from mne_denoise.dss.denoisers import AverageBias

    bias = AverageBias(axis="epochs")

    dss = DSS(bias=bias, n_components=1, normalize_input=False)
    sources = dss.fit_transform(data)  # (1, n_times, n_epochs)

    # Average across epochs
    source_avg = sources[0].mean(axis=1)  # (n_times,)

    # Source average should correlate with ERP
    correlation = np.abs(np.corrcoef(source_avg, erp)[0, 1])
    assert correlation > 0.9, f"ERP correlation {correlation} too low"


def test_dss_mne_raw_extracts_line_noise():
    """DSS should extract line noise from MNE Raw (functional test)."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 8, 5000
    sfreq = 500.0

    # Create 50 Hz line noise in channels 0-3
    t = np.arange(n_samples) / sfreq
    line_noise = np.sin(2 * np.pi * 50 * t)

    # Known mixing
    mixing_weights = np.zeros(n_channels)
    mixing_weights[:4] = [1, 0.8, 0.5, 0.2]
    mixing_weights = mixing_weights / np.linalg.norm(mixing_weights)

    # Create data
    noise = rng.standard_normal((n_channels, n_samples)) * 0.3
    data = noise + 2 * np.outer(mixing_weights, line_noise)

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    raw = mne.io.RawArray(data, info, verbose=False)

    # Use line noise bias (notch method)
    from mne_denoise.dss.denoisers import LineNoiseBias

    bias = LineNoiseBias(freq=50, sfreq=sfreq, method="iir", bandwidth=2)

    dss = DSS(bias=bias, n_components=1, normalize_input=False)
    sources = dss.fit_transform(raw)

    # Top source should correlate with line noise
    correlation = np.abs(np.corrcoef(sources[0], line_noise)[0, 1])
    assert correlation > 0.85, f"Line noise correlation {correlation} too low"


def test_dss_mne_epochs_extracts_known_erp():
    """DSS should extract known ERP from MNE Epochs (functional test)."""
    rng = np.random.default_rng(42)
    n_channels, n_times, n_epochs = 6, 100, 40
    sfreq = 100.0

    # Create known ERP
    erp = np.zeros(n_times)
    erp[45:55] = np.hanning(10) * 3  # Strong peak

    # Known mixing: ERP in first 3 channels
    mixing_weights = np.array([1, 0.6, 0.3, 0, 0, 0])
    mixing_weights = mixing_weights / np.linalg.norm(mixing_weights)

    # Create data (MNE format: n_epochs, n_channels, n_times)
    noise = rng.standard_normal((n_epochs, n_channels, n_times)) * 0.4
    signal = np.outer(mixing_weights, erp)[np.newaxis, :, :]  # (1, n_ch, n_times)
    data = noise + signal  # Broadcast signal to all epochs

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    epochs = mne.EpochsArray(data, info, verbose=False)

    from mne_denoise.dss.denoisers import AverageBias

    bias = AverageBias(axis="epochs")

    dss = DSS(bias=bias, n_components=1, normalize_input=False)
    sources = dss.fit_transform(epochs)  # (n_epochs, 1, n_times)

    # Average across epochs
    source_avg = sources[:, 0, :].mean(axis=0)  # (n_times,)

    # Should correlate with ERP
    correlation = np.abs(np.corrcoef(source_avg, erp)[0, 1])
    assert correlation > 0.9, f"Epochs ERP correlation {correlation} too low"


def test_dss_mne_evoked_extracts_known_signal():
    """DSS should work with MNE Evoked and extract known signal."""
    rng = np.random.default_rng(42)
    n_channels, n_times = 6, 200
    sfreq = 100.0

    # Create known oscillatory signal at 10 Hz
    t = np.arange(n_times) / sfreq
    signal = np.sin(2 * np.pi * 10 * t)

    # Known mixing
    mixing_weights = np.array([1, 0.5, 0.2, 0, 0, 0])
    mixing_weights = mixing_weights / np.linalg.norm(mixing_weights)

    # Create data
    noise = rng.standard_normal((n_channels, n_times)) * 0.3
    data = noise + 2 * np.outer(mixing_weights, signal)

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    evoked = mne.EvokedArray(data, info, verbose=False)

    from mne_denoise.dss.denoisers import BandpassBias

    bias = BandpassBias(freq_band=(8, 12), sfreq=sfreq)

    dss = DSS(bias=bias, n_components=1, normalize_input=False)
    sources = dss.fit_transform(evoked)

    # Top source should correlate with signal
    correlation = np.abs(np.corrcoef(sources[0], signal)[0, 1])
    assert correlation > 0.85, f"Evoked signal correlation {correlation} too low"


def test_dss_reconstruction_preserves_signal():
    """DSS transform + inverse_transform should preserve signal content."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 8, 500

    # Create data with known structure
    t = np.linspace(0, 1, n_samples)
    signal1 = np.sin(2 * np.pi * 5 * t)  # 5 Hz
    signal2 = np.sin(2 * np.pi * 10 * t)  # 10 Hz

    data = np.zeros((n_channels, n_samples))
    data[0] = signal1 * 2
    data[1] = signal1 + signal2
    data[2:] = rng.standard_normal((n_channels - 2, n_samples)) * 0.1

    dss = DSS(bias=lambda x: x, n_components=n_channels, normalize_input=False)
    sources = dss.fit_transform(data)
    reconstructed = dss.inverse_transform(sources)

    # Reconstruction should match centered original
    data_centered = data - data.mean(axis=1, keepdims=True)
    assert_allclose(reconstructed, data_centered, atol=0.1)


def test_dss_mne_epochs_inverse_transform_with_normalization():
    """inverse_transform should work with MNE Epochs format sources."""
    rng = np.random.default_rng(42)
    n_channels, n_times, n_epochs = 6, 50, 10
    sfreq = 100.0

    # Create data (MNE format: n_epochs, n_channels, n_times)
    data = rng.standard_normal((n_epochs, n_channels, n_times))

    info = mne.create_info(
        ch_names=[f"EEG{i:03d}" for i in range(n_channels)], sfreq=sfreq, ch_types="eeg"
    )
    epochs = mne.EpochsArray(data, info, verbose=False)

    from mne_denoise.dss.denoisers import AverageBias

    dss = DSS(bias=AverageBias(axis="epochs"), n_components=3, normalize_input=True)

    # Fit and transform - sources will be (n_epochs, n_components, n_times)
    sources = dss.fit_transform(epochs)

    assert sources.shape == (n_epochs, 3, n_times)

    # Now inverse_transform with MNE epochs format sources
    reconstructed = dss.inverse_transform(sources)

    # Should produce (n_epochs, n_channels, n_times)
    assert reconstructed.shape == (n_epochs, n_channels, n_times)


def test_dss_full_rank_reconstruction_exact_match():
    """DSS with n_components=n_channels should reconstruct data exactly (minus mean)."""
    rng = np.random.default_rng(42)
    n_channels, n_samples = 5, 500
    data = rng.standard_normal((n_channels, n_samples)) * 1e-6  # uV scale

    # Use no-op bias
    dss = DSS(bias=lambda x: x, n_components=n_channels, normalize_input=True)
    dss.fit(data)
    sources = dss.transform(data)
    rec = dss.inverse_transform(sources)

    # Comparison against centered data
    data_centered = data - data.mean(axis=1, keepdims=True)

    # Tolerances for floating point arithmetic
    # Relative tolerance 1e-7 is reasonable for float64
    assert_allclose(rec, data_centered, rtol=1e-7, atol=1e-25)


def test_dss_inverse_transform_mne_format_3d():
    """inverse_transform should detect MNE epochs format (n_epochs, n_comps, n_times)."""
    rng = np.random.default_rng(42)
    n_channels, n_times, n_epochs = 8, 100, 5
    n_components = 4

    # Create numpy 3D data (channels, times, epochs)
    data = rng.standard_normal((n_channels, n_times, n_epochs))

    dss = DSS(bias=lambda x: x, n_components=n_components, normalize_input=True)
    sources = dss.fit_transform(data)  # (n_components, n_times, n_epochs)

    assert sources.shape == (n_components, n_times, n_epochs)

    # Now manually transpose to MNE epochs format
    sources_mne_format = np.transpose(
        sources, (2, 0, 1)
    )  # (n_epochs, n_comps, n_times)

    # inverse_transform should detect this and handle it
    reconstructed = dss.inverse_transform(sources_mne_format)

    # Should produce (n_epochs, n_channels, n_times) since it detected MNE format
    assert reconstructed.shape == (n_epochs, n_channels, n_times)


def test_dss_normalization_with_different_scales():
    """Test DSS normalization with channels at vastly different scales."""
    rng = np.random.RandomState(42)
    n_samples = 200
    n_channels = 3

    # Create data with vastly different scales
    data = rng.randn(n_channels, n_samples)
    data[0] *= 1e-6  # "Gradiometer" scale
    data[1] *= 1  # "Magnetometer" scale
    data[2] *= 1000  # "EEG" scale

    class SimpleBias:
        def apply(self, x):
            return x

    # Fit without normalization
    dss_raw = DSS(n_components=3, bias=SimpleBias(), normalize_input=False)
    dss_raw.fit(data)

    # Fit with normalization
    dss_norm = DSS(n_components=3, bias=SimpleBias(), normalize_input=True)
    dss_norm.fit(data)

    # Check that channel norms were computed correctly
    assert dss_norm.channel_norms_ is not None
    assert dss_norm.channel_norms_.shape == (n_channels,)
    # Norms should reflect the scales
    assert (
        dss_norm.channel_norms_[0]
        < dss_norm.channel_norms_[1]
        < dss_norm.channel_norms_[2]
    )

    # Transform and reconstruct
    sources_norm = dss_norm.transform(data)
    assert sources_norm.shape == (3, n_samples)

    rec_norm = dss_norm.inverse_transform(sources_norm)
    data_centered = data - data.mean(axis=1, keepdims=True)
    assert_allclose(data_centered, rec_norm, atol=1e-5 * 1000, rtol=1e-5)


def test_dss_weighted_fit_ignores_outliers():
    """Test DSS fit with weights to mask out outliers."""
    rng = np.random.RandomState(42)
    n_samples = 200
    n_channels = 4
    data = rng.randn(n_channels, n_samples)

    # Create data with outliers in second half
    data_clean = data.copy()
    data_bad = data.copy()
    data_bad[0, 100:] = 1e6  # Huge outlier

    weights = np.ones(n_samples)
    weights[100:] = 0  # Ignore outlier region

    # Fit DSS on bad data with weights masking outliers
    dss = DSS(n_components=2, bias=lambda x: x, normalize_input=False)
    dss.fit(data_bad, weights=weights)

    # Fit DSS on clean data (first half only)
    dss_clean = DSS(n_components=2, bias=lambda x: x, normalize_input=False)
    dss_clean.fit(data_clean[:, :100])

    # Filters should be nearly identical (up to sign flip)
    f1 = dss.filters_
    f2 = dss_clean.filters_

    for i in range(2):
        corr = np.corrcoef(f1[i], f2[i])[0, 1]
        assert abs(corr) > 0.99, f"Filter {i} correlation {corr} too low"


def test_dss_cov_method_options():
    """Test DSS with different covariance method options."""
    rng = np.random.RandomState(42)
    n_samples = 2000
    n_channels = 3
    data = rng.randn(n_channels, n_samples)

    # Test numpy path with shrinkage
    dss = DSS(
        n_components=2,
        bias=lambda x: x,
        cov_method="shrinkage",
        cov_kws={"shrinkage": 0.1},
    )
    dss.fit(data)
    assert dss.filters_.shape == (2, 3)

    # Test MNE path with auto method
    info = mne.create_info(n_channels, 1000.0, "eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    dss_mne = DSS(
        n_components=2,
        bias=lambda x: x,
        cov_method="auto",
        cov_kws=None,
    )
    dss_mne.cov_method = "empirical"
    dss_mne.fit(raw)
    assert dss_mne.filters_.shape == (2, 3)


def test_dss_preserves_scale():
    """DSS reconstruction should preserve physical signal scale (Microvolts)."""
    sfreq = 1000
    n_channels = 10
    n_times = 5000
    t = np.arange(n_times) / sfreq

    signal_scale = 5e-6
    data = np.random.randn(n_channels, n_times) * 1e-7  # noise
    data[0:3, :] += signal_scale * np.sin(2 * np.pi * 10 * t)

    from mne_denoise.dss.denoisers import LinearDenoiser

    class IdentityBias(LinearDenoiser):
        def apply(self, data):
            return data

    bias = IdentityBias()
    dss = DSS(
        bias=bias, n_components=n_channels, normalize_input=False, return_type="raw"
    )
    reconstructed = dss.fit_transform(data)

    rms_orig = np.sqrt(np.mean(data**2))
    rms_rec = np.sqrt(np.mean(reconstructed**2))

    assert_allclose(rms_orig, rms_rec, rtol=0.05)


def test_dss_get_normalized_patterns():
    """Test the newly added get_normalized_patterns method in DSS."""
    from mne_denoise.dss.denoisers import LinearDenoiser

    class IdentityBias(LinearDenoiser):
        def apply(self, data):
            return data

    data = np.random.randn(10, 1000)
    dss = DSS(bias=IdentityBias(), n_components=2)
    dss.fit(data)
    norm_patterns = dss.get_normalized_patterns()
    assert norm_patterns.shape == (10, 2)
    assert_allclose(np.linalg.norm(norm_patterns, axis=0), 1.0)
