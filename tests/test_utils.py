"""Tests for MNE object reconstruction utilities."""

import mne
import numpy as np

from mne_denoise.utils import reconstruct_mne_object


def test_reconstruct_mne_object_array_passthrough():
    arr = np.random.randn(3, 50)
    out = reconstruct_mne_object(arr, None, "array")
    assert out is arr


def test_reconstruct_mne_object_none_orig():
    arr = np.random.randn(3, 50)
    out = reconstruct_mne_object(arr, "dummy", "array")
    assert out is arr


def test_reconstruct_mne_object_raw_reconstruction():
    info = mne.create_info(ch_names=["C1", "C2"], sfreq=100.0, ch_types="eeg")
    raw = mne.io.RawArray(np.random.randn(2, 200), info)
    new_data = np.random.randn(2, 200)
    out = reconstruct_mne_object(new_data, raw, "raw")
    assert isinstance(out, mne.io.RawArray)
    np.testing.assert_array_almost_equal(out.get_data(), new_data)


def test_reconstruct_mne_object_raw_with_annotations():
    info = mne.create_info(ch_names=["C1"], sfreq=100.0, ch_types="eeg")
    raw = mne.io.RawArray(np.random.randn(1, 200), info)
    raw.set_annotations(
        mne.Annotations(onset=[0.5], duration=[0.1], description=["bad"])
    )
    new_data = np.random.randn(1, 200)
    out = reconstruct_mne_object(new_data, raw, "raw")
    assert len(out.annotations) > 0


def test_reconstruct_mne_object_raw_preserves_first_sample_and_picks():
    """Copy-based restoration retains Raw identity and untouched channels."""
    info = mne.create_info(["C1", "C2", "STIM"], 100.0, ["eeg", "eeg", "stim"])
    original = np.random.randn(3, 200)
    raw = mne.io.RawArray(original, info, first_samp=41, verbose=False)
    replacement = np.random.randn(2, 200)
    out = reconstruct_mne_object(replacement, raw, "raw", picks=np.array([0, 1]))
    assert out.first_samp == 41
    np.testing.assert_allclose(out.get_data()[:2], replacement)
    np.testing.assert_array_equal(out.get_data()[2], original[2])
    np.testing.assert_array_equal(raw.get_data(), original)


def test_reconstruct_mne_object_epochs_reconstruction():
    info = mne.create_info(ch_names=["C1", "C2"], sfreq=100.0, ch_types="eeg")
    data_3d = np.random.randn(5, 2, 100)
    events = np.column_stack([np.arange(5) * 100, np.zeros(5, int), np.ones(5, int)])
    epochs = mne.EpochsArray(data_3d, info, events=events, event_id={"stim": 1})
    new_data = np.random.randn(5, 2, 100)
    out = reconstruct_mne_object(new_data, epochs, "epochs")
    assert isinstance(out, mne.EpochsArray)
    np.testing.assert_array_almost_equal(out.get_data(), new_data)


def test_reconstruct_mne_object_epochs_preserves_selection_and_drop_log():
    """Epoch bookkeeping survives copy-based reconstruction."""
    info = mne.create_info(["C1", "C2", "STIM"], 100.0, ["eeg", "eeg", "stim"])
    data = np.random.randn(4, 3, 80)
    events = np.column_stack([np.arange(4) * 100, np.zeros(4, int), np.ones(4, int)])
    epochs = mne.EpochsArray(data, info, events=events, verbose=False)
    epochs.drop([1], reason="USER")
    replacement = np.random.randn(3, 2, 80)
    out = reconstruct_mne_object(replacement, epochs, "epochs", picks=np.array([0, 1]))
    np.testing.assert_array_equal(out.selection, epochs.selection)
    assert out.drop_log == epochs.drop_log
    np.testing.assert_allclose(out.get_data()[:, :2], replacement)
    np.testing.assert_array_equal(out.get_data()[:, 2], epochs.get_data()[:, 2])


def test_reconstruct_mne_object_evoked_reconstruction():
    info = mne.create_info(ch_names=["C1", "C2"], sfreq=100.0, ch_types="eeg")
    data_2d = np.random.randn(2, 100)
    evoked = mne.EvokedArray(data_2d, info, tmin=-0.1, nave=10, comment="test")
    new_data = np.random.randn(2, 100)
    out = reconstruct_mne_object(new_data, evoked, "evoked")
    assert isinstance(out, mne.EvokedArray)
    assert out.nave == 10
    assert out.comment == "test"


def test_reconstruct_mne_object_unknown_type_passthrough():
    arr = np.random.randn(3, 50)
    info = mne.create_info(ch_names=["C1", "C2", "C3"], sfreq=100.0, ch_types="eeg")
    raw = mne.io.RawArray(arr, info)
    out = reconstruct_mne_object(arr, raw, "unknown")
    assert out is arr
