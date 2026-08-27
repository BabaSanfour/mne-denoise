"""Tests for the shared input validation helpers in mne_denoise._validation."""

from __future__ import annotations

import numpy as np
import pytest

from mne_denoise._validation import (
    check_channel_first_data,
    check_channel_layout,
    check_chunk_size,
    check_matching_sfreq,
    check_option,
    check_positive_integer,
    check_positive_real,
    resolve_sample_window,
    resolve_sfreq,
)


@pytest.fixture()
def rng():
    """Shared random generator."""
    return np.random.default_rng(0)


# ---------------------------------------------------------------------------
# check_positive_integer
# ---------------------------------------------------------------------------


def test_positive_integer_accepts_python_and_numpy_integers():
    """Positive integer validation accepts and normalizes integer scalars."""
    assert check_positive_integer(1, name="count") == 1
    value = check_positive_integer(np.int64(2), name="count")
    assert value == 2
    assert isinstance(value, int)


@pytest.mark.parametrize(
    ("value", "error"),
    [(True, TypeError), (1.5, TypeError), (0, ValueError), (-1, ValueError)],
)
def test_positive_integer_rejects_invalid_values(value, error):
    """Booleans, non-integers, and non-positive values are rejected."""
    with pytest.raises(error, match="count must be a positive integer"):
        check_positive_integer(value, name="count")


# ---------------------------------------------------------------------------
# check_positive_real
# ---------------------------------------------------------------------------


def test_positive_real_returns_python_float():
    """Valid real values are normalized to plain Python floats."""
    for value, expected in [(1, 1.0), (1.5, 1.5), (np.float64(0.5), 0.5)]:
        result = check_positive_real(value, name="width")
        assert result == expected
        assert isinstance(result, float)


@pytest.mark.parametrize("value", [True, "1.0", None])
def test_positive_real_rejects_invalid_types(value):
    """Booleans and non-real values are rejected with a type error."""
    with pytest.raises(TypeError, match="width must be a positive, finite number"):
        check_positive_real(value, name="width")


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, -np.inf])
def test_positive_real_rejects_invalid_values(value):
    """Non-finite and non-positive values are rejected."""
    with pytest.raises(ValueError, match="width must be a positive, finite number"):
        check_positive_real(value, name="width")


# ---------------------------------------------------------------------------
# check_option
# ---------------------------------------------------------------------------


def test_option_accepts_an_allowed_value():
    """Allowed categorical values pass unchanged."""
    value = "auto"
    assert check_option(value, name="blend", allowed=("auto", "constant")) is value


def test_option_rejects_an_unallowed_value_with_context():
    """The error identifies the parameter, choices, and received value."""
    with pytest.raises(ValueError) as exc_info:
        check_option("invalid", name="blend", allowed=("auto", "constant"))
    message = str(exc_info.value)
    assert all(part in message for part in ("blend", "auto", "constant"))
    assert "received value" in message


# ---------------------------------------------------------------------------
# check_matching_sfreq
# ---------------------------------------------------------------------------


def test_matching_sfreq_accepts_exact_and_close_values():
    """Exact and default-tolerance matches are accepted."""
    check_matching_sfreq(250.0, 250.0, name="X")
    check_matching_sfreq(250.0005, 250.0, name="X")


def test_matching_sfreq_accepts_missing_metadata():
    """Missing input or fitted metadata does not create a mismatch."""
    check_matching_sfreq(None, 250.0, name="X")
    check_matching_sfreq(250.0, None, name="X")


def test_matching_sfreq_rejects_a_meaningful_mismatch():
    """A mismatch reports the estimator and both frequencies."""
    with pytest.raises(ValueError, match="X.*transform sfreq=251.*fitted sfreq=250"):
        check_matching_sfreq(251.0, 250.0, name="X")


def test_matching_sfreq_honours_strict_custom_tolerance():
    """Caller-controlled tolerance supports strict fitted contracts."""
    check_matching_sfreq(100.0 + 5e-13, 100.0, name="X", rtol=0.0, atol=1e-12)
    with pytest.raises(ValueError, match="transform sfreq"):
        check_matching_sfreq(100.0 + 2e-12, 100.0, name="X", rtol=0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# check_channel_first_data
# ---------------------------------------------------------------------------


def test_accepts_continuous_and_epoched(rng):
    """Both supported layouts pass and are returned as float64."""
    continuous = check_channel_first_data(rng.standard_normal((4, 100)), name="X")
    epoched = check_channel_first_data(rng.standard_normal((3, 4, 100)), name="X")
    assert continuous.dtype == np.float64
    assert epoched.shape == (3, 4, 100)


def test_converts_integer_input():
    """Integer input is converted rather than rejected."""
    out = check_channel_first_data(np.ones((3, 10), dtype=int), name="X")
    assert out.dtype == np.float64


def test_epochs_can_be_disallowed(rng):
    """allow_epochs=False rejects three-dimensional input."""
    with pytest.raises(ValueError, match="Expected a 2-D channel-first array"):
        check_channel_first_data(
            rng.standard_normal((3, 4, 100)), name="X", allow_epochs=False
        )


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (np.ones(10), "2-D or 3-D"),
        (np.ones((2, 2, 2, 2)), "2-D or 3-D"),
        (np.ones((1, 100)), "at least two channels"),
        (np.ones((4, 1)), "at least two time samples"),
        (np.full((4, 100), np.nan), "finite"),
        (np.full((4, 100), np.inf), "finite"),
    ],
)
def test_rejects_invalid_data(data, message):
    """Shape, size, and finiteness preconditions are enforced."""
    with pytest.raises(ValueError, match=message):
        check_channel_first_data(data, name="X")


def test_error_messages_name_the_algorithm():
    """The caller's name appears in the message, not a hard-coded one."""
    with pytest.raises(ValueError, match="SNS requires at least two channels"):
        check_channel_first_data(np.ones((1, 100)), name="SNS")
    with pytest.raises(ValueError, match="BSS-CCA requires at least two channels"):
        check_channel_first_data(np.ones((1, 100)), name="BSS-CCA")


def test_minimum_sizes_are_configurable(rng):
    """Callers can demand more channels or samples than the default."""
    data = rng.standard_normal((3, 100))
    check_channel_first_data(data, name="X", min_channels=3)
    with pytest.raises(ValueError, match="at least two channels"):
        check_channel_first_data(data, name="X", min_channels=4)


def test_empty_epoch_axis_is_rejected():
    """A zero-epoch array carries no data."""
    with pytest.raises(ValueError, match="at least one epoch"):
        check_channel_first_data(np.ones((0, 4, 100)), name="X")


# ---------------------------------------------------------------------------
# resolve_sample_window
# ---------------------------------------------------------------------------


def test_sample_window_resolves_samples_and_seconds():
    """Both explicit units use one shared half-open conversion contract."""
    assert resolve_sample_window((-2, 3), unit="samples") == (-2, 3)
    assert resolve_sample_window((-0.015, 0.025), unit="seconds", sfreq=100.0) == (
        -2,
        2,
    )


@pytest.mark.parametrize("sfreq", [0.0, -1.0, np.nan, np.inf])
def test_sample_window_rejects_invalid_sfreq(sfreq):
    """A supplied sampling frequency must be positive and finite."""
    with pytest.raises(ValueError, match="sfreq must be a positive, finite number"):
        resolve_sample_window((0, 1), unit="samples", sfreq=sfreq)


@pytest.mark.parametrize(
    ("window", "unit", "sfreq", "match"),
    [
        ((-1.5, 2), "samples", None, "must be integers"),
        ((2, -1), "samples", None, "strictly less"),
        ((0, 0.001), "seconds", 100.0, "empty or reversed"),
        ((-1, 2), "seconds", None, "sfreq is required"),
        ((-1, 2), "minutes", None, "window_unit"),
        ((-(2**63) - 1, 2), "samples", None, "signed 64-bit"),
        ((-1e308, 1e308), "seconds", 1e308, "finite sample range"),
    ],
)
def test_sample_window_rejects_invalid_contracts(window, unit, sfreq, match):
    """Invalid windows fail consistently at the shared package boundary."""
    with pytest.raises((TypeError, ValueError), match=match):
        resolve_sample_window(window, unit=unit, sfreq=sfreq)


# ---------------------------------------------------------------------------
# check_chunk_size
# ---------------------------------------------------------------------------


def test_chunk_size_accepts_none_and_integers():
    """None means 'all at once'; integers are normalized."""
    assert check_chunk_size(None) is None
    assert check_chunk_size(np.int64(64)) == 64
    assert isinstance(check_chunk_size(64), int)


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (True, TypeError),
        (1.5, TypeError),
        ("64", TypeError),
        (0, ValueError),
        (-1, ValueError),
    ],
)
def test_chunk_size_rejects_invalid(value, error):
    """Booleans, non-integers, and non-positive values are rejected."""
    with pytest.raises(error, match="chunk_size must be a positive integer or None"):
        check_chunk_size(value)


# ---------------------------------------------------------------------------
# resolve_sfreq
# ---------------------------------------------------------------------------


def test_resolve_returns_declared_value_when_it_is_the_only_source():
    """A declared value is returned when container metadata is absent."""
    assert resolve_sfreq(250.0, None) == 250.0


def test_resolve_returns_container_value_when_it_is_the_only_source():
    """Container metadata is used when no value was declared."""
    assert resolve_sfreq(None, 250.0) == 250.0


def test_resolve_returns_the_container_value_when_both_sources_agree():
    """Agreeing sources collapse to one effective value."""
    assert resolve_sfreq(250.0, 250.0) == 250.0


def test_resolve_rejects_disagreement():
    """A declared value is never silently discarded."""
    with pytest.raises(ValueError, match="disagrees with MNE info sfreq"):
        resolve_sfreq(100.0, 250.0)


def test_resolve_reports_a_missing_value():
    """With nothing to go on, the caller learns what needed it."""
    with pytest.raises(ValueError, match="sfreq is required when lag_seconds is used"):
        resolve_sfreq(None, None, context="lag_seconds")


def test_resolve_rejects_a_missing_required_value():
    """The default required contract rejects two missing sources."""
    with pytest.raises(ValueError, match="^sfreq is required$"):
        resolve_sfreq(None, None)


def test_resolve_can_allow_a_missing_value():
    """Optional sampling frequencies return None rather than raising."""
    assert resolve_sfreq(None, None, required=False) is None


@pytest.mark.parametrize(
    "value",
    [-1.0, np.nan, np.inf, True, "250"],
)
def test_resolve_rejects_invalid_declared_sfreq(value):
    """The declared source uses the shared positive-real contract."""
    error = TypeError if isinstance(value, (bool, str)) else ValueError
    with pytest.raises(error, match="sfreq must be a positive, finite number"):
        resolve_sfreq(value, None)


@pytest.mark.parametrize(
    "value",
    [-1.0, np.nan, np.inf, True, "250"],
)
def test_resolve_rejects_invalid_data_sfreq(value):
    """Container metadata uses the same shared positive-real contract."""
    error = TypeError if isinstance(value, (bool, str)) else ValueError
    with pytest.raises(error, match="sfreq must be a positive, finite number"):
        resolve_sfreq(None, value)


# ---------------------------------------------------------------------------
# check_channel_layout
# ---------------------------------------------------------------------------


def test_channel_layout_accepts_a_match():
    """Identical names and counts pass."""
    check_channel_layout(
        "X",
        n_channels=2,
        fitted_n_channels=2,
        ch_names=("a", "b"),
        fitted_ch_names=("a", "b"),
    )


def test_channel_layout_rejects_reordering():
    """Order is part of the layout, not just membership."""
    with pytest.raises(ValueError, match="names/order differ from fit"):
        check_channel_layout(
            "SNS",
            n_channels=2,
            fitted_n_channels=2,
            ch_names=("b", "a"),
            fitted_ch_names=("a", "b"),
        )


def test_channel_layout_rejects_a_count_mismatch():
    """Array input has no names, so the count is the only check."""
    with pytest.raises(ValueError, match="X has 3 channels; fitted data had 2"):
        check_channel_layout("X", n_channels=3, fitted_n_channels=2)


def test_channel_layout_skips_names_for_arrays():
    """A fitted-on-array estimator does not demand names."""
    check_channel_layout(
        "X",
        n_channels=2,
        fitted_n_channels=2,
        ch_names=("a", "b"),
        fitted_ch_names=None,
    )
