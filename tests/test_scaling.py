"""
Tests for scale_responses.

The 'minmax' and 'zscore' methods are checked against scikit-learn's
MinMaxScaler and StandardScaler, which the notebooks used previously.
"""

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from dgsa.utils.scaling import scale_responses


@pytest.fixture
def responses():
    """Responses with very different magnitudes per column."""
    rng = np.random.default_rng(100)
    return np.column_stack([
        rng.normal(0, 1, 50),
        rng.normal(1e6, 1e5, 50),
        rng.uniform(-5, 5, 50),
    ])


def test_raw_returns_unchanged_copy(responses):
    scaled = scale_responses(responses, method='raw')
    np.testing.assert_array_equal(scaled, responses)
    assert scaled is not responses


def test_minmax_matches_sklearn(responses):
    np.testing.assert_allclose(
        scale_responses(responses, method='minmax'),
        MinMaxScaler().fit_transform(responses),
    )


def test_zscore_matches_sklearn(responses):
    np.testing.assert_allclose(
        scale_responses(responses, method='zscore'),
        StandardScaler().fit_transform(responses),
    )


@pytest.mark.parametrize("method", ['minmax', 'zscore'])
def test_constant_column_scaled_to_zero(responses, method):
    responses[:, 1] = 5.0
    scaled = scale_responses(responses, method=method)
    np.testing.assert_array_equal(scaled[:, 1], 0.0)
    assert np.all(np.isfinite(scaled))


def test_invalid_method_raises(responses):
    with pytest.raises(ValueError):
        scale_responses(responses, method='bad')
