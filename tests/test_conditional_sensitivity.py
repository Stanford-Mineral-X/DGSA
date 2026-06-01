"""
Tests for conditional parameter sensitivity against the Park2016 reference results.

Both analyses are re-run with identical parameters to the notebook
(random_seed=100, n_draws=3000, alpha=0.95, n_bins=3) via the park2016 fixture.
Standardized values are compared with rtol=0.10 — higher than single sensitivity
because bootstrap variance compounds across the (n_params, n_params, n_clusters, n_bins)
array.

NaN positions (diagonal: a parameter conditioned on itself) are validated
separately before the allclose check, since assert_allclose does not
handle NaN positions consistently.
"""

import numpy as np
import pytest


def test_conditional_diagonal_is_nan_l1norm(park2016):
    standardized = park2016["conditional"]["conditional_l1norm"]["standardized"]
    assert np.all(np.isnan(np.diag(standardized)))


def test_conditional_diagonal_is_nan_ASL(park2016):
    standardized = park2016["conditional"]["conditional_ASL"]["standardized"]
    assert np.all(np.isnan(np.diag(standardized)))


def test_conditional_l1norm_standardized(park2016):
    result = park2016["conditional"]["conditional_l1norm"]["standardized"]
    ref = park2016["ref"]["conditional_l1norm"]["standardized"]

    # NaN positions must match before comparing values
    np.testing.assert_array_equal(np.isnan(result), np.isnan(ref))

    # Compare non-NaN values
    mask = ~np.isnan(ref)
    np.testing.assert_allclose(result[mask], ref[mask], rtol=0.10)


def test_conditional_ASL_standardized(park2016):
    result = park2016["conditional"]["conditional_ASL"]["standardized"]
    ref = park2016["ref"]["conditional_ASL"]["standardized"]

    np.testing.assert_array_equal(np.isnan(result), np.isnan(ref))

    mask = ~np.isnan(ref)
    np.testing.assert_allclose(result[mask], ref[mask], rtol=0.10)


def test_conditional_invalid_method(park2016):
    from dgsa.computation.conditional_parameter_sensitivity import conditional_parameter_sensitivity
    with pytest.raises(ValueError):
        conditional_parameter_sensitivity(
            parameter_values=park2016["parameter_values"],
            parameter_names=park2016["ref"]["parameter_names"],
            clustering=park2016["ref"]["clustering"],
            method="bad_method",
        )
