"""
Tests for single parameter sensitivity against the Park2016 reference results.

Both analyses are re-run with identical parameters to the notebook
(random_seed=100, n_draws=3000, alpha=0.95) via the park2016 fixture.
Standardized values are compared with rtol=0.05 to allow for acceptable
bootstrap variance across runs.
"""

import numpy as np
import pytest


def test_single_l1norm_standardized(park2016):
    result = park2016["single"]["single_l1norm"]["standardized"]
    ref = park2016["ref"]["single_l1norm"]["standardized"]
    np.testing.assert_allclose(result, ref, rtol=0.05)


def test_single_ASL_standardized(park2016):
    result = park2016["single"]["single_ASL"]["standardized"]
    ref = park2016["ref"]["single_ASL"]["standardized"]
    np.testing.assert_allclose(result, ref, rtol=0.05)


def test_single_invalid_method(park2016):
    from dgsa.computation.single_parameter_sensitivity import single_parameter_sensitivity
    with pytest.raises(ValueError):
        single_parameter_sensitivity(
            parameter_values=park2016["parameter_values"],
            clustering=park2016["ref"]["clustering"],
            method="bad_method",
        )
