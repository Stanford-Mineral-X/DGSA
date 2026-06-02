"""
Shared test fixtures for DGSA tests.

Reference results were generated from the Park2016 dataset using:
    random_seed=100, n_draws=3000, alpha=0.95, n_clusters=3, n_bins=3
and stored as a frozen snapshot in tests/fixtures/ so notebook reruns
cannot silently change what the tests compare against.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dgsa.computation.kmedoids import kmedoids
from dgsa.computation.single_parameter_sensitivity import single_parameter_sensitivity
from dgsa.computation.conditional_parameter_sensitivity import conditional_parameter_sensitivity

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def park2016():
    """Load frozen reference results and re-run analyses with identical parameters."""

    # Load frozen reference
    with open(FIXTURES / "Park2016_DGSA_results.pkl", "rb") as f:
        ref = pickle.load(f)

    # Load inputs
    parameter_values = pd.read_csv(FIXTURES / "Park2016_parameters.csv").to_numpy()
    distance_matrix = pd.read_csv(
        FIXTURES / "Park2016_distance_matrix.csv", header=None
    ).to_numpy()
    parameter_names = ref["parameter_names"]
    clustering = ref["clustering"]  # use reference clustering for sensitivity tests

    # Re-run kmedoids with fixed random_seed for reproducibility
    # We compare against the reference cluster_assignments in test_clustering.py
    clustering_rerun = kmedoids(
        distance_matrix=distance_matrix,
        n_clusters=3,
        n_rep=5,
        max_iterations=50,
        random_seed=100,
    )

    # Re-run single sensitivity with same parameters as notebook
    single = single_parameter_sensitivity(
        parameter_values=parameter_values,
        clustering=clustering,
        alpha=0.95,
        n_draws=3000,
        random_seed=100,
        method="l1norm_and_ASL",
    )

    # Re-run conditional sensitivity with same parameters as notebook
    conditional = conditional_parameter_sensitivity(
        parameter_values=parameter_values,
        parameter_names=parameter_names,
        clustering=clustering,
        alpha=0.95,
        n_bins=3,
        n_draws=3000,
        random_seed=100,
        method="l1norm_and_ASL",
    )

    return {
        "ref": ref,
        "parameter_values": parameter_values,
        "distance_matrix": distance_matrix,
        "clustering_rerun": clustering_rerun,
        "single": single,
        "conditional": conditional,
    }
