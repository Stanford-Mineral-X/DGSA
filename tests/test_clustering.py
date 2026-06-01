"""
Tests for kmedoids clustering against the Park2016 reference results.

kmedoids is non-deterministic (no random_seed argument), so we cannot
require bit-exact reproduction. Instead we check structural invariants
that must hold regardless of the random initialisation:
  - every sample is assigned to exactly one cluster
  - cluster counts sum to n_samples
  - medoid indices fall within valid range
  - the cost (sum of distances from each point to its medoid) is at most
    as large as the reference cost, allowing a tolerance for a different
    but equally valid solution
"""

import numpy as np
import pytest


N_SAMPLES = 1000
N_CLUSTERS = 3


def _cost(distance_matrix, clustering):
    """Sum of distances from each point to its assigned medoid."""
    medoids = clustering["medoid_indices"]
    assignments = clustering["cluster_assignments"]
    return sum(
        distance_matrix[i, medoids[assignments[i]]] for i in range(len(assignments))
    )


def test_cluster_assignments_cover_all_samples(park2016):
    assignments = park2016["clustering_rerun"]["cluster_assignments"]
    assert len(assignments) == N_SAMPLES


def test_cluster_labels_are_valid(park2016):
    assignments = park2016["clustering_rerun"]["cluster_assignments"]
    assert set(assignments).issubset(set(range(N_CLUSTERS)))


def test_n_points_sums_to_n_samples(park2016):
    n_points = park2016["clustering_rerun"]["n_points"]
    assert n_points.sum() == N_SAMPLES


def test_medoid_indices_in_range(park2016):
    medoids = park2016["clustering_rerun"]["medoid_indices"]
    assert len(medoids) == N_CLUSTERS
    assert np.all(medoids >= 0) and np.all(medoids < N_SAMPLES)


def test_clustering_cost_within_tolerance_of_reference(park2016):
    """
    A different random init may find the same or better solution.
    We allow the rerun cost to be at most 5% higher than the reference,
    accepting that occasional worse initialisations are possible.
    """
    dm = park2016["distance_matrix"]
    ref_cost = _cost(dm, park2016["ref"]["clustering"])
    rerun_cost = _cost(dm, park2016["clustering_rerun"])
    assert rerun_cost <= ref_cost * 1.05
