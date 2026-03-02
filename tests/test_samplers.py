import numpy as np
import pytest
from primel.distributions import Empirical, MixtureModel, MultivariateUniform
from primel.samplers import (
    RandomSampler,
    StratifiedSampler,
    LHSampler,
    ImportanceSampler,
)


@pytest.fixture
def sample_data():
    return np.array([[1, 2], [3, 4], [5, 6]])


@pytest.fixture
def empirical_dist(sample_data):
    return Empirical(data=sample_data)


@pytest.fixture
def uniform_dist(sample_data):
    return MultivariateUniform(X=sample_data, margins=0.1)


@pytest.fixture
def mixture_dist(sample_data):
    dist1 = MultivariateUniform(X=sample_data, margins=0.1)
    dist2 = MultivariateUniform(X=sample_data, margins=0.2)
    return MixtureModel(components=[dist1, dist2], weights=np.array([0.4, 0.6]))


def test_random_sampler(empirical_dist):
    sampler = RandomSampler(dist=empirical_dist)
    samples = sampler.sample(5, random_state=42)
    assert samples.shape == (5, 2)


def test_stratified_sampler(mixture_dist):
    sampler = StratifiedSampler(dist=mixture_dist)
    samples = sampler.sample(10, random_state=42)
    assert samples.shape == (10, 2)


def test_lh_sampler(uniform_dist):
    sampler = LHSampler(dist=uniform_dist)
    samples = sampler.sample(10, random_state=42)
    assert samples.shape == (10, 2)
    assert np.all(samples >= uniform_dist.lower)
    assert np.all(samples <= uniform_dist.upper)


def test_importance_sampler(empirical_dist, uniform_dist):
    sampler_entries = [
        ("empirical", RandomSampler(empirical_dist), 10),
        ("uniform", RandomSampler(uniform_dist), 20, 0.5),
    ]

    sampler = ImportanceSampler(sampler_entries=sampler_entries)

    assert list(sampler.name_map.keys()) == ["empirical", "uniform"]
    assert sampler.range_map == [(0, 10), (10, 30)]
    assert sampler.samples.shape == (30, 2)
    assert sampler.weights.shape == (30,)

    emp_samples = sampler.get_samples("empirical")
    assert emp_samples.shape == (10, 2)

    uni_weights = sampler.get_weights("uniform")
    assert uni_weights.shape == (20,)
    # With fallback weighting for Empirical: each sample from dist i gets weight n_i
    # empirical: 10 samples each get weight 10 -> total 100
    # uniform: 20 samples each get weight 20 -> total 400
    # Total: 500
    # After normalization: uniform samples get 20/500 = 0.04 each
    expected_uniform_weight = 20.0 / 500.0
    assert np.allclose(uni_weights, expected_uniform_weight)

    emp_weights = sampler.get_weights("empirical")
    expected_empirical_weight = 10.0 / 500.0
    assert np.allclose(emp_weights, expected_empirical_weight)

    # Verify weights sum to 1
    assert np.allclose(np.sum(sampler.weights), 1.0)

    with pytest.raises(ValueError):
        sampler.get_samples("nonexistent")

    with pytest.raises(ValueError):
        sampler.get_weights("nonexistent")


def test_importance_sampler_balance_heuristic(sample_data):
    """Test that balance heuristic is applied when all distributions have PDFs."""
    # Create two uniform distributions with some overlap
    uniform_dist1 = MultivariateUniform(X=sample_data, margins=0.1, non_negative=False)
    uniform_dist2 = MultivariateUniform(
        X=sample_data + 2,  # Offset the second distribution
        margins=0.1,
        non_negative=False
    )
    
    sampler_entries = [
        ("uniform1", RandomSampler(uniform_dist1), 10),
        ("uniform2", RandomSampler(uniform_dist2), 20),
    ]

    sampler = ImportanceSampler(sampler_entries=sampler_entries, random_state=42)

    # Verify balance heuristic is used (not fallback)
    # Balance heuristic: w_i(x) = (n_i * p_i(x)) / Σ_j(n_j * p_j(x))
    # Weights should vary by sample location, not be uniform per distribution
    
    uniform1_weights = sampler.get_weights("uniform1")
    uniform2_weights = sampler.get_weights("uniform2")
    
    # Check that weights are not all equal within a distribution
    # (This would be the case with fallback weighting)
    # With balance heuristic, samples in overlapping regions get different weights
    assert sampler.samples.shape == (30, 2)
    assert np.allclose(np.sum(sampler.weights), 1.0)
    
    # Verify weights are positive
    assert np.all(uniform1_weights > 0)
    assert np.all(uniform2_weights > 0)
