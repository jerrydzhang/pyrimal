import numpy as np
import pytest
import torch

# Skip all tests in this module if physo is not installed
physo = pytest.importorskip("physo")

from primel.adapters.physo import PhySOAdapter
from primel.distributions import MultivariateUniform
from primel.samplers import ImportanceSampler, RandomSampler, LHSampler


@pytest.fixture
def sampler():
    dist = MultivariateUniform(X=np.random.rand(10, 2), margins=0.1)
    sampler_entries = [("train", RandomSampler(dist), 10)]
    return ImportanceSampler(sampler_entries=sampler_entries)


@pytest.fixture
def reference_dist():
    return MultivariateUniform(X=np.random.rand(10, 2), margins=0.1)


@pytest.fixture
def multi_component_sampler():
    """Sampler with multiple components (train + local + global)."""
    dist = MultivariateUniform(X=np.random.rand(20, 2), margins=0.1)
    sampler_entries = [
        ("train", RandomSampler(dist), 50),
        ("kde", RandomSampler(dist), 20),
        ("uniform", LHSampler(dist), 30),
    ]
    return ImportanceSampler(sampler_entries=sampler_entries)


def test_physo_adapter_init(sampler, reference_dist):
    adapter = PhySOAdapter(sampler=sampler, reference_distribution=reference_dist)
    assert adapter.lambda_ == 1.0
    assert adapter.exponent == 1.0
    assert adapter.mean_center_on is None


def test_physo_adapter_reward_config(sampler, reference_dist):
    adapter = PhySOAdapter(sampler=sampler, reference_distribution=reference_dist)
    config = adapter.get_reward_config()

    assert "reward_function" in config
    assert config["zero_out_unphysical"] is True
    assert config["zero_out_duplicates"] is False

    # Test reward function returns value in [0, 1]
    reward_fn = config["reward_function"]
    y_pred = np.random.randn(len(sampler.samples))
    reward = reward_fn(np.zeros_like(y_pred), y_pred)
    assert 0 <= reward <= 1


def test_physo_adapter_learning_config(sampler, reference_dist):
    adapter = PhySOAdapter(sampler=sampler, reference_distribution=reference_dist)
    config = adapter.get_learning_config()

    assert "learning_config" in config  # ✓ Verify top-level key exists
    assert (
        "rewards_computer" in config["learning_config"]
    )  # ✓ Check correct nested level


def test_physo_adapter_reward_with_torch_tensor(sampler, reference_dist):
    adapter = PhySOAdapter(sampler=sampler, reference_distribution=reference_dist)
    reward_fn = adapter.get_reward_config()["reward_function"]

    # Create torch tensor y_pred
    y_pred_torch = torch.randn(len(sampler.samples))
    reward = reward_fn(np.zeros(len(sampler.samples)), y_pred_torch)
    assert 0 <= reward <= 1


def test_physo_adapter_validates_prediction_length(sampler, reference_dist):
    adapter = PhySOAdapter(sampler=sampler, reference_distribution=reference_dist)
    reward_fn = adapter.get_reward_config()["reward_function"]

    # Wrong length - should raise ValueError
    y_pred_wrong = np.random.randn(len(sampler.samples) - 5)
    with pytest.raises(ValueError, match="does not match"):
        reward_fn(np.zeros_like(y_pred_wrong), y_pred_wrong)


def test_physo_adapter_multi_component_sampler(multi_component_sampler, reference_dist):
    """Verify reward works with multi-component sampler (train + local + global)."""
    adapter = PhySOAdapter(
        sampler=multi_component_sampler,
        reference_distribution=reference_dist,
    )
    reward_fn = adapter.get_reward_config()["reward_function"]

    # PhySO would provide predictions for all 100 samples (50 + 20 + 30)
    y_pred_all = np.random.randn(len(multi_component_sampler.samples))
    reward = reward_fn(np.zeros(len(y_pred_all)), y_pred_all)
    assert 0 <= reward <= 1
