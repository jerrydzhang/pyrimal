from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Self

import numpy as np
from jernerics.experiment import Experiment

from primel.adapters.physo import PhySOAdapter
from primel.distributions import Empirical, GaussianKDE, MultivariateUniform
from primel.samplers import ImportanceSampler, LHSampler, RandomSampler
from primel.fitness import induced_kl_divergence

import physo


DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass
class PhySOExperiment(Experiment):
    random_state: int
    metrics: dict[str, Any] = field(default_factory=dict)

    def setup_data(self: Self, config: Dict[str, Any]) -> np.ndarray:
        """Load data from CSV file."""
        data_path = DATA_DIR / config["data_file"]
        data = np.loadtxt(data_path, delimiter=",")
        return data

    def train(self: Self, data: np.ndarray, config: dict) -> None:
        """Train PhySO model with multi-component sampling."""
        # Create distributions
        empirical_dist = Empirical(data=data)
        gaussian_kde_dist = GaussianKDE(
            X=data, bandwidth=config["model__kde_bandwidth"]
        )
        uniform_dist = MultivariateUniform(X=data, margins=0.1, non_negative=False)

        # Create multi-component sampler (train + kde + uniform)
        sampler = ImportanceSampler(
            sampler_entries=[
                (
                    "train",
                    RandomSampler(empirical_dist),
                    data.shape[0],
                ),  # All training points
                (
                    "kde",
                    RandomSampler(gaussian_kde_dist),
                    config.get("model__n_kde", 100),
                ),  # Local exploration
                (
                    "uniform",
                    LHSampler(uniform_dist),
                    config.get("model__n_uniform", 100),
                ),  # Global exploration
            ],
            random_state=self.random_state,
        )

        # Create PhySO adapter
        adapter = PhySOAdapter(
            sampler=sampler,
            reference_distribution=gaussian_kde_dist,
            lambda_=config.get("lambda_", 1.0),
            exponent=config.get("exponent", 1.0),
            mean_center_on="train",
        )

        # Run PhySO for each trial
        for trial_idx in range(config["n_trials"]):
            # PhySO expects X with shape (n_dim, n_samples)
            X = data.T
            y = np.zeros(data.shape[0])

            # Generate X_names from config or auto-generate
            X_names = config.get("X_names", [f"x{i}" for i in range(data.shape[1])])

            # Run PhySO symbolic regression
            expression = physo.SR(
                X=X,
                y=y,
                X_names=X_names,
                op_names=["add", "sub", "mul", "div", "sqrt", "neg", "n2", "inv"],
                run_config=adapter.get_learning_config(),
                stop_after_n_epochs=config.get("epochs", 100),
                parallel_mode=False,
            )

            # Compute metrics
            trial_metrics = self._compute_metrics(
                expression, sampler, gaussian_kde_dist
            )
            self.metrics[f"trial_{trial_idx}"] = trial_metrics

    def _compute_metrics(
        self: Self, expression, sampler: ImportanceSampler, ref_dist: GaussianKDE
    ) -> dict:
        """Compute KL divergence for discovered expression."""
        # Evaluate expression on sampler samples
        samples = sampler.samples
        # PhySO expressions are callable
        f_vals = expression(samples)

        # Compute KL divergence
        kl_divergence = induced_kl_divergence(
            f_vals,
            sampler,
            ref_dist,
            mean_center_on="train",
        )

        return {
            "expression": expression,
            "infix_pretty": str(expression),
            "kl_divergence": kl_divergence,
        }

    def evaluate(self: Self, model: Any, data: np.ndarray, config: dict) -> dict:
        """Aggregate metrics across trials."""
        total_kl_div = 0.0
        for trial_idx in range(config["n_trials"]):
            trial_metrics = self.metrics.get(f"trial_{trial_idx}", {})
            total_kl_div += trial_metrics.get("kl_divergence", 0.0)

        total_kl_div /= config["n_trials"]

        self.metrics["total"] = {
            "num_trials": config["n_trials"],
            "avg_kl_divergence": total_kl_div,
        }
        return self.metrics


def get_experiment(config: dict) -> Experiment:
    """Factory function to create PhySO experiment."""
    return PhySOExperiment(**config)
