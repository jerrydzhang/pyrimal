from dataclasses import dataclass

import numpy as np
import torch
import physo.physym.reward as reward_module

from primel.fitness import induced_kl_divergence
from primel.samplers import ImportanceSampler
from primel.distributions import Distribution


@dataclass
class PhySOAdapter:
    """PhySO adapter wrapping induced_kl_divergence as reward signal.

    Uses ALL samples from ImportanceSampler (training + local + global regions).
    GECCO weights are computed internally by induced_kl_divergence based on
    reference distribution PDF values. Sampler.weights are NOT passed to PhySO
    to avoid double-weighting.
    """

    sampler: ImportanceSampler
    reference_distribution: Distribution
    lambda_: float = 1.0
    exponent: float = 1.0
    mean_center_on: str | list[str] | None = None

    def get_reward_config(self) -> dict:
        def kl_reward(y_target, y_pred, y_weights=1.0):
            # Validate prediction count matches sampler sample count
            if len(y_pred) != len(self.sampler.samples):
                raise ValueError(
                    (
                        f"Number of predictions ({len(y_pred)}) does not "
                        f"match number of samples ({len(self.sampler.samples)})."
                    )
                )

            # Convert torch tensor to numpy
            y_pred_np = (
                y_pred.detach().cpu().numpy()
                if torch.is_tensor(y_pred)
                else np.asarray(y_pred)
            )

            # Compute KL divergence using induced_kl_divergence
            kl_value = induced_kl_divergence(
                f_vals=y_pred_np,
                sampler=self.sampler,
                reference_dist=self.reference_distribution,
                lambda_=self.lambda_,
                exponent=self.exponent,
                mean_center_on=self.mean_center_on,
            )

            # Transform to reward: squash to [0, 1] (higher = better)
            reward = 1.0 / (1.0 + kl_value)
            return reward

        return {
            "reward_function": kl_reward,
            "zero_out_unphysical": True,
            "zero_out_duplicates": False,
        }

    def get_learning_config(self) -> dict:
        """Return run_config for PhySO.SR.

        Returns a dict with reward_config, learning_config, priors_config,
        cell_config, and free_const_opti_args keys.
        """
        import torch

        return {
            "reward_config": self.get_reward_config(),
            "learning_config": {
                "rewards_computer": reward_module.make_RewardsComputer(
                    **self.get_reward_config()
                ),
                "batch_size": 1000,
                "max_time_step": 35,
                "n_epochs": 1000000000,
                "gamma_decay": 0.7,
                "entropy_weight": 0.005,
                "risk_factor": 0.05,
                "get_optimizer": lambda model: torch.optim.Adam(
                    model.parameters(), lr=0.0025
                ),
                "observe_units": False,
            },
            "priors_config": [
                ("HardLengthPrior", {"max_length": 35, "min_length": 4}),
            ],
            "cell_config": {
                "hidden_size": 128,
                "is_lobotomized": False,
                "n_layers": 1,
            },
            "free_const_opti_args": {
                "loss": "MSE",
                "method": "LBFGS",
                "method_args": {
                    "lbfgs_func_args": {
                        "line_search_fn": "strong_wolfe",
                        "max_iter": 4,
                    },
                    "n_steps": 15,
                    "tol": 1e-08,
                },
            },
        }
