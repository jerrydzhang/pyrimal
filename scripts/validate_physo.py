#!/usr/bin/env python
"""Local validation script for PhySO adapter with KL divergence reward."""

import argparse
from pathlib import Path

import numpy as np

# Fail early if physo is not installed
import physo

from primel.adapters.physo import PhySOAdapter
from primel.distributions import Empirical, GaussianKDE, MultivariateUniform
from primel.samplers import ImportanceSampler, LHSampler, RandomSampler
from primel.fitness import induced_kl_divergence

import physo


def main():
    parser = argparse.ArgumentParser(
        description="Validate PhySO adapter with KL divergence reward"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="zhong/f01/f01.csv",
        help="Path to CSV data file (relative to data/ directory)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    # Load data
    data_path = Path(__file__).parent.parent / "data" / args.data
    data = np.loadtxt(data_path, delimiter=",")

    # Setup distributions
    empirical_dist = Empirical(data=data)
    gaussian_kde_dist = GaussianKDE(X=data, bandwidth=0.1)
    uniform_dist = MultivariateUniform(X=data, margins=0.1, non_negative=False)

    # Setup multi-component sampler
    sampler = ImportanceSampler(
        sampler_entries=[
            ("train", RandomSampler(empirical_dist), data.shape[0]),
            ("kde", RandomSampler(gaussian_kde_dist), 100),
            ("uniform", LHSampler(uniform_dist), 100),
        ],
        random_state=args.seed,
    )

    # Print sample breakdown
    print(
        f"Samples: {data.shape[0]} train + 100 kde + 100 uniform = {len(sampler.samples)} total"
    )

    # Create PhySO adapter
    adapter = PhySOAdapter(
        sampler=sampler,
        reference_distribution=gaussian_kde_dist,
        lambda_=1.0,
        exponent=1.0,
        mean_center_on="train",
    )

    # Run PhySO symbolic regression
    X = data.T
    y = np.zeros(data.shape[0])
    X_names = [f"x{i}" for i in range(data.shape[1])]

    expression = physo.SR(
        X=X,
        y=y,
        X_names=X_names,
        op_names=["add", "sub", "mul", "div", "sqrt", "neg", "n2", "inv"],
        run_config=adapter.get_learning_config(),
        stop_after_n_epochs=args.epochs,
        parallel_mode=False,
    )

    # Compute KL divergence
    f_vals = expression(sampler.samples)
    kl_divergence = induced_kl_divergence(
        f_vals,
        sampler,
        gaussian_kde_dist,
        mean_center_on="train",
    )

    # Print results
    print(f"Expression: {expression}")
    print(f"KL score: {kl_divergence:.4f}")


if __name__ == "__main__":
    main()
