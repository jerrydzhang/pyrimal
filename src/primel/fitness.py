from typing import Tuple, List

import numpy as np

from .samplers import ImportanceSampler
from .distributions import Distribution


def gecco_like_weights(y: np.ndarray) -> np.ndarray:
    """Compute GECCO-style weights that upweight low-density (global) samples.

    This matches the Julia implementation:
    ```julia
    function gecco_like_weights(y::AbstractVector{T}) where {T<:Real}
        thresh = (maximum(y)+minimum(y))/2
        n_glob = sum(y.<thresh)
        n = size(y)[1]
        w_glob = (n-n_glob)/n_glob
        weights = ones(size(y)[1])
        weights[y.<thresh].=w_glob
        return(weights)
    end
    ```

    Samples below the threshold (low density, typically uniform/global samples)
    get upweighted by w_glob = (n-n_glob)/n_glob.

    Args:
        y: Reference distribution density values at sample points

    Returns:
        weights: Array of weights, with low-density samples upweighted
    """
    thresh = (np.max(y) + np.min(y)) / 2
    n = len(y)
    n_glob = np.sum(y < thresh)

    if n_glob == 0:
        return np.ones(n)

    w_glob = (n - n_glob) / n_glob
    weights = np.ones(n)
    weights[y < thresh] = w_glob

    return weights


def induced_kl_divergence(
    f_vals: np.ndarray,
    sampler: ImportanceSampler,
    reference_dist: Distribution,
    lambda_: float = 1.0,
    exponent: float = 1.0,
    mean_center_on: str | List[str] | None = None,
) -> float:
    f_vals = f_vals.astype(np.float64)
    epsilon = 1e-15

    if mean_center_on is not None:
        if isinstance(mean_center_on, str):
            mean_center_on = [mean_center_on]

        center_mask = np.zeros(len(f_vals), dtype=bool)
        for name in mean_center_on:
            try:
                idx = sampler.name_map[name]
                start, end = sampler.range_map[idx]
                center_mask[start:end] = True
            except KeyError:
                raise ValueError(
                    f"Component ''{name}'' provided in `mean_center_on` was not found in the sampler."
                )

        if np.any(center_mask):
            centering_mean = np.mean(f_vals[center_mask])
            f_vals = f_vals - centering_mean

    # Compute the pdfs
    candidate_dist_unnorm = np.exp(-lambda_ * np.abs(f_vals) ** exponent)
    reference_dist_unnorm = reference_dist.pdf(sampler.samples)

    # Normalize the pdfs
    candidate_dist_norm = candidate_dist_unnorm / (
        np.sum(candidate_dist_unnorm) + epsilon
    )
    reference_dist_norm = reference_dist_unnorm / (
        np.sum(reference_dist_unnorm) + epsilon
    )

    unweighted_fitness = (
        reference_dist_norm * np.log((reference_dist_norm + epsilon))
    ) - (reference_dist_norm * np.log((candidate_dist_norm + epsilon)))

    gecco_weights = gecco_like_weights(reference_dist_unnorm)

    # Match Julia implementation: Use ONLY GECCO weights
    # Julia code (dispatch.jl line 59-61):
    #   gw = gecco_like_weights(y)
    #   loss_func = loss_factory(SKL, X_train=Xm, loss_weights=gw)
    # The importance sampling weights 'w' are generated but NOT used in loss.
    #
    # Alternative (matches paper description "w(z) = μ(z)q̃_g(z)"):
    #   combined_weights = sampler.weights * gecco_weights
    # This combines balance heuristic (Veach) with GECCO weights, which is
    # theoretically superior for variance reduction but not what Julia does.
    combined_weights = gecco_weights

    # Compute the induced KL divergence
    fitness_per_sample = combined_weights * unweighted_fitness
    kl_divergence = np.sum(fitness_per_sample).item()

    # Exponentiate the loss to match Julia implementation: loss = exp(loss)
    return np.exp(kl_divergence)


def train_val_variance_split(
    f_vals: np.ndarray,
    sampler: ImportanceSampler,
    train_component_names: str | List[str],
    mean_center: bool = True,
) -> Tuple[float, float]:
    if isinstance(train_component_names, str):
        train_component_names = [train_component_names]

    train_mask = np.zeros(len(f_vals), dtype=bool)
    for name in train_component_names:
        try:
            train_idx = sampler.name_map[name]
            start, end = sampler.range_map[train_idx]
            train_mask[start:end] = True
        except KeyError:
            raise ValueError(
                f"Component ''{name}'' given for `train_component_names` was not found in the sampler."
            )

    if not np.any(train_mask):
        raise ValueError(
            "No training samples found for the given `train_component_names`."
        )

    if mean_center:
        train_mean = np.mean(f_vals[train_mask])
        f_vals = f_vals - train_mean

    var_train = np.var(f_vals[train_mask]).item()
    var_not_train = np.var(f_vals[~train_mask]).item()

    return var_not_train, var_train
