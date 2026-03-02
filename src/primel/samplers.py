import dataclasses
from dataclasses import InitVar, dataclass, field
from typing import Dict, List, Protocol, Self, Sequence, Tuple, TypeVar

import numpy as np
from scipy.stats import qmc

from .distributions import Distribution, MixtureModel, MultivariateUniform

__all__ = [
    "Sampler",
    "RandomSampler",
    "StratifiedSampler",
    "LHSampler",
    "ImportanceSampler",
]

T_Distribution = TypeVar("T_Distribution", bound=Distribution)


@dataclass
class Sampler(Protocol[T_Distribution]):
    dist: T_Distribution

    def sample(
        self: Self,
        n_samples: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        raise NotImplementedError


@dataclass
class RandomSampler:
    dist: Distribution

    def sample(
        self: Self,
        n_samples: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        return self.dist.sample(n_samples, random_state)


@dataclass
class StratifiedSampler:
    dist: MixtureModel

    def sample(
        self: Self,
        n_samples: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(random_state)

        samples_per_component = rng.multinomial(
            n_samples,
            self.dist._weights,
        )
        component_random_states = rng.integers(
            0,
            2**31,
            size=len(self.dist.components),
        )

        samples_list: List[np.ndarray] = []
        for count, component, component_rs in zip(
            samples_per_component,
            self.dist.components,
            component_random_states,
        ):
            if count == 0:
                continue

            samples_list.append(
                component.sample(
                    count,
                    random_state=component_rs,
                )
            )

        return np.vstack(samples_list)


@dataclass
class LHSampler:
    dist: MultivariateUniform

    def sample(
        self: Self,
        n_samples: int,
        random_state: int | None = None,
    ) -> np.ndarray:
        sampler = qmc.LatinHypercube(d=self.dist.shape[1], rng=random_state)
        samples = sampler.random(n=n_samples)
        scaled_samples = qmc.scale(samples, self.dist.lower, self.dist.upper)
        return scaled_samples


SamplerEntry = Tuple[str, Sampler, int]
WeightedSamplerEntry = Tuple[str, Sampler, int, float]


@dataclass
class ImportanceSampler:
    sampler_entries: InitVar[Sequence[SamplerEntry | WeightedSamplerEntry]]

    name_map: Dict[str, int] = field(init=False)
    range_map: List[Tuple[int, int]] = field(init=False)
    n_samples_per_dist: List[int] = field(init=False)

    samples: np.ndarray = field(init=False)
    weights: np.ndarray = field(init=False)
    samplers: List[Sampler] = field(init=False)

    random_state: int | None = None

    def __post_init__(
        self: Self,
        sampler_entries: Sequence[SamplerEntry | WeightedSamplerEntry],
    ):
        self.samplers = []
        self.name_map: Dict[str, int] = {}
        self.range_map: List[Tuple[int, int]] = []
        self.n_samples_per_dist: List[int] = []

        samples_list: List[np.ndarray] = []

        pointer = 0
        for i, entry in enumerate(sampler_entries):
            if len(entry) == 4:
                name, sampler, n_samples, _ = entry
                # Ignore user-provided weights - we compute them with balance heuristic
            elif len(entry) == 3:
                name, sampler, n_samples = entry
            else:
                raise ValueError(
                    "Sampler entry must be of the form "
                    "(name, sampler, n_samples) or "
                    "(name, sampler, n_samples, weight)."
                )

            self.samplers.append(sampler)
            self.n_samples_per_dist.append(n_samples)

            if n_samples > 0:
                samples = sampler.sample(n_samples, random_state=self.random_state)
                samples_list.append(samples)

            self.range_map.append((pointer, pointer + n_samples))
            self.name_map[name] = i
            pointer += n_samples

        if samples_list:
            self.samples = np.vstack(samples_list)
            # Compute balance heuristic weights
            self._compute_balance_heuristic_weights()
        else:
            self.samples = np.array([])
            self.weights = np.array([])
            if sampler_entries:
                try:
                    d = sampler_entries[0][1].dist.shape[1]
                    self.samples = np.empty((0, d))
                except Exception:
                    pass  # keep samples as empty array
    
    def _compute_balance_heuristic_weights(self: Self) -> None:
        """
        Compute weights using Veach's balance heuristic (SIGGRAPH '95, Eq. 12):
        w_i(x) = (n_i * p_i(x)) / Σ_j(n_j * p_j(x))
        
        This ensures optimal variance reduction when combining multiple 
        importance sampling distributions.
        
        Special handling for Empirical distributions: Since they don't have
        a well-defined PDF, we skip the balance heuristic and use uniform
        weighting (n_i / N_total) as a fallback.
        """
        from .distributions import Empirical
        
        n_total_samples = len(self.samples)
        
        # Check if any distribution is Empirical (no PDF available)
        has_empirical = any(
            isinstance(sampler.dist, Empirical) 
            for sampler in self.samplers
        )
        
        if has_empirical:
            # Fall back to simple uniform weighting: n_i / N_total
            # This is not optimal but is unbiased
            weights = np.zeros(n_total_samples)
            for i, n_i in enumerate(self.n_samples_per_dist):
                start, end = self.range_map[i]
                if start < end:
                    weights[start:end] = n_i
            self.weights = weights / (np.sum(weights) + 1e-15)
            return
        
        # Compute balance heuristic for distributions with PDFs
        weights = np.zeros(n_total_samples)
        
        # For each sample, compute weighted probability under all distributions
        for i, (sampler, n_i) in enumerate(zip(self.samplers, self.n_samples_per_dist)):
            start, end = self.range_map[i]
            if start >= end:
                continue
            
            # Get samples from this distribution
            samples_i = self.samples[start:end]
            
            # Compute denominator: Σ_j(n_j * p_j(x)) for each sample
            denominator = np.zeros(len(samples_i))
            for j, (sampler_j, n_j) in enumerate(zip(self.samplers, self.n_samples_per_dist)):
                if n_j == 0:
                    continue
                # Evaluate probability density of distribution j at samples from i
                p_j = sampler_j.dist.pdf(samples_i)
                denominator += n_j * p_j
            
            # Compute numerator: n_i * p_i(x)
            p_i = sampler.dist.pdf(samples_i)
            numerator = n_i * p_i
            
            # Balance heuristic weight with numerical stability
            epsilon = 1e-15
            weights[start:end] = numerator / (denominator + epsilon)
        
        # Normalize weights to sum to 1
        self.weights = weights / (np.sum(weights) + 1e-15)

    def get_samples(self: Self, name: str) -> np.ndarray:
        if name not in self.name_map:
            raise ValueError(f"Sampler with name '{name}' not found.")

        index = self.name_map[name]
        start, end = self.range_map[index]
        return self.samples[start:end]

    def get_weights(self: Self, name: str) -> np.ndarray:
        if name not in self.name_map:
            raise ValueError(f"Sampler with name '{name}' not found.")

        index = self.name_map[name]
        start, end = self.range_map[index]
        return self.weights[start:end]

    def reweight_by_dist(self: Self, dist: Distribution) -> None:
        pdf_values = dist.pdf(self.samples)
        mid = (np.max(pdf_values) - np.min(pdf_values)) / 2.0
