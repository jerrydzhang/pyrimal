# Stack Research

**Domain:** PhySO (Physics Symbolic Optimization) custom fitness functions
**Researched:** 2026-03-02
**Confidence:** HIGH

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| PhySO | 1.2.0 | RL-based symbolic regression | Only library with dimensional analysis + deep RL; supports custom rewards via `reward_function` config |
| PyTorch | >= 1.11.0 | Auto-differentiation for reward computation | Required by PhySO; custom rewards must use torch tensors |
| NumPy | >= 1.24 | Data handling | PhySO dependency; already in project |
| SymPy | >= 1.12 | Symbolic expression manipulation | PhySO uses for expression export; already in project |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scikit-learn | >= 1.0 | Benchmark analysis | Feynman benchmark scripts; already in project |
| pandas | >= 2.0 | Data loading | CSV handling in PhySO; already in project |
| matplotlib | >= 3.7 | Visualization | Training curves via `RunVisualiser`; already in project |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| pytest | Unit testing | Already in project |
| Hypothesis | Property-based testing | For verifying level set preservation |

## Installation

```bash
# PhySO (add to pyproject.toml as optional dependency)
uv add physo  # or: pip install physo

# Via conda (alternative)
conda install -c conda-forge physo
```

## Custom Reward Function Integration

### Mechanism

PhySO exposes custom rewards via the `reward_config` in run configuration:

```python
import physo
import physo.physym.reward as reward_module

def custom_kl_divergence_reward(y_target, y_pred, y_weights=1.):
    """Custom reward using induced KL divergence."""
    # Must return float in [0, 1] range (higher = better)
    # Must use torch tensors for auto-diff
    kl_div = compute_induced_kl_divergence(y_target, y_pred, y_weights)
    reward = 1.0 / (1.0 + kl_div)  # Squash to [0, 1]
    return reward

reward_config = {
    "reward_function": custom_kl_divergence_reward,
    "zero_out_unphysical": True,
    "zero_out_duplicates": False,
}

learning_config = {
    "rewards_computer": physo.physym.reward.make_RewardsComputer(**reward_config),
    # ... other config
}

run_config = {"learning_config": learning_config, ...}
```

### Reward Function Contract

| Requirement | Details |
|-------------|---------|
| Signature | `fn(y_target, y_pred, y_weights=1.) -> float` |
| Input types | `torch.tensor` (not numpy) |
| Output range | `[0, 1]` where 1 = perfect fit |
| Differentiability | Must use torch ops for gradient flow |

### Alternative: candidate_wrapper

For post-processing expression output before reward:

```python
def candidate_wrapper(func, X):
    """Apply transformation to f(X) before reward computation."""
    y_pred = func(X)
    # Transform output, e.g., normalize, apply threshold
    return transformed_y_pred

expression, logs = physo.SR(X, y, candidate_wrapper=candidate_wrapper, ...)
```

### Alternative: y_weights

For per-sample weighting in reward:

```python
y_weights = compute_importance_weights(X, y)  # Shape: (n_samples,)
expression, logs = physo.SR(X, y, y_weights=y_weights, ...)
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| PhySO | gplearn | When genetic programming preferred over RL; already implemented |
| PhySO | PySR | When Julia runtime acceptable; multi-objective optimization |
| PhySO | DSO (deep-symbolic-optimization) | When physics units not needed; pure RL approach |
| Custom reward | candidate_wrapper | When transformation needed on output, not on reward metric |
| Custom reward | y_weights | When only per-sample importance differs, same metric |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Modifying PhySO source | Breaks updates, hard to maintain | `reward_function` config parameter |
| Numpy in reward function | Breaks auto-differentiation | PyTorch tensors throughout |
| Reward outside [0, 1] | Breaks RL training stability | Squash with `1/(1+x)` or sigmoid |
| CUDA for PhySO | Bottleneck is constant optimization, not GPU compute | CPU mode (default) |

## Stack Patterns by Variant

**If using dimensional analysis (physics problems):**
- Enable `PhysicalUnitsPrior` in `priors_config`
- Pass `X_units`, `y_units` to `physo.SR()`
- Use `config1` preset as starting point

**If dimensionless (pure math):**
- Omit `*_units` parameters
- Use `config2` preset
- Disable `PhysicalUnitsPrior`

**If multi-dataset (Class SR):**
- Use `physo.ClassSR()` instead of `physo.SR()`
- Pass `multi_X`, `multi_y` as lists of datasets
- Free constants can be dataset-specific

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| PhySO 1.2.0 | Python 3.8-3.12 | July 2025 update added 3.12 support |
| PhySO 1.2.0 | PyTorch >= 1.11 | Required for auto-diff |
| PhySO 1.2.0 | NumPy 2.x | Supported as of July 2025 |
| PhySO + gplearn | Same process | Both optional; can coexist |

## Key Implementation Notes

### Adapter Pattern (from PROJECT.md context)

```python
class PhySOAdapter:
    """Wraps induced_kl_divergence as PhySO reward signal."""
    
    def __init__(self, fitness_fn):
        self.fitness_fn = fitness_fn
    
    def __call__(self, y_target, y_pred, y_weights=1.):
        # Convert fitness (lower = better) to reward (higher = better)
        fitness = self.fitness_fn(y_target, y_pred, y_weights)
        reward = 1.0 / (1.0 + fitness)
        return reward
```

### Config Selection

| Config | Batch Size | Use Case |
|--------|------------|----------|
| config0 | 1,000 | Demos, quick tests |
| config1 | 10,000 | SR with dimensional analysis |
| config2 | 5,000 | Dimensionless SR |

## Sources

- **PhySO GitHub**: https://github.com/WassimTenachi/PhySO — Official repository
- **PhySO Docs**: https://physo.readthedocs.io — Reward function API, config reference
- **PhySO reward.py source**: `physo/physym/reward.py` — `SquashedNRMSE`, `make_RewardsComputer`
- **PhySO config0.py source**: `physo/config/config0.py` — Example reward_config structure
- **PhySO Papers**: arXiv:2303.03192 (SR), arXiv:2312.01816 (Class SR) — Methodology

---
*Stack research for: PhySO custom fitness integration*
*Researched: 2026-03-02*
