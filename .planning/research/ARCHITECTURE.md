# Architecture Research

**Domain:** PhySO Adapter for Symbolic Regression
**Researched:** 2026-03-02
**Confidence:** MEDIUM

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Experiment Layer                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │ GpLearnExperiment│  │ PhySOExperiment  │                 │
│  └────────┬─────────┘  └────────┬─────────┘                 │
│           │                     │                            │
├───────────┴─────────────────────┴────────────────────────────┤
│                      Adapter Layer                           │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │  GPLearnAdapter  │  │   PhySOAdapter   │                 │
│  │  ┌────────────┐  │  │  ┌────────────┐  │                 │
│  │  │get_fitness │  │  │  │reward_fn   │  │                 │
│  │  └────────────┘  │  │  └────────────┘  │                 │
│  └────────┬─────────┘  └────────┬─────────┘                 │
│           │                     │                            │
├───────────┴─────────────────────┴────────────────────────────┤
│                       Core Layer                             │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │Importance   │  │Distribution │  │induced_kl_divergence│  │
│  │Sampler      │  │(KDE, Empir.)│  │fitness function     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| GPLearnAdapter | Wrap `induced_kl_divergence` for genetic programming fitness | Dataclass returning `make_fitness()` callable |
| PhySOAdapter | Provide reward signal for RL-based expression search | Candidate wrapper + sample weighting |
| ImportanceSampler | Combine multiple sampling strategies with balance heuristic | Stratified sampling from multiple distributions |
| induced_kl_divergence | Core fitness measuring distribution alignment | KL divergence between candidate and reference |

## PhySO Integration Architecture

### Key Finding: PhySO Reward Interface

**Confidence:** MEDIUM (based on official documentation, not source code inspection)

PhySO does **NOT** expose a direct custom reward function interface like GPLearn's `make_fitness()`. Instead, PhySO uses an internal reward function based on fitting accuracy (MSE-like). The extension points are:

1. **`candidate_wrapper`**: A callable that wraps the candidate function's output before reward computation
2. **`y_weights`**: Weights applied to data points during reward computation
3. **Configuration**: Hyperparameters controlling the RL training process

### Recommended PhySO Adapter Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                    PhySOAdapter                              │
├─────────────────────────────────────────────────────────────┤
│  sampler: ImportanceSampler                                  │
│  reference_distribution: Distribution                        │
│  lambda_: float = 1.0                                        │
│  exponent: float = 1.0                                       │
│  mean_center_on: str | None = None                          │
├─────────────────────────────────────────────────────────────┤
│  + get_candidate_wrapper() -> Callable                      │
│  + get_y_weights() -> np.ndarray                            │
│  + compute_reward(f_vals) -> float                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    physo.SR() Call                           │
├─────────────────────────────────────────────────────────────┤
│  expression, logs = physo.SR(                               │
│      X, y,                                                   │
│      y_weights = adapter.get_y_weights(),                   │
│      candidate_wrapper = adapter.get_candidate_wrapper(),   │
│      ...                                                     │
│  )                                                           │
└─────────────────────────────────────────────────────────────┘
```

### Adapter Interface Comparison

| Aspect | GPLearn | PhySO |
|--------|---------|-------|
| Custom fitness entry point | `make_fitness(function, greater_is_better)` | No direct equivalent |
| Fitness function signature | `(y_true, y_pred, sample_weight) -> float` | Internal, not exposed |
| Sample weighting | Via `sample_weight` parameter | Via `y_weights` parameter |
| Output transformation | N/A | Via `candidate_wrapper` |
| Training control | Per-generation callbacks | Early stopping via `stop_reward` |

## Data Flow

### GPLearn Flow (Existing)

```
ImportanceSampler.samples
        ↓
GPLearnAdapter.get_fitness()
        ↓
gplearn.fitness.make_fitness(_induced_kl_divergence)
        ↓
SymbolicRegressor.fit() → Evolution loop
        ↓
Each generation: evaluate fitness via adapter
        ↓
induced_kl_divergence(f_vals, sampler, ref_dist)
        ↓
Selection → Crossover → Mutation
```

### PhySO Flow (Proposed)

```
ImportanceSampler.samples
        ↓
PhySOAdapter.get_y_weights() + get_candidate_wrapper()
        ↓
physo.SR(X, y, y_weights=..., candidate_wrapper=...)
        ↓
RL Training Loop (internal to PhySO)
        ↓
For each candidate expression:
    1. Evaluate f(X) → f_vals
    2. Apply candidate_wrapper(f, X) → wrapped_f_vals
    3. Compute internal reward (MSE-like with y_weights)
        ↓
Policy gradient update
        ↓
Return best expression
```

### Key Data Flow Differences

1. **GPLearn**: Adapter provides the *entire* fitness function
2. **PhySO**: Adapter provides *inputs* (weights, wrapper) to internal reward

## Architectural Patterns

### Pattern 1: Candidate Wrapper for Implicit Constraints

**What:** Use `candidate_wrapper` to transform expression output before PhySO's internal reward computation.

**When to use:** When you need the expression to satisfy implicit constraints (e.g., `f(X) ≈ 0`).

**Trade-offs:**
- (+) Allows influencing reward without modifying PhySO internals
- (-) Indirect control; wrapper output still goes through MSE-like reward
- (-) May interfere with free constant optimization (requires differentiable wrapper)

**Example:**
```python
def get_candidate_wrapper(self):
    def wrapper(func, X):
        f_vals = func(X)
        # Transform to work with induced KL divergence concept
        # Note: This is conceptually different from GPLearn's direct fitness
        return f_vals
    return wrapper
```

### Pattern 2: Sample Weighting for Importance Sampling

**What:** Use `y_weights` to implement importance sampling-based weighting.

**When to use:** When different samples should contribute differently to the reward.

**Trade-offs:**
- (+) Directly supported by PhySO
- (+) Affects both reward computation and free constant optimization
- (-) Cannot change the *form* of the reward function

**Example:**
```python
def get_y_weights(self) -> np.ndarray:
    # Use GECCO-like weights to upweight low-density regions
    from primel.fitness import gecco_like_weights
    ref_pdf = self.reference_distribution.pdf(self.sampler.samples)
    return gecco_like_weights(ref_pdf)
```

### Pattern 3: Hybrid Evaluation (Post-Hoc)

**What:** Run PhySO with default reward, then evaluate candidates using `induced_kl_divergence` post-hoc.

**When to use:** When direct reward customization is not feasible.

**Trade-offs:**
- (+) Clean separation of concerns
- (+) Can use any evaluation metric
- (-) PhySO optimizes for wrong objective during search
- (-) May not find optimal expressions for KL-based fitness

**Example:**
```python
# Run PhySO with default settings
expression, logs = physo.SR(X, y, ...)

# Evaluate with custom fitness
f_vals = expression(torch.tensor(X))
kl_score = induced_kl_divergence(f_vals, sampler, ref_dist)
```

## Anti-Patterns

### Anti-Pattern 1: Assuming Direct Reward Function Access

**What people do:** Try to pass a custom reward function directly to PhySO.

**Why it's wrong:** PhySO doesn't expose this interface. The reward function is internal to the RL training loop.

**Do this instead:** Use `candidate_wrapper` and `y_weights` to influence reward computation indirectly.

### Anti-Pattern 2: Ignoring Free Constant Optimization

**What people do:** Create a `candidate_wrapper` that's not differentiable.

**Why it's wrong:** PhySO uses gradient-based optimization (L-BFGS) for free constants. Non-differentiable wrappers break this.

**Do this instead:** Write wrappers in PyTorch with differentiable operations.

### Anti-Pattern 3: Treating PhySO Like GPLearn

**What people do:** Expect the same adapter pattern to work for both libraries.

**Why it's wrong:** GPLearn uses genetic programming with pluggable fitness; PhySO uses RL with internal reward.

**Do this instead:** Design separate adapter patterns for each library's extension points.

## Build Order Implications

Based on the architecture analysis, recommended build order:

### Phase 1: Core Adapter Interface
1. **Define abstract adapter protocol** - Common interface both adapters implement
2. **PhySOAdapter dataclass** - Holds sampler, reference distribution, parameters

### Phase 2: PhySO Integration
3. **Implement `get_y_weights()`** - Map importance sampling to PhySO weights
4. **Implement `get_candidate_wrapper()`** - PyTorch-differentiable wrapper
5. **Basic PhySO experiment** - Validate adapter produces reasonable results

### Phase 3: Evaluation & Comparison
6. **Post-hoc KL evaluation** - Compare PhySO expressions using `induced_kl_divergence`
7. **Benchmark against GPLearn** - Same data, different optimizers

### Phase 4: Refinement (if needed)
8. **Custom reward exploration** - If PhySO's architecture allows deeper modification
9. **Hyperparameter tuning** - PhySO configs optimized for KL-based objectives

## Component Boundaries

| Boundary | Owner | Interface |
|----------|-------|-----------|
| Core → GPLearn | GPLearnAdapter | `get_fitness() -> Fitness` |
| Core → PhySO | PhySOAdapter | `get_y_weights()`, `get_candidate_wrapper()` |
| Adapter → Experiment | Experiment class | Adapter instance, config params |
| PhySO → Adapter | Adapter | Data (samples, weights), functions (wrapper) |

## Sources

- PhySO Documentation: https://physo.readthedocs.io/en/latest/r_sr.html
- PhySO GitHub: https://github.com/WassimTenachi/PhySO
- PhySO Features: https://physo.readthedocs.io/en/latest/r_features.html
- Existing GPLearn adapter: `src/primel/adapters/gplearn/adapter.py`
- Core fitness function: `src/primel/fitness.py`

---
*Architecture research for: PhySO adapter integration*
*Researched: 2026-03-02*
