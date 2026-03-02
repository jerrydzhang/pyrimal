# Architecture

**Analysis Date:** 2026-03-01

## Pattern Overview

**Overall:** Layered Architecture with Adapter Pattern

**Key Characteristics:**
- Separation of core abstractions from ML library integrations
- Protocol-based design for extensible distributions and samplers
- Adapter pattern for integrating with symbolic regression libraries
- Configuration-driven experiment execution

## Layers

**Core Layer (`src/primel/`):**
- Purpose: Fundamental abstractions for probability, sampling, and expression trees
- Location: `src/primel/`
- Contains: Distribution implementations, sampling strategies, expression tree representation, fitness evaluation, early stopping logic
- Depends on: `numpy`, `scipy`
- Used by: Adapter layer, experiment layer

**Adapter Layer (`src/primel/adapters/`):**
- Purpose: Integrations with external symbolic regression libraries
- Location: `src/primel/adapters/`
- Contains: Library-specific adapters that map core abstractions to ML library APIs
- Depends on: Core layer, `gplearn` (optional)
- Used by: Experiment layer

**Experiment Layer (`experiments/`):**
- Purpose: Experiment configuration, execution, and result management
- Location: `experiments/`
- Contains: Experiment implementations, YAML configuration files, data generation scripts
- Depends on: Core layer, adapter layer, `jernerics.experiment`
- Used by: CLI entry point, researchers running experiments

## Data Flow

**Experiment Execution Flow:**

1. **Data Loading**: Training data loaded from disk (`.npy` or `.csv` files)
2. **Distribution Creation**: Data converted to multiple distributions (Empirical, GaussianKDE, MultivariateUniform)
3. **Sampler Setup**: ImportanceSampler configured with multiple sampling strategies and distributions
4. **Adapter Configuration**: Adapter initialized with sampler, reference distribution, early stopping, and fitness function
5. **Model Training**: Symbolic regression model trained using samples from importance sampler
6. **Fitness Evaluation**: Model predictions evaluated using induced KL divergence fitness function
7. **Early Stopping**: Training stops when variance thresholds met (if configured)
8. **Metrics Computation**: KL divergence, generations, success metrics recorded
9. **Result Saving**: Metrics and models saved to disk

**State Management:**
- Distributions: Stateless, created from data
- Samplers: Stateful, hold samples and weights
- ExpressionTrees: Stateful, hold tree structure and can be evaluated
- EarlyStopping: Stateful, maintains variance check history

## Key Abstractions

**Distribution Protocol:**
- Purpose: Interface for all probability distributions
- Examples: `src/primel/distributions.py` (Empirical, GaussianKDE, MultivariateUniform, MixtureModel)
- Pattern: Protocol defining `pdf()` and `sample()` methods

**Sampler Protocol:**
- Purpose: Interface for all sampling strategies
- Examples: `src/primel/samplers.py` (RandomSampler, StratifiedSampler, LHSampler)
- Pattern: Protocol defining `sample()` method wrapping a Distribution

**ImportanceSampler:**
- Purpose: Combines multiple sampling strategies with importance weighting
- Examples: `src/primel/samplers.py`
- Pattern: Composite pattern with balance heuristic weighting (Veach's method)

**ExpressionTree:**
- Purpose: Represent symbolic mathematical expressions as tree structures
- Examples: `src/primel/tree.py`
- Pattern: Tree data structure with evaluation and simplification capabilities

**Adapter Interface:**
- Purpose: Bridge between core abstractions and ML library fitness functions
- Examples: `src/primel/adapters/gplearn/adapter.py` (GPLearnAdapter)
- Pattern: Adapter pattern providing `get_fitness()` method for library integration

## Entry Points

**CLI Entry Point:**
- Location: `src/primel/run.py`
- Triggers: Command line execution with `--name`, `--train_data_path`, and other parameters
- Responsibilities: Parse arguments, load data, create Experiment instance, execute experiment

**Experiment Entry Point:**
- Location: `experiments/gplearn_experiment.py` (GpLearnExperiment)
- Triggers: Called by `run.py` or directly imported
- Responsibilities: Setup data, configure distributions/samplers/adapter, train model, compute metrics

**Python Package Entry Point:**
- Location: `src/primel/__init__.py`
- Triggers: `import primel`
- Responsibilities: Export public API (currently empty)

## Error Handling

**Strategy:** Use of Python exceptions and warnings

**Patterns:**
- Input validation raises `ValueError` with descriptive messages (e.g., invalid query points in distributions)
- Numerical issues handled with `np.errstate` context managers for graceful degradation
- Warnings issued for potential issues (e.g., zero-volume spaces in MultivariateUniform)
- Early stopping checks return boolean rather than raising exceptions

## Cross-Cutting Concerns

**Logging:** None detected (uses print statements in CLI)
**Validation:** Explicit validation in distribution methods with clear error messages
**Authentication:** Not applicable (scientific computing package)

---

*Architecture analysis: 2026-03-01*
