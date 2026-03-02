# Codebase Structure

**Analysis Date:** 2026-03-01

## Directory Layout

```
pyrimel/
├── build/                    # Build artifacts (not committed)
├── data/                     # Training datasets
│   └── zhong/               # Zhong benchmark dataset files
├── experiments/             # Experiment configurations and implementations
│   ├── scripts/            # Data generation and parameter search scripts
│   └── *.yaml              # Experiment configuration files
├── examples/                # Example notebooks and scripts
│   └── gplearn/            # GPLearn usage examples
├── results/                 # Experiment outputs (not committed)
├── src/                     # Source code
│   └── primel/             # Main package
│       ├── adapters/       # ML library adapters
│       │   ├── gplearn/   # GPLearn integration
│       │   └── symbolicregression/ # Julia integration (placeholder)
│       ├── __init__.py
│       ├── distributions.py
│       ├── early_stopping.py
│       ├── fitness.py
│       ├── patch_kde.py
│       ├── run.py
│       ├── samplers.py
│       └── tree.py
└── tests/                  # Test suite
    └── adapters/           # Adapter-specific tests
```

## Directory Purposes

**`src/primel/`:**
- Purpose: Main package containing core library code
- Contains: Probability distributions, sampling strategies, expression trees, fitness functions, adapters
- Key files: `distributions.py`, `samplers.py`, `tree.py`, `fitness.py`

**`src/primel/adapters/`:**
- Purpose: Integrations with external symbolic regression libraries
- Contains: Library-specific adapter implementations
- Key files: `gplearn/adapter.py`, `gplearn/model.py`

**`experiments/`:**
- Purpose: Experiment configuration and execution logic
- Contains: Experiment classes, YAML configs, data generation scripts
- Key files: `gplearn_experiment.py`, `*.yaml`

**`tests/`:**
- Purpose: Test suite for all components
- Contains: Unit tests for distributions, samplers, trees, fitness, adapters
- Key files: `test_distributions.py`, `test_samplers.py`, `test_tree.py`, `test_fitness.py`

**`examples/`:**
- Purpose: Demonstration notebooks and example scripts
- Contains: Jupyter notebooks showing library usage
- Key files: `gplearn/gplearn_basic.ipynb`

**`data/`:**
- Purpose: Training datasets for experiments
- Contains: Benchmark datasets (e.g., Zhong benchmarks)
- Key files: `zhong/f*/*.csv`

**`results/`:**
- Purpose: Output directory for experiment results (not tracked in git)
- Contains: Generated model files, metrics, logs
- Committed: No

## Key File Locations

**Entry Points:**
- `src/primel/run.py`: CLI entry point for running experiments
- `experiments/gplearn_experiment.py`: Main experiment implementation

**Configuration:**
- `pyproject.toml`: Project metadata and dependencies
- `flake.nix`: Nix-based development environment
- `experiments/*.yaml`: Experiment configurations

**Core Logic:**
- `src/primel/distributions.py`: Probability distribution implementations
- `src/primel/samplers.py`: Sampling strategies and importance sampling
- `src/primel/tree.py`: Expression tree representation and manipulation
- `src/primel/fitness.py`: KL divergence fitness function

**Testing:**
- `tests/`: Root test directory
- `tests/test_*.py`: Unit tests for each module
- `tests/adapters/`: Adapter-specific tests

## Naming Conventions

**Files:**
- Module files: `lowercase_with_underscores.py` (e.g., `distributions.py`, `early_stopping.py`)
- Test files: `test_<module>.py` (e.g., `test_samplers.py`, `test_fitness.py`)
- Config files: `descriptive_name.yaml` (e.g., `gplearn_wide_search.yaml`)

**Directories:**
- Package directories: `lowercase_with_underscores` (e.g., `src/primel/`, `experiments/`)
- Adapter directories: `libraryname_lowercase` (e.g., `gplearn/`, `symbolicregression/`)

**Classes:**
- Distribution classes: `PascalCase` (e.g., `Empirical`, `GaussianKDE`, `MultivariateUniform`)
- Sampler classes: `PascalCase` with `Sampler` suffix (e.g., `RandomSampler`, `ImportanceSampler`)
- Adapter classes: `PascalCase` with `Adapter` suffix (e.g., `GPLearnAdapter`)
- Experiment classes: `PascalCase` with `Experiment` suffix (e.g., `GpLearnExperiment`)

**Functions:**
- Private functions: `_lowercase_with_leading_underscore` (e.g., `_validate_query_points`)
- Public functions: `lowercase_with_underscores` (e.g., `induced_kl_divergence`, `simplify_tree`)

## Where to Add New Code

**New Distribution:**
- Primary code: Add class to `src/primel/distributions.py`
- Tests: Add tests to `tests/test_distributions.py`
- Export: Add to `__all__` list in `distributions.py`

**New Sampler:**
- Primary code: Add class to `src/primel/samplers.py`
- Tests: Add tests to `tests/test_samplers.py`
- Export: Add to `__all__` list in `samplers.py`

**New Adapter:**
- Implementation: Create new directory in `src/primel/adapters/` (e.g., `newlib/`)
- Adapter class: `adapters/newlib/adapter.py`
- Model class: `adapters/newlib/model.py` (if custom model needed)
- Export: Update `adapters/__init__.py` to expose the adapter
- Tests: Create `tests/adapters/test_newlib.py`

**New Experiment:**
- Implementation: Add to `experiments/` directory (e.g., `newlib_experiment.py`)
- Configuration: Create YAML config file in `experiments/` directory
- Data: Place datasets in `data/` directory if needed

**Utilities:**
- Shared helpers: Add to relevant core module (e.g., tree utilities to `tree.py`)
- If general-purpose: Create new file in `src/primel/` (e.g., `utils.py`)

## Special Directories

**`build/`:**
- Purpose: Build artifacts from setuptools
- Generated: Yes
- Committed: No

**`results/`:**
- Purpose: Experiment output directory
- Generated: Yes
- Committed: No (ignored by .gitignore)

**`.venv/`:**
- Purpose: Python virtual environment
- Generated: Yes
- Committed: No (ignored by .gitignore)

**`__pycache__/`:**
- Purpose: Python bytecode cache
- Generated: Yes
- Committed: No (ignored by .gitignore)

**`.planning/codebase/`:**
- Purpose: Codebase documentation generated by GSD
- Generated: Yes
- Committed: Yes

---

*Structure analysis: 2026-03-01*
