# Coding Conventions

**Analysis Date:** 2026-03-01

## Naming Patterns

**Files:**
- `snake_case.py` for modules (e.g., `samplers.py`, `distributions.py`, `fitness.py`)
- `__init__.py` for package initialization

**Functions:**
- `snake_case` for function names (e.g., `sample()`, `induced_kl_divergence()`, `train_val_variance_split()`)
- Private methods prefixed with underscore (e.g., `_subtree_size()`, `_compute_balance_heuristic_weights()`)
- Factory methods use `from_` prefix (e.g., `from_weights()` in `GaussianKDE`)

**Variables:**
- `snake_case` for local variables and parameters (e.g., `random_state`, `query_points`, `sampler_entries`)
- Constants use UPPER_CASE when they're module-level (e.g., `epsilon = 1e-15` for numerical stability)
- Type variables use `T_` prefix (e.g., `T_Distribution`)

**Types:**
- `PascalCase` for classes (e.g., `Distribution`, `ImportanceSampler`, `ExpressionTree`)
- `Protocol` suffix for protocol definitions (e.g., `Sampler` protocol is defined as a Protocol)
- Protocol classes use `Protocol[T]` generic syntax

**Constants:**
- No explicit constant module
- Magic numbers are often given descriptive names (e.g., `epsilon = 1e-15` for numerical stability)

## Code Style

**Formatting:**
- Tool: ruff (version >=0.13.0 in dev dependencies)
- No explicit ruff.toml or .ruff.toml - using ruff defaults
- 4-space indentation
- Maximum line length: Not specified (using ruff default of 88)
- Trailing commas: Used in multi-line lists and function arguments

**Linting:**
- Tool: ruff (version >=0.13.0)
- Configuration: Using ruff defaults (no explicit config file found)

## Import Organization

**Order:**
1. Standard library imports (e.g., `import argparse`, `import json`)
2. Third-party imports (e.g., `import numpy as np`, `from scipy.stats import gaussian_kde`)
3. Local imports (e.g., `from .distributions import Distribution`, `from .samplers import ImportanceSampler`)

**Style:**
- One import per line preferred
- Use `import X` for module imports
- Use `from X import Y` for specific imports
- Use `as` aliasing: `import numpy as np`
- Group imports by blank lines between standard/third-party/local

**Path Aliases:**
- Relative imports using `from .module import Class` for same-package imports
- Absolute imports using package name for cross-package imports: `from primel.distributions import Distribution`
- No custom import aliases configured

**__all__ exports:**
- Explicit `__all__` lists used in modules to control public API (e.g., `src/primel/samplers.py`, `src/primel/distributions.py`)

## Error Handling

**Patterns:**
- Raise `ValueError` for invalid arguments with descriptive messages (e.g., "Sampler with name 'X' not found.")
- Raise `NotImplementedError` for abstract methods that must be overridden (e.g., `Distribution.pdf()` in `Empirical`)
- Use warnings for non-critical issues: `warnings.warn()` with category `UserWarning`
- Check conditions early and fail fast with clear error messages

**Validation functions:**
- Private helper functions for input validation (e.g., `_validate_query_points()` in `distributions.py`)
- Validate array shapes and dimensions before processing

**Numerical stability:**
- Use small epsilon values for division stability (e.g., `epsilon = 1e-15` in `samplers.py` and `fitness.py`)
- Use `with np.errstate(invalid='ignore', divide='ignore')` context manager for numpy operations (e.g., in `tree.py` line 147)

## Logging

**Framework:** No logging framework used - uses `print()` statements in `run.py` for output
- Example: `print(f"Starting experiment {args.name} with parameters: {parameters}")`

**Patterns:**
- Print statements only in CLI entry points (`run.py`)
- No logging in core library modules (`samplers.py`, `distributions.py`, etc.)
- Debug statements commented out rather than removed (e.g., line 42 in `early_stopping.py`: `# print(f"Early stopping check: {varience}, stop: {result}")`)

## Comments

**When to Comment:**
- Explaining non-obvious algorithms (e.g., balance heuristic formula in `samplers.py` lines 169-178)
- Documenting workarounds or patches (e.g., `patch_kde.py` lines 5-6 explaining picklable replacement)
- Cross-referencing Julia implementation for verification (e.g., `fitness.py` lines 96-106)
- Explaining numerical stability choices (e.g., epsilon values)

**JSDoc/TSDoc:**
- Use Google-style docstrings for classes and functions
- Docstrings include:
  - Brief description
  - `Args:` section listing parameters with types
  - `Returns:` section describing return value and type
  - `Raises:` section for exceptions (when applicable)
  - Example docstrings in `Empirical` class (lines 46-83) and `induced_kl_divergence()` (lines 9-45)

**Comment patterns:**
- Multi-line docstrings with triple quotes
- Inline comments for complex calculations (e.g., `fitness.py` lines 96-106)
- Commented-out debug code preserved rather than removed
- Reference comments linking to external implementations (e.g., Julia code)

## Function Design

**Size:** No strict size limit, but most functions are under 50 lines. Exceptions:
- `_simplify_at_index()` in `tree.py` (~80 lines)
- `_compute_balance_heuristic_weights()` in `samplers.py` (~65 lines)
- Large functions are typically refactored with nested helper functions

**Parameters:**
- Positional parameters first, keyword parameters last
- Optional parameters with default values at the end
- `random_state: int | None = None` pattern for reproducibility throughout
- Type annotations on all parameters
- Use `Self` return type annotation for instance methods that return `self` (Python 3.11+)

**Return Values:**
- Always type-annotated (e.g., `-> float`, `-> np.ndarray`, `-> None`)
- Use `.item()` to extract scalar from numpy arrays (e.g., in `fitness.py` lines 110, 145, 146)
- Tuple returns for multiple values (e.g., `-> Tuple[float, float]` in `train_val_variance_split()`)

## Module Design

**Exports:**
- Explicit `__all__` lists in modules (`src/primel/samplers.py`, `src/primel/distributions.py`)
- Lazy loading in `__init__.py` using `__getattr__()` for optional dependencies (`src/primel/adapters/__init__.py`)
- No wildcard imports (`from module import *`)

**Barrel Files:**
- `__init__.py` files in directories for package structure
- Lazy imports for optional adapters to avoid import errors
- `src/primel/__init__.py` is currently empty

**Type system:**
- Extensive use of `Protocol` for interface definitions (e.g., `Distribution`, `Sampler`)
- Use of `dataclass` decorator for data-heavy classes
- Use of `InitVar` for computed fields in dataclasses
- Union types with `|` syntax (Python 3.10+) for optional types (e.g., `int | None`)

**Generics:**
- TypeVar for generic protocols (e.g., `T_Distribution = TypeVar("T_Distribution", bound=Distribution)`)
- Bounded TypeVars for constraints

---

*Convention analysis: 2026-03-01*
