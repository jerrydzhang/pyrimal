# Testing Patterns

**Analysis Date:** 2026-03-01

## Test Framework

**Runner:**
- pytest 8.4.2+
- Config: Using pytest defaults (no pytest.ini or pyproject.toml [tool.pytest] section found)
- No explicit test configuration discovered

**Assertion Library:**
- pytest's built-in assertions
- `np.testing.assert_array_equal()` for numpy array comparisons
- `np.allclose()` for floating-point comparisons

**Run Commands:**
```bash
pytest              # Run all tests
pytest tests/       # Run tests in tests directory
pytest -v           # Verbose output
pytest tests/test_tree.py  # Run specific test file
pytest -k "test_simplify"  # Run tests matching pattern
```

**Test execution:**
- Tests located in `tests/` directory
- No coverage tool configured in pyproject.toml
- No test runner configuration files found

## Test File Organization

**Location:**
- All tests in separate `tests/` directory at project root
- Test files mirror source structure (e.g., `tests/test_tree.py` tests `src/primel/tree.py`)
- Subdirectory structure mirrors package: `tests/adapters/test_gplearn.py` tests `src/primel/adapters/gplearn/`

**Naming:**
- Test files: `test_*.py` pattern (e.g., `test_tree.py`, `test_samplers.py`, `test_distributions.py`)
- Test classes: `TestClassName` (e.g., `TestExpressionTree`, `TestSimplifyTree`)
- Test functions: `test_function_name()` (e.g., `test_tree_creation_and_str()`, `test_evaluate()`)
- Fixture functions: Descriptive names without `test_` prefix (e.g., `pos_data`, `empirical_dist`, `sample_data`)

**Structure:**
```
tests/
├── test_tree.py
├── test_samplers.py
├── test_fitness.py
├── test_distributions.py
└── adapters/
    └── test_gplearn.py
```

## Test Structure

**Suite Organization:**
```python
class TestClassName:
    """Tests for the ClassName class methods."""

    def test_method_one(self):
        """Test method one behavior."""
        pass

    def test_method_two(self, fixture1, fixture2):
        """Test method two with fixtures."""
        pass
```

Example from `tests/test_tree.py`:
```python
class TestExpressionTree:
    """Tests for the ExpressionTree class methods."""

    def test_tree_creation_and_str(self):
        """Test basic tree creation and its string representation."""
        tree = ExpressionTree("add", lambda a, b: a + b, 2)
        tree.add_node("x0", lambda x: x, 0)
        tree.add_node("const", 2.0, 0)

        assert str(tree).startswith("add")
        assert "x0" in str(tree)
        assert "const" in str(tree)
        assert len(tree.nodes) == 3
```

**Patterns:**
- **Setup pattern:** Use `@pytest.fixture` decorators for common test data
- **Teardown pattern:** No explicit teardown - pytest handles fixture cleanup automatically
- **Assertion pattern:**
  - Simple assertions: `assert condition`
  - Array comparisons: `np.testing.assert_array_equal(result, expected)`
  - Floating-point: `np.allclose(actual, expected)`
  - Exception testing: `with pytest.raises(ExceptionType):`
  - Type checking: `assert isinstance(result, type)`

**Example with fixtures and exception testing:**
```python
@pytest.fixture
def sample_data():
    return np.array([[1, 2], [3, 4], [5, 6]])

def test_importance_sampler(sample_data):
    sampler = ImportanceSampler(sampler_entries=entries)

    with pytest.raises(ValueError):
        sampler.get_samples("nonexistent")
```

## Mocking

**Framework:** No explicit mocking framework used - tests use real instances

**Patterns:**
- No mocking of classes or functions
- Tests instantiate real objects with controlled data (e.g., `Empirical(data=sample_data)`)
- Use `random_state` parameter for deterministic random behavior (e.g., `sampler.sample(10, random_state=42)`)

**What to Mock:**
- Not applicable - mocking not used in current test suite

**What NOT to Mock:**
- Distribution classes (test real implementations)
- Sampler classes (test real behavior)
- Math functions (use numpy functions directly)

## Fixtures and Factories

**Test Data:**
```python
@pytest.fixture
def sample_data():
    return np.array([[1, 2], [3, 4], [5, 6]])

@pytest.fixture
def empirical_dist(sample_data):
    return Empirical(data=sample_data)

@pytest.fixture
def sampler():
    dist = MultivariateUniform(X=np.random.rand(10, 2), margins=0.1)
    sampler_entries = [("all", RandomSampler(dist), 10)]
    return ImportanceSampler(sampler_entries=sampler_entries)
```

**Location:**
- Fixtures defined at module level in test files
- Each test file defines its own fixtures
- No shared fixtures across test files (no conftest.py found)

**Fixture patterns:**
- Simple data fixtures: Return static arrays or values
- Dependent fixtures: Use other fixtures as parameters (e.g., `empirical_dist(sample_data)`)
- Complex fixtures: Build composed objects for testing (e.g., `sampler` fixture)

**Example of complex fixture with random data:**
```python
@pytest.fixture
def mixture_dist(sample_data):
    dist1 = MultivariateUniform(X=sample_data, margins=0.1)
    dist2 = MultivariateUniform(X=sample_data, margins=0.2)
    return MixtureModel(components=[dist1, dist2], weights=np.array([0.4, 0.6]))
```

## Coverage

**Requirements:** None enforced - no coverage tool configured in pyproject.toml

**View Coverage:**
```bash
# No coverage configuration found
# Would need to add pytest-cov or coverage.py to dev dependencies
```

**Coverage files in .gitignore:**
- `.coverage`, `.coverage.*`
- `htmlcov/`
- `cover/`
- These are ignored but no coverage tool is currently configured

## Test Types

**Unit Tests:**
- Scope: Individual class methods and functions
- Approach: Test isolated behavior with controlled inputs
- Examples:
  - `test_tree_creation_and_str()` in `test_tree.py`
  - `test_empirical_distribution()` in `test_distributions.py`
  - `test_random_sampler()` in `test_samplers.py`

**Integration Tests:**
- Scope: Interaction between multiple classes
- Approach: Test combined behavior of samplers, distributions, and fitness functions
- Examples:
  - `test_importance_sampler_balance_heuristic()` in `test_samplers.py` - tests sampler + distribution interaction
  - `test_induced_kl_divergence()` in `test_fitness.py` - tests fitness function with sampler and reference distribution

**E2E Tests:**
- Framework: Not used - no end-to-end or system-level tests
- No tests that run full experiments or workflows

**Optional dependency testing:**
- Use `pytest.importorskip()` for tests requiring optional dependencies
- Example from `tests/adapters/test_gplearn.py`:
  ```python
  # Skip all tests in this module if gplearn is not installed
  gplearn = pytest.importorskip("gplearn")
  ```

## Common Patterns

**Async Testing:**
- Not applicable - no async/await code in codebase

**Error Testing:**
```python
# Pattern from test_fitness.py and test_samplers.py
def test_function_with_error():
    with pytest.raises(ValueError):
        function_with_invalid_args()

# Testing NotImplementedError
def test_empirical_pdf_raises(sample_data):
    dist = Empirical(data=sample_data)
    with pytest.raises(NotImplementedError):
        dist.pdf(sample_data)
```

**Deterministic Random Testing:**
```python
# Use random_state parameter for reproducibility
def test_sampler_with_random_state():
    samples = sampler.sample(10, random_state=42)
    # Will always produce same results
```

**Numpy Array Testing:**
```python
# Pattern from test_tree.py
result = tree.evaluate(pos_data)
expected = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
np.testing.assert_array_equal(result, expected)

# Floating-point comparison
assert np.allclose(actual, expected)
```

**Parameterized Testing:**
- Not extensively used - most tests use fixtures instead
- Some tests test multiple conditions in single test method (e.g., `test_simplify_nested()` in `test_tree.py` tests multiple tree structures)

**Testing Mathematical Properties:**
```python
# Pattern from test_tree.py - verify mathematical invariants
def test_simplify_div_x_x(self, pos_data):
    """Test rule: div(x, x) -> 1"""
    tree = ExpressionTree("div", np.divide, 2)
    tree.add_node("x", lambda x: x, 0)
    tree.add_node("x", lambda x: x, 0)

    simplify_tree(tree, pos_data)

    assert len(tree.nodes) == 1
    assert tree.nodes[0].name == "constant"
    assert tree.nodes[0].value == 1
```

**Edge Case Testing:**
```python
# Test empty array handling from test_fitness.py
dist = Empirical(data=np.array([]))
empty_sampler_entries = [("train", RandomSampler(dist), 0)]
sampler_no_train = ImportanceSampler(sampler_entries=empty_sampler_entries)
with pytest.raises(ValueError):
    train_val_variance_split(np.array([]), sampler_no_train, train_component_names="train")
```

---

*Testing analysis: 2026-03-01*
