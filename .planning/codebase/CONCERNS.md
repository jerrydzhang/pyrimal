# Codebase Concerns

**Analysis Date:** 2026-03-01

## Tech Debt

**Large Model File:**
- Issue: `src/primel/adapters/gplearn/model.py` is 665 lines, containing complex genetic programming logic mixed with tree conversion and parallel evolution
- Files: `src/primel/adapters/gplearn/model.py`
- Impact: Difficult to maintain, understand, and modify; increased cognitive load for developers
- Fix approach: Extract tree conversion functions (`_build_tree`, `_build_program`) into a separate module; split `_parallel_evolve` into smaller functions; separate the `ImplicitSymbolicRegressor` class into smaller cohesive classes (evolution logic, validation, tree management)

**Monkey Patch Dependency:**
- Issue: Directly patches `BaseSymbolic._validate_data` from gplearn as workaround for upstream issue #303
- Files: `src/primel/adapters/gplearn/model.py` (lines 30-36)
- Impact: Creates tight coupling to gplearn's internal implementation; upstream changes could break compatibility; workaround may mask actual problem
- Fix approach: Monitor gplearn issue #303; consider forking or contributing fix upstream; implement adapter pattern to abstract away validation differences

## Known Bugs

**Unreachable Return Statement:**
- Symptoms: Duplicate return statement at end of `EarlyStopping.check()` method
- Files: `src/primel/early_stopping.py` (line 46)
- Trigger: Code path never executes; second return is unreachable
- Workaround: None needed - cosmetic bug only
- Fix approach: Remove the duplicate `return result` statement on line 46

## Security Considerations

**Silent Exception Swallowing:**
- Risk: Bare `except Exception:` in `__post_init__` catches and suppresses all errors without logging or context
- Files: `src/primel/samplers.py` (line 165)
- Current mitigation: None - errors are silently ignored
- Recommendations: Replace with specific exception types; add logging; provide meaningful error messages to help diagnose issues

**Error Handling Without Context:**
- Risk: Exception handlers catch specific errors but don't log or provide diagnostic information
- Files: `src/primel/run.py` (lines 39, 45), `src/primel/fitness.py` (lines 69, 131)
- Current mitigation: Errors are caught but only minimal error messages printed
- Recommendations: Add structured logging; include stack traces in debug mode; consider re-raising with additional context

## Performance Bottlenecks

**Tree Simplification Restart Loop:**
- Problem: `simplify_tree()` restarts from beginning after each simplification, leading to O(n²) worst-case complexity
- Files: `src/primel/tree.py` (lines 272-280)
- Cause: Restart-based approach ensures parent context but requires re-traversal
- Improvement path: Implement single-pass algorithm with post-order traversal; track modifications and process efficiently without full restarts

**Expensive KL Divergence Computation:**
- Problem: Computes PDF for all samples across all distributions for every fitness evaluation
- Files: `src/primel/fitness.py` (lines 204-230)
- Cause: Nested loops compute denominator for each sample across all distributions
- Improvement path: Cache PDF computations; vectorize operations across samples when possible; consider approximations for large numbers of distributions

**Deep Copy in Parallel Evolution:**
- Problem: Deep copies program objects for every program in every generation
- Files: `src/primel/adapters/gplearn/model.py` (line 214)
- Cause: Need to preserve original program state before simplification
- Improvement path: Implement copy-on-write; use shallow copies where safe; avoid deep copying entire program structure

**Repeated Tree Evaluation:**
- Problem: `simplify_tree()` evaluates entire subtree at multiple nodes to check simplification conditions
- Files: `src/primel/tree.py` (lines 209, 215, 265)
- Cause: Checks conditions by evaluating subtrees at multiple points
- Improvement path: Cache evaluation results; track monotonic properties more efficiently; memoize subtree evaluations

## Fragile Areas

**Tree Structure Mismatch:**
- Files: `src/primel/adapters/gplearn/_tree.py`
- Why fragile: Uses Node class with `children` attribute while `src/primel/tree.py` uses flat array representation; two different tree representations exist without conversion layer
- Safe modification: Always use conversion functions; document which representation is expected where; add type checking
- Test coverage: Gaps - no integration tests verifying conversion correctness between representations

**Data Loading with Minimal Validation:**
- Files: `src/primel/run.py` (lines 43-47)
- Why fragile: Only checks if file exists, doesn't validate data shape, type, or content
- Safe modification: Add data validation checks; verify array dimensions; validate numeric types
- Test coverage: Low - only tests FileNotFoundError, doesn't test invalid data formats

**ImportanceSampler Initialization Complexity:**
- Files: `src/primel/samplers.py` (lines 118-166)
- Why fragile: Complex initialization logic with multiple paths and silent failure handling
- Safe modification: Add validation for sampler entry formats; improve error messages; separate initialization into smaller methods
- Test coverage: Medium - has tests but may not cover all edge cases

**Balance Heuristic Weight Computation:**
- Files: `src/primel/samplers.py` (lines 168-231)
- Why fragile: Complex numerical computation with epsilon values; has special case for Empirical distributions; silent fallback on errors
- Safe modification: Add input validation; document numerical stability considerations; add tests for edge cases
- Test coverage: Low - limited tests for weight computation correctness

## Scaling Limits

**Memory Usage with Large Populations:**
- Current capacity: Population size configurable but memory usage grows linearly with population_size × generations
- Limit: Not explicitly tested for large-scale experiments; may hit memory limits with >10,000 population or 100+ generations
- Scaling path: Implement low_memory mode; add memory profiling; support streaming/generational garbage collection

**Computation Time with Complex Trees:**
- Current capacity: Single generation time depends on population size, tree complexity, and sample count
- Limit: No explicit timeouts or progress reporting; long-running experiments may hang without feedback
- Scaling path: Add progress callbacks; implement early stopping based on time; support checkpoint/resume

## Dependencies at Risk

**gplearn:**
- Risk: Unmaintained library (last release 2019); relies on monkey patch for compatibility; may not work with newer sklearn versions
- Impact: Core genetic programming functionality would break
- Migration plan: Monitor for fork or alternative; consider extracting core GP logic; implement custom symbolic regression based on research paper

## Missing Critical Features

**Comprehensive Logging:**
- Problem: No centralized logging; minimal error context; difficult to debug production issues
- Blocks: Troubleshooting failed experiments; understanding convergence behavior; debugging optimization issues

**Progress Reporting:**
- Problem: Long-running experiments provide no feedback until completion
- Blocks: User experience; ability to monitor experiment health; early intervention on problematic runs

**Checkpointing/Resume:**
- Problem: No ability to save and resume experiment state
- Blocks: Long experiments lost if interrupted; inability to explore intermediate results; wasted computation time

## Test Coverage Gaps

**Untested Core Modules:**
- What's not tested: `src/primel/adapters/gplearn/model.py` (no tests), `src/primel/run.py` (no tests), `src/primel/early_stopping.py` (no tests), `src/primel/patch_kde.py` (no tests)
- Files: `src/primel/adapters/gplearn/model.py`, `src/primel/run.py`, `src/primel/early_stopping.py`, `src/primel/patch_kde.py`
- Risk: Bugs in core functionality go undetected; refactoring is risky; integration issues may surface late
- Priority: High

**Untested Edge Cases:**
- What's not tested: Empty datasets, single-sample datasets, extreme parameter values, numerical stability edge cases
- Files: All test files lack these scenarios
- Risk: Production failures on edge cases; numerical instability in corner cases
- Priority: Medium

**Integration Testing:**
- What's not tested: End-to-end workflow from data loading through model training; adapter integration; early stopping integration
- Files: No integration test suite
- Risk: Component-level tests pass but system fails; integration bugs discovered late
- Priority: High

**Performance Testing:**
- What's not tested: Benchmarks for scaling with data size, population size, tree complexity
- Files: No performance test suite
- Risk: Performance regressions go unnoticed; inability to set realistic expectations
- Priority: Low

---

*Concerns audit: 2026-03-01*
