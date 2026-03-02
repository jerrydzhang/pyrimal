# Pitfalls Research

**Domain:** Expression tree simplification with level set preservation
**Researched:** 2026-03-02
**Confidence:** HIGH (based on code analysis and mathematical principles)

## Critical Pitfalls

### Pitfall 1: Context-Blind Constant Removal

**What goes wrong:**
Removing constants from anywhere in the tree, not distinguishing between constants on the final output vs. constants in intermediate positions. The current implementation removes `add/sub/mul/div` with constants at any node, but:
- `f(x) + 5 → f(x)` is safe (constant shifts output, preserves zero-set topology)
- `f(x + 5) → f(x)` is NOT safe (constant changes what gets fed to f)

**Why it happens:**
The simplification rules operate node-by-node without tracking whether the current node is at the "output edge" of the expression or nested within other operations. A constant at position `f(g(x) + c)` affects what `f` sees, not just the final output.

**How to avoid:**
Track "output proximity" during traversal. Only remove additive/multiplicative constants when no further operations compose on top. Implement bottom-up simplification with context passing: each node knows if its parent will apply a non-trivial operation to its result.

**Warning signs:**
- Simplified expressions produce different zero-crossings than original
- Early stopping triggers on expressions that shouldn't be "done"
- Validation variance spikes after simplification on certain inputs

**Phase to address:**
Phase 1 (Fix tree simplification) - this is the core issue

---

### Pitfall 2: Monotonic Unary Removal Without Zero-Preservation Verification

**What goes wrong:**
The code removes `log`, `sqrt`, `exp` unconditionally because they're monotonic. But monotonic ≠ zero-preserving. 
- `log(x)` maps `x=1` to `0`, not `x=0`
- `exp(x)` never produces zero at all
- `sqrt(x)` only preserves zero at `x=0`

Removing these changes WHERE zeros occur, not just the output scale.

**Why it happens:**
Conflating "preserves order" with "preserves level sets." A monotonic function `g` preserves that `f(x) < f(y)` implies `g(f(x)) < g(f(y))`, but `g(f(x)) = 0` is a different equation than `f(x) = 0`.

**How to avoid:**
For level set preservation (zero-set topology), only remove operations that satisfy `g(0) = 0` AND `g⁻¹(0) = {0}`. This means:
- `add(x, c)` and `sub(x, c)`: safe (zero shifts by constant, topology same)
- `mul(x, c)` and `div(x, c)` for c ≠ 0: safe (zero scaled, topology same)
- `log(x)`, `exp(x)`: NOT safe (zeros move or disappear)
- `sqrt(x)`, `square(x)`: only safe if already at zero

**Warning signs:**
- Expressions that should have zeros don't after simplification
- Fitness evaluations differ significantly pre/post simplification
- Solutions found in wrong regions of input space

**Phase to address:**
Phase 1 (Fix tree simplification) - requires rethinking which unary ops are removable

---

### Pitfall 3: Sampling-Based Monotonicity Assumptions

**What goes wrong:**
For trig functions (`sin`, `cos`, `tan`), the code samples `X` and checks if the range is under `2π` to assume monotonicity. This is fragile:
- The sample `X` may not cover the full input range the expression will see
- Edge cases near period boundaries can flip the decision
- Training data vs. validation data vs. test data may have different ranges

**Why it happens:**
True monotonicity analysis requires symbolic reasoning about the input domain, which is hard. Sampling is easy but only proves monotonicity on the sampled points, not globally.

**How to avoid:**
1. Be conservative: if you can't PROVE monotonicity symbolically, don't remove the operation
2. If using sampling, require monotonicity on a strictly larger range than training data
3. Track the domain assumptions explicitly and re-check if inputs change
4. Consider not removing trig functions at all (they're usually not spurious)

**Warning signs:**
- Simplification decisions vary between runs with different random seeds
- Expressions behave differently on validation vs. training data
- Sudden failures when input distribution shifts

**Phase to address:**
Phase 1 (Fix tree simplification) - remove or harden the sampling-based checks

---

### Pitfall 4: Division Simplification Creating Undefined Behavior

**What goes wrong:**
The code removes `mul(x, c)` and `div(x, c)` for `c ≠ 0`. But for division by a subtree (not constant), the current code doesn't handle:
- `x / x → 1` is only valid where `x ≠ 0`
- The simplified expression may be defined where the original wasn't

This changes the level sets by adding/removing singularities.

**Why it happens:**
Symbolic simplification often ignores domain restrictions. But for numerical evaluation and level set preservation, WHERE an expression is undefined matters as much as WHAT it computes.

**How to avoid:**
1. For `x / x → 1`: require proving `x ≠ 0` on the input domain, or don't simplify
2. Track domain restrictions through simplification
3. Consider using safe division that returns NaN/Inf at singularities, and preserve those

**Warning signs:**
- NaN/Inf values appear or disappear after simplification
- Fitness becomes undefined on previously valid inputs
- Optimizer exploits spurious singularities

**Phase to address:**
Phase 1 (Fix tree simplification) - add domain tracking or remove risky simplifications

---

### Pitfall 5: Identical Subtree Folding Without Sign Analysis

**What goes wrong:**
The code simplifies `x * x → x` when `x ≥ 0`. But:
- `x - x → 0` is always safe (zeros preserved)
- `x + x → x` changes the scale but preserves zeros
- `x * x → x` is only valid for `x ∈ {0, 1}`, not all `x ≥ 0`
- `x / x → 1` is only valid for `x ≠ 0`

The current `square` check `(result >= 0).all()` is wrong for `x * x → x`.

**Why it happens:**
Confusion between "the subtree is non-negative" and "the transformation preserves level sets." `x * x` and `x` have the same sign but different magnitudes (except at 0 and 1).

**How to avoid:**
- `x + x → x`: NOT level-set preserving (scales by 2, moves zeros unless already at zero)
- `x * x → x`: NOT level-set preserving (only equal at 0 and 1)
- `x - x → 0`: Preserves zeros but loses all other information - use carefully
- `x / x → 1`: Only valid where `x ≠ 0`

For level set preservation, the only safe identical-subtree simplification is `x - x → 0` (and even that changes the function dramatically).

**Warning signs:**
- Simplified expressions have different magnitudes than original
- Zero-finding produces different roots
- Fitness landscape changes unexpectedly

**Phase to address:**
Phase 1 (Fix tree simplification) - audit and fix identical subtree rules

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Sampling-based monotonicity checks | Fast, easy to implement | Incorrect on edge cases, data-dependent | Never for level set preservation |
| Removing all unary monotonic ops | Simpler expressions | Changes zero locations | Only for operations with g(0)=0 |
| Ignoring domain restrictions | Simpler code | Undefined behavior, NaN propagation | Never |
| Top-down simplification | Natural traversal order | Can't track output context | Only if you track context separately |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Re-sampling X for every simplification check | Slow simplification on large trees | Cache evaluation results per subtree | Trees with >100 nodes |
| Restart-from-beginning on each simplification | O(n²) simplification passes | Track dirty nodes, only recheck affected | Multiple simplifications per tree |
| Evaluating full tree for each node check | Redundant computation | Memoize subtree evaluations | Deep trees |

## Security Mistakes

Not applicable - this is a mathematical library with no external inputs beyond the training data.

## "Looks Done But Isn't" Checklist

- [ ] **Simplification preserves zeros:** Verify f(x)=0 ⟺ simplify(f)(x)=0 for test cases
- [ ] **Simplification preserves undefined points:** Verify NaN/Inf behavior matches
- [ ] **Constant removal is context-aware:** Constants at root only, not in subtrees
- [ ] **Unary removal verified:** Only remove ops where g(0)=0 AND g⁻¹(0)={0}
- [ ] **Property-based tests pass:** Hypothesis finds no counterexamples to level set preservation

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Context-blind constant removal | MEDIUM | Add context parameter to simplify function, pass "is_output" flag |
| Wrong unary removal rules | HIGH | Redesign which operations are removable, may need new analysis |
| Sampling-based monotonicity | LOW | Remove sampling checks, be conservative |
| Division domain issues | MEDIUM | Add domain tracking or remove non-constant division simplification |
| Identical subtree bugs | LOW | Fix the specific rules, add tests |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Context-blind constant removal | Phase 1 (Fix tree simplification) | Property test: simplify preserves zeros on all generated expressions |
| Monotonic unary removal | Phase 1 (Fix tree simplification) | Unit tests for each unary op, verify g(0)=0 condition |
| Sampling-based monotonicity | Phase 1 (Fix tree simplification) | Remove sampling, verify no test regressions |
| Division domain issues | Phase 1 (Fix tree simplification) | Test with expressions containing divisions, check NaN handling |
| Identical subtree bugs | Phase 1 (Fix tree simplification) | Symbolic verification of x+x, x*x, x-x, x/x transformations |
| Missing property tests | Phase 2 (Hypothesis tests) | Run Hypothesis for 1000+ iterations on level set preservation |

## Sources

- Code analysis of `/home/jerry/Research/code/pyrimel/src/primel/tree.py`
- Project context from `/home/jerry/Research/code/pyrimel/.planning/PROJECT.md`
- Mathematical principles of level sets and monotonic functions
- Computer algebra system simplification literature (SymPy, Mathematica documentation)

---
*Pitfalls research for: Tree simplification with level set preservation*
*Researched: 2026-03-02*
