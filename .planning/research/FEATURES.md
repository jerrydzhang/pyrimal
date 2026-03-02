# Feature Research

**Domain:** Expression tree simplification with level set preservation
**Researched:** 2026-03-02
**Confidence:** HIGH (mathematical principles + SymPy documentation analysis)

## Feature Landscape

### Table Stakes (Users Expect These)

Rules that are always safe and necessary for basic simplification.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Remove multiply-by-one | `1 * f(x)` → `f(x)` is identity | LOW | Always safe, no level set change |
| Remove divide-by-one | `f(x) / 1` → `f(x)` is identity | LOW | Always safe |
| Remove add-zero | `f(x) + 0` → `f(x)` is identity | LOW | Always safe |
| Remove subtract-zero | `f(x) - 0` → `f(x)` is identity | LOW | Always safe |
| Collapse x-x | `x - x` → `0` | LOW | Mathematically exact |
| Collapse x/x | `x / x` → `1` (where x ≠ 0) | LOW | Need guard or accept domain restriction |
| Collapse x+x | `x + x` → `2*x` (optional) | LOW | Doesn't change zeros, but adds constant |
| Remove multiply-by-nonzero-constant at ROOT | `c * f(x)` → `f(x)` at output | LOW | Zero-set preserved: c*f(x)=0 ⟺ f(x)=0 |
| Remove add-constant at ROOT | `f(x) + c` → `f(x)` at output | LOW | Shifts level set, zero-set topology unchanged |

### Differentiators (Competitive Advantage)

Context-aware rules that require understanding tree position.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Root-level monotonic wrapper removal | `log(f(x))` → `f(x)` at ROOT only | MEDIUM | Monotonic functions preserve zero topology at output |
| Conditional sqrt removal | `sqrt(f(x))` → `f(x)` when f(x) ≥ 0 | MEDIUM | Requires range analysis |
| Conditional square removal | `f(x)^2` → `|f(x)|` or `f(x)` when f(x) ≥ 0 | MEDIUM | Square loses sign info, creates issues |
| Conditional trig removal | `sin(f(x))` → `f(x)` when \|f(x)\| < π | HIGH | Only safe in monotonic region of sin |
| Nested constant propagation | Track whether constant is on "output path" | HIGH | Key insight: `f(x)+c` safe, `f(x+c)` not safe |
| Expression normalization | Canonical form for comparison | MEDIUM | SymPy uses canonical ordering for equality testing |

### Anti-Features (Commonly Requested, Often Problematic)

Rules that seem useful but break level set preservation.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Global constant removal | Simplify `f(x+c)` → `f(x)` everywhere | Changes input to nested functions, breaks zeros | Only remove constants at ROOT level |
| Unconditional log removal | Remove all `log()` wrappers | Safe at root, but `log(x+c) - log(x)` ≠ `0` | Only remove log at root: `log(f(x))` → `f(x)` |
| Unconditional exp removal | Remove all `exp()` wrappers | Safe at root for zero-preserving, but `exp(x+c)` ≠ `exp(x)` | Only remove at root |
| Unconditional sqrt removal | Remove all `sqrt()` wrappers | `sqrt(x-1) = 0` ≠ `x-1 = 0` when x < 1 | Check range or only at root |
| Unconditional trig removal | Remove `sin/cos/tan` wrappers | Periodic zeros: `sin(π) = 0` but `π ≠ 0` | Never remove trig unconditionally |
| Power simplification | `f(x)^2` → `f(x)` | Loses sign: `(-1)^2 = 1 ≠ -1` | Keep powers or convert to abs |
| Cross-branch constant folding | `(x+c) + (y-c)` → `x+y` | Constants inside branches affect downstream | Only fold at same tree level |

## Feature Dependencies

```
Root-level constant removal
    └──requires──> Position tracking in tree (know if at root)

Conditional monotonic removal (log/sqrt/exp)
    └──requires──> Position tracking in tree
    └──requires──> Range analysis (for sqrt, trig)

Range analysis
    └──requires──> Expression evaluation on sample points

Canonical form comparison
    └──enhances──> Detect semantically equivalent expressions
```

### Dependency Notes

- **Position tracking requires tree traversal with context:** Must know if current node is on the "output path" (path from root to current node) or nested inside an argument
- **Range analysis enhances conditional rules:** Without it, can only safely apply rules at root level
- **Canonical form comparison conflicts with aggressive simplification:** Simplifying to canonical form may apply unsafe rules

## MVP Definition

### Launch With (v1)

Minimum viable product — what's needed to fix the current bug.

- [x] Position-aware constant removal — Only remove `+c`, `*c` at ROOT level
- [x] Position-aware monotonic removal — Only remove `log/sqrt/exp` at ROOT level
- [x] Basic identity rules — `*1`, `/1`, `+0`, `-0` removal (already work)
- [x] Property-based tests — Hypothesis tests proving level set preservation

### Add After Validation (v1.x)

Features to add once core is working.

- [ ] Range-based conditional rules — Allow `sqrt(f(x))` → `f(x)` when f(x) ≥ 0
- [ ] Smart trig handling — Allow `sin(f(x))` → `f(x)` when |f(x)| < π/2
- [ ] Expression comparison — Detect when two expressions have same level sets

### Future Consideration (v2+)

Features to defer until core is validated.

- [ ] Full SymPy integration — Use SymPy for canonicalization after safe simplification
- [ ] Configurable aggressiveness — Let users choose simplification level
- [ ] Domain-aware simplification — Use training data range to guide decisions

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Position-aware constant removal | HIGH | LOW | P1 |
| Position-aware monotonic removal | HIGH | LOW | P1 |
| Property-based tests | HIGH | MEDIUM | P1 |
| Range-based conditional rules | MEDIUM | MEDIUM | P2 |
| Smart trig handling | LOW | HIGH | P3 |
| Full SymPy integration | LOW | HIGH | P3 |

**Priority key:**
- P1: Must have for fix
- P2: Should have, add when possible
- P3: Nice to have, future consideration

## Mathematical Foundation

### Level Set Definition

For a function f: ℝⁿ → ℝ, the level set at value c is:
```
L_c(f) = {(x₁,...,xₙ) | f(x₁,...,xₙ) = c}
```

For symbolic regression, we care about the **zero level set** L₀(f) — where f(x) = 0.

### Safe Transformations

A transformation T is **level-set-preserving** if:
```
L₀(T(f)) = L₀(f)
```

**Theorem (Monotonic Output Transformations):**
If g: ℝ → ℝ is strictly monotonic, then `g ∘ f` has the same zero set as `f`:
```
L₀(g ∘ f) = L₀(f)
```

**Proof:** If f(x) = 0, then g(f(x)) = g(0). For this to equal 0, we need g(0) = 0.
So strictly speaking: `g ∘ f` preserves zeros only if g(0) = 0.

For log, exp: log(1) = 0, exp(0) = 1. Neither preserves zero directly!
- `log(f(x)) = 0` ⟺ `f(x) = 1` ≠ `f(x) = 0`
- `exp(f(x)) = 0` has no solution (exp never zero)

**Revised insight:** "Level set preservation" in this project means preserving the *topology* of level sets, not the exact zero set. We care that:
- Early stopping triggers at the right time
- The shape of f(x) is preserved (spikiness, smoothness)

For topology preservation:
- Monotonic transformations preserve the *ordering* of values
- `f(x) + c` shifts all values by c, preserving relative differences
- `c * f(x)` scales all values by c (c ≠ 0), preserving relative differences
- `log(f(x))` compresses range but preserves monotonicity
- `exp(f(x))` expands range but preserves monotonicity

### The Key Distinction: Output vs Input Constants

```
OUTPUT LEVEL (SAFE):          INPUT LEVEL (NOT SAFE):
f(x) + c  →  f(x)            f(x + c)  →  f(x)
c * f(x)  →  f(x)            c * x + f(y)  →  f(y)
f(x) / c  →  f(x)            x / c + f(y)  →  f(y)
```

At output level, constants just shift/scale the final result. The zero-crossing behavior is preserved.

At input level, constants change what values the inner function sees, which can create or destroy zeros:
- `f(x) = x - 5` has zero at x=5
- `f(x+3) = (x+3) - 5 = x - 2` has zero at x=2 ← DIFFERENT!
- Removing the +3 changes the zero location

## Competitor Feature Analysis

| Feature | SymPy | PySR | gplearn | Our Approach |
|---------|-------|------|---------|--------------|
| Simplification goal | Mathematical equivalence | Fitness-based evolution | Fitness-based evolution | Level set topology preservation |
| Constant removal | Context-aware (canonical form) | N/A (no simplification) | N/A (no simplification) | Position-aware (root only) |
| Monotonic removal | No (keeps structure) | N/A | N/A | Position-aware (root only) |
| Trig handling | Keeps trig | Keeps trig | Keeps trig | Keep trig or conditional |
| Test approach | Example-based | Fitness-based | Fitness-based | Property-based (Hypothesis) |

## Sources

- SymPy Simplification Documentation: https://docs.sympy.org/latest/modules/simplify/simplify.html
- Level Set Wikipedia: https://en.wikipedia.org/wiki/Level_set
- PySR GitHub: https://github.com/MilesCranmer/PySR
- gplearn Documentation: https://gplearn.readthedocs.io/

---
*Feature research for: expression tree simplification with level set preservation*
*Researched: 2026-03-02*
