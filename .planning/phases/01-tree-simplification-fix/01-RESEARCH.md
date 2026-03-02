# Phase 1: Tree Simplification Fix - Research

**Researched:** 2026-03-02
**Domain:** Expression tree simplification with level set preservation
**Confidence:** HIGH

## Summary

This phase addresses position-aware tree simplification that preserves level set topology. The core insight is that simplifications like removing constants (`+c`, `*c`) and monotonic functions (`log`, `sqrt`, `exp`) are only valid at the **root/output edge** of the expression tree, not at nested positions where they would change the input to inner functions.

**Primary recommendation:** Implement a `depth` parameter in `_simplify_at_index` to track position context. Root-level (depth=0) allows all simplifications; nested positions allow only identity rules (`*1`, `/1`, `+0`, `-0`).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- All simplification rules must preserve level set topology (where f(x) = 0)
- No flexibility on correctness — this is a rigid mathematical requirement
- Position-aware rules: constants (+c, *c) and monotonic ops (log, sqrt, exp) only removable at root/output edge

### OpenCode's Discretion
- Property-based test coverage breadth (Hypothesis)
- Error message detail level
- Backward compatibility approach for non-breaking cases

### Deferred to Research
- **x+x Rule Handling** — researcher must investigate:
  1. Is there a mathematically safe position-aware fix for `x + x → x`?
  2. If not, remove the rule entirely
- ROADMAP specifies this rule is NOT level-set preserving as currently implemented

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| SIMP-01 | Position-aware constant removal — only remove `+c`, `*c` at root/output edge | Pass `depth` param to `_simplify_at_index`; check `depth == 0` before constant removal |
| SIMP-02 | Position-aware monotonic removal — only remove log/sqrt/exp at root level | Same depth-tracking approach; guard lines 204-206 with `depth == 0` check |
| SIMP-03 | Basic identity rules preserved — `*1`, `/1`, `+0`, `-0` removal still works | These are safe at any depth; no change needed beyond keeping existing logic |
| SIMP-04 | Property-based tests with Hypothesis proving level set preservation | Use `st.recursive()` for tree generation; test `simplify(f)(x) == 0 ⟺ f(x) == 0` |
| SIMP-05 | Remove or fix `x + x → x` rule (NOT level-set preserving) | **Recommendation: REMOVE entirely** — no safe position-aware fix exists |
| SIMP-06 | Fix sampling-based monotonicity checks (fragile, data-dependent) | **Recommendation: Interval arithmetic** — propagate min/max bounds through tree |
| TEST-01 | Unit tests for position-aware constant removal | Test that `add(x, 5)` simplifies at root but `log(add(x, 5))` doesn't remove the +5 |
| TEST-02 | Unit tests for position-aware monotonic removal | Test that `log(x)` simplifies at root but `add(log(x), 1)` doesn't remove the log |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| hypothesis | latest | Property-based testing | Industry standard for Python PBT, excellent recursive strategy support |
| pytest | >=8.4.2 | Test framework | Already in dev dependencies |

### Supporting
| Library | Purpose | When to Use |
|---------|---------|-------------|
| numpy | Array operations for test data | Already core dependency |

### Installation
```bash
uv add --dev hypothesis
```

## Architecture Patterns

### Recommended Approach: Depth-Tracking

The core architectural change is adding a `depth` parameter to track position in the tree:

```python
def simplify_tree(tree: ExpressionTree, X: np.ndarray) -> None:
    def _simplify_at_index(index: int, depth: int) -> bool:
        if index >= len(tree.nodes):
            return False
            
        node = tree.nodes[index]
        
        if node.arity == 1:
            # ONLY at root level (depth == 0)
            if depth == 0 and node.name in {"log", "sqrt", "exp"}:
                tree.replace_node_with_child(index, 0)
                return True
            # ... trig/square checks remain data-dependent (SIMP-06)
            
        elif node.arity == 2:
            # Constant removal ONLY at root level
            if depth == 0:
                if node.name in {"add", "sub"}:
                    # ... existing constant removal logic
                elif node.name in {"mul", "div"}:
                    # ... existing constant removal logic
            
            # Identity rules safe at any depth
            # ... existing x-x, x/x logic
            
            # x + x rule: REMOVE entirely (SIMP-05)
            # Don't implement this rule
```

### Pattern: Recursive Tree Traversal with Context

```python
# When recursing into children, increment depth
def _simplify_at_index(index: int, depth: int) -> bool:
    # ... simplification logic ...
    
    # After simplification, recurse into children with depth + 1
    child_index = index + 1
    for _ in range(node.arity):
        if _simplify_at_index(child_index, depth + 1):
            return True  # Restart
        child_index += tree._subtree_size(child_index)
```

### Anti-Patterns to Avoid

- **Unconditional monotonic removal:** The current code removes log/sqrt/exp everywhere — this breaks level sets when nested
- **Data-dependent checks without bounds:** The trig/square range checks are fragile; they depend on X which may not cover the full input space
- **x + x → x simplification:** Mathematically equivalent to `2*x → x` which requires position-awareness AND fails when nested

## Mathematical Analysis: x + x Rule

**The Problem:**
- `f(x) + f(x) = 2 * f(x)`
- Simplifying to `f(x)` is equivalent to removing the constant multiplier 2
- This is only valid at root level (like other constant removal)

**Why It's Harder Than It Seems:**
- At root: `2*f(x) = 0 ⟺ f(x) = 0` ✓ Safe
- Nested: `log(2*f(x))` vs `log(f(x))` — zeros may shift ✗ Not safe

**Position-aware fix analysis:**
```python
# At root level (depth == 0):
# f(x) + f(x) → f(x)  IS safe (equivalent to removing constant 2)

# At nested level (depth > 0):
# f(x) + f(x) → f(x)  IS NOT safe in general
```

**Recommendation:** **REMOVE the rule entirely.** While a position-aware fix exists, it provides minimal benefit and adds complexity. The rule is not essential for simplification effectiveness.

## SIMP-06: Fixing Sampling-Based Checks

**Current Problem (lines 207-218):**
```python
elif node.name in {"sin", "cos", "tan"}:
    result = tree.evaluate(X, index + 1)
    if np.abs(result.max() - result.min()) < 2 * np.pi:
        tree.replace_node_with_child(index, 0)
```

This is fragile because:
1. X may not cover the full input space
2. The check is only valid for the sampled data
3. Different X values could give different simplification results

**Recommended Solution: Interval Arithmetic**

Implement simple interval bounds propagation:

```python
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Interval:
    lo: float
    hi: float
    
    def __add__(self, other: 'Interval') -> 'Interval':
        return Interval(self.lo + other.lo, self.hi + other.hi)
    
    def __mul__(self, other: 'Interval') -> 'Interval':
        return Interval(
            min(self.lo*other.lo, self.lo*other.hi, self.hi*other.lo, self.hi*other.hi),
            max(self.lo*other.lo, self.lo*other.hi, self.hi*other.lo, self.hi*other.hi)
        )
    
    @property
    def width(self) -> float:
        return self.hi - self.lo
```

Then propagate intervals through the tree to get conservative bounds on ranges.

**Alternative (simpler):** Remove trig/square simplification entirely. These rules are opportunistic optimizations, not correctness requirements. The position-aware rules (SIMP-01, SIMP-02) are the critical fixes.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Property-based test strategies | Custom generator logic | `hypothesis.strategies.recursive()` | Handles shrinking, edge cases, composability |
| Interval arithmetic (if needed) | Full interval library | Simple dataclass with bounds | Only need basic arithmetic, not full IEEE 1788 |

## Common Pitfalls

### Pitfall 1: Forgetting to Restart After Simplification
**What goes wrong:** After `replace_node_with_child`, indices shift; continuing traversal uses stale indices
**Why it happens:** The current code correctly restarts, but depth-tracking adds complexity
**How to avoid:** Always return `True` after any modification to trigger restart from index 0
**Warning signs:** IndexError, wrong nodes being simplified

### Pitfall 2: Confusing "Safe at Root" with "Always Safe"
**What goes wrong:** Applying root-only rules at nested positions
**Why it happens:** The `depth` parameter is easy to forget in recursive calls
**How to avoid:** Every rule must explicitly check `depth == 0` or be provably safe at all depths
**Warning signs:** Tests pass with simple trees but fail with nested structures

### Pitfall 3: Hypothesis Tests That Pass Trivially
**What goes wrong:** Generated trees are too simple to trigger edge cases
**Why it happens:** `recursive()` with low `max_leaves` produces shallow trees
**How to avoid:** Use `max_leaves=20` or higher; use `@settings(max_examples=500)` for coverage
**Warning signs:** 100% pass rate with no shrinking ever occurring

## Code Examples

### Hypothesis Strategy for Expression Trees

```python
import hypothesis.strategies as st
from hypothesis import given, assume
import numpy as np
from primel.tree import ExpressionTree, Node

# Leaf strategies
leaves = st.one_of([
    st.builds(lambda: Node("x", lambda X: X, 0)),
    st.builds(lambda v: Node("const", v, 0), st.floats(min_value=-10, max_value=10)),
])

# Unary operators
unary_ops = st.sampled_from([
    ("log", np.log),
    ("sqrt", np.sqrt),
    ("exp", np.exp),
    ("sin", np.sin),
    ("cos", np.cos),
])

# Binary operators  
binary_ops = st.sampled_from([
    ("add", np.add),
    ("sub", np.subtract),
    ("mul", np.multiply),
    ("div", np.divide),
])

# Recursive tree strategy
@st.composite
def expression_trees(draw, max_leaves=15):
    def extend(base):
        return st.one_of([
            # Unary node wrapping base
            st.builds(
                lambda op, child: [Node(op[0], op[1], 1)] + child,
                unary_ops, base
            ),
            # Binary node with two base children
            st.builds(
                lambda op, left, right: [Node(op[0], op[1], 2)] + left + right,
                binary_ops, base, base
            ),
        ])
    
    node_list = draw(st.recursive(
        st.builds(lambda n: [n], leaves),
        extend,
        max_leaves=max_leaves
    ))
    return ExpressionTree.init_from_list(node_list)

# Property test for level set preservation
@given(tree=expression_trees(), X=st.builds(np.array, st.lists(st.floats(-10, 10), min_size=5, max_size=20)))
def test_simplify_preserves_level_sets(tree, X):
    X = X.reshape(-1, 1)  # Shape for single feature
    
    # Evaluate original
    with np.errstate(invalid='ignore', divide='ignore'):
        original = tree.evaluate(X)
    
    # Copy and simplify
    # (Need to implement tree copying first)
    simplified_tree = copy_tree(tree)
    simplify_tree(simplified_tree, X)
    
    with np.errstate(invalid='ignore', divide='ignore'):
        simplified = simplified_tree.evaluate(X)
    
    # Key property: zeros are preserved
    # f(x) == 0 ⟺ simplify(f)(x) == 0
    original_zeros = np.isclose(original, 0, atol=1e-10)
    simplified_zeros = np.isclose(simplified, 0, atol=1e-10)
    
    # Must have same zero locations (within tolerance)
    np.testing.assert_array_equal(original_zeros, simplified_zeros)
```

### Position-Aware Simplification (Core Change)

```python
def simplify_tree(tree: ExpressionTree, X: np.ndarray) -> None:
    def _simplify_at_index(index: int, depth: int) -> bool:
        if index >= len(tree.nodes):
            return False
            
        node = tree.nodes[index]

        if node.arity == 1:
            # Monotonic ops: ONLY safe at root (depth == 0)
            if depth == 0 and node.name in {"log", "sqrt", "exp"}:
                tree.replace_node_with_child(index, 0)
                return True
            elif node.name in {"sin", "cos", "tan"}:
                # SIMP-06: Consider removing or using interval analysis
                result = tree.evaluate(X, index + 1)
                if np.abs(result.max() - result.min()) < 2 * np.pi:
                    tree.replace_node_with_child(index, 0)
                    return True
            elif node.name == "square":
                result = tree.evaluate(X, index + 1)
                if (result >= 0).all():
                    tree.replace_node_with_child(index, 0)
                    return True

        elif node.arity == 2:
            if node.name not in {"add", "sub", "mul", "div"}:
                return False

            left_index = index + 1
            right_index = index + 1 + tree._subtree_size(index + 1)
            
            if right_index >= len(tree.nodes):
                return False
                
            left = tree.nodes[left_index]
            right = tree.nodes[right_index]

            # Constant removal: ONLY at root (depth == 0)
            if depth == 0:
                if node.name in {"add", "sub"}:
                    if left.arity == 0 and isinstance(left.value, (int, float)):
                        tree.replace_node_with_child(index, 1)
                        return True
                    elif right.arity == 0 and isinstance(right.value, (int, float)):
                        tree.replace_node_with_child(index, 0)
                        return True
                
                elif node.name in {"mul", "div"}:
                    if left.arity == 0 and isinstance(left.value, (int, float)):
                        if left.value != 0:
                            tree.replace_node_with_child(index, 1)
                            return True
                    elif right.arity == 0 and isinstance(right.value, (int, float)):
                        if right.value != 0:
                            tree.replace_node_with_child(index, 0)
                            return True
            
            # Identity rules: Safe at any depth
            if tree._is_subtree_equal(left_index, right_index):
                if node.name == "sub":  # x - x = 0
                    tree.replace_subtree(index, "constant", 0.0, 0)
                    return True
                elif node.name == "div":  # x / x = 1
                    tree.replace_subtree(index, "constant", 1.0, 0)
                    return True
                # SIMP-05: x + x rule REMOVED - not level-set preserving
                # mul(x, x) requires data check, keep but at root only
                elif node.name == "mul" and depth == 0:
                    result = tree.evaluate(X, left_index)
                    if (result >= 0).all():
                        tree.replace_node_with_child(index, 0)
                        return True

        return False

    # Main loop with depth tracking
    index = 0
    depth = 0  # Start at root
    while index < len(tree.nodes):
        if _simplify_at_index(index, depth):
            index = 0  # Restart
            depth = 0
        else:
            # Move to next node, update depth
            # (Depth calculation requires tracking parent context)
            index += 1
```

**Note:** The depth calculation in the main loop requires additional bookkeeping. A cleaner approach is to make `_simplify_at_index` fully recursive:

```python
def simplify_tree(tree: ExpressionTree, X: np.ndarray) -> None:
    def _simplify_subtree(index: int, depth: int) -> tuple[bool, int]:
        """
        Simplify subtree starting at index.
        Returns (made_change, next_index_after_subtree).
        """
        if index >= len(tree.nodes):
            return False, index
            
        node = tree.nodes[index]
        made_change = False
        
        # Try simplification at current node
        if _try_simplify_at(index, depth):
            return True, 0  # Signal restart
        
        # Recurse into children with incremented depth
        child_index = index + 1
        for _ in range(node.arity):
            changed, next_idx = _simplify_subtree(child_index, depth + 1)
            if changed:
                return True, 0  # Signal restart
            child_index = next_idx
        
        return False, index + tree._subtree_size(index)
    
    # Keep simplifying until no changes
    while True:
        changed, _ = _simplify_subtree(0, 0)
        if not changed:
            break
```

## Open Questions

1. **Should we remove trig/square simplification entirely (SIMP-06)?**
   - What we know: Current approach is data-dependent and fragile
   - What's unclear: Whether interval analysis is worth the implementation cost
   - Recommendation: **Remove for v1**. These are optimizations, not correctness requirements. Add interval analysis in v2 if performance demands it.

2. **How to handle tree copying for Hypothesis tests?**
   - What we know: Need to compare original vs simplified
   - What's unclear: ExpressionTree doesn't have `__copy__` or `__deepcopy__`
   - Recommendation: Implement `copy()` method that creates new list of Nodes

## Sources

### Primary (HIGH confidence)
- Hypothesis docs (https://hypothesis.readthedocs.io) - recursive strategies, builds, assume
- Current codebase analysis - tree.py:176-281, test_tree.py

### Secondary (MEDIUM confidence)
- Mathematical analysis of level set preservation (self-derived)

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Hypothesis is well-documented, pytest already in use
- Architecture: HIGH - Depth-tracking is a standard pattern, math is clear
- Pitfalls: HIGH - Based on direct code analysis and common recursion mistakes

**Research date:** 2026-03-02
**Valid until:** 30 days (stable domain)
