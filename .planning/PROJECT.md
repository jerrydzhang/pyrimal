# Pyrimel PhySO Integration

## What This Is

A symbolic regression library that uses importance sampling and KL divergence-based fitness to discover physical equations. Currently supports gplearn; adding PhySO support and fixing tree simplification logic.

## Core Value

Correct level set preservation during tree simplification - expressions must maintain their zero-set topology through simplification so early stopping doesn't trigger false positives from spiky artifacts.

## Requirements

### Validated

- ✓ GPLearn adapter with induced_kl_divergence fitness function — existing
- ✓ ExpressionTree representation with evaluation — existing
- ✓ Importance sampling with multiple distribution strategies — existing
- ✓ Early stopping based on train/validation variance split — existing
- ✓ Basic tree simplification rules — existing (but flawed)

### Active

- [ ] Fix tree simplification to preserve level sets correctly
- [ ] Property-based tests (Hypothesis) to verify level set preservation
- [ ] PhySO adapter using induced_kl_divergence as reward signal
- [ ] Integration tests for PhySO adapter with existing experiment framework

### Out of Scope

- Changes to the core fitness function algorithm — working as intended
- New distribution types — not needed for this work
- UI/visualization improvements — out of scope

## Context

**Current state:**
- GPLearn adapter wraps `induced_kl_divergence` as a custom fitness metric
- Tree simplification removes monotonic operations (log, sqrt, exp) and constant operations (add, sub, mul, div with constants)
- Early stopping checks if training variance is low but custom-sampled validation points have high variance
- Spiky expressions from over-aggressive simplification trigger false early stops

**Problem with current simplification:**
- Rules are too aggressive and context-insensitive
- `f(x) + 5` → `f(x)` is safe (constant on output)
- `f(x+5)` → `f(x)` is NOT safe (constant affects input to nested functions)
- Need to track whether a constant is modifying the final output vs being an intermediate value

**PhySO integration:**
- PhySO uses deep reinforcement learning for symbolic regression
- Need to replace its default reward with `induced_kl_divergence`
- Similar adapter pattern to GPLearn

## Constraints

- **Level set preservation**: Simplification must not change zero-set topology (level curves up to constant offset)
- **PhySO compatibility**: Adapter must work with PhySO's RL training loop
- **Test coverage**: Property-based tests must prove correctness, not just examples

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Hypothesis for testing | Property-based testing discovers edge cases we'd miss with example tests | — Pending |
| Simplification first | Broken simplification causes false early stops, blocking PhySO work | — Pending |
| Context-aware simplification | Track position in tree to know if constant is on output vs intermediate | — Pending |

---
*Last updated: 2026-03-02 after initialization*
