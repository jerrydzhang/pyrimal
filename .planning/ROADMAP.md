# Roadmap: Pyrimel PhySO Integration

## Overview

This roadmap delivers correct tree simplification (preserving level set topology) followed by PhySO adapter integration with induced KL divergence as the reward signal. The simplification fix is foundational—without it, early stopping triggers false positives from spiky artifacts, making PhySO results unreliable.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Tree Simplification Fix** - Position-aware simplification preserving level set topology (completed 2026-03-02)
- [ ] **Phase 2: PhySO Adapter** - PhySO integration with induced KL divergence reward

## Phase Details

### Phase 1: Tree Simplification Fix
**Goal**: Tree simplification preserves level set topology during expression manipulation
**Depends on**: Nothing (first phase)
**Requirements**: SIMP-01, SIMP-02, SIMP-03, SIMP-04, SIMP-05, SIMP-06, TEST-01, TEST-02
**Success Criteria** (what must be TRUE):
  1. Constants (`+c`, `*c`) only removed at root/output edge, never in nested positions
  2. Monotonic operations (log, sqrt, exp) only removed at root level where shift doesn't affect zeros
  3. Basic identity rules (`*1`, `/1`, `+0`, `-0`) still work everywhere
  4. Property-based tests (Hypothesis) prove level set preservation across random expression trees
  5. `x + x → x` rule removed or fixed (NOT level-set preserving)
**Plans**: 2 plans

Plans:
- [x] 01-01: Position-aware simplification with depth tracking (SIMP-01, SIMP-02, SIMP-05, SIMP-06, TEST-01, TEST-02)
- [x] 01-02: Property-based testing with Hypothesis (SIMP-04)

### Phase 2: PhySO Adapter
**Goal**: PhySO can use induced KL divergence as its reward signal for symbolic regression
**Depends on**: Phase 1
**Requirements**: PHYS-01, PHYS-02, PHYS-03, PHYS-04, PHYS-05, TEST-03, TEST-04
**Success Criteria** (what must be TRUE):
  1. PhySOAdapter class wraps `induced_kl_divergence` as PhySO reward via `make_RewardsComputer()`
  2. ImportanceSampler samples pass through to PhySO training via `y_weights`
  3. Local validation script/notebook demonstrates end-to-end PhySO training with KL divergence fitness
  4. Unit tests verify reward computation matches expected behavior
**Plans**: TBD

Plans:
- [ ] 02-01: Implement PhySOAdapter with custom reward wrapper
- [ ] 02-02: Add sampler integration and validation script

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Tree Simplification Fix | 2/2 | Complete | 2026-03-02 |
| 2. PhySO Adapter | 0/2 | Not started | - |

---
*Roadmap created: 2026-03-02*
*Total v1 requirements: 15*
