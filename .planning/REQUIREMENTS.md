# Requirements: Pyrimel PhySO Integration

**Defined:** 2026-03-02
**Core Value:** Correct level set preservation during tree simplification - expressions must maintain their zero-set topology so early stopping doesn't trigger false positives

## v1 Requirements

### Tree Simplification

- [x] **SIMP-01**: Position-aware constant removal — only remove `+c`, `*c` at root/output edge
- [x] **SIMP-02**: Position-aware monotonic removal — only remove log/sqrt/exp at root level
- [x] **SIMP-03**: Basic identity rules preserved — `*1`, `/1`, `+0`, `-0` removal still works
- [x] **SIMP-04**: Property-based tests with Hypothesis proving level set preservation
- [x] **SIMP-05**: Remove or fix `x + x → x` rule (NOT level-set preserving)
- [x] **SIMP-06**: Fix sampling-based monotonicity checks (fragile, data-dependent)

### PhySO Adapter

- [ ] **PHYS-01**: PhySOAdapter class following gplearn adapter pattern
- [ ] **PHYS-02**: Wrap `induced_kl_divergence` as PhySO reward via `make_RewardsComputer()`
- [ ] **PHYS-03**: Basic PhySO experiment runner
- [ ] **PHYS-04**: Local test script or marimo notebook for PhySO adapter validation
- [ ] **PHYS-05**: Sampler integration — pass ImportanceSampler samples to PhySO

### Testing

- [x] **TEST-01**: Unit tests for position-aware constant removal
- [x] **TEST-02**: Unit tests for position-aware monotonic removal
- [ ] **TEST-03**: Unit tests for PhySO adapter fitness computation
- [ ] **TEST-04**: Local validation script in place of full integration tests

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Advanced Simplification

- **SIMP-07**: Range-based conditional rules — allow `sqrt(f(x)) → f(x)` when f(x) ≥ 0
- **SIMP-08**: Smart trig handling — allow `sin(f(x)) → f(x)` when |f(x)| < π/2
- **SIMP-09**: Expression comparison — detect when two expressions have same level sets

### Enhanced PhySO

- **PHYS-06**: Dimensional analysis integration with importance sampling
- **PHYS-07**: Multi-dataset (Class SR) support
- **PHYS-08**: Benchmark comparison framework against gplearn

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Changes to induced_kl_divergence algorithm | Core fitness function works as intended |
| New distribution types | Not needed for this work |
| UI/visualization improvements | Out of scope for library work |
| Global constant removal (`f(x+c) → f(x)` everywhere) | Breaks level set preservation |
| Unconditional log/exp/sqrt removal | Changes input to nested functions, breaks zeros |
| `x + x → x` simplification | NOT level-set preserving |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| SIMP-01 | Phase 1: Tree Simplification Fix | Complete |
| SIMP-02 | Phase 1: Tree Simplification Fix | Complete |
| SIMP-03 | Phase 1: Tree Simplification Fix | Complete |
| SIMP-04 | Phase 1: Tree Simplification Fix | Complete |
| SIMP-05 | Phase 1: Tree Simplification Fix | Complete |
| SIMP-06 | Phase 1: Tree Simplification Fix | Complete |
| TEST-01 | Phase 1: Tree Simplification Fix | Complete |
| TEST-02 | Phase 1: Tree Simplification Fix | Complete |
| PHYS-01 | Phase 2: PhySO Adapter | Pending |
| PHYS-02 | Phase 2: PhySO Adapter | Pending |
| PHYS-03 | Phase 2: PhySO Adapter | Pending |
| PHYS-04 | Phase 2: PhySO Adapter | Pending |
| PHYS-05 | Phase 2: PhySO Adapter | Pending |
| TEST-03 | Phase 2: PhySO Adapter | Pending |
| TEST-04 | Phase 2: PhySO Adapter | Pending |

**Coverage:**
- v1 requirements: 15 total
- Mapped to phases: 15
- Unmapped: 0 ✓

**Phase Summary:**
- Phase 1 (Tree Simplification Fix): 8 requirements (SIMP-01 to SIMP-06, TEST-01, TEST-02)
- Phase 2 (PhySO Adapter): 7 requirements (PHYS-01 to PHYS-05, TEST-03, TEST-04)

---
*Requirements defined: 2026-03-02*
*Last updated: 2026-03-02 after initial definition*
