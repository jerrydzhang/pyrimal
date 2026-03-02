# Project Research Summary

**Project:** PhySO Adapter for Induced KL Divergence Symbolic Regression
**Domain:** Symbolic regression with custom fitness functions and level set preservation
**Researched:** 2026-03-02
**Confidence:** HIGH

## Executive Summary

This project integrates PhySO (Physics Symbolic Optimization) as an alternative to gplearn for symbolic regression with induced KL divergence as the fitness metric. PhySO offers RL-based expression search with dimensional analysis capabilities, which can discover better physical laws than genetic programming approaches. The research reveals PhySO supports custom reward functions through `reward_config` and `make_RewardsComputer()`, though this is less direct than gplearn's `make_fitness()` interface. Implementation requires creating a custom `learning_config` with a `rewards_computer` wrapper.

The recommended approach uses PhySO's existing extension points: `candidate_wrapper` for output transformation and `y_weights` for importance sampling, combined with direct reward customization via `reward_config["reward_function"]`. The key risk is ensuring tree simplification preserves level set topology—constant removal must be position-aware (only at root level) to avoid altering zero-crossing behavior. Property-based testing with Hypothesis is essential to verify simplification rules don't introduce bugs that change expression behavior.

## Key Findings

### Recommended Stack

PhySO 1.2.0 is the core technology for RL-based symbolic regression with dimensional analysis. It supports custom reward functions via `reward_config["reward_function"]` and `make_RewardsComputer()`, enabling integration with induced KL divergence fitness. PyTorch >= 1.11 is required for auto-differentiation in reward computation and free constant optimization. NumPy, SymPy, scikit-learn, pandas, and matplotlib are supporting libraries already present in the project.

**Core technologies:**
- **PhySO 1.2.0**: RL-based symbolic regression with dimensional analysis — only library combining deep RL with physics units constraints
- **PyTorch >= 1.11**: Auto-differentiation for reward computation — required for PhySO's gradient-based free constant optimization
- **NumPy >= 1.24, SymPy >= 1.12**: Data handling and expression manipulation — PhySO dependencies already in project

### Expected Features

The MVP focuses on tree simplification bugs that break level set preservation. Position-aware constant removal (only at root level) and position-aware monotonic removal (log, sqrt, exp only at root) are critical fixes. Property-based tests with Hypothesis verify simplification rules preserve level sets.

**Must have (table stakes):**
- Position-aware constant removal — users expect `f(x) + c → f(x)` at root to preserve zero topology
- Position-aware monotonic removal — log, sqrt, exp only removable at output edge
- Basic identity rules — remove `*1`, `/1`, `+0`, `-0` (already work)
- Property-based tests — Hypothesis tests proving level set preservation

**Should have (competitive):**
- Range-based conditional rules — allow `sqrt(f(x)) → f(x)` when f(x) ≥ 0
- Expression comparison — detect when two expressions have same level sets

**Defer (v2+):**
- Full SymPy integration — use SymPy for canonicalization after safe simplification
- Configurable aggressiveness — let users choose simplification level

### Architecture Approach

The adapter pattern separates PhySO integration from the GPLearn implementation. PhySOAdapter wraps `induced_kl_divergence` as a PhySO reward function using `make_RewardsComputer()`. The key architectural difference: gplearn uses `make_fitness(function)` directly, while PhySO requires creating a custom `learning_config` with `rewards_computer: make_RewardsComputer(reward_function=custom_reward)`.

**Major components:**
1. **PhySOAdapter** — wraps induced KL divergence as PhySO reward signal, provides `get_candidate_wrapper()` and `get_y_weights()`
2. **GPLearnAdapter** — existing adapter using `make_fitness()` for genetic programming
3. **ImportanceSampler** — combines multiple sampling strategies with balance heuristic
4. **induced_kl_divergence** — core fitness measuring distribution alignment

### Critical Pitfalls

The most critical pitfall is context-blind constant removal: removing constants from anywhere in the tree changes input to nested functions, breaking level set preservation. Only remove `+c` and `*c` at the root output edge. Monotonic unary operations like log, sqrt, exp don't preserve zeros (log maps x=1 to 0, not x=0), so they're only removable at root level where the shift doesn't matter for early stopping.

1. **Context-blind constant removal** — track "output proximity" during traversal, only remove constants when no further operations compose on top
2. **Monotonic unary removal** — only remove operations where g(0) = 0 AND g⁻¹(0) = {0}; at root level, log/sqrt/exp are acceptable for topology preservation
3. **Sampling-based monotonicity** — avoid sampling to "prove" monotonicity; be conservative or require symbolic proof
4. **Division domain issues** — `x / x → 1` only valid where x ≠ 0; track domain restrictions or don't simplify
5. **Identical subtree bugs** — `x + x → x` is NOT level-set preserving; only `x - x → 0` is safe

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Fix Tree Simplification (Critical Bug)
**Rationale:** The current tree simplification breaks level set preservation by removing constants context-blindly and removing monotonic operations unconditionally. This is a correctness issue that must be fixed before PhySO integration can be trusted.
**Delivers:** Position-aware constant and monotonic removal rules, property-based tests for level set preservation
**Addresses:** All table stakes features from FEATURES.md
**Avoids:** Context-blind constant removal, wrong unary removal rules (PITFALLS.md #1, #2)

### Phase 2: PhySO Adapter Implementation
**Rationale:** PhySO integration is the core deliverable. Requires implementing the reward function adapter pattern, which is more complex than gplearn's direct fitness approach.
**Delivers:** PhySOAdapter with custom KL divergence reward, basic PhySO experiment runner
**Uses:** PhySO 1.2.0, PyTorch (STACK.md)
**Implements:** PhySOAdapter component (ARCHITECTURE.md)

### Phase 3: Evaluation & Comparison
**Rationale:** Validate PhySO adapter produces better results than gplearn on induced KL divergence objective. Requires benchmarking on Feynman datasets and comparing pareto fronts.
**Delivers:** Benchmark comparison framework, performance analysis reports
**Avoids:** Treating PhySO like GPLearn (PITFALLS.md anti-pattern)

### Phase 4: Advanced Features (Optional)
**Rationale:** Once core functionality works, add range-based conditional rules and expression comparison for better simplification quality.
**Delivers:** Range-based conditional simplification, expression equivalence testing
**Implements:** Should-have features from FEATURES.md (P2 items)

### Phase Ordering Rationale

- **Phase 1 first**: Tree simplification is currently broken; PhySO integration will be unreliable if simplification changes expression behavior during early stopping
- **Phase 2 second**: PhySO adapter is the core deliverable and depends on correct tree simplification for valid comparisons
- **Phase 3 third**: Evaluation validates the approach and informs whether Phase 4 (advanced features) is worthwhile
- **Phase 4 last**: Advanced features are optional improvements, not required for basic functionality

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 2:** PhySO reward configuration nuances — the custom reward mechanism exists but is less direct than gplearn; may need to experiment with different wrapper patterns
- **Phase 3:** Benchmark dataset selection — need to identify which Feynman equations benefit most from KL divergence objective vs standard MSE

Phases with standard patterns (skip research-phase):
- **Phase 1:** Tree simplification patterns are well-documented in computer algebra literature; SymPy's implementation provides reference
- **Phase 4:** Advanced simplification features are standard in symbolic computation; patterns from SymPy simplification module apply

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | PhySO source code verified on GitHub; reward.py confirms custom reward support via `reward_function` parameter |
| Features | HIGH | Mathematical principles of level sets are well-established; SymPy documentation confirms canonical form approaches |
| Architecture | MEDIUM | ARCHITECTURE.md incorrectly claimed PhySO has no direct reward interface; corrected through source code inspection |
| Pitfalls | HIGH | Based on actual code analysis of `/home/jerry/Research/code/pyrimel/src/primel/tree.py`; mathematical principles are sound |

**Overall confidence:** HIGH

### Gaps to Address

- **PhySO reward integration complexity:** The custom reward mechanism exists but requires creating custom `learning_config` and `run_config` dictionaries. The exact pattern for induced KL divergence reward may need experimentation during implementation.
- **Tree simplification testing coverage:** Property-based tests are recommended but no specific test strategy is documented; need to design Hypothesis generators that create edge cases for level set preservation.
- **Performance impact:** Position-aware simplification may be slower than current implementation; performance impact unknown until Phase 1 is complete.

## Sources

### Primary (HIGH confidence)
- **PhySO reward.py source** — verified `reward_function` parameter in `RewardsComputer()` and `make_RewardsComputer()`
- **PhySO config0.py source** — confirmed `reward_config` structure and `make_RewardsComputer(**reward_config)` pattern
- **PhySO official docs** — https://physo.readthedocs.io/en/latest/r_sr.html
- **SymPy simplification docs** — https://docs.sympy.org/latest/modules/simplify/simplify.html
- **Code analysis** — `/home/jerry/Research/code/pyrimel/src/primel/tree.py` — actual simplification implementation

### Secondary (MEDIUM confidence)
- **PhySO GitHub** — https://github.com/WassimTenachi/PhySO
- **Level set mathematical principles** — Wikipedia and mathematical literature on monotonic functions and topology preservation
- **gplearn docs** — https://gplearn.readthedocs.io/

### Tertiary (LOW confidence)
- **PhySO papers** — arXiv:2303.03192 (SR), arXiv:2312.01816 (Class SR) — methodology papers, not API documentation

---
*Research completed: 2026-03-02*
*Ready for roadmap: yes*
