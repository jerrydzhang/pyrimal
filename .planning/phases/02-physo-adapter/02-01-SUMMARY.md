---
phase: 02-physo-adapter
plan: 01
subsystem: adapters
tags: [physo, symbolic-regression, adapter-pattern, reward-function]

# Dependency graph
requires:
  - phase: 01-tree-simplification-fix
    provides: induced_kl_divergence function for fitness computation
provides:
  - PhySOAdapter dataclass that wraps induced_kl_divergence as PhySO's reward signal
  - Unit tests verifying adapter initialization and reward computation
affects: 02-physo-adapter-02, 02-physo-adapter-03

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Adapter pattern: Dataclass with sampler, reference distribution, and fitness parameters
    - Reward transformation: KL divergence (lower = better) → reward (higher = better) via 1/(1+x)
    - Lazy imports: physo submodule loaded on-demand via __getattr__

key-files:
  created:
    - src/primel/adapters/physo/__init__.py
    - src/primel/adapters/physo/adapter.py
    - tests/adapters/test_physo.py
  modified:
    - src/primel/adapters/__init__.py

key-decisions:
  - "Reward function uses ALL samples from ImportanceSampler (training + local + global regions), not just training points"
  - "GECCO weights are computed internally by induced_kl_divergence; sampler.weights NOT passed to PhySO to avoid double-weighting"
  - "Reward validation ensures PhySO provides predictions for all sampled points (len(y_pred) == len(sampler.samples))"
  - "No get_y_weights() method created (would cause double-weighting with GECCO weights)"

patterns-established:
  - "PhySO adapter pattern: dataclass with get_reward_config() and get_learning_config() methods"
  - "Reward function signature: (y_target, y_pred, y_weights) -> float in [0,1] range"
  - "Torch to numpy conversion at adapter boundary for PhySO compatibility"

requirements-completed: [PHYS-01, PHYS-02, PHYS-05, TEST-03]

# Metrics
duration: 2min
completed: 2026-03-03T01:15:30Z
---

# Phase 02 Plan 01: PhySOAdapter Summary

**PhySOAdapter dataclass wrapping induced_kl_divergence as PhySO's reward signal via make_RewardsComputer(), with unit tests validating reward computation, length validation, and multi-component sampler behavior**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-03T01:13:18Z
- **Completed:** 2026-03-03T01:15:30Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- PhySOAdapter dataclass created with sampler, reference_distribution, lambda_, exponent, and mean_center_on parameters
- Reward function wraps induced_kl_divergence and transforms KL divergence (lower = better) to reward (higher = better) via 1/(1+x) transformation
- Reward function validates prediction count matches sampler sample count to catch sampler/expression mismatch bugs
- Reward function handles torch tensor to numpy conversion for PhySO compatibility
- get_learning_config() returns PhySO's rewards_computer via make_RewardsComputer()
- Docstring documents sampling architecture (uses ALL samples from ImportanceSampler, not just training)
- Unit tests verify initialization, reward config, learning config, length validation, torch tensor conversion, and multi-component sampler behavior

## task Commits

Each task was committed atomically:

1. **task 1: Create PhySOAdapter dataclass with reward config** - `89c5cfa` (feat)
2. **task 2: Add unit tests for PhySOAdapter** - `cd555b0` (test)

**Plan metadata:** `lmn012o` (docs: complete plan)

## Files Created/Modified

- `src/primel/adapters/physo/__init__.py` - Module init exporting PhySOAdapter
- `src/primel/adapters/physo/adapter.py` - PhySOAdapter dataclass with get_reward_config() and get_learning_config()
- `src/primel/adapters/__init__.py` - Added lazy import for physo submodule
- `tests/adapters/test_physo.py` - Unit tests for adapter initialization, reward config, learning config, length validation, torch tensor conversion, and multi-component sampler

## Decisions Made

- Reward function uses ALL samples from ImportanceSampler (training + local + global regions), not just training points, matching the Julia reference implementation behavior
- GECCO weights are computed internally by induced_kl_divergence based on reference distribution PDF values; sampler.weights are NOT passed to PhySO to avoid double-weighting
- No get_y_weights() method created because it would cause double-weighting with GECCO weights computed internally by induced_kl_divergence
- Length validation is critical for catching sampler/expression mismatch bugs early
- No logging added per user decision (no logging infrastructure needed)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

PhySOAdapter complete and tested. Ready for next plan in phase (02-02: PhySO experiment runner or 02-03: Validation script).

The adapter correctly:
- Wraps induced_kl_divergence as PhySO's reward signal
- Uses ALL samples from ImportanceSampler (training + local + global regions)
- Validates prediction count matches sampler sample count
- Converts torch tensors to numpy for PhySO compatibility
- Returns reward values in [0, 1] range
- Handles multi-component samplers with train + local + global regions

No blockers or concerns identified.

---
*Phase: 02-physo-adapter*
*Completed: 2026-03-03*

## Self-Check: PASSED

All created files verified:
- src/primel/adapters/physo/__init__.py ✓
- src/primel/adapters/physo/adapter.py ✓
- tests/adapters/test_physo.py ✓

All commits verified:
- 89c5cfa (feat: create PhySOAdapter dataclass with reward config) ✓
- cd555b0 (test: add unit tests for PhySOAdapter) ✓
