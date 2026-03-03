---
phase: 02-physo-adapter
plan: 02
subsystem: experiments
tags: [physo, experiment, validation, jernerics]

# Dependency graph
requires:
  - phase: 02-01
    provides: PhySOAdapter with KL divergence reward
provides:
  - PhySOExperiment class following jernerics Experiment pattern for HPC deployment
  - Validation script for end-to-end PhySO adapter testing
affects: 02-physo-adapter-03

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Experiment pattern: Dataclass with setup_data, train, evaluate methods
    - Multi-component sampler: train + kde + uniform for comprehensive sampling
    - PhySO run_config structure: reward_config, learning_config, priors_config, cell_config, free_const_opti_args

key-files:
  created:
    - experiments/physo_experiment.py
    - scripts/validate_physo.py
  modified:
    - src/primel/adapters/physo/adapter.py

key-decisions:
  - "Used zhong/f01/f01.csv as default data file (zhong/constant.csv does not exist)"
  - "Updated PhySOAdapter.get_learning_config() to return complete run_config with all required keys for PhySO.SR"

patterns-established:
  - "PhySO experiment pattern follows gplearn_experiment.py structure with jernerics Experiment base class"
  - "Multi-component ImportanceSampler with explicit sample counts per component"
  - "Validation script pattern with CLI arguments and minimal output"

requirements-completed: [PHYS-03, PHYS-04, TEST-04]

# Metrics
duration: 12min
completed: 2026-03-03T01:30:33Z
---

# Phase 02 Plan 02: PhySO Experiment Runner and Validation Script Summary

**PhySOExperiment class following jernerics Experiment pattern for HPC deployment, multi-component sampler (train + kde + uniform), and validation script for end-to-end PhySO adapter testing**

## Performance

- **Duration:** 12 min
- **Started:** 2026-03-03T01:18:21Z
- **Completed:** 2026-03-03T01:30:33Z
- **Tasks:** 3 (2 main tasks + 1 bug fix)
- **Files modified:** 3

## Accomplishments

- PhySOExperiment class created following gplearn_experiment.py pattern for jernerics HPC deployment
- Multi-component ImportanceSampler with train + kde + uniform components for comprehensive sampling
- PhySOAdapter integration with KL divergence reward through run_config
- _compute_metrics method calculates KL divergence for discovered expressions
- evaluate method aggregates metrics across trials
- get_experiment factory function for jernerics compatibility
- Local validation script with CLI arguments (--data, --epochs, --seed)
- Validation script prints sample breakdown, discovered expression, and KL score
- Minimal output format per user decision (no verbose logging)

## task Commits

Each task was committed atomically:

1. **task 1: Create PhySO experiment runner** - `b468f6c` (feat)
2. **task 2: Create PhySO validation script** - `98098b5` (feat)
3. **Fix: PhySOAdapter returns complete run_config** - `c80729b` (fix)

## Files Created/Modified

- `experiments/physo_experiment.py` - PhySOExperiment class with setup_data, train, _compute_metrics, evaluate, and get_experiment factory
- `scripts/validate_physo.py` - Local validation script with CLI interface and minimal output
- `src/primel/adapters/physo/adapter.py` - Updated get_learning_config() to return complete run_config with reward_config, learning_config, priors_config, cell_config, and free_const_opti_args

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] PhySOAdapter.get_learning_config() incomplete for PhySO.SR**
- **Found during:** task 2 (testing validation script)
- **Issue:** PhySO.SR requires complete run_config dict with reward_config, learning_config, priors_config, cell_config, and free_const_opti_args keys. The adapter only returned learning_config with rewards_computer, causing KeyError.
- **Fix:** Updated get_learning_config() to return full run_config dict with all required keys:
  - reward_config: reward_function and related settings
  - learning_config: rewards_computer and learning parameters (batch_size, max_time_step, n_epochs, gamma_decay, entropy_weight, risk_factor, get_optimizer, observe_units)
  - priors_config: HardLengthPrior
  - cell_config: network architecture (hidden_size, is_lobotomized, n_layers)
  - free_const_opti_args: constant optimization settings
- **Files modified:** src/primel/adapters/physo/adapter.py
- **Verification:** PhySO.SR successfully runs with adapter config, completes training epochs, returns discovered expression
- **Committed in:** c80729b (task 2 fix commit)

**2. [Rule 3 - Blocking] Used existing data file instead of non-existent zhong/constant.csv**
- **Found during:** task 1 (creating experiment runner)
- **Issue:** Plan specified default data_file as "zhong/constant.csv" but this file does not exist in data directory
- **Fix:** Used "zhong/f01/f01.csv" as default in both experiment and validation script
- **Files modified:** experiments/physo_experiment.py, scripts/validate_physo.py
- **Verification:** Both files load data successfully from zhong/f01/f01.csv
- **Committed in:** b468f6c (task 1 commit), 98098b5 (task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 bug, 1 blocking)
**Impact on plan:** Both fixes essential for correctness and functionality. No scope creep.

## Issues Encountered

None - all issues resolved via auto-fix rules.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

PhySO experiment runner and validation script complete and tested. Ready for next plan in phase (02-03: Additional PhySO features or next phase).

The experiment correctly:
- Follows jernerics Experiment pattern for HPC deployment
- Creates multi-component sampler (train + kde + uniform) with explicit sample counts
- Integrates PhySOAdapter with complete run_config
- Runs physo.SR and computes KL divergence metrics
- Supports configurable parameters (epochs, bandwidth, n_kde, n_uniform, n_trials, lambda_, exponent)

The validation script correctly:
- Accepts CLI arguments (--data, --epochs, --seed)
- Uses zhong dataset by default (zhong/f01/f01.csv)
- Creates multi-component sampler with explicit configuration
- Prints sample breakdown, discovered expression, and KL score
- Has minimal output format (no verbose logging)

No blockers or concerns identified.

---
*Phase: 02-physo-adapter*
*Completed: 2026-03-03*

## Self-Check: PASSED

All created files verified:
- experiments/physo_experiment.py ✓
- scripts/validate_physo.py ✓

All commits verified:
- b468f6c (feat: create PhySO experiment runner) ✓
- 98098b5 (feat: create PhySO validation script) ✓
- c80729b (fix: PhySOAdapter returns complete run_config) ✓
