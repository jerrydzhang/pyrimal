---
phase: 01-tree-simplification-fix
plan: 01
subsystem: simplification
tags: [tree-simplification, level-set-topology, position-aware, depth-tracking]

# Dependency graph
requires:
  - phase: none
    provides: Initial ExpressionTree and simplify_tree implementation
provides:
  - Position-aware simplify_tree with depth parameter and depth-gated simplification rules
  - Test suite covering constant/monotonic removal at root vs nested positions
affects:
  - 01-tree-simplification-fix/01-02
  - 02-physo-adapter/*

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Depth-tracking recursive simplification
    - Root-level guards for topology-preserving rules
    - Identity rules at any depth

key-files:
  created: []
  modified:
    - src/primel/tree.py - Added depth parameter to simplify_tree
    - tests/test_tree.py - Added position-aware test cases

key-decisions:
  - Constants (+c, *c) only removed at root (depth == 0) to preserve level set topology
  - Monotonic ops (log, sqrt, exp) only removed at root (depth == 0) to preserve level set topology
  - Removed x + x rule entirely (not level-set preserving)
  - Removed trig/square sampling-based checks (fragile, data-dependent)
  - Identity rules (x-x, x/x) work at any depth (correctness rules)

patterns-established:
  - Pattern: Depth parameter in recursive tree operations to track position from root
  - Pattern: Depth-gated simplification rules for topology preservation
  - Pattern: Separate correctness rules (identity) from topology rules (constants/monotonic)

requirements-completed: [SIMP-01, SIMP-02, SIMP-03, SIMP-05, SIMP-06, TEST-01, TEST-02]

# Metrics
duration: 4 min
completed: 2026-03-02T22:56:29Z
---

# Phase 1 Plan 1: Position-Aware Tree Simplification Summary

**Position-aware tree simplification with depth tracking to preserve level set topology, preventing spurious zero sets from early stopping false positives**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-02T22:52:12Z
- **Completed:** 2026-03-02T22:56:29Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Implemented depth-tracking recursive simplification in `simplify_tree`
- Added depth-gated rules: constants and monotonic ops only removed at root (depth == 0)
- Removed x + x rule (not level-set preserving) and trig/square sampling-based checks
- Preserved identity rules (x-x, x/x) at any depth for correctness
- Added comprehensive test suite for position-aware behavior (TEST-01, TEST-02)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add depth parameter to simplify_tree** - `b34abee` (feat)
2. **Task 2: Add unit tests for position-aware behavior** - `0ec6b8a` (test)

**Plan metadata:** (pending)

## Files Created/Modified

- `src/primel/tree.py` - Modified simplify_tree to track depth through recursion, gated constant/monotonic removal with depth == 0 checks, removed x+x rule and trig/square sampling
- `tests/test_tree.py` - Added test_constant_removal_at_root, test_constant_removal_nested, test_monotonic_removal_at_root, test_monotonic_removal_nested, test_identity_rules_at_any_depth, test_x_plus_x_rule_removed, updated existing tests for new behavior

## Decisions Made

- Constants (+c, *c) only removed at root level: Removing nested constants changes where expressions equal zero, breaking level set topology
- Monotonic ops (log, sqrt, exp) only removed at root level: Shifts from nested monotonic ops affect zero locations
- Removed x + x rule: While 2x doesn't change zeros, the rule was removed because it doesn't preserve level sets in all cases (not level-set preserving)
- Removed trig/square sampling checks: Sampling-based rules are fragile and data-dependent per research findings
- Identity rules at any depth: x-x -> 0 and x/x -> 1 are correctness rules that work anywhere

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Position-aware simplification complete and tested
- Ready for Plan 02 (add tree.copy() method for safe tree duplication)
- No blockers or concerns

## Self-Check: PASSED

- SUMMARY.md created and exists at `.planning/phases/01-tree-simplification-fix/01-01-SUMMARY.md`
- Task commit `b34abee` exists
- Task commit `0ec6b8a` exists
- All tests pass (21/21)
- STATE.md updated with decisions
- ROADMAP.md updated with plan progress
- REQUIREMENTS.md marked requirements complete

---
*Phase: 01-tree-simplification-fix*
*Completed: 2026-03-02*
