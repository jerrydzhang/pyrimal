# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-02)

**Core value:** Correct level set preservation during tree simplification - expressions must maintain their zero-set topology through simplification so early stopping doesn't trigger false positives from spiky artifacts.
**Current focus:** Phase 2 - PhySO Adapter

## Current Position

Phase: 2 of 2 (PhySO Adapter)
Plan: 1 of 3 in current phase
Status: Plan 01 complete, continuing Phase 2
Last activity: 2026-03-03 — Completed 02-01 PhySO adapter plan

Progress: [██████░░░░░] 66% (Phase 1 complete, 1/3 plans in Phase 2 complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 3
- Average duration: 6.3 min
- Total execution time: 0.3 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| Phase 01-tree-simplification-fix | 2 | 19 min | 9.5 min |
| Phase 02-physo-adapter | 1 | 2 min | 2 min |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Roadmap: Simplification first approach - broken simplification blocks PhySO work
- Testing: Hypothesis for property-based testing to verify level set preservation
- [Phase 01-tree-simplification-fix]: Constants and monotonic ops only removed at root (depth == 0) — Removing nested constants/monotonic ops changes where expressions equal zero, breaking level set topology for early stopping detection
- [Phase 01-tree-simplification-fix]: Removed x + x rule and trig/square sampling checks — x + x not level-set preserving; trig/square sampling fragile and data-dependent per research
- [Phase 01-tree-simplification-fix]: Level set preservation means if f(x1)==f(x2), then simplify(f)(x1)==simplify(f)(x2) — Equivalence classes preserved; NaN→real values acceptable (domain extension)
- [Phase 02-physo-adapter]: PhySOAdapter uses ALL samples from ImportanceSampler (training + local + global regions), not just training points — Matches Julia reference implementation for fitness computation across full distribution
- [Phase 02-physo-adapter]: GECCO weights computed internally by induced_kl_divergence; sampler.weights NOT passed to PhySO — Avoids double-weighting since GECCO weights already capture importance via reference PDF
- [Phase 02-physo-adapter]: No get_y_weights() method in PhySOAdapter — Would cause double-weighting with GECCO weights computed internally by induced_kl_divergence
- [Phase 02-physo-adapter]: Reward function validates len(y_pred) == len(sampler.samples) — Critical for catching sampler/expression mismatch bugs early

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-03-03
Stopped at: Completed 02-01-PLAN.md (PhySOAdapter dataclass with reward config and unit tests)
Resume file: None
