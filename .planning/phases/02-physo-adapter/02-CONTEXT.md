# Phase 2: PhySO Adapter - Context

**Gathered:** 2026-03-02
**Status:** Ready for planning

<domain>
## Phase Boundary

PhySO integration with induced KL divergence reward signal. The adapter wraps `induced_kl_divergence` as PhySO's reward via `make_RewardsComputer()`, passes ImportanceSampler samples via `y_weights`, and provides validation tooling. Creating the core fitness function and sampling infrastructure are separate (already exist).

</domain>

<decisions>
## Implementation Decisions

### Validation format
- Python script (not marimo notebook)
- Uses zhong dataset by default, CLI-configurable for other datasets
- Demonstrates basic end-to-end: runs adapter, outputs discovered expression and KL score
- Minimal output (expression + score, no verbose logging)
- ALSO include jernerics experiment alongside validation script
- Jernerics experiment follows existing gplearn pattern for HPC deployment

### Error handling
- PhySO-specific errors: propagate (let exceptions bubble up with context)
- NaN/Inf in predictions: let it raise naturally
- ImportanceSampler: no special handling needed (robust as-is)
- No logging infrastructure needed

### API ergonomics
- OpenCode's discretion on exact API design
- Not tied to GPLearnAdapter pattern — go with most ergonomic approach
- Research item: confirm relationship between sampler.weights and PhySO's y_weights
  - Check how gplearn adapter currently handles this
  - Likely: weights only used for KL divergence calculation, not passed to training

### Testing scope
- Use real PhySO in tests (mocking doesn't test meaningful integration)
- Test what is needed — pragmatic approach
- Cover both input validation and numerical edge cases
- Tests live in tests/ directory (e.g., tests/test_physo_adapter.py)

### OpenCode's Discretion
- Exact adapter class structure (dataclass vs regular class vs factory)
- Parameter surface and naming
- How to expose reward to PhySO (method returning callable, full config, etc.)
- Exact test file organization within tests/

</decisions>

<specifics>
## Specific Ideas

- "I am not sold on the gplearn adapter pattern — go with whatever seems most ergonomic"
- Jernerics library handles HPC deployment as long as you follow the example created for gplearn

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-physo-adapter*
*Context gathered: 2026-03-02*
