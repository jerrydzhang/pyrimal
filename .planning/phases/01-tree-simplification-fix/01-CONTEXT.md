# Phase 1: Tree Simplification Fix - Context

**Gathered:** 2026-03-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Tree simplification preserves level set topology during expression manipulation. The core requirement is rigid and universal: simplifications must not violate the math. All rules must preserve where expressions equal zero.

</domain>

<decisions>
## Implementation Decisions

### Mathematical Strictness
- All simplification rules must preserve level set topology (where f(x) = 0)
- No flexibility on correctness — this is a rigid mathematical requirement
- Position-aware rules: constants (+c, *c) and monotonic ops (log, sqrt, exp) only removable at root/output edge

### x+x Rule Handling
- **Decision deferred to research** — researcher must investigate:
  1. Is there a mathematically safe position-aware fix for `x + x → x`?
  2. If not, remove the rule entirely
- ROADMAP specifies this rule is NOT level-set preserving as currently implemented

### OpenCode's Discretion
- Property-based test coverage breadth (Hypothesis)
- Error message detail level
- Backward compatibility approach for non-breaking cases

</decisions>

<specifics>
## Specific Ideas

- "Do whatever simplifications just make sure they do not violate the math" — user emphasis on mathematical correctness over convenience

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 01-tree-simplification-fix*
*Context gathered: 2026-03-02*
