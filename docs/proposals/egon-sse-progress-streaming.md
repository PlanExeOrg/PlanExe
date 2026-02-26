# SSE Progress Streaming — Implementation Plan

**Assignee:** Egon  
**Feature:** 5.1 from MCP Interface Roadmap  
**Target:** PlanExeOrg/PlanExe repository

## Problem

Users running long plans (10–20 minutes) get zero feedback until completion. They see only `"state": "processing"` with no visibility into what the agent is doing.

## Proposed Solution

Add a `log_lines` array to the `plan_status` response containing the last N lines of agent stdout/stderr (tail). This gives users live feedback without polling complexity.

## Technical Scope

### Files to Modify

| File | Changes |
|------|---------|
| `mcp_cloud/schemas.py` | Add `log_lines: list[str]` to `PlanStatusOutput` schema |
| `mcp_cloud/handlers.py` | Populate `log_lines` from agent output in `handle_plan_status` |
| `mcp_cloud/db_queries.py` | Possibly add helper to fetch tail from agent output table |
| `worker_plan/worker_plan_api.py` | Ensure agent stdout/stderr is captured to DB |

### Schema Change

```python
class PlanStatusOutput(BaseModel):
    plan_id: UUID
    state: PlanState
    progress_percentage: float
    created_at: datetime
    updated_at: datetime
    prompt_excerpt: str
    result: Optional[dict] = None
    error: Optional[dict] = None
    log_lines: list[str] = []  # NEW: last 50 lines of agent output
```

### Implementation Steps

1. **Verify output capture:** Confirm where agent stdout/stderr is stored (likely `agent_output` table or similar)
2. **Add DB query:** Create `_get_plan_log_tail(plan_id, lines=50)` in `db_queries.py`
3. **Update schema:** Add `log_lines` field to `PlanStatusOutput`
4. **Wire handler:** In `handle_plan_status`, fetch tail and populate field
5. **Test:** Verify field appears in `plan_status` response for running and completed plans

### Edge Cases

- If no output exists yet: return empty array `[]`
- If output is shorter than 50 lines: return all available
- Truncate individual lines at 500 chars to prevent huge payloads

## Success Criteria

- `plan_status` returns `log_lines: ["...", "..."]` with last 50 lines
- Works for both `processing` and `completed` states
- No performance impact on `plan_status` call (<50ms extra)
- Documented in MCP interface spec

## Effort Estimate

~2–3 hours  
PR type: implementation (not docs-only)
