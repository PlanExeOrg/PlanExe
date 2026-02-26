# Webhook Notifications — Implementation Plan

**Assignee:** Bubba  
**Feature:** 5.2 from MCP Interface Roadmap  
**Target:** PlanExeOrg/PlanExe repository

## Problem

Users must poll `plan_status` to know when a plan completes. This is inefficient for long-running plans and doesn't support CI/CD integrations.

## Proposed Solution

Add optional `webhook_url` parameter to `plan_create`. When the plan transitions to `completed` or `failed`, POST a JSON payload to that URL.

## Technical Scope

### Files to Modify

| File | Changes |
|------|---------|
| `mcp_cloud/schemas.py` | Add `webhook_url: Optional[str]` to `PlanCreateInput` |
| `mcp_cloud/handlers.py` | Pass `webhook_url` to plan creation; trigger webhook on completion |
| `worker_plan/worker_plan_api.py` | Emit event when plan completes (for webhook dispatch) |
| `mcp_cloud/webhooks.py` | NEW: Handle async webhook delivery with retry logic |

### Schema Change

```python
class PlanCreateInput(BaseModel):
    prompt: str
    model_profile: Optional[str] = "baseline"
    user_api_key: Optional[str] = None
    webhook_url: Optional[str] = None  # NEW
```

### Payload POSTed to webhook_url

```json
{
  "plan_id": "uuid",
  "state": "completed",
  "progress_percentage": 100,
  "created_at": "2026-02-26T12:00:00Z",
  "completed_at": "2026-02-26T12:15:00Z",
  "result": { ... },
  "error": null
}
```

### Implementation Steps

1. **Add schema:** Include `webhook_url` in `PlanCreateInput`
2. **Store webhook:** Persist `webhook_url` in `plan_metadata` column
3. **Emit event:** In worker, call webhook dispatcher when plan reaches terminal state
4. **Create dispatcher:** `webhooks.py` with POST + retry (3 attempts, exponential backoff)
5. **Log results:** Record webhook delivery status in `plan_metadata`
6. **Test:** Create plan with webhook_url, verify POST received

### Security Considerations

- Validate `webhook_url` is HTTPS (or localhost for dev)
- Add `webhook_secret` header for receiver validation
- Rate limit webhook dispatch to prevent abuse

### Edge Cases

- If webhook URL unreachable: log error, don't fail the plan
- If plan is stopped via `plan_stop`: optionally send "cancelled" state
- If user provides invalid URL: fail at plan creation with validation error

## Success Criteria

- `plan_create` accepts `webhook_url` parameter
- Plan completion triggers POST to URL within 30 seconds
- Retry logic handles transient failures (3 retries, exponential backoff)
- Webhook delivery status logged for debugging

## Effort Estimate

~4–5 hours  
PR type: implementation (not docs-only)

## Notes

- This can be done in parallel with Egon's SSE work (different files, no conflicts)
- Bubba should coordinate with Simon on whether webhook secrets are needed
