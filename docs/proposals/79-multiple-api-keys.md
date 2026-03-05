# Proposal 79 — Multiple API Keys

## Status

Implemented (2026-03-05)

## Origin

From `docs/plan.md`:

> As of 2026-mar-02, the user can only have 1 api key.
> Now I have quite a few places where PlanExe is integrated, and
> I want to see stats on what api key is being used, and cost.
> Resetting the api key, and all the integrations dies.

---

## Problem

Today each user has at most one active `UserApiKey`. Regenerating that key revokes it — instantly breaking every integration that used it (MCP clients, CI pipelines, HTTP scripts). There is also no way to see which key triggered a plan or how much each integration is costing.

## Goals

1. Allow up to 10 active API keys per user.
2. Each key carries an optional human-readable label (e.g. "Claude Code", "CI pipeline").
3. Plans and credit ledger entries are attributed to the key that created them.
4. The account page shows per-key stats: plans created, credits used, last used.
5. Keys can be individually revoked without affecting the others.

## Non-Goals

- Per-key rate limiting or spending caps (future work).
- Key scoping / permissions (all keys have full account access).
- Renaming the database table `user_api_key` (it stays as-is).

---

## Design

### Database Schema Changes

All new columns are **nullable** so existing rows are unaffected and no data backfill is needed.

#### `user_api_key` table — add `label`

| Column | Type | Default | Purpose |
|--------|------|---------|---------|
| `label` | `VARCHAR(128)` | `NULL` | Human-readable name for the key |

In the model (`database_api/model_user_api_key.py`):

```python
# New column — optional human label for the key.
label = db.Column(db.String(128), nullable=True)
```

Display fallback: if `label` is NULL or empty, the UI shows `"Untitled"`.

#### `task_item` table — add `api_key_id`

| Column | Type | Default | Purpose |
|--------|------|---------|---------|
| `api_key_id` | `VARCHAR(36)` | `NULL` | UUID of the `UserApiKey` that created this plan |

In the model (`database_api/model_planitem.py`):

```python
# Which API key created this plan (NULL for legacy/frontend plans).
api_key_id = db.Column(db.String(36), nullable=True, index=True)
```

#### `credit_history` table — add `api_key_id`

| Column | Type | Default | Purpose |
|--------|------|---------|---------|
| `api_key_id` | `VARCHAR(36)` | `NULL` | UUID of the key whose plan incurred this charge |

In the model (`database_api/model_credit_history.py`):

```python
# Which API key's plan incurred this charge (NULL for purchases/legacy).
api_key_id = db.Column(db.String(36), nullable=True, index=True)
```

### Schema Migrations

Follow the existing `ALTER TABLE ADD COLUMN IF NOT EXISTS` pattern used throughout the codebase. A single new function runs on startup in all three services.

#### New function: `ensure_multi_api_key_columns()`

```python
def ensure_multi_api_key_columns() -> None:
    """Add columns for multi-API-key support (idempotent)."""
    statements = (
        "ALTER TABLE user_api_key ADD COLUMN IF NOT EXISTS label VARCHAR(128)",
        "ALTER TABLE task_item ADD COLUMN IF NOT EXISTS api_key_id VARCHAR(36)",
        "ALTER TABLE credit_history ADD COLUMN IF NOT EXISTS api_key_id VARCHAR(36)",
    )
    with db.engine.begin() as conn:
        for stmt in statements:
            try:
                conn.execute(text(stmt))
            except Exception as exc:
                logger.warning("Schema update failed for %s: %s", stmt, exc, exc_info=True)
```

**Where it is called** (three places, mirroring the existing pattern):

| File | Call site |
|------|-----------|
| `mcp_cloud/db_setup.py` | Module-level `with app.app_context():` block (line 76–77) — after `ensure_planitem_stop_columns()` |
| `worker_plan_database/app.py` | Inside `startup_worker()` (line 991) — after `ensure_fractional_credit_columns()` |
| `frontend_multi_user/src/app.py` | Inside `_setup_db()` (line 506) — after `_ensure_user_account_columns()` |

### Backend Auth — Return `api_key_id`

**File:** `mcp_cloud/auth.py` — `_resolve_user_from_api_key()`

Currently returns:

```python
return {
    "user_id": str(user.id),
    "credits_balance": float(user.credits_balance or 0),
}
```

Change to also include the key's UUID:

```python
return {
    "user_id": str(user.id),
    "credits_balance": float(user.credits_balance or 0),
    "api_key_id": str(api_key.id),
}
```

No callers inspect unknown keys in this dict, so adding a field is safe.

### Plan/Billing Attribution

#### `mcp_cloud/handlers.py` — `handle_plan_create()`

Currently passes metadata:

```python
{"user_id": str(user_context["user_id"])} if user_context else None
```

Change to also forward the key ID:

```python
{
    "user_id": str(user_context["user_id"]),
    "api_key_id": user_context.get("api_key_id"),
} if user_context else None
```

#### `mcp_cloud/db_queries.py` — `_create_plan_sync()`

Currently creates the PlanItem without `api_key_id`. Add:

```python
plan = PlanItem(
    prompt=prompt,
    state=PlanState.pending,
    user_id=metadata.get("user_id", "admin") if metadata else "admin",
    api_key_id=metadata.get("api_key_id") if metadata else None,   # ← new
    parameters=parameters,
)
```

Also include `api_key_id` in the `event_context` dict for observability.

#### `worker_plan_database/app.py` — `_charge_usage_credits_once()`

When creating the `CreditHistory` ledger entry (line 565–572), propagate `api_key_id` from the task:

```python
ledger = _new_model(
    CreditHistory,
    user_id=user.id,
    delta=-charged_credits,
    reason="plan_created_with_usage_cost" if success else "plan_failed_usage_cost",
    source="usage_billing",
    external_id=str(task_id),
    api_key_id=getattr(task, "api_key_id", None),   # ← new
)
```

### Frontend — Multi-Key Management

**File:** `frontend_multi_user/src/app.py`

#### `_get_or_create_api_key()` — accept optional label

Current signature:

```python
def _get_or_create_api_key(self, user: UserAccount) -> str:
```

New signature:

```python
def _get_or_create_api_key(self, user: UserAccount, label: Optional[str] = None) -> str:
```

Changes:
- Remove the early-return guard that checks for an existing active key (the whole point is to allow multiple keys).
- Instead, enforce a max-10-key limit:

```python
active_count = UserApiKey.query.filter_by(user_id=user.id, revoked_at=None).count()
if active_count >= 10:
    return ""
```

- Pass `label` when creating the `UserApiKey`:

```python
api_key = _new_model(
    UserApiKey,
    user_id=user.id,
    key_hash=key_hash,
    key_prefix=key_prefix,
    label=(label or "").strip()[:128] or None,
)
```

#### Account route — new POST actions

Replace the single `regenerate_api_key` action with two actions:

| Action | Behavior |
|--------|----------|
| `create_api_key` | Call `_get_or_create_api_key(user, label=request.form.get("label"))`. If the returned key is non-empty, store in session for one-time display. If empty (limit reached), flash an error. |
| `revoke_api_key` | Receive `key_id` from the form. Look up the `UserApiKey` by id, verify it belongs to the current user and is not already revoked. Set `revoked_at = now`. |

Keep the existing `regenerate_api_key` action working for backwards compatibility (it revokes all keys and creates a new one), but the UI will no longer send it.

#### Account route — per-key stats

Compute stats to pass to the template:

```python
active_keys = (
    UserApiKey.query
    .filter_by(user_id=user.id, revoked_at=None)
    .order_by(UserApiKey.created_at.asc())
    .all()
)

# Per-key plan counts
from sqlalchemy import func
plan_counts = dict(
    db.session.query(PlanItem.api_key_id, func.count(PlanItem.id))
    .filter(PlanItem.api_key_id.in_([str(k.id) for k in active_keys]))
    .group_by(PlanItem.api_key_id)
    .all()
)

# Per-key credit usage (sum of negative deltas)
credit_usage = dict(
    db.session.query(CreditHistory.api_key_id, func.sum(CreditHistory.delta))
    .filter(
        CreditHistory.api_key_id.in_([str(k.id) for k in active_keys]),
        CreditHistory.delta < 0,
    )
    .group_by(CreditHistory.api_key_id)
    .all()
)
```

Pass `active_keys`, `plan_counts`, `credit_usage`, and `can_create_key = len(active_keys) < 10` to the template.

#### First-login auto-create

The OAuth callback currently calls `_get_or_create_api_key(user)` which returns empty string if a key exists. With the guard removed, it would create a new key on every login. Fix: only call it if the user has zero active keys:

```python
has_key = UserApiKey.query.filter_by(user_id=user.id, revoked_at=None).first() is not None
if not has_key:
    new_api_key = self._get_or_create_api_key(user, label="Default")
    if new_api_key:
        session["new_api_key"] = new_api_key
```

### Frontend — Account Template

**File:** `frontend_multi_user/templates/account.html`

Replace the current single-key section with a multi-key table and create form.

#### Key table

```html
<table class="payments-table">
    <thead>
        <tr>
            <th>Label</th>
            <th>Key</th>
            <th>Created</th>
            <th>Last Used</th>
            <th>Plans</th>
            <th>Credits Used</th>
            <th></th>
        </tr>
    </thead>
    <tbody>
        {% for key in active_keys %}
        <tr>
            <td>{{ key.label or "Untitled" }}</td>
            <td><span class="key-prefix">{{ key.key_prefix }}...</span></td>
            <td>{{ key.created_at.strftime("%Y-%m-%d") if key.created_at else "—" }}</td>
            <td>{{ key.last_used_at.strftime("%Y-%m-%d") if key.last_used_at else "Never" }}</td>
            <td>{{ plan_counts.get(key.id|string, 0) }}</td>
            <td>{{ format_credits(credit_usage.get(key.id|string, 0)) }}</td>
            <td>
                <form method="POST" style="margin:0;">
                    <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
                    <input type="hidden" name="action" value="revoke_api_key">
                    <input type="hidden" name="key_id" value="{{ key.id }}">
                    <button class="btn-acct btn-acct-secondary" style="color:#991b1b;"
                        onclick="return confirm('Revoke this key? Integrations using it will stop working.')">
                        Revoke
                    </button>
                </form>
            </td>
        </tr>
        {% endfor %}
    </tbody>
</table>
```

#### New-key one-time display

Kept as-is (the `new_api_key` session variable pattern already works).

#### Create key form

```html
{% if can_create_key %}
<form method="POST" style="margin-top: 12px; display: flex; gap: 8px; align-items: flex-end;">
    <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
    <input type="hidden" name="action" value="create_api_key">
    <div class="billing-input-group">
        <label for="key-label">Label (optional)</label>
        <input type="text" name="label" id="key-label" maxlength="128" placeholder="e.g. Claude Code"
            style="width: 200px; padding: 8px 10px; border: 1px solid var(--color-border); border-radius: var(--radius); font-size: 0.9rem;">
    </div>
    <button class="btn-acct btn-acct-primary">Create API Key</button>
</form>
{% else %}
<p class="key-hint" style="margin-top: 12px;">
    Maximum of 10 keys reached. Revoke an unused key to create a new one.
</p>
{% endif %}
```

---

## Files Changed

| File | Change |
|------|--------|
| `database_api/model_user_api_key.py` | Add `label` column |
| `database_api/model_planitem.py` | Add `api_key_id` column |
| `database_api/model_credit_history.py` | Add `api_key_id` column |
| `mcp_cloud/db_setup.py` | Add `ensure_multi_api_key_columns()`, call on startup |
| `mcp_cloud/auth.py` | Return `api_key_id` in user_context dict |
| `mcp_cloud/handlers.py` | Pass `api_key_id` in metadata to `_create_plan_sync` |
| `mcp_cloud/db_queries.py` | Store `api_key_id` on `PlanItem` |
| `worker_plan_database/app.py` | Call migration on startup; store `api_key_id` on `CreditHistory` ledger entries |
| `frontend_multi_user/src/app.py` | Multi-key limit, `create_api_key`/`revoke_api_key` actions, per-key stats, label parameter |
| `frontend_multi_user/templates/account.html` | Multi-key table, create form, per-key stats display |

---

## Backward Compatibility

- **All new columns are nullable.** Existing single-key users are unaffected.
- **`api_key_id = NULL`** on old plans and ledger entries simply means "created before multi-key tracking" — these rows are excluded from per-key stats and attributed to no specific key.
- **Label fallback:** NULL or empty labels display as "Untitled" in the UI.
- **First-login behavior:** Unchanged — a first-time OAuth user still gets one auto-generated key (now labeled "Default").
- **MCP API contract:** `user_api_key` field in `plan_create`/`plan_list` works exactly as before. No new required fields.

---

## Verification

1. **Existing single-key user:** Log in, verify account page shows the existing key in the table with "Untitled" label. Old plans show 0 in the per-key stats (since they have no `api_key_id`).
2. **Create multiple keys:** Use the create form to add keys with different labels. Verify each gets a unique prefix and is listed in the table.
3. **Revoke one key:** Click Revoke on one key. Verify it disappears from the table. Verify other keys still work via MCP.
4. **Per-key attribution:** Create plans using different API keys. Verify the account page shows correct plan counts and credit usage per key.
5. **Max limit:** Create 10 keys. Verify the "Create API Key" button is replaced by the limit message.
6. **Run existing tests:** `pytest database_api/tests/ mcp_cloud/tests/ frontend_multi_user/tests/ -v` — all pass.
7. **Migration idempotency:** Restart each service multiple times. Verify no errors from `ALTER TABLE ADD COLUMN IF NOT EXISTS`.

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Existing `_get_or_create_api_key` callers assume at most one key | Audit all call sites (OAuth callback, `regenerate_api_key` action). OAuth callback is updated to check `has_key` before calling. |
| Performance of per-key stats queries on large accounts | Queries use indexed `api_key_id` columns and are bounded by the 10-key limit. Stats are computed per page load, not in a hot loop. |
| `api_key_id` is `VARCHAR(36)` instead of a UUID foreign key | Matches the existing `user_id` pattern on `PlanItem` (also a string, not a FK). Avoids FK constraint complexity across services. Index provides lookup performance. |
| User accidentally revokes their only key | The UI shows a confirmation dialog. Creating a new key is one click away. |

---

## Current Status (2026-03-05)

**Implemented and working.** All goals from this proposal are complete, plus several enhancements discovered during implementation.

### What was built

**Core multi-key support (as proposed):**
- Up to 10 active API keys per user with optional names.
- Click-to-edit key renaming (pencil icon → inline form with Save/Cancel/Escape).
- Three-dot overflow menu per key with "Reset secret" and "Delete".
- `api_key_id` tracked on `PlanItem`, `CreditHistory`, and `TokenMetrics`.
- Per-key stats on the account page: **LLM Calls**, **Credits Used**, **Last Used**.
- API key secret visibility controlled by `PLANEXE_API_KEY_SHOW_ONCE` env var.

**Enhancements beyond the original proposal:**

| Enhancement | Details |
|-------------|---------|
| `TokenMetrics.api_key_id` column | Each LLM call records which API key was active. The account page queries `TokenMetrics.api_key_id` directly (not joined through `PlanItem`), so retrying a plan with a different key does not move historical LLM call counts. |
| Incremental billing | Credits are charged during plan execution via `_charge_incremental_usage()` on each progress heartbeat, not just at completion. Uses `source="usage_billing_progress"` entries. |
| Billing archival on retry | When a plan is retried, old incremental billing entries are archived (`source` changed from `"usage_billing_progress"` to `"usage_billing_settled"`), not deleted. This lets the new run charge from zero while preserving the previous key's credit history. |
| Auth-disabled key resolution | When `PLANEXE_MCP_REQUIRE_AUTH=false` (local dev), the HTTP server still resolves any provided API key for attribution (`last_used_at`, per-key billing) — it just never rejects unauthenticated requests. |
| Plan retry re-attribution | `plan_retry` via MCP updates `plan.user_id` and `plan.api_key_id` to the caller's key. New LLM calls and billing entries go to the new key. |
| Worker `api_key_id` context | `set_current_api_key_id()` is set alongside `set_current_task_id()` and `set_current_user_id()` at pipeline start, flowing through to all `TokenMetrics` rows. |
| Separate per-key stat queries | LLM calls and credit usage queries are in independent try/except blocks so one failure doesn't zero out the other. |
| PostgreSQL index race condition handling | Each `CREATE INDEX IF NOT EXISTS` statement is wrapped in its own try/except to handle the `UniqueViolation` race when multiple gunicorn workers start simultaneously. |
| Admin user identity fix | Frontend routes use the admin's `UserAccount` UUID (not the username string) for `PlanItem.user_id`, so billing resolves correctly. `_admin_user_ids()` returns both old and new values for backward-compatible queries. |

### Files changed (beyond original proposal)

| File | Additional changes |
|------|-------------------|
| `database_api/model_token_metrics.py` | Added `api_key_id` column |
| `worker_plan/worker_plan_internal/llm_util/token_instrumentation.py` | Added `set_current_api_key_id` / `get_current_api_key_id` context functions |
| `worker_plan/worker_plan_internal/llm_util/token_metrics_store.py` | Added `api_key_id` parameter to `record_token_usage()` |
| `worker_plan/worker_plan_internal/llm_util/track_activity.py` | Passes `api_key_id` to `record_token_usage()` |
| `mcp_cloud/http_server.py` | Auth-disabled key resolution; `plan_retry` injects authenticated key |
| `mcp_cloud/db_queries.py` | `_retry_failed_plan_sync` accepts `caller_metadata`, archives old billing entries |
| `docs/frontend_multi_user_guide.md` | Full billing/per-key stats documentation |
| `frontend_multi_user/AGENTS.md` | Per-key stats architecture, retry archival |
| `worker_plan_database/AGENTS.md` | TokenMetrics attribution, billing archival |
| `mcp_cloud/AGENTS.md` | Auth-disabled resolution, retry attribution |
