# frontend_multi_user agent instructions

Scope: Flask-based multi-user UI with Postgres backing, admin UI, and queue
logic. Talks to `worker_plan` for execution and uses shared `database_api`
models. Keep interfaces stable across services.

## Guidelines
- Preserve the `worker_plan` API contract defined in `worker_plan/app.py`;
  update the UI calls if routes or response shapes change.
- Keep Postgres wiring and env defaults stable (see root `AGENTS.md` for keys);
  prefer `SQLALCHEMY_DATABASE_URI`, otherwise fall back to
  `PLANEXE_FRONTEND_MULTIUSER_DB_*` and `PLANEXE_POSTGRES_*`.
- Maintain required admin auth (`PLANEXE_FRONTEND_MULTIUSER_ADMIN_USERNAME`
  / `PLANEXE_FRONTEND_MULTIUSER_ADMIN_PASSWORD`) and Flask login flow.
- Continue using `database_api.planexe_db_singleton.db`; do not create new
  SQLAlchemy engines or sessions here.
- Keep `.env` loading via `PlanExeDotEnv` and `update_os_environ()` early so
  local/debug behavior is consistent.
- If schema usage changes (e.g., new PlanItem columns), update the
  `_ensure_planitem_artifact_columns()` helper and keep changes backward
  compatible.
- Index creation (`CREATE INDEX IF NOT EXISTS`) must wrap each statement in
  its own try/except — PostgreSQL has a race condition when multiple gunicorn
  workers start simultaneously and all try to create the same index.
- Artifact storage model:
  - Use `PlanItem.run_activity_overview_json` as primary UI cost/usage source.
  - Keep `PlanItem.run_track_activity_jsonl` internal/admin-only.
  - User zip downloads should serve layout-versioned snapshots directly for
    new tasks; sanitize legacy snapshots only.
- Admin navbar UX invariant:
  - Keep the top-right corner behavior as a location-based toggle.
  - On `/`, users see `Admin Panel` in the top-right corner.
  - On `/admin`, keep `Dashboard` in that same top-right corner location, with
    `Logout` immediately to its left (`Logout`, then `Dashboard`).
- Do not store run state in module-level globals; fetch state from Postgres or
  `worker_plan` per request.
- Admin user identity:
  - Flask-Login stores the admin username string (e.g. `"admin"`) as the
    session user_id.
  - The admin's `UserAccount` row uses a deterministic UUID
    (`uuid5(NAMESPACE_URL, "planexe-admin-pref:{username}")`), created lazily
    by `_get_current_user_account()`.  The lazy creation is wrapped in
    try/except with rollback + re-fetch to handle gunicorn worker races.
    The `/account` route has an additional inline fallback that retries
    creation if `_get_current_user_account()` returns `None`, so the admin
    is never logged out just because the `UserAccount` row is missing.
  - When creating PlanItem records, always use the admin's UserAccount UUID as
    `user_id` (not the username string). The billing system in
    `worker_plan_database` resolves `user_id` via `uuid.UUID()`, so a plain
    string like `"admin"` fails silently and skips billing.
  - Use `_admin_user_ids()` when querying PlanItem rows for admin — it returns
    both the old username string and the UUID so old and new plans both appear.
  - When creating plans, set `api_key_id` from the user's first active key so
    per-key statistics work on the account page.
- Per-key stats on the account page:
  - **LLM Calls**: queries `TokenMetrics.api_key_id` directly (not joined
    through PlanItem). Each TokenMetrics row records which key was active at
    LLM call time, so stats are immutable per-row.
  - **Credits Used**: queries `CreditHistory.api_key_id` with `delta < 0`.
    Includes all billing sources (incremental, settled, and final).
  - Both queries are in separate try/except blocks so one failure doesn't
    zero out the other.
- Plan retry in the frontend (`/plan/retry`) archives old incremental billing
  entries (`usage_billing_progress` → `usage_billing_settled`) instead of
  deleting them, preserving the original key's credit history.
- Forbidden imports: `worker_plan_internal`, `worker_plan.app`,
  `frontend_single_user`, `open_dir_server`.

## Testing
- No automated tests currently. If you change UI or DB flow, add a unit test
  close to the logic when feasible and run `python test.py` from repo root.
