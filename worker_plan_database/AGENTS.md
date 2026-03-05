# worker_plan_database agent instructions

Scope: database-backed worker that polls Postgres `PlanItem` rows, runs the
PlanExe pipeline, and updates task state/progress.

## Guidelines
- Keep database access wired through `database_api.planexe_db_singleton.db`;
  do not create new engine/session instances here.
- Preserve the startup sequence in `worker_plan_database/app.py`:
  `.env` loading, logging setup, Flask app config, then `db.init_app(app)`.
- Maintain the DB connection logic:
  - Prefer `SQLALCHEMY_DATABASE_URI` when set.
  - Otherwise build from `PLANEXE_POSTGRES_*` (see root `AGENTS.md` for keys).
- Keep worker identity and required env checks intact (`PLANEXE_WORKER_ID`,
  `PLANEXE_IFRAME_GENERATOR_CONFIRMATION_*`).
- When changing schema usage, add columns in a backward-compatible way and
  ensure `ensure_planitem_artifact_columns()` or related helpers are updated.
- Billing depends on `PlanItem.user_id` being a valid UUID that maps to a
  `UserAccount` row. `_charge_usage_credits_once()` parses `user_id` via
  `uuid.UUID()` — if it's a plain string (e.g. `"admin"`), the lookup fails
  silently and no `CreditHistory` entry is created. Callers creating PlanItem
  records must use a real UserAccount UUID.
- Per-key attribution: both `CreditHistory.api_key_id` and
  `TokenMetrics.api_key_id` are set from `PlanItem.api_key_id` during billing
  and LLM execution respectively. If the PlanItem has no `api_key_id`, entries
  will have `NULL` and the account page per-key stats won't include them.
- TokenMetrics api_key_id: set via `set_current_api_key_id()` context variable
  at pipeline start (alongside `set_current_task_id` and `set_current_user_id`).
  Each LLM call records which API key was active, so retrying a plan with a
  different key does not retroactively move historical LLM call counts.
- Incremental billing: credits are charged during plan execution (not just at
  the end) via `_charge_incremental_usage()`, called from each progress
  heartbeat in `_handle_task_completion()`. This uses
  `source="usage_billing_progress"` entries. The final
  `_charge_usage_credits_once()` subtracts already-charged incremental amounts
  and creates a `source="usage_billing"` entry for the remainder + success
  fee. This prevents abuse where users abort plans to avoid charges.
- Billing on retry: when a plan is retried, old `usage_billing_progress`
  entries are archived to `source="usage_billing_settled"` (not deleted).
  This lets the new run charge from zero while preserving the old key's
  credit history. `_sum_already_charged_credits()` only counts
  `usage_billing_progress` entries, so settled entries are excluded.
- Artifact storage model:
  - Persist `track_activity.jsonl` into `PlanItem.run_track_activity_jsonl`
    (+ bytes in `run_track_activity_bytes`).
  - Persist `activity_overview.json` into `PlanItem.run_activity_overview_json`.
  - Build `run_zip_snapshot` without `track_activity.jsonl` and set
    `run_artifact_layout_version` for new runs.
- Forbidden imports: `worker_plan.app`, `frontend_*`, `open_dir_server`.

## Testing
- No automated tests currently. If you change worker behavior, add a unit test
  close to the logic when feasible and run `python test.py` from repo root.
