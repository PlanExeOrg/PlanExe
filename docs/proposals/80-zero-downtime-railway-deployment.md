# Proposal 80 — Zero-Downtime Railway Deployment

## Status

Proposed

## Problem

Deploying to Railway currently causes disruption:

- **MCP clients** (Claude Code, Cursor, etc.) have active HTTP connections to `mcp.planexe.org`. When the `mcp_cloud` container restarts, in-flight `plan_create` calls fail, SSE progress streams disconnect, and download URLs return 502 during the restart window.
- **Frontend users** on `home.planexe.org` get 502 errors while `frontend_multi_user` rebuilds and restarts.
- **Running plans** are abandoned mid-execution when `worker_plan_database` restarts. The task stays in `processing` state indefinitely (or until a heartbeat timeout marks it failed), wasting the user's credits.

Railway rebuilds each service from its Dockerfile on every deploy. During the build + restart window (30-120 seconds per service), the old container is stopped and the new one is not yet healthy.

## Current Deployment Topology

```
Internet
  |
  +-- mcp.planexe.org    --> Railway: mcp_cloud        (FastAPI/Uvicorn, port 8001)
  +-- home.planexe.org   --> Railway: frontend_multi_user (Flask/Gunicorn, port 5000)
  |
  +-- (internal) worker_plan          (FastAPI/Uvicorn, port 8000)
  +-- (internal) worker_plan_database (Flask, no HTTP — polls DB)
  +-- (internal) database_postgres    (PostgreSQL 16)
```

### Current `railway.toml` settings

| Service | Restart Policy | Health Check |
|---------|---------------|--------------|
| `mcp_cloud` | `ON_FAILURE` (10 retries) | `/healthcheck` (100s timeout) |
| `frontend_multi_user` | `NEVER` | 10s interval |
| `worker_plan` | `NEVER` | 10s interval |
| `worker_plan_database` | `NEVER` | 10s interval |

### What happens today on deploy

1. Railway receives a git push (or manual redeploy trigger).
2. All services with matching `watchPatterns` are rebuilt in parallel.
3. For each service, Railway stops the old container (sends SIGTERM, waits ~10s, SIGKILL) then starts the new one.
4. Traffic is routed to the new container only after health check passes.
5. During the gap between old-container-down and new-container-healthy, Railway returns 502 to all requests.

---

## Solution

A combination of Railway-side and application-side changes to eliminate or minimize downtime.

### Part 1: Deploy Order (Manual Checklist)

Services have a dependency graph. Deploying in the wrong order can cause cascading failures (e.g., `mcp_cloud` starts before `database_postgres` migration finishes). The safe order:

```
1. database_postgres     (no downtime — Railway Postgres addon persists across deploys)
2. worker_plan           (stateless HTTP API, no active connections from end users)
3. worker_plan_database  (background worker — see Part 3 for graceful drain)
4. mcp_cloud             (public-facing — see Part 2)
5. frontend_multi_user   (public-facing — see Part 2)
```

Steps 4 and 5 are independent and can run in parallel, but each individually needs the zero-downtime treatment from Part 2.

For routine code changes that only affect one service, deploy just that service. The `watchPatterns` in each `railway.toml` already control which services rebuild on a git push.

### Part 2: Zero-Downtime for Public-Facing Services

#### Option A: Railway Replicas with Rolling Deploy (Recommended)

Railway supports running multiple replicas per service with rolling deploys. When a new version is deployed, Railway:

1. Starts new replica(s) alongside the old ones.
2. Waits for new replica(s) to pass health checks.
3. Drains traffic from old replica(s) — stops routing new requests but allows in-flight requests to complete.
4. Stops old replica(s).

**Configuration changes needed:**

`mcp_cloud/railway.toml`:
```toml
[build]
builder = "DOCKERFILE"
dockerfilePath = "/mcp_cloud/Dockerfile"
watchPatterns = ["/mcp_cloud/**", "/database_api/**", "/worker_plan/worker_plan_api/**"]
context = "."

[deploy]
startCommand = "python -m mcp_cloud.http_server"
healthcheckPath = "/healthcheck"
healthcheckTimeout = 100
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10
numReplicas = 2
```

`frontend_multi_user/railway.toml`:
```toml
[build]
builder = "DOCKERFILE"
dockerfilePath = "/frontend_multi_user/Dockerfile"
watchPatterns = ["/frontend_multi_user/**", "/worker_plan/worker_plan_api/**", "/database_api/**"]
context = "."

[deploy]
healthcheckPath = "/healthcheck"
healthcheckTimeout = 100
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3
numReplicas = 2
```

**Cost:** 2x for these two services. On Railway's Pro plan, this is typically the cheapest path to zero downtime.

**Why it works:** Both services are stateless. `mcp_cloud` uses FastAPI/Uvicorn (all state is in Postgres). `frontend_multi_user` uses Flask/Gunicorn with session cookies (session data is server-side but non-critical — a user simply re-authenticates if their session lands on the new replica).

#### Option B: Blue-Green via Railway Environments (Alternative)

If replicas are not available or too costly:

1. Create a `staging` environment in Railway that mirrors `production`.
2. Deploy to `staging` first. Smoke-test against the staging URL.
3. Swap the custom domain (`mcp.planexe.org`, `home.planexe.org`) from the production service to the staging service via Railway dashboard or CLI.
4. Domain swap is near-instant (Railway updates its reverse proxy, no DNS change needed).

**Drawback:** Manual process, requires maintaining two environments.

### Part 3: Graceful Shutdown for `worker_plan_database`

The background worker is the highest-risk service during deploys. It runs long tasks (10-20 minutes) and has no HTTP interface for Railway to drain.

#### Current behavior on SIGTERM

`worker_plan_database` runs `start_task_monitor()` which is a `while True` loop with `process_pending_tasks()`. On SIGTERM:
- Python raises `SystemExit` (or the process is killed after Railway's grace period).
- If a task is mid-execution, the subprocess running the pipeline is killed.
- The task stays in `processing` state in the database.

#### Proposed: SIGTERM handler with graceful drain

Add a signal handler that:
1. Sets a `_shutdown_requested` flag.
2. The main loop checks this flag and stops picking up new tasks.
3. Waits for the current task (if any) to finish, up to a configurable timeout.
4. If the timeout expires, marks the in-progress task as `failed` with a clear message so it can be retried.

**Code change in `worker_plan_database/app.py`:**

```python
import signal
import threading

_shutdown_requested = threading.Event()

def _handle_sigterm(signum, frame):
    logger.info("SIGTERM received. Will stop after current task completes.")
    _shutdown_requested.set()

signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)
```

Update `start_task_monitor()`:

```python
def start_task_monitor():
    logger.info("Started monitoring database for pending tasks.")
    try:
        last_heartbeat_time = time.time()
        while not _shutdown_requested.is_set():
            processed_something = process_pending_tasks()
            # Use Event.wait() instead of time.sleep() so SIGTERM wakes us immediately
            _shutdown_requested.wait(timeout=1 if processed_something else 5)

            new_heartbeat_time = time.time()
            if processed_something:
                last_heartbeat_time = new_heartbeat_time
            if new_heartbeat_time - last_heartbeat_time > HEARTBEAT_INTERVAL_IN_SECONDS:
                last_heartbeat_time = new_heartbeat_time
                with app.app_context():
                    WorkerItem.upsert_heartbeat(worker_id=WORKER_ID)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Stopping task monitor...")
    except Exception as e:
        logger.critical(f"Unhandled exception in task monitor: {e}", exc_info=True)
    finally:
        logger.info("Task monitor shut down.")
        logging.shutdown()
```

**Railway grace period:** Increase the SIGTERM-to-SIGKILL grace period so the worker has time to finish its current task. Railway's default is 10 seconds, which is far too short for a 10-20 minute pipeline run.

In the Railway dashboard (per-service settings), set the drain timeout / shutdown grace period to a value that covers the worst case. Unfortunately, `railway.toml` does not currently expose a `shutdownGracePeriodSeconds` key — this must be set in the Railway dashboard or via the Railway API.

As a fallback, the worker should mark its current task as failed on forced shutdown, so the user can retry via `plan_retry`:

```python
def _mark_current_task_failed_on_shutdown():
    """Best-effort: mark current processing task as failed so user can retry."""
    with app.app_context():
        task = PlanItem.query.filter_by(
            state=PlanState.processing,
        ).filter(
            PlanItem.parameters.contains({"_worker_id": WORKER_ID})
        ).first()
        if task:
            task.state = PlanState.failed
            task.progress_message = "Worker restarted during execution. Use plan_retry to re-run."
            db.session.commit()
            logger.info("Marked task %s as failed due to shutdown.", task.id)
```

Call this in the `finally` block of `start_task_monitor()`.

### Part 4: SSE Reconnection Resilience

MCP clients monitoring plan progress via SSE (`GET /sse/plan/{plan_id}`) will lose their connection during a deploy. The SSE implementation already sends a `retry: 5000` hint (line 95 of `sse.py`), which tells well-behaved SSE clients (including `EventSource` in browsers) to automatically reconnect after 5 seconds.

**Current state:** This already works. After the new container passes its health check, the client reconnects and resumes receiving progress events. No code change needed.

**Improvement (optional):** Add `Last-Event-Id` support so clients can resume from where they left off instead of getting the current snapshot. This is a nice-to-have but not required — the current behavior (reconnect and get latest state) is sufficient since SSE events are idempotent status snapshots, not an ordered event log.

### Part 5: MCP Client Retry Guidance

MCP clients that get a 502 during a deploy should retry. The `plan_create` handler is idempotent in the sense that each call creates a new plan — there is no risk of duplicate execution. For `plan_status`, `plan_file_info`, and `plan_list`, retries are naturally safe (read-only operations).

This is primarily a documentation concern. The MCP server instructions already tell clients to poll `plan_status` periodically, which implicitly handles transient failures.

### Part 6: Database Migrations During Deploy

Schema migrations run at service startup (`ensure_*_columns()` functions). These use `ALTER TABLE ADD COLUMN IF NOT EXISTS` which is:
- **Idempotent:** Safe to run from multiple services simultaneously.
- **Non-blocking in PostgreSQL:** `ADD COLUMN` with no default does not lock the table.
- **Backward compatible:** New columns are nullable, so the old code (still running during rolling deploy) is not affected.

No changes needed. The current migration pattern is already deploy-safe.

---

## Deploy Checklist

### Before deploying

- [ ] Verify the change is on the correct branch and CI passes.
- [ ] Check `watchPatterns` in `railway.toml` — confirm only the intended services will rebuild.
- [ ] If schema migrations are included, verify they are additive (nullable columns, `IF NOT EXISTS`).

### During deploy

- [ ] If deploying `worker_plan_database`: check that no critical long-running tasks are in `processing` state. If there are, either wait for them to finish or accept that they will be retried.
- [ ] Monitor Railway deploy logs for health check pass on each service.
- [ ] Check `/healthcheck` on both `mcp.planexe.org` and `home.planexe.org` after deploy.

### After deploy

- [ ] Verify `mcp.planexe.org/mcp/tools` returns the tool list (confirms MCP is operational).
- [ ] Verify `home.planexe.org/healthcheck` returns `{"status": "ok", "database": "ok"}`.
- [ ] Check for any tasks stuck in `processing` state that may need `plan_retry`.
- [ ] Spot-check SSE: `curl -N https://mcp.planexe.org/sse/plan/<recent-plan-id>` returns heartbeats.

---

## Files Changed

| File | Change |
|------|--------|
| `mcp_cloud/railway.toml` | Add `numReplicas = 2`, add `healthcheckTimeout` |
| `frontend_multi_user/railway.toml` | Add `numReplicas = 2`, add `healthcheckPath`, `healthcheckTimeout` |
| `worker_plan_database/app.py` | Add SIGTERM handler, graceful drain in `start_task_monitor()`, mark-failed-on-shutdown |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| 2x replica cost for mcp_cloud and frontend | Can scale back to 1 replica during low-traffic periods. Railway bills per-usage, so idle replicas cost very little. |
| Worker SIGTERM handler doesn't fire (SIGKILL before handler runs) | The `_mark_current_task_failed_on_shutdown()` fallback in `finally` handles this. Even without it, the existing heartbeat timeout mechanism will eventually mark stale tasks as failed. |
| Rolling deploy routes traffic to new replica before migrations finish | Migrations are idempotent `ADD COLUMN IF NOT EXISTS` — they complete in milliseconds and are safe to run concurrently. The health check only passes after the Flask/FastAPI app fully initializes (including migrations). |
| SSE connections drop during deploy | Already handled: `retry: 5000` hint causes automatic reconnection. The new replica serves the reconnected client with current state. |
| Session cookies from old frontend replica are invalid on new replica | Flask sessions use a server-side secret key (from env var). As long as `SECRET_KEY` is the same across replicas (it is — set via Railway shared variables), sessions are valid on any replica. |
| Database connection pool exhaustion with 2x replicas | Each service uses SQLAlchemy with `pool_recycle=280` and `pool_pre_ping=True`. Default pool size is 5 connections per worker. With 2 replicas x 4 gunicorn workers = 40 connections max for frontend. Railway Postgres supports hundreds of connections. |
