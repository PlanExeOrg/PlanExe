# End-to-end test plan (Docker + optional LLM access)

## Context

Unit tests mock the database and external calls. They verify wiring but not the actual behavior of the full stack. End-to-end tests run against the real Docker services (database, worker, MCP server) and optionally invoke LLMs. They are slow, non-deterministic (when LLMs are involved), and costly — so they should be run selectively, not on every commit.

This proposal catalogs the end-to-end test scenarios that would provide the most confidence, grouped by whether they require LLM access.

---

## Tests that do NOT require LLM access

These tests exercise the MCP server, database, and worker interactions without invoking any LLM. They are deterministic and cheap.

### 1. Pipeline version mismatch on resume

**Goal:** Verify that `plan_resume` rejects a snapshot created by a different pipeline version.

**Steps:**
1. Create a plan via `plan_create`.
2. Let the worker pick it up and fail it (e.g. via `plan_stop` shortly after start, or by configuring an invalid LLM model).
3. Manually update `PIPELINE_VERSION` in the database parameters to a different value (or patch the constant before resuming).
4. Call `plan_resume`.
5. Assert: MCP returns `PIPELINE_VERSION_MISMATCH` error.

**Variant — worker-side check:**
1. Bypass the MCP-layer check (e.g. manually set `parameters["pipeline_version"]` to match current).
2. But ensure the `001-3-planexe_metadata.json` in the zip snapshot has a different version.
3. Let the worker pick up the resumed plan.
4. Assert: worker sets plan to failed with progress_message containing "Not resumable".

### 2. Pipeline version mismatch on resume — legacy plan (no version)

**Goal:** Verify that plans created before pipeline versioning cannot be resumed.

**Steps:**
1. Create a plan and let it fail.
2. Remove `pipeline_version` from the plan's parameters in the database.
3. Call `plan_resume`.
4. Assert: MCP returns `PIPELINE_VERSION_MISMATCH` error.

### 3. Resume with matching pipeline version

**Goal:** Verify that `plan_resume` accepts a snapshot with matching pipeline version and requeues it.

**Steps:**
1. Create a plan via `plan_create`.
2. Let the worker start, then stop it via `plan_stop`.
3. Confirm plan is in `failed` state with `plan_status`.
4. Call `plan_resume`.
5. Assert: returns successfully with `state: "pending"` and `resume_count: 1`.

### 4. Resume a non-failed plan

**Goal:** Verify `plan_resume` rejects plans that are not in failed state.

**Steps:**
1. Create a plan via `plan_create` (will be in `pending` state).
2. Call `plan_resume` immediately.
3. Assert: returns `PLAN_NOT_RESUMABLE` error.

### 5. Retry clears artifacts, resume preserves them

**Goal:** Verify the fundamental difference between `plan_retry` and `plan_resume`.

**Steps:**
1. Create a plan and let it partially complete before stopping.
2. Record the list of intermediary files via `plan_status`.
3. Call `plan_retry` — verify the plan is reset to pending with no prior artifacts.
4. Let it partially complete again and stop.
5. Call `plan_resume` — verify the plan is reset to pending with prior artifacts preserved.

---

## Tests that require LLM access

These tests invoke real LLMs and are non-deterministic, slow (~10-20 min per plan), and incur cost. Run sparingly.

### 6. Full plan create to completion

**Goal:** Smoke test the happy path end-to-end.

**Steps:**
1. Call `example_prompts` to get a sample prompt.
2. Call `plan_create` with the sample prompt and `model_profile: "baseline"`.
3. Poll `plan_status` until `completed`.
4. Call `plan_file_info` with `artifact: "report"` — assert `download_url` is present.
5. Call `plan_file_info` with `artifact: "zip"` — assert `download_url` is present.
6. Download the report and verify it is valid HTML containing expected sections.
7. Download the zip and verify `001-3-planexe_metadata.json` is present with correct `pipeline_version`.

### 7. Resume after mid-generation failure

**Goal:** Verify that resume actually skips completed steps and finishes the plan.

**Steps:**
1. Create a plan and let it run until ~50% progress.
2. Call `plan_stop`.
3. Wait for plan to reach `failed` state.
4. Record `steps_completed` from `plan_status`.
5. Call `plan_resume`.
6. Poll `plan_status` until `completed`.
7. Assert: the plan completed successfully and the final report is valid.
8. Assert: `resume_count` is 1 in the response.

### 8. Multiple resumes

**Goal:** Verify that a plan can be resumed more than once.

**Steps:**
1. Create a plan and stop it at ~30% progress.
2. Resume and stop it again at ~60% progress.
3. Resume a second time and let it complete.
4. Assert: `resume_count` is 2.
5. Assert: the final report is valid.

### 9. Plan list reflects resume state

**Goal:** Verify that `plan_list` correctly shows plans that have been resumed.

**Steps:**
1. Create two plans — let one complete normally and stop the other.
2. Resume the stopped plan.
3. Call `plan_list`.
4. Assert: both plans appear with correct states and progress.

---

## Implementation notes

- These tests should live in a dedicated directory (e.g. `tests/e2e/`) separate from unit tests.
- They require `docker compose up` with healthy `mcp_cloud`, `worker_plan_database`, and `database_postgres` services.
- LLM tests require valid model configuration (e.g. baseline profile with at least one available model).
- Consider a pytest marker (e.g. `@pytest.mark.e2e` and `@pytest.mark.e2e_llm`) to separate no-LLM from LLM tests.
- Non-LLM tests could run in CI on every PR. LLM tests should be triggered manually or on a schedule.
