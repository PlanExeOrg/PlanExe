# worker_plan agent instructions

Scope: FastAPI worker service and shared `worker_plan_internal`/`worker_plan_api`
packages used by frontends and database workers. Keep interfaces stable across
consumers.

## Guidelines
- Preserve the public API contract in `worker_plan/app.py`:
  - Keep request/response shapes and endpoint paths backward compatible.
  - Avoid renaming response fields like `run_id`, `run_dir`, `display_run_dir`.
- Artifact contract: `/runs/{run_id}/zip` must not include
  `track_activity.jsonl` in downloadable zips.
- Maintain the run directory conventions (`PlanExe_...`) and environment-driven
  paths (`PLANEXE_RUN_DIR`, `PLANEXE_HOST_RUN_DIR`, `PLANEXE_CONFIG_PATH`).
- When changing pipeline behavior, keep the subprocess invocation in
  `start_pipeline_subprocess` consistent with `worker_plan_internal`.
- Keep `PlanExeDotEnv.load().update_os_environ()` early so `.env` overrides work.
- CRITICAL: `worker_plan_api` must stay lightweight. Allowed imports are
  stdlib, `typing`, `dataclasses`, `enum`, and `pydantic`. Do not import
  `llama_index`, `fastapi`, `httpx`, `numpy`, `pandas`, or `torch` there.
- Keep planning logic in `worker_plan_internal` (pipeline stages:
  prompt parsing -> LLM planning -> file/report output).
- Other services may import `worker_plan_api` for types/helpers; they must not
  import `worker_plan_internal`.
- If new environment variables or endpoints are added, update
  `worker_plan/README.md` and any Railway docs.

## LLM Error Handling
- When an LLM call fails inside a pipeline task, raise `LLMChatError` (from
  `worker_plan_internal.llm_util.llm_errors`), **not** a bare `ValueError`.
  `LLMChatError` preserves the root cause and generates an `error_id` UUID for
  cross-referencing logs with `usage_metrics.jsonl` rows.
- Standard pattern (30+ call sites):
  ```python
  from worker_plan_internal.llm_util.llm_errors import LLMChatError
  try:
      result = llm_executor.run(execute_function)
  except PipelineStopRequested:
      raise
  except Exception as e:
      llm_error = LLMChatError(cause=e)
      logger.debug(f"LLM chat interaction failed [{llm_error.error_id}]: {e}")
      logger.error(f"LLM chat interaction failed [{llm_error.error_id}]", exc_info=True)
      raise llm_error from e
  ```
- For tasks with multiple LLM calls, use the `message` parameter to
  distinguish them: `LLMChatError(cause=e, message="LLM chat interaction 2 failed")`.
- Error classification: `classify_error()` in `usage_metrics.py` maps raw
  exception strings to short categories (`invalid_json`, `timeout`,
  `rate_limit`, etc.) stored in `usage_metrics.jsonl`. Unknown errors preserve
  a truncated `error_detail` field.

## Testing
- Prefer unit tests over manual server checks. Run `python test.py` from repo
  root; worker tests live under `worker_plan/worker_plan_internal/**/tests` and
  `worker_plan/worker_plan_api/tests`.
- Error handling tests: `worker_plan/tests/test_llm_errors.py` and
  `worker_plan/tests/test_usage_metrics.py`.
