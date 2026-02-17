# worker_plan agent instructions

Scope: FastAPI worker service and shared `worker_plan_internal`/`worker_plan_api`
packages used by frontends and database workers. Keep interfaces stable across
consumers.

## Guidelines
- Preserve the public API contract in `worker_plan/app.py`:
  - Keep request/response shapes and endpoint paths backward compatible.
  - Avoid renaming response fields like `run_id`, `run_dir`, `display_run_dir`.
- Maintain UUID run directory conventions (with legacy compatibility where explicitly required) and environment-driven
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

## Testing
- Prefer unit tests over manual server checks. Run `python test.py` from repo
  root; worker tests live under `worker_plan/worker_plan_internal/**/tests` and
  `worker_plan/worker_plan_api/tests`.
