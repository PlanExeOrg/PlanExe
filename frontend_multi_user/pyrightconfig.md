# Pyright setup for `frontend_multi_user`

## TL;DR
- Scope: This config is local to `frontend_multi_user` and only affects editor analysis, not runtime.
- Key settings: `venvPath: "."`, `venv: ".venv"`, `extraPaths: ["..", "../worker_plan", "src"]`.
- Effect: Fixes editor import resolution for `database_api.*`, `worker_plan_api.*`, `worker_plan_internal.*`, and local `src/` modules.

## What Pyright is
Pyright is a strict static type checker and language server for Python, used by editors like VS Code and Cursor to provide diagnostics, IntelliSense, and import resolution without running the code.

## Why this file exists
- Cursor/Pyright could not resolve imports when editing `frontend_multi_user/src/app.py`.
- The imports come from multiple locations outside this directory:
  - `database_api.*` lives in `../database_api`
  - `worker_plan_api.*` lives in `../worker_plan/worker_plan_api`
  - `worker_plan_internal.*` lives in `../worker_plan/worker_plan_internal`
  - `planexe_modelviews` lives in `src/`
- Runtime already worked (docker and `.venv` with `PYTHONPATH`), so this change aligns editor analysis with reality.

## Why Pyright is the right fix
- Purely analytical: no `sys.path` hacks or code changes just to appease the editor.
- Mirrors actual resolution once `extraPaths` is set, so Cursor matches what runs in docker and `.venv`.
- Scoped and reversible: lives in this folder only and doesn't affect runtime behavior.

## What it does
- Points Pyright at the local virtual environment (`venvPath: "."`, `venv: ".venv"`).
- Adds `..` to `extraPaths` so Pyright sees `database_api`.
- Adds `../worker_plan` to `extraPaths` so Pyright sees `worker_plan_api` and `worker_plan_internal`.
- Adds `src` to `extraPaths` so Pyright sees local modules like `planexe_modelviews`.

## Maintenance
- If you move or rename folders, update `pyrightconfig.json` paths.
- If you switch or rename the interpreter/venv, update `venvPath`/`venv` to match.
- Reload/restart the language server after edits so the changes take effect.
