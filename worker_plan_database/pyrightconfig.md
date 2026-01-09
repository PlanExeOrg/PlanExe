# Pyright setup for `worker_plan_database`

## TL;DR
- Scope: This config is local to `worker_plan_database` and only affects editor analysis, not runtime.
- Key settings: `pythonVersion: "3.13"`, `extraPaths: ["../worker_plan", "..", ".venv/lib/python3.13/site-packages"]`.
- Effect: Fixes editor import resolution for `worker_plan_api.*`, `worker_plan_internal.*`, and `database_api.*`.

## What Pyright is
Pyright is a strict static type checker and language server for Python, used by editors like VS Code and Cursor to provide diagnostics, IntelliSense, and import resolution without running the code.

## Why this file exists
- Cursor/Pyright could not resolve imports when editing `worker_plan_database/app.py`.
- The imports come from multiple locations outside this directory:
  - `worker_plan_api.*` lives in `../worker_plan/worker_plan_api`
  - `worker_plan_internal.*` lives in `../worker_plan/worker_plan_internal`
  - `database_api.*` lives in `../database_api`
- Runtime already worked (docker and `.venv` with `PYTHONPATH`), so this change aligns editor analysis with reality.

## Why Pyright is the right fix
- Purely analytical: no `sys.path` hacks or code changes just to appease the editor.
- Mirrors actual resolution once `extraPaths` is set, so Cursor matches what runs in docker and `.venv`.
- Scoped and reversible: lives in this folder only and doesn't affect runtime behavior.

## What it does
- Sets `pythonVersion: "3.13"` to match the project's Python version.
- Adds `../worker_plan` to `extraPaths` so Pyright sees `worker_plan_api` and `worker_plan_internal`.
- Adds `..` to `extraPaths` so Pyright sees `database_api`.
- Adds `.venv/lib/python3.13/site-packages` to `extraPaths` so Pyright sees installed packages.

## Maintenance
- If you move or rename folders, update `pyrightconfig.json` paths.
- If you upgrade Python, update `pythonVersion` and the site-packages path.
- Reload/restart the language server after edits so the changes take effect.
