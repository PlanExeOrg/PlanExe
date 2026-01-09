# Pyright setup for `frontend_single_user`

## TL;DR
- Scope: This config is local to `frontend_single_user` and only affects editor analysis, not runtime.
- Key settings: `venvPath: "."`, `venv: ".venv"`, `extraPaths: ["../worker_plan"]` (also in `executionEnvironments`).
- Effect: Fixes editor import resolution for `worker_plan_api.*` without touching code or folder layout.

## What Pyright is
Pyright is a strict static type checker and language server for Python, used by editors like VS Code and Cursor to provide diagnostics, IntelliSense, and import resolution without running the code.

## Why this file exists
- Cursor/Pyright could not resolve imports like `worker_plan_api.*` when editing `frontend_single_user/app.py`.
- The code lives in `frontend_single_user/`, but `worker_plan_api` lives next door in `../worker_plan`, so the editor needed a path hint.
- Runtime already worked (docker and `.venv`), so this change is purely to align editor analysis with reality.

## Why Pyright is the right fix
- Purely analytical: no `sys.path` hacks or code changes just to appease the editor.
- Mirrors actual resolution once `extraPaths` is set, so Cursor matches what runs in docker and `.venv`.
- Scoped and reversible: lives in this folder only and doesn't affect runtime behavior.

## What it does
- Points Pyright at the local virtual environment (`venvPath: "."`, `venv: ".venv"`).
- Adds `../worker_plan` to `extraPaths` (also mirrored inside `executionEnvironments`) so Pyright sees `worker_plan_api`.

## Maintenance
- If you move or rename folders, update `pyrightconfig.json` paths.
- If you switch or rename the interpreter/venv, update `venvPath`/`venv` to match.
- Reload/restart the language server after edits so the changes take effect.
