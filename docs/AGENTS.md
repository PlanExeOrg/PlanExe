# How PlanExe-docs Builds and Publishes to docs.planexe.org

This document describes how the [PlanExe-docs](https://github.com/PlanExeOrg/PlanExe-docs) repository takes content from **this directory** (`PlanExe/docs/`) and publishes it to [https://docs.planexe.org](https://docs.planexe.org).

## Overview

- **Content source**: This directory (`PlanExe/docs/`). All Markdown files, images, and assets here become the published documentation.
- **Build & deploy**: The [PlanExe-docs](https://github.com/PlanExeOrg/PlanExe-docs) repo. It holds MkDocs config, GitHub Actions workflow, and build scripts.
- **Output**: Static site served via **GitHub Pages** at **docs.planexe.org**.

## Pipeline (CI)

1. **Trigger**  
   The [Deploy Documentation](https://github.com/PlanExeOrg/PlanExe-docs/blob/main/.github/workflows/deploy.yml) workflow runs when:
   - There is a **push to `main`** on PlanExe-docs, or
   - It is started **manually** (`workflow_dispatch`), or
   - A **`repository_dispatch`** event `docs-updated` is sent (e.g. when PlanExe is updated and you want to redeploy docs).

   **Pushing only to PlanExe does not by itself update docs.planexe.org.** This repo has a workflow (`.github/workflows/docs-update.yml`) that runs on push to `main` when `docs/` changes and sends `repository_dispatch` to PlanExe-docs. For that to work you must add a secret in **PlanExe** (see below). Otherwise, after editing `PlanExe/docs/`, either run the Deploy workflow **manually** in PlanExe-docs, or push to PlanExe-docs `main` (e.g. after syncing content) to deploy.

2. **Checkout**  
   - PlanExe-docs repo (workflow, `mkdocs.yml`, `requirements.txt`, etc.).  
   - PlanExe repo into `planexe-source/` (so this `docs/` directory is available).

3. **Build**  
   - `mkdir -p docs` in the PlanExe-docs workspace.  
   - `cp -r planexe-source/docs/* docs/` — all content from **this** `PlanExe/docs/` directory is copied into PlanExe-docs’ `docs/` folder.  
   - `mkdocs build --site-dir site` — MkDocs (Material theme, config from `mkdocs.yml`) builds the site into `site/`.

4. **Deploy**  
   - The [peaceiris/actions-gh-pages](https://github.com/peaceiris/actions-gh-pages) action publishes the `site/` directory to the **gh-pages** branch of PlanExe-docs.  
   - Custom domain **docs.planexe.org** is set via `cname: docs.planexe.org` in the workflow.  
   - GitHub Pages serves the site from that branch, so updates appear at **https://docs.planexe.org**.

## Key files

| What | Where |
|------|--------|
| Doc content (you edit here) | `PlanExe/docs/` (this directory) |
| MkDocs config, theme, plugins | PlanExe-docs `mkdocs.yml` |
| Deploy workflow | PlanExe-docs `.github/workflows/deploy.yml` |
| Build dependencies | PlanExe-docs `requirements.txt` |
| Frontpage | `PlanExe/docs/index.md` (used as site index) |

## Local preview

To build and preview the same site locally:

1. Clone both PlanExe and PlanExe-docs.  
2. From PlanExe-docs, run `python build.py` (optionally set `PLANEXE_REPO` if PlanExe is not at `../PlanExe`).  
   - This copies `PlanExe/docs/` into a temp `docs/` dir, runs `mkdocs build`, and writes output to `site/`.  
3. Run `python serve.py` to serve `site/` at `http://127.0.0.1:18525/`.

## Auto-deploy from PlanExe (optional)

To have the live site update when you push to **PlanExe** `main` with changes under `docs/`:

1. In **PlanExe** repo: **Settings → Secrets and variables → Actions** → **New repository secret**.
2. Name: `PLANEXE_DOCS_DISPATCH_TOKEN`. Value: a [Personal Access Token](https://github.com/settings/tokens) (or fine-grained PAT) with **repo** scope for **PlanExeOrg/PlanExe-docs** (or at least permission to trigger workflows in PlanExe-docs).
3. Push to **PlanExe** `main` with changes under `docs/`. The workflow `.github/workflows/docs-update.yml` runs and sends `repository_dispatch` to PlanExe-docs; PlanExe-docs then checks out PlanExe, copies `docs/`, builds, and deploys.

If the secret is not set, the "Notify docs deploy" workflow in PlanExe will fail at the dispatch step. You can still update the live site by running the **Deploy Documentation** workflow manually in PlanExe-docs (Actions → Deploy Documentation → Run workflow), or by pushing to PlanExe-docs `main`.

## Summary

Edits in **PlanExe/docs/** are what get published. PlanExe-docs orchestrates copy → MkDocs build → GitHub Pages deploy to **docs.planexe.org**. Push to PlanExe-docs `main`, trigger the Deploy workflow manually in PlanExe-docs, or set up `PLANEXE_DOCS_DISPATCH_TOKEN` in PlanExe so pushes to `docs/` auto-trigger the deploy.
