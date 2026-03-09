# Model Routing Post-Mortem: Simon's 26 February 2026 PlanExe Refactor

**Author:** Larry (analysis), Egon (rubric refinement)  
**Date:** 2026-02-26  
**Status:** Draft — for Simon's review

---

## Executive Summary

Simon shipped 64 commits across 16 PRs on 26 February 2026, touching 88 files with 8,269 insertions and 2,137 deletions. This was exceptional productivity. The purpose of this post-mortem is not to question Simon's choices — the work is architecturally sound, well-tested, and thoroughly documented. The purpose is to identify a repeatable **model routing pattern** that can reduce API costs on future refactors of similar scale without sacrificing quality or causing re-work.

The core finding: **two task clusters (module split + external rename) justified Opus due to file size and cross-file dependency. The remaining 60%+ of commits (docs, tests, small fixes, internal renames) could have executed on Haiku or Minimax with a well-structured plan from Opus.**

---

## Pricing Reference (per 1M tokens)

| Model | Input (≤200K) | Output (≤200K) | Context |
|-------|--------------|----------------|---------|
| Minimax M2.5 | $0.30 | $1.10 | 196K |
| Haiku 4.5 | $1.00 | $5.00 | 200K |
| Sonnet 4.6 | $3.00 | $15.00 | 1M |
| Opus 4.6 | $5.00 | $25.00 | 1M |

> **Note:** Sonnet and Opus prices **double** past 200K tokens. Haiku and Minimax do not double.

---

## Complexity Rubric

Rate each task 1–5 on four dimensions. Sum for model recommendation.

| Dimension | 1 | 2 | 3 | 4 | 5 |
|-----------|---|---|---|---|---|
| **File size** | <100 lines | 100–300 | 300–600 | 600–1000 | 1000+ |
| **Semantic complexity** | rename/replace | simple logic | new function | architectural | cross-file refactor |
| **Ambiguity** | crystal clear + line numbers | minor choices | some design | significant decisions | open-ended |
| **Context dependency** | self-contained | 1 file | 1 module | multi-module | whole codebase |

**Score → Model:**
- 4–7: **Minimax** (mechanical execution)
- 8–11: **Haiku** (guided execution)
- 12–15: **Sonnet** (moderate complexity)
- 16–20: **Opus** (planning, large files, architectural)

---

## Task Cluster Analysis

### Key file sizes (actual, post-refactor)

| File | Lines |
|------|-------|
| mcp_local/planexe_mcp_local.py | 1,055 |
| mcp_cloud/http_server.py | 1,089 |
| mcp_cloud/handlers.py | 554 |
| mcp_cloud/tool_models.py | 298 |
| mcp_cloud/db_queries.py | 304 |
| mcp_cloud/schemas.py | 239 |
| mcp_cloud/app.py (thin facade) | 176 |
| mcp_cloud/download_tokens.py | 152 |
| mcp_cloud/auth.py | 50 |

---

### Cluster 1: Module Split (app.py → 10 focused modules)
*PRs: #91 vicinity — `9f1a7db`*

| Dimension | Score | Notes |
|-----------|-------|-------|
| File size | 5 | Original app.py was 76KB monolith |
| Semantic complexity | 5 | Architectural — split into 10 modules with correct imports |
| Ambiguity | 3 | High-level goal clear, but module boundaries required design decisions |
| Context dependency | 5 | Whole codebase — all callers needed updating |
| **Total** | **18** | **→ Opus** |

**Estimated tokens:** ~150K input (reading full monolith + callers) + 30K output = 180K tokens  
**Cost at Opus:** 150K×$0.0050 + 30K×$0.0250 = **$0.75 + $0.75 = ~$1.50**  
**Could cheaper model execute it?** Yes — with Opus writing a surgical plan (module boundaries, exact file/line splits), Sonnet could execute. Saves ~50%.  
**Confidence (Sonnet executes):** 4/5  
**Retry factor:** Low — plan is precise enough for Sonnet

---

### Cluster 2: External API Rename (task_id → plan_id, TASK_* → PLAN_*)
*PRs: #88, #89, #92, #101 — commits `3663bc6`, `0dbe1af`, `0f2e9cc`, `3624db7`*

| Dimension | Score | Notes |
|-----------|-------|-------|
| File size | 5 | Hits planexe_mcp_local.py (1,055 lines) and http_server.py (1,089 lines) |
| Semantic complexity | 3 | Rename is mechanical, but must not break backward compat aliases during transition |
| Ambiguity | 2 | Clear goal — but alias removal timing was a design decision |
| Context dependency | 5 | Full stack: MCP cloud, MCP local, tool_models, schemas, test files |
| **Total** | **15** | **→ Sonnet (planning pass), Haiku (execution)** |

**Estimated tokens:** ~200K input (reading all affected files) + 20K output = 220K tokens  
**Cost at Opus:** 200K×$0.0050 + 20K×$0.0250 = **$1.00 + $0.50 = ~$1.50**  
**Cost at Sonnet plan + Haiku execute:** ~$0.75 + ~$0.25 = **~$1.00**  
**Savings:** ~33% — modest because rename is fast even at Opus  
**Confidence (Sonnet/Haiku):** 5/5 for mechanical rename, 4/5 for alias decisions  
**Retry factor:** Low — if Haiku misses a rename, it's a quick grep-and-fix

---

### Cluster 3: Performance Optimizations (deferred columns, column selection)
*PRs: #93, #95, #96 — commits `5b3c479`, `b4a27d8`, `b3cefab`, `c13e0b6`*

| Dimension | Score | Notes |
|-----------|-------|-------|
| File size | 4 | db_queries.py (304 lines), http_server.py (1,089 lines) |
| Semantic complexity | 4 | SQLAlchemy deferred loading, column_property — non-trivial |
| Ambiguity | 3 | Goal clear, but SQLAlchemy patterns require deep understanding |
| Context dependency | 4 | DB model, HTTP handlers, MCP local all interdependent |
| **Total** | **15** | **→ Sonnet** |

**Estimated tokens:** ~80K input + 15K output = 95K tokens  
**Cost at Opus:** ~$0.40 + $0.375 = **~$0.78**  
**Cost at Sonnet:** ~$0.24 + $0.225 = **~$0.47**  
**Savings:** ~40%  
**Confidence (Sonnet):** 4/5  
**Retry factor:** Medium — deferred loading bugs can be subtle

---

### Cluster 4: Security / Auth Hardening (CORS, fail-hard, rate limiting)
*PRs: #92, #93 — commits `73457d4`, `642a759`, `d39167e`, `52d426b`*

| Dimension | Score | Notes |
|-----------|-------|-------|
| File size | 3 | auth.py (50 lines), http_server.py (1,089 lines) |
| Semantic complexity | 3 | New validation + rate limiter module |
| Ambiguity | 2 | Specs clear (fail hard, CORS default) |
| Context dependency | 3 | http_server.py + auth.py |
| **Total** | **11** | **→ Haiku** |

**Estimated tokens:** ~40K input + 10K output = 50K tokens  
**Cost at Opus:** ~$0.20 + $0.25 = **~$0.45**  
**Cost at Haiku:** ~$0.04 + $0.05 = **~$0.09**  
**Savings:** ~80%  
**Confidence (Haiku):** 4/5  
**Retry factor:** Low — specs are explicit

---

### Cluster 5: Documentation Updates (README, AGENTS.md, MCP interface spec, proposals)
*PRs: #100, #101, #97 — commits `587dccf`, `3624db7`, `ba6e7d4`, `843b98d`, `5b3c479`*

| Dimension | Score | Notes |
|-----------|-------|-------|
| File size | 2 | Mostly markdown, <300 lines per file |
| Semantic complexity | 1 | Writing/updating docs |
| Ambiguity | 2 | Some judgment calls on framing |
| Context dependency | 2 | Read code, write docs — no code changes |
| **Total** | **7** | **→ Minimax** |

**Estimated tokens:** ~30K input + 10K output = 40K tokens  
**Cost at Opus:** ~$0.15 + $0.25 = **~$0.40**  
**Cost at Minimax:** ~$0.009 + $0.011 = **~$0.02**  
**Savings:** ~95%  
**Confidence (Minimax):** 5/5  
**Retry factor:** None — docs are easy to review and fix

---

### Cluster 6: Tests (plan_list, test file renames, TYPE_CHECKING fixes)
*PRs: #92, #94 — commits `ad0d339`, `006fc93`, `dd61a58`*

| Dimension | Score | Notes |
|-----------|-------|-------|
| File size | 2 | Test files ~100-200 lines each |
| Semantic complexity | 2 | Writing tests against known API |
| Ambiguity | 1 | Clear — test the documented behavior |
| Context dependency | 2 | One module per test file |
| **Total** | **7** | **→ Minimax / Haiku** |

**Estimated tokens:** ~20K input + 15K output = 35K tokens  
**Cost at Opus:** ~$0.10 + $0.375 = **~$0.475**  
**Cost at Haiku:** ~$0.02 + $0.075 = **~$0.095**  
**Savings:** ~80%  
**Confidence (Haiku):** 5/5  
**Retry factor:** None — tests are deterministic

---

### Cluster 7: Glama / Registry Work
*PRs: #98, parts of #100 — commits `fa811d9`, `587dccf`*

| Dimension | Score | Notes |
|-----------|-------|-------|
| File size | 1 | Config files, small |
| Semantic complexity | 1 | File placement, JSON config |
| Ambiguity | 2 | Trial and error on Glama claim |
| Context dependency | 1 | Self-contained |
| **Total** | **5** | **→ Minimax** |

**Cost at Opus:** ~$0.20  
**Cost at Minimax:** ~$0.01  
**Savings:** ~95%

---

## Cost Summary

| Cluster | Actual model used | Est. cost at Opus | Est. cost optimal | Savings |
|---------|------------------|-------------------|-------------------|---------|
| Module split | Opus | ~$1.50 | ~$0.75 (Opus plan + Sonnet exec) | ~50% |
| External rename | Opus | ~$1.50 | ~$1.00 (Sonnet + Haiku) | ~33% |
| Performance opts | Opus | ~$0.78 | ~$0.47 (Sonnet) | ~40% |
| Security hardening | Opus | ~$0.45 | ~$0.09 (Haiku) | ~80% |
| Documentation | Opus | ~$0.40 | ~$0.02 (Minimax) | ~95% |
| Tests | Opus | ~$0.475 | ~$0.095 (Haiku) | ~80% |
| Glama/registry | Opus | ~$0.20 | ~$0.01 (Minimax) | ~95% |
| **Total** | Opus throughout | **~$5.30** | **~$2.47** | **~53%** |

> **Important note:** These are estimates based on typical token usage patterns. Actual usage depends on session length, context carried between tasks, compaction events, and whether tasks were batched or separate sessions. Simon's actual spend would reflect his real session structure.

---

## Key Findings

**1. Opus was fully justified for planning the module split and external rename.**  
Both involved 1,000+ line files and cross-codebase changes. Opus needed to read `http_server.py` (1,089 lines) and `planexe_mcp_local.py` (1,055 lines) top-to-bottom to produce surgical plans. This is exactly the Opus use case.

**2. Execution of those plans could have shifted to Sonnet/Haiku in a fresh session.**  
Once Opus produces a plan with exact file paths and line numbers, a cheaper model executes it without needing the full planning context. Fresh session = no context drag from prior planning work.

**3. Docs, tests, small fixes, and Glama work are Minimax/Haiku territory.**  
These represent roughly 40% of the commits. Routing them to Minimax ($0.30/$1.10) vs Opus ($5/$25) is a ~95% cost reduction per task.

**4. The >200K token price jump is the real risk.**  
If a session carrying the full app.py monolith context rolls past 200K tokens, Opus input cost doubles from $5 to $10/1M and output from $25 to $37.50. Starting fresh sessions at task boundaries is the single most impactful session hygiene practice.

---

## Recommendations

### The Two-Phase Pattern
```
Phase 1 (Opus, new session):
  - Read all large files relevant to the task
  - Write a surgical plan: file paths, line numbers, exact changes, decisions made
  - End session

Phase 2 (Sonnet/Haiku/Minimax, fresh session):
  - Load only the plan document + target files
  - Execute mechanically
  - No context from Phase 1 carried over
```

### Task Routing Quick Reference
- **Opus**: files >400 lines, cross-module architectural decisions, ambiguous design calls
- **Sonnet**: files 200–600 lines, moderate logic changes, executing a clear plan on complex files
- **Haiku**: files <200 lines, test writing, security config with clear specs, executing rename plans
- **Minimax**: documentation, registry work, boilerplate, simple renames in small files

### When to Start a New Session
- After writing a plan (don't execute in the same session)
- When context exceeds ~150K tokens (approaching the doubling threshold)
- When switching from planning to execution
- When switching from one task cluster to another

---

## What This Is NOT

This is not a suggestion that Simon's workflow was wrong. Using Opus throughout a complex refactor guarantees quality and avoids re-work. The cost of one Haiku failure that requires a Sonnet debugging session can easily exceed the savings. The rubric exists to help **future planning** — knowing upfront which tasks need Opus for the plan, which can execute on Haiku, and where session breaks save money.

Simon shipped 16 PRs in one day. That productivity is worth optimizing around, not second-guessing.

---

*Post-mortem written by Larry. Rubric refinements by Egon. Authorized by Mark.*  
*Next step: Submit as docs-only proposal PR to PlanExeOrg/PlanExe for Simon's review.*
