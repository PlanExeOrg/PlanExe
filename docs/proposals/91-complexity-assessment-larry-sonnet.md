# Complexity Assessment: Simon's 26 Feb 2026 PlanExe Refactor
## Larry's View (Claude Sonnet 4.6)

**Author:** Larry (Claude Sonnet 4.6)  
**Date:** 2026-02-26  
**Purpose:** Independent complexity scoring for model routing post-mortem  
**For Simon's review:** Please score each cluster 1–5 on actual difficulty and compare to my estimates

---

## Rubric Reminder

| Dimension | 1 | 2 | 3 | 4 | 5 |
|-----------|---|---|---|---|---|
| File size | <100 lines | 100–300 | 300–600 | 600–1000 | 1000+ |
| Semantic complexity | rename/replace | simple logic | new function | architectural | cross-file refactor w/ deps |
| Ambiguity | crystal clear + line numbers | minor choices | some design calls | significant decisions | open-ended |
| Context dependency | self-contained | 1 file | 1 module | multi-module | whole codebase |

**Score → Model:** 4–7 Minimax · 8–11 Haiku · 12–15 Sonnet · 16–20 Opus

---

## Task Cluster Assessments

### Cluster A: Module Split (app.py monolith → 10 focused modules)
*Key commit: `9f1a7db` — "refactor: split mcp_cloud/app.py into 10 focused modules"*

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| File size | **5** | Original app.py was documented as 76KB — massive monolith |
| Semantic complexity | **5** | Architectural: defining module boundaries, splitting responsibilities, fixing all import chains, ensuring no circular deps |
| Ambiguity | **3** | Goal clear (split the monolith), but WHERE to draw module lines required genuine design judgment — e.g., does `download_tokens.py` own its rate limiter or does `http_server.py`? |
| Context dependency | **5** | Every file that imports anything from app.py needed updating; changes ripple through mcp_local too |
| **Total** | **18** | **→ Opus justified for planning AND likely for execution** |

**Larry's verdict:** This is the clearest Opus use case in the whole day. A 76KB monolith with import chains across a cloud+local architecture — Sonnet would have made boundary errors on this one. Opus earns its rate here.

---

### Cluster B: External API Rename (task_id → plan_id, TASK_* → PLAN_*)
*Key commits: `0dbe1af`, `0f2e9cc`, `3663bc6`, `1e714e2`, `befa6cb`*

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| File size | **5** | Touches planexe_mcp_local.py (1,055 lines) and http_server.py (1,089 lines) — both 1,000+ line files |
| Semantic complexity | **3** | Mechanically a rename, but must maintain backward-compat aliases during transition, update error codes, update test fixtures, keep external API stable |
| Ambiguity | **2** | The goal was crystal clear. The one design decision (alias removal timing) was small |
| Context dependency | **5** | Full stack: mcp_cloud handlers, schemas, tool_models, mcp_local, test files, docs |
| **Total** | **15** | **→ Sonnet for planning, Haiku for execution** |

**Larry's verdict:** The file sizes push this toward Opus, but the actual work is mechanical once you understand the scope. The right play: Opus reads both 1,000-line files top-to-bottom, maps every occurrence, writes a hit list. Haiku or Sonnet executes from that hit list. The backward-compat alias timing decision was the only thing requiring real judgment.

---

### Cluster C: Performance Optimizations (deferred columns, column selection)
*Key commits: `5b3c479`, `b4a27d8`, `b3cefab`, `c13e0b6`, `2e12c47`*

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| File size | **4** | db_queries.py (304 lines), http_server.py (1,089 lines) — one very large file |
| Semantic complexity | **4** | SQLAlchemy deferred column loading is non-trivial. `column_property` vs `deferred()` patterns, session alive/commit interactions, lazy load timing — this is domain-specific knowledge |
| Ambiguity | **3** | Goal clear (don't load 25MB+ per query), but the implementation pattern (has_* properties, column_property checks) required design calls |
| Context dependency | **4** | DB model layer, HTTP handlers, MCP local, worker — multi-module |
| **Total** | **15** | **→ Sonnet** |

**Larry's verdict:** This one I'd give to Sonnet. The SQLAlchemy patterns are specific enough that Haiku might fumble the session-alive edge cases (as the bug fix `c13e0b6` suggests — someone had to fix a session-alive issue after the initial perf PR). Sonnet handles this, but fresh session for the fix work.

---

### Cluster D: Security & Auth Hardening (CORS, fail-hard, rate limiting)
*Key commits: `73457d4`, `642a759`, `d39167e`, `52d426b`*

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| File size | **3** | auth.py is only 50 lines; most changes are in http_server.py but confined to specific functions |
| Semantic complexity | **3** | New validation logic (fail-hard on missing secrets), new rate limiter bucket — well-understood patterns |
| Ambiguity | **2** | Specs explicit: fail loudly in prod, CORS default to known domains, rate limit at 10/60s |
| Context dependency | **3** | auth.py + http_server.py — single module |
| **Total** | **11** | **→ Haiku** |

**Larry's verdict:** With a clear spec ("raise RuntimeError if PLANEXE_API_KEY_SECRET not set when AUTH_REQUIRED=true"), Haiku can write this. The rate limiter is a copy-paste adaptation of the existing MCP limiter with different thresholds. Solid Haiku territory.

---

### Cluster E: Railway & Deployment Fixes (env vars, TYPE_CHECKING, column_property)
*Key commits: `992e0a6`, `006fc93`, `dd61a58`, `9085ffd`*

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| File size | **3** | Changes confined to specific methods in medium-sized files |
| Semantic complexity | **2** | TYPE_CHECKING import guard, column_property placement after class body — known Python patterns |
| Ambiguity | **2** | Deployment failures with clear error messages guide the fix |
| Context dependency | **3** | SQLAlchemy model file + Railway config |
| **Total** | **10** | **→ Haiku** |

**Larry's verdict:** These are bug fixes with clear error messages pointing to the root cause. Haiku with the error message and the relevant file section can nail these without needing the full codebase context.

---

### Cluster F: Documentation (README, AGENTS.md, MCP interface spec, proposals)
*Key commits: `587dccf`, `3624db7`, `ba6e7d4`, `843b98d`, `cf89030`, `5acaa23`*

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| File size | **2** | Markdown files, all under 300 lines |
| Semantic complexity | **1** | Writing docs, updating spec, fixing typos |
| Ambiguity | **2** | "More accurate explanation of PlanExe" requires some judgment on framing |
| Context dependency | **2** | Read code → write docs, no code changes |
| **Total** | **7** | **→ Minimax** |

**Larry's verdict:** Minimax. The only judgment call is "what's the most accurate way to describe PlanExe?" — that's a writing task, not a coding task. Feed Minimax the relevant code sections and ask for accurate documentation.

---

### Cluster G: Tests (plan_list tests, test file renames, TYPE_CHECKING test fixes)
*Key commits: `ad0d339`, test file renames in `befa6cb`*

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| File size | **2** | Test files typically 100–200 lines |
| Semantic complexity | **2** | Writing tests against documented behavior — mock setup, assertion patterns |
| Ambiguity | **1** | Tests verify known behavior; no design decisions |
| Context dependency | **2** | One module under test per file |
| **Total** | **7** | **→ Minimax/Haiku** |

**Larry's verdict:** These are the clearest Minimax opportunity in the whole day. The behavior is already documented. Feed Minimax the handler function and the expected outputs; it writes the tests. No judgment required.

---

### Cluster H: Glama/Registry Work
*Key commits: `fa811d9` (Glama experiments), parts of `587dccf`*

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| File size | **1** | Config files, JSON |
| Semantic complexity | **1** | File placement, JSON config |
| Ambiguity | **2** | Trial-and-error on claiming — some uncertainty, but no code |
| Context dependency | **1** | Self-contained |
| **Total** | **5** | **→ Minimax** |

**Larry's verdict:** Pure Minimax. This is basically following instructions on a website and placing config files. If anything, a human assistant could do this faster than any AI.

---

## Summary Table

| Cluster | File size | Semantic | Ambiguity | Context | **Total** | **Model** |
|---------|-----------|----------|-----------|---------|-----------|-----------|
| A: Module split | 5 | 5 | 3 | 5 | **18** | Opus |
| B: API rename | 5 | 3 | 2 | 5 | **15** | Sonnet plan / Haiku exec |
| C: Perf opts | 4 | 4 | 3 | 4 | **15** | Sonnet |
| D: Security | 3 | 3 | 2 | 3 | **11** | Haiku |
| E: Deploy fixes | 3 | 2 | 2 | 3 | **10** | Haiku |
| F: Docs | 2 | 1 | 2 | 2 | **7** | Minimax |
| G: Tests | 2 | 2 | 1 | 2 | **7** | Minimax |
| H: Glama | 1 | 1 | 2 | 1 | **5** | Minimax |

---

## Overall Assessment

**Simon's work was genuinely complex.** Clusters A and B alone represent the most architecturally difficult type of refactor: reading 1,000+ line files, maintaining backward compatibility during renames across a full cloud+local stack, and splitting a monolith cleanly.

**Opus was justified for the planning phase of clusters A and B.** Not because the execution is impossible for cheaper models, but because the *planning* — reading the full files, understanding all the interdependencies, producing a surgical hit list — requires Opus's ability to hold a large codebase in context without losing the thread.

**~55% of the commit count (clusters D-H) was Haiku/Minimax territory** with good plans in hand.

**The two-phase pattern saves the most money on exactly this type of day:**
1. Opus: read the big files, write precise plans (file + line + exact change)
2. Fresh session, Haiku/Minimax: execute from the hit list

---

## Questions for Simon

1. Was the module split (cluster A) actually as ambiguous as I scored it (3/5 on ambiguity)? Or did you have a clear spec for where each function should land?
2. Did the SQLAlchemy deferred column work (cluster C) require significant domain knowledge, or was it more mechanical than it looks from the diff?
3. Were the security/auth changes (cluster D) spec'd out explicitly before you started, or did you need to design as you went?
4. Would you say the overall complexity was higher, lower, or about right compared to my scores?

---

*Written by Larry (Claude Sonnet 4.6). For comparison with Egon (Minimax M2.5) and Bubba (Haiku 4.5) assessments.*
