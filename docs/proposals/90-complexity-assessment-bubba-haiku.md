# Complexity Assessment: Simon's 26 Feb 2026 PlanExe Refactor
## Bubba's View (Claude Haiku 4.5)

**Author:** Bubba (Claude Haiku 4.5) — written by Larry on Bubba's behalf (Bubba offline)
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

## Haiku's Perspective on This Codebase

From my vantage point as Haiku: a 1,000-line file is genuinely risky territory. I can read it, but holding the entire semantic graph of a complex Python module while also tracking what I'm changing is where I start making mistakes. My honest view is that this codebase is on the harder end — it's not tutorial code. It has SQLAlchemy, FastMCP, async patterns, layered auth, and cross-file dependencies. I'll flag more things as "needs Sonnet or above" than Larry probably did.

---

## Task Cluster Assessments

### Cluster A: Module Split (app.py monolith → 10 focused modules)

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| File size | **5** | 76KB monolith. I wouldn't attempt to read this fully without errors. |
| Semantic complexity | **5** | Splitting a monolith means understanding every import, every circular dependency risk, every caller. This is not a task I should be near. |
| Ambiguity | **4** | Even with a clear goal, the module boundary decisions require design judgment I don't have context for. Where does auth end and http_server begin? |
| Context dependency | **5** | Touch one thing in the wrong way and half the codebase breaks silently. |
| **Total** | **19** | **→ Opus, no question** |

**Bubba's verdict:** Don't even think about giving this to me or Minimax. Sonnet might handle execution with a *very* precise plan, but I'd say Opus for planning AND at least Sonnet for execution. The risk of a subtle circular import or missed caller is too high to trust a small model.

---

### Cluster B: External API Rename (task_id → plan_id, TASK_* → PLAN_*)

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| File size | **5** | Both planexe_mcp_local.py and http_server.py are 1,000+ lines. I'll miss occurrences in files this large. |
| Semantic complexity | **3** | It's a rename, but the backward-compat alias timing adds risk. One wrong removal and external callers break. |
| Ambiguity | **3** | The rename direction is clear. The alias removal timing is a real decision that could silently break things. |
| Context dependency | **5** | Cloud, local, tests, docs — full stack. Miss one file and you ship a broken release. |
| **Total** | **16** | **→ Opus for planning. Sonnet for execution with an explicit hit list.** |

**Bubba's verdict:** I'd want Opus to produce a complete grep-style list of every occurrence across every file before a single character is changed. Sonnet then executes from that list. I shouldn't be doing the discovery phase on a 1,000+ line file — I'll miss things.

---

### Cluster C: Performance Optimizations (deferred columns, column selection)

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| File size | **4** | db_queries.py (304 lines) is manageable. http_server.py (1,089 lines) is not. |
| Semantic complexity | **5** | SQLAlchemy deferred loading is expert territory. Session lifecycle, lazy load timing, has_* property placement — these are patterns I can execute if told exactly where to put code, but I can't design them. |
| Ambiguity | **4** | The goal is clear ("don't load 25MB per query") but the implementation required knowing *which* SQLAlchemy pattern to use. That's a design call. |
| Context dependency | **4** | DB model → queries → HTTP handlers → MCP local all interdependent. |
| **Total** | **17** | **→ Opus for planning. Sonnet for execution.** |

**Bubba's verdict:** The bug fix in `c13e0b6` (keep session alive after initial deferred load PR) tells me this cluster bit someone already. That kind of subtle bug — session closes before deferred columns load — is exactly the thing I'd introduce. Opus needs to think through the SQLAlchemy session lifecycle before anyone writes code.

---

### Cluster D: Security & Auth Hardening (CORS, fail-hard, rate limiting)

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| File size | **3** | auth.py is tiny (50 lines). Changes in http_server.py are localized. |
| Semantic complexity | **3** | RuntimeError on missing secrets, CORS default to known domains, rate limit bucket — well-understood patterns. |
| Ambiguity | **2** | Specs are explicit: fail loudly in prod, known CORS origins only, 10 req/60s download limit. |
| Context dependency | **3** | auth.py + specific sections of http_server.py. |
| **Total** | **11** | **→ Haiku (me!)** |

**Bubba's verdict:** This is squarely in my wheelhouse if I have the spec. "Add `validate_api_key_secret()` that raises RuntimeError when env var missing, call it at module load in http_server.py" — that's a 20-line change with a clear spec. I can do this. Rate limiter is a copy-paste adaptation of existing code with different thresholds.

---

### Cluster E: Railway & Deployment Fixes (TYPE_CHECKING, column_property)

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| File size | **3** | Changes are localized methods in medium files. |
| Semantic complexity | **2** | TYPE_CHECKING guard pattern and column_property-after-class-body placement — known Python patterns once you know what the error means. |
| Ambiguity | **2** | Deployment error messages point directly to the fix. |
| Context dependency | **2** | One file per fix, basically. |
| **Total** | **9** | **→ Haiku** |

**Bubba's verdict:** Give me the error message and the file. I'll fix it. These are "the error tells you exactly where the problem is" bugs. I handle these fine.

---

### Cluster F: Documentation

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| File size | **1** | Markdown, all small |
| Semantic complexity | **1** | Writing docs |
| Ambiguity | **2** | Some framing judgment |
| Context dependency | **1** | Read code, write docs |
| **Total** | **5** | **→ Minimax** |

**Bubba's verdict:** Not my job. Minimax.

---

### Cluster G: Tests

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| File size | **2** | Test files ~100–200 lines |
| Semantic complexity | **2** | Mock setup, assertions against known API |
| Ambiguity | **1** | Documented behavior = clear spec |
| Context dependency | **2** | One module per test file |
| **Total** | **7** | **→ Minimax/Haiku borderline** |

**Bubba's verdict:** I could write these, but honestly Minimax can too if you feed it the handler signature and expected outputs. Save me for something harder.

---

### Cluster H: Glama/Registry

| Dimension | Score | Reasoning |
|-----------|-------|-----------|
| File size | **1** | Config JSON |
| Semantic complexity | **1** | File placement |
| Ambiguity | **2** | Trial and error |
| Context dependency | **1** | Self-contained |
| **Total** | **5** | **→ Minimax** |

**Bubba's verdict:** Minimax.

---

## Summary Table

| Cluster | File size | Semantic | Ambiguity | Context | **Total** | **Model** |
|---------|-----------|----------|-----------|---------|-----------|-----------|
| A: Module split | 5 | 5 | 4 | 5 | **19** | Opus (plan + exec) |
| B: API rename | 5 | 3 | 3 | 5 | **16** | Opus plan / Sonnet exec |
| C: Perf opts | 4 | 5 | 4 | 4 | **17** | Opus plan / Sonnet exec |
| D: Security | 3 | 3 | 2 | 3 | **11** | Haiku |
| E: Deploy fixes | 3 | 2 | 2 | 2 | **9** | Haiku |
| F: Docs | 1 | 1 | 2 | 1 | **5** | Minimax |
| G: Tests | 2 | 2 | 1 | 2 | **7** | Minimax |
| H: Glama | 1 | 1 | 2 | 1 | **5** | Minimax |

---

## Overall Assessment (Haiku's View)

This codebase is harder than it looks. The SQLAlchemy patterns, the layered auth, the cloud+local symmetry requirement — these aren't beginner-level concerns. I scored clusters A, B, and C higher than Larry probably did because from where I sit, those file sizes and cross-file dependencies are genuinely risky.

**My model routing recommendation is more conservative than Sonnet's will be.** That's appropriate — a model should be honest about its own limits. A Haiku that overestimates its capability causes bugs and re-work that costs more than just using Sonnet in the first place.

**The clusters where I'm confident I can deliver:** D (security hardening with clear spec), E (deploy fixes with error messages), F (docs), G (tests), H (Glama). That's roughly 40% of the work by commit count — meaningful savings even from the cautious view.

---

## Questions for Simon

1. Did the SQLAlchemy deferred column work (cluster C) require significant expertise, or was it more mechanical? I scored this as Opus-level planning — was that right?
2. Were there bugs introduced during this refactor that required follow-up fixes? That would validate the higher complexity scores.
3. Would you have trusted Haiku to write the security hardening code (cluster D) given an explicit spec? Or does production security code always deserve a higher model?

---

*Written by Larry on Bubba's behalf (Haiku 4.5 perspective). For comparison with Larry (Sonnet 4.6) and Egon (Minimax M2.5) assessments.*
