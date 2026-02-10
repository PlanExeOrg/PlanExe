# Plan: "Smart On The Fly" Agent Routing (Business vs Software)

This is a concrete implementation plan for making PlanExe's agent behavior adapt **on the fly** to whether the user's request is primarily a **business plan** or a **software plan**, with different *levers*, *gates*, and *deliverables* per type.

## 1) Current State (What This Repo Already Does)

PlanExe already has multiple "early classification" concepts and quality gates that we can build on:

- **Purpose classification (business/personal/other)**: `worker_plan/worker_plan_internal/assume/identify_purpose.py` produces `002-6-identify_purpose.md` and is already used downstream (e.g., SWOT prompt selection).

- **Plan type classification (digital/physical)**: `worker_plan/worker_plan_internal/assume/identify_plan_type.py` produces `002-8-plan_type.md`. Note: it intentionally labels most software development as "physical" (because it assumes a physical workspace/devices).

- **Levers pipeline**: `worker_plan/worker_plan_internal/lever/*` produces potential levers -> deduped -> enriched -> "vital few" -> scenarios/strategic decisions.

- **Quality gates already exist**:

  - Redline gate / premise attack: `worker_plan/worker_plan_internal/diagnostics/*`

  - Self-audit checklist includes "Lacks Technical Depth", "Legal Minefield", "External Dependencies", etc.: `worker_plan/worker_plan_internal/self_audit/self_audit.py`

- **MCP interface is tools-only** and supports `task_create -> task_status -> task_file_info/task_download`: `mcp_cloud/app.py`, `mcp_local/planexe_mcp_local.py`, and `docs/planexe_mcp_interface.md`.

- **LLM configuration is externalized** (profiles in `llm_config.json`, default via `DEFAULT_LLM` env var; keys from `.env`): `worker_plan/worker_plan_internal/llm_factory.py`, `worker_plan/worker_plan_internal/utils/planexe_llmconfig.py`, `worker_plan/worker_plan_api/planexe_dotenv.py`.

### The gap
We do **not** currently classify "business plan vs software plan" as a first-class routing decision, even though:

- the downstream artifacts and "what good looks like" differ heavily, and

- the SelfAudit's "Lacks Technical Depth" (#9) is a strong hint we *want* deeper software gating when appropriate.

## 2) Target Behavior (What "Smart On The Fly" Means)

Given a single prompt, PlanExe should:

1) **Determine focus**: business plan vs software plan (or hybrid).

2) **Select a planning track**:

   - Business track: market/GTM/unit economics/ops/legal emphasis

   - Software track: requirements/architecture/security/testing/deployment/observability emphasis

   - Hybrid: do both, but explicitly separate them and sequence decisions

3) **Use different levers + different "gates"**:

   - Levers = "what knobs can we turn?"

   - Gates = "what must be true before we proceed / what is a NO-GO?"

4) **Surface the decision early** so downstream tasks can be shaped accordingly (and so the user can override it).

## 3) Proposed New Classification: Plan Focus

### 3.1 Output schema (conceptual)
Add a structured classification step that outputs:

- `plan_focus`: `business | software | hybrid | unknown`

- `confidence`: `high | medium | low`

- `reasons`: short bullets grounded in the user prompt

- `missing_info`: short list (used to ask clarifying questions *only when needed*)

- `override_hint`: a single sentence telling the user how to override (e.g., "Say: 'Treat this as a software plan'")

### 3.2 Inputs
Use the user prompt plus existing early outputs:

- `plan.txt` (user prompt)

- `purpose.md` (business/personal/other)

- `plan_type.md` (digital/physical)

### 3.3 Decision rules (practical)
Use a two-stage approach:

1) **Cheap deterministic heuristic** (fast, no LLM):

   - If prompt contains strong software signals (APIs, architecture, codebase, deployment, infra, testing, SLOs, data model, auth, migrations, etc.), mark `software` unless business signals dominate.

   - If prompt contains strong business signals (pricing, GTM, CAC/LTV, TAM/SAM/SOM, margins, channel, sales motion, market positioning, competition, fundraising), mark `business`.

   - If both are strong, mark `hybrid`.

2) **LLM tie-breaker** only when heuristic confidence is low.

This keeps cost and latency down and avoids adding fragility.

## 4) Track-Specific Levers (What We Generate)

The "IdentifyPotentialLevers" stage is the most obvious place to diverge by track.

### 4.1 Software plan lever set (examples)
Levers that must exist (or be strongly represented) for software-focused prompts:

1) Product scope slicing & release strategy

2) Architecture & service boundaries (monolith/modular/services)

3) Data model & consistency strategy

4) Integration strategy (3rd parties, protocols, contracts)

5) Security/privacy posture (authn/authz, secrets, threat model)

6) Reliability targets (SLOs/SLAs), observability, incident response

7) Testing strategy (unit/integration/e2e), CI/CD, environments

8) Deployment strategy (cloud/on-prem), rollout/rollback

### 4.2 Business plan lever set (examples)
Levers that must exist (or be strongly represented) for business-focused prompts:

1) Target segment & positioning

2) Pricing & packaging

3) Channel strategy (PLG/sales/partners/marketplaces)

4) Unit economics & cost structure

5) Operating model & hiring plan

6) Regulatory/legal constraints (if applicable)

7) Customer discovery & validation strategy

8) Competitive differentiation & moat

### 4.3 Hybrid
Hybrid plans should *explicitly* separate:

- Business model decisions (what to build + why + how to sell)

- Software execution decisions (how to build + how to ship + how to operate)

## 5) Track-Specific Gates (What We Must Verify)

PlanExe already has a strong "gate" concept via SelfAudit + diagnostics. The plan here is to **re-weight and re-frame** the gating based on track, without breaking existing output contracts.

### 5.1 Software gates (NO-GO style)
Before committing to "execute":

- Requirements clarity: scoped MVP + non-goals

- Architecture artifacts exist: interfaces/contracts + data model + integration map

- Security: threat model + authn/authz + secrets strategy

- Testability: acceptance criteria + test plan

- Operations: deployment plan + monitoring + incident response

- Dependencies: critical third parties have fallback or mitigation

### 5.2 Business gates (NO-GO style)

- Clear ICP + buyer/user distinction

- Pricing hypothesis + rough unit economics

- Channel feasibility (how customers actually arrive)

- Validation plan (customer discovery / pilots)

- Legal/regulatory feasibility (as needed)

- Operational capacity (team, hiring, suppliers)

## 6) Where This Fits in the Pipeline (Minimal Disruption)

Do not change the public service contracts (per repo guardrails). Instead:

- Insert the Plan Focus decision **after** `IdentifyPurposeTask` and `PlanTypeTask`, and **before** lever generation.

- Feed the Plan Focus markdown into:

  - IdentifyPotentialLevers

  - Risks/assumptions framing

  - ReviewPlan and SelfAudit emphasis (so software plans get stronger #9/#17/#14 behavior)

No MCP interface changes are required: the client still sends one prompt to `task_create`.

## 7) MCP/Client UX ("Smart On The Fly" for Agents)

### 7.1 mcp_cloud / mcp_local
Keep tools-only behavior. "Smartness" lives in PlanExe's pipeline and in how prompts are structured.

### 7.2 Prompt examples
Add/curate prompt examples that clearly represent:

- a software build (backend + frontend + deployment + requirements)

- a business plan (GTM + pricing + ops + financial model)

- a hybrid "build a SaaS" prompt that forces the split

This improves agent behavior without requiring new tools.

## 8) Implementation Phases (Deliverables-First)

Phase 0 - Doc-only (this file)

- Document the target behavior, levers, gates, and integration points.

Phase 1 - Deterministic Plan Focus classifier

- Add a small, dependency-free classifier (stdlib only) in `worker_plan_internal` (not `worker_plan_api`).

- Unit-test it with a dozen prompts (software/business/hybrid).

Phase 2 - LLM tie-breaker (optional)

- Add a structured output model for low-confidence cases only.

- Ensure it's robust across providers in `llm_config.json` (structured output required).

Phase 3 - Track-aware lever and gate prompting

- Update the lever-generation query to include "Plan Focus" context.

- Re-weight SelfAudit framing for software vs business (without changing the checklist items or output format).

Phase 4 - Measure + iterate

- Add lightweight telemetry in logs: detected focus + confidence + user override (if any).

- Evaluate false positives/negatives against real prompts.

## 9) Validation Strategy

- Unit tests for classifier determinism (no LLM required).

- "Golden prompt" fixtures: a small set of prompts whose Plan Focus classification should remain stable.

- Manual smoke runs using `speed_vs_detail=ping` and `speed_vs_detail=fast` via MCP tools (keeps cost down).

## 10) Guardrails (Must Not Break)

- Keep `worker_plan_api` lightweight: no new heavy deps or service imports.

- Keep `worker_plan` HTTP endpoints backward compatible.

- Do not touch `open_dir_server` allowlist/path validation unless explicitly asked.

- Do not change MCP to advertise tasks protocol ("Run as task") - tools-only stays.
