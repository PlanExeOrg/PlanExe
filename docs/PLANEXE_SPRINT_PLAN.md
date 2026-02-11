# PlanExe Sprint Plan (2‑Week Outline from Simon + Bot)

**Source:** #openclaw channel responses from Simon + Simon Strandgaard Bot (2026‑02‑08)

## Executive Summary
Simon wants to shift PlanExe toward a **ranking/metrics pipeline** for plans and a **production‑ready MCP cloud** with a **Stripe top‑up flow**. The bot supplied a detailed technical plan for **KPI extraction + Elo ranking**, including DB schema changes, endpoints, and a light UI/exports. Simon’s core concern: **don’t let plans game a static weighted sum**. Use a **2‑phase LLM evaluation** (raw KPI extraction → pairwise comparison) to produce Elo updates.

---

## Stated Priorities (from Simon)
**Priority 1:** Get `mcp_cloud` live on Railway
**Priority 2:** Stripe top‑up flow + credit UI (quick win)
**Priority 3:** KPI extraction scaffolding (foundation for ranking)

---

## Two‑Week Sprint Outline (Bot)
### Week 1
- **Mon**: Implement Stripe top‑up flow + credit UI
- **Tue**: Add `/api/credit` and webhook handler
- **Wed**: Write minimal “metrics viewer” page for debugging
- **Thu**: Refactor `mcp_cloud` to call KPI module after plan generation and store metrics
- **Fri**: Deploy to Railway; smoke test (create plan → credit deducted)

### Week 2
- **Mon**: Start KPI extraction module from plan JSON
- **Tue**: Add `plan_metrics` API endpoint; unit tests
- **Wed**: Expand ranking (Elo), leaderboard UI
- **Thu**: Add export endpoint (`/api/export`) for top‑N plans
- **Fri**: Documentation + demo video

---

## Ranking Engine Design (Key Takeaways)
### Why: Prevent Gaming
Static linear weights can be gamed. Solution: **two‑phase LLM evaluation**.

### Phase 1 – Raw KPI Extraction (0‑1 continuous)
**KPI examples:**
- novelty_score (embedding similarity)
- prompt_quality (LLM rating)
- technical_completeness (WBS depth + dependencies)
- feasibility (cost vs budget)
- impact_estimate (ROI placeholder)

### Phase 2 – Pairwise LLM Comparison
LLM compares KPI vectors for Plan A vs Plan B → returns Likert preference.
Use that to update **Elo** for both plans.

### Sampling Strategy
Avoid O(N²) comparisons.
- Bucket‑wise tournament (100 plans per bucket)
- Optional ANN nearest‑neighbor pre‑filter (PGVector/Faiss)
- Sample ~10–20 neighbors per new plan

---

## Concrete Implementation Items
### DB
- `plan_corpus` table (seeded from PlanExe-web examples)
- `plan_metrics` table with KPI columns + `elo` + optional comment

### Backend Endpoints
- `POST /api/rank` (rank a plan, update Elo)
- `GET /api/leaderboard` (top plans by Elo)
- `GET /api/export` (JSON dump of top‑N)
- (optional) `GET /metrics/<plan_id>`

### UI
- `rankings.html` (simple Jinja table pulling from leaderboard)

### Testing
- `docker-compose.test.yml` to smoke test `/api/rank → /api/leaderboard`

---

## Gaps / Open Questions to Resolve
1. **Which priority gets staffed first?** (MCP cloud vs Stripe vs ranking)
2. **LLM provider for KPI extraction?** (OpenAI/Anthropic/Gemini) and budget limits
3. **Plan corpus source of truth?** (PlanExe-web YAML vs DB vs MCP outputs)
4. **Vector store decision?** (PGVector vs Faiss vs none)
5. **KPI definitions** (weights, normalizations, thresholds) – final spec needed
6. **Human‑in‑the‑loop?** (store LLM justification + manual audit of random samples)
7. **Security & abuse**: Rate‑limit `/api/rank` so users can’t spam Elo
8. **Deployment**: Who owns Railway creds and rollout timeline?

---

## Suggested Next Actions (for Larry)
- Confirm with Simon which **priority** is highest to start coding first
- Ask whether **LLM scoring** should use OpenAI or Claude
- Decide if **vector store** is required for v1 or can be deferred
- Draft the **migration scripts** + `plan_metrics` schema
- Build the **ranking MVP** (KPI stub + Elo updates + leaderboard API)

---

## Notes
- Simon’s concern: **weighted sum can be gamed** → must use **LLM‑derived KPIs + pairwise comparison**.
- Bot offered to provide ready‑made diffs for ranking engine components.
