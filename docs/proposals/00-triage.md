# PlanExe Proposal Triage — 80:20 Analysis & Strategic Gaps

**Date:** 26 February 2026  
**Authors:** Larry + Egon  
**Status:** Ready for Review & Discussion  

---

## Overview

Simon asked us to triage the proposal space with an 80:20 lens. This document captures:

1. **Which proposals deliver outsized value** — the 20% that unlock 80% of the architecture
2. **Related proposals nearby in the graph** — which ones can reuse artifacts or reasoning
3. **High-leverage parameter tweaks** — code tweaks and second/third order effects
4. **Critical gaps** — what's missing that should be proposed
5. **Relevant questions** — what Simon might not be asking yet
6. **Actionable tasks** — what we can execute proactively

We focused on recent proposals ("67+" cluster) plus the validation/orchestration story that **FermiSanityCheck** will unlock.

---

## High-Leverage Proposals (The 20%)

These 5-7 proposals drive 80% of architectural value:

| # | Title | Lines | Why It Matters | Role |
|---|-------|-------|----------------|------|
| **07** | Elo Ranking System | 1,751 | Core ranking mechanism for comparing plan quality across all downstream use cases | Foundation |
| **63-66** | Orchestration Cluster (Luigi integration, post-plan orchestration, enrichment swarm) | 2,000+ | Controls how PlanExe schedules, retries, and enriches the Luigi DAG. Any new validation task (like FermiSanityCheck) ripples here | Critical Path |
| **62** | Agent-First Frontend Discoverability | 609 | Defines agent UX; depends on ranking (#07) and validation signals | Interface |
| **69** | Arcgentica Agent Patterns | 279 | Hardens PlanExe via recursion + typed returns + soft eval; now references FermiSanityCheck | Hardening |
| **41** | Autonomous Execution of Plan | 200+ | Distributed execution of plans; feeds back to ranking and reporting | Execution |
| **05** | Semantic Plan Search Graph | 200+ | Retrieval and discoverability; powers agent discovery (#62) | Search |

**These interlock around:** Planning quality signals (#07, #69, FermiSanityCheck) → Orchestration (#63-66) → Interfaces (#62, #41, #05).

---

## Reuse Opportunities & Adjacent Clusters

### Proposal Pairs That Share Heuristics

**#07 Elo Ranking + #62 Agent-First Frontend**
- Both score/rank plans but likely with independent heuristics
- **Quick win:** Extract ranking weights (cost, feasibility, confidence) to shared module
- Both proposals reference the same quality signals
- FermiSanityCheck flags become features in both scoring systems

**#38 Risk Propagation Network + #44 Investor Grade Audit Pack**
- Both model risk but likely with different terminology
- **Quick win:** Create unified `RiskModel` Pydantic schema; both consume it
- Eliminates duplicate code, improves consistency
- Single source of truth for risk visualization

### Validation Cluster (Interconnected)

**#69 Arcgentica + #56 Adversarial Red Team + #43 Assumption Drift Monitor**
- FermiSanityCheck = front-line validation (quantitative grounding)
- Red-team (#56) = logic/feasibility check (qualitative, adversarial)
- Drift monitor (#43) = tracking invalidation over time
- All three consume validation_report.json and escalate to human review

**#63-66 Orchestration Cluster (Internal Coherence)**
- #63 (Luigi integration), #64 (post-plan orchestration), #66 (enrichment swarm) all touch DAG
- Currently unclear how they fit together
- **Quick win:** Add relationship map to each showing dependencies and integration points

---

## 80:20 Quick Wins (Parameter Tweaks, No Rewrites)

### Category 1: Ranking Tuning

**#07 Elo Ranking: Adjust Scoring Weights**
- **Current:** Unknown weighting between cost, feasibility, risk, confidence
- **Quick win:** Expose weights as configurable; test 3 profiles (cost-optimized, risk-minimized, time-critical)
- **Effort:** 2-4 hours
- **Impact:** Same engine serves different user priorities; agents can ask "what's your priority?"
- **Integration:** FermiSanityCheck confidence becomes a penalty/bonus in ranking

**#07 + #62: Share Ranking Heuristics**
- **Current:** Likely duplicated ranking logic across ranking (#07) and UI discovery (#62)
- **Quick win:** Extract heuristics to shared module
- **Effort:** 2-3 hours
- **Impact:** Consistent rankings across all surfaces; single source of truth

### Category 2: DAG & Distribution Tuning

**#32 Gantt Parallelization: Parameterize Lane Constraints**
- **Current:** Unknown lane count, cost thresholds
- **Quick win:** Make lane count, parallel cost ceiling, task-splitting configurable
- **Effort:** 1-2 hours
- **Impact:** One algorithm covers 80% of Gantt use cases (startup vs. enterprise scale)
- **Integration:** Reuse FermiSanityCheck heuristics for duration plausibility

**#36-37 Monte Carlo: Adjust Distribution Models**
- **Current:** Likely simple distributions for uncertainty
- **Quick win:** Expose distribution type (uniform, normal, triangular) per cost category
- **Effort:** 1-2 hours
- **Impact:** More realistic cost/timeline distributions without rewriting

### Category 3: Alignment & Terminology

**#38 + #44: Risk Terminology Unification**
- **Current:** Both define risk differently
- **Quick win:** Create shared `RiskModel` schema; both reference it
- **Effort:** 2-3 hours
- **Impact:** Eliminate duplicate code, improve consistency, enable shared visualization

**#63-66: Orchestration Relationship Mapping**
- **Current:** Three proposals all touch orchestration; unclear how they fit
- **Quick win:** Add "Relationship Map" section to each showing integration order
- **Effort:** 1 hour
- **Impact:** Clear implementation order, reduced confusion

---

## Second/Third Order Effects

### Direct (1st Order)
- **FermiSanityCheck validates quantified claims** immediately after MakeAssumptions
- Prevents garbage-in → garbage-out in downstream tasks
- Reduces manual Simon review overhead

### 2nd Order (Downstream Reliability)
- **#36-37 Monte Carlo:** With validated bounds, success probability distributions are grounded in evidence, not guesses
- **#38 Risk Propagation:** With better assumptions, risk models catch *real* threats
- **#41 Autonomous Execution:** Plans with validated assumptions have higher real-world success
- **#43 Assumption Drift:** Baseline assumptions are now credible; drift detection has a trustworthy reference
- **#44 Investor Audit:** Reports cite evidence-backed numbers → credibility increases
- **#07 Ranking:** FermiSanityCheck confidence becomes a scoring feature; better plans rank higher

### 3rd Order (Strategic Implications)

Once assumptions are validated + execution tracked + drift monitored:

- **PlanExe becomes a learning system**
  - Early execution failures → identify assumption gaps → improve MakeAssumptions prompts
  - Drift detection → trigger re-planning before catastrophic failure
  - Execution feedback → inform future FermiSanityCheck heuristics
  
- **Trust compounding:** Validated → Executed → Monitored → Improved → Trusted (virtuous cycle)

- **Competitive moat:** If execution feedback loops work, PlanExe plans become *more valuable over time* (vs. one-shot consulting)

- **Validation observability:** Dashboard tracking how many plans pass/fail FermiSanityCheck each week; historical trends inform product roadmap

---

## Critical Gaps in Existing Proposals

### Gap 1: Execution Feedback Loop (Missing Entirely)

**What's missing:**
- #41 (Autonomous Execution) says "run the plan" but doesn't specify how execution metrics feed back into planning
- #43 (Assumption Drift) says "monitor" but doesn't spec *when* to re-plan vs. *when* to adjust-on-the-fly

**Impact:** Without feedback, PlanExe is write-once (create, execute, done). With feedback, it's learning (create, execute, learn, re-plan).

**Recommendation:** Create **#55 Execution Feedback Loop** proposal defining:
- KPI tracking during execution (% of plan met, timeline vs. actual, cost vs. actual)
- Criteria for "assumption invalidated" (what % deviation triggers re-plan?)
- Re-planning strategy (full re-plan vs. localized fix?)
- Feedback → prompt improvement (how execution data improves future MakeAssumptions)

---

### Gap 2: Cost Estimation Validation (Order-of-Magnitude Reconciliation)

**What's missing:**
- #34 (Finance top-down) and #35 (bottom-up) estimate costs but don't validate against each other
- No proposal says "if top-down ≠ bottom-up by >2×, escalate to human"

**Impact:** Cost overruns (most common plan failure) aren't caught early.

**Recommendation:** Extend FermiSanityCheck to **estimation reconciliation**:
- Extract top-down and bottom-up cost estimates
- Flag if diverge by >50% (signal of missing scope or hidden assumptions)
- Do the same for timeline

---

### Gap 3: Stakeholder Alignment Pre-Plan (Assumption Validation)

**What's missing:**
- MakeAssumptions generates assumptions but doesn't validate with stakeholders first
- Risk: plan is based on wrong assumptions; stakeholders reject after weeks of work

**Impact:** Wasted plan generation effort + stakeholder friction.

**Recommendation:** Create **#57 Assumption Validation with Stakeholders** proposal:
- After MakeAssumptions, surface top 5 assumptions for stakeholder thumbs-up/down
- If 2+ rejected, re-generate before investing in full plan
- Saves 90% of downstream rework

---

### Gap 4: Plan Complexity Budgeting (Don't Over-Plan)

**What's missing:**
- No proposal caps plan complexity
- Risk: 3-month projects get 50 WBS tasks + governance phases (overkill)

**Impact:** High plan generation cost + user friction for simple projects.

**Recommendation:** Create **#58 Complexity-Gated Pipeline** proposal:
- After MakeAssumptions, estimate project complexity (cost × duration × stakeholders)
- Route simple projects → "fast track" (WBS only, no expert review)
- Route medium → "standard" (WBS + governance)
- Route complex → "full" (everything)
- Reduces cost/time for 60% of plans by 70%

---

### Gap 5: Software Plan Specialization (Code Bridge)

**What's missing:**
- PlanExe optimized for business/organizational plans
- Software projects need code-specific artifacts (architecture, API contracts, testing, deployment)
- #06 (Adopt on the fly) tries but doesn't go deep

**Impact:** Software founders get generic business plans, not actionable dev roadmaps.

**Recommendation:** Create **#59 Software Plan Specialization** proposal:
- Detect "this is a software project" in MakeAssumptions
- Route through software-specific WBS (architecture → MVP → iteration)
- Generate deployment checklist, testing strategy, tech debt budget
- Integrate with code search/LODA for architectural patterns

---

## Missing Proposals (Should Exist)

| # | Title | Why Needed | Effort | Leverage |
|---|-------|-----------|--------|----------|
| **55** | Execution Feedback Loop | Close the learn-from-execution cycle | High | **High** (enables learning moat) |
| **56** | Adversarial Red-Team Reality Check | Questions assumptions from hostile angle | High | **Medium** (complements Fermi) |
| **57** | Assumption Validation with Stakeholders | De-risk assumptions early | Medium | **High** (saves 80% rework) |
| **58** | Complexity-Gated Pipeline | Don't over-engineer simple projects | Medium | **High** (70% cost reduction for 60% of plans) |
| **59** | Software Plan Specialization | Bridge PlanExe ↔ code | High | **Medium** (critical for dev teams) |
| **60** | Plan Versioning & Comparison | Track assumption/scope changes | Low | **Medium** (useful for ongoing projects) |
| **61** | Validation Observability Dashboard | Real-time view of live plans | Medium | **Medium** (stakeholder trust) |
| **62** | Aggregate Data Strategy | How to retain/use execution data | High | **High** (data moat) |
| **63** | Competitive Positioning | vs. manual consulting | Medium | **High** (informs GTM) |
| **64** | Megaproject Scalability | 10-year plans with quarterly re-planning | High | **Medium** (TAM expansion) |
| **65** | Cost Reduction Curve | Margin improvement with scale | Medium | **High** (profitability) |

---

## Strategic Questions Simon Might Not Be Asking

### Question 1: Unit Economics of Plan Generation

**Context:** Proposals discuss implementation effort but don't tie to revenue.

**Unanswered:**
- How much does a plan cost to generate (tokens, compute)?
- How much value does a plan create for users (time saved, better decisions)?
- At what project size does PlanExe ROI break even?
- Does ROI flip at $100k vs. $1M vs. $100M projects?

**Why it matters:**
- If generating a plan costs $500 in API calls, you need $5k+ project value to be worthwhile
- Determines TAM and go-to-market strategy
- Affects which proposals are worth building (e.g., #58 Complexity-Gating becomes *essential*)

**Recommendation:** Create cost accounting for plan generation (token count per task, cost breakdown). Model unit economics by project size. Use that to prioritize proposals.

---

### Question 2: Data Moat Strategy

**Context:** PlanExe collects execution data (via #41-43), but there's no proposal about what to do with it.

**Unanswered:**
- Do you retain execution data? How long?
- Can you compare user outcomes to benchmarks ("your project ran 10% over budget; industry is 15% over")?
- Can you use aggregate data to improve MakeAssumptions? (e.g., "e-commerce projects have 3× cost overruns → adjust baselines")
- Is data a defensible moat, or just overhead?

**Why it matters:**
- If you have a moat, PlanExe gets *better over time* (vs. static)
- Justifies continued investment in execution feedback loops
- Competitors can't easily copy without historical data

**Recommendation:** Create **#62 Aggregate Data Strategy** proposal defining what data to retain, how to anonymize for benchmarking, how aggregate insights improve plan quality.

---

### Question 3: Competitive Positioning vs. Consulting

**Context:** A human consultant also creates plans; they're just slower and more expensive.

**Unanswered:**
- What does PlanExe do *better* than human consulting?
- Speed (same quality, faster)? Comprehensiveness? Consistency? Complement?
- If PlanExe is "fast but lower quality," TAM = small projects only
- If "same quality, faster, cheaper," direct displacement (larger TAM)
- If "complement," revenue model is add-on (lower ceiling)

**Why it matters:** Informs go-to-market messaging and TAM sizing.

**Recommendation:** Create **#63 Competitive Positioning** proposal defining quality bar vs. manual, competitive advantages, and strategic positioning.

---

### Question 4: Megaproject Scalability (10-Year Plans)

**Context:** Most proposals assume 6-24 month projects.

**Unanswered:**
- Can PlanExe generate 10-year plans with quarterly re-planning?
- Can it handle strategic pivots (market shifts → re-scope everything)?
- Can it track across 40+ quarterly re-plans?

**Why it matters:**
- If yes, TAM includes infrastructure, government, aerospace (high-value)
- If no, capped at 2-3 year projects (smaller TAM)

**Recommendation:** Create **#64 Megaproject Scalability** proposal defining quarterly re-plan mechanics, scope change propagation, and cost/token budgets.

---

### Question 5: Margin Structure & Cost Reduction Curve

**Context:** Early PlanExe usage is expensive (high API calls). Improves with volume?

**Unanswered:**
- Can caching/retrieval (LODA, semantic search) reduce generation cost?
- If you've planned 100 SaaS startups, does plan 101 cost 50% less (template reuse)?
- Does margin improve *compound* with scale?

**Why it matters:**
- If margin improves with scale, you have a winner (more volume → cheaper → more volume)
- If flat, need high volume to be profitable

**Recommendation:** Create **#65 Cost Reduction Curve** proposal modeling token usage plan 1 vs. plan 100, estimating savings from semantic search + template reuse, setting margin targets.

---

## Relevant Tasks We Can Execute Now

### Task 1: Proposal Dependency Graph (Executable)

**What:** Build visual/code showing which proposals depend on which.

**Current state:** Unknown.

**Output:** Visual DAG + implementation order.

**Why it matters:** Shows critical paths, which proposals unblock 3+ others.

**Effort:** 4-6 hours.

---

### Task 2: Estimation Reconciliation (FermiSanityCheck Extension)

**What:** Validate top-down vs. bottom-up cost/time estimates.

**Current state:** FermiSanityCheck validates individual claims; doesn't cross-check.

**Output:** Flag when top-down ≠ bottom-up by >50%; diagnostic suggestions.

**Why it matters:** Cost overruns are #1 plan failure mode.

**Effort:** 2-3 hours (once FermiSanityCheck built).

---

### Task 3: Proposal Sizing Matrix (Quick Win Inventory)

**What:** 2×2 matrix for all 54 proposals: effort (1-10) vs. impact (1-10).

**Current state:** Unknown.

**Output:** Top-left quadrant (low effort, high impact) = quick wins to tackle first.

**Why it matters:** Transparent prioritization; efficient resource allocation.

**Effort:** 3-4 hours.

---

### Task 4: FermiSanityCheck Implementation (Core)

**What:** Build the Luigi task + QuantifiedAssumption schema + validation report.

**Current state:** Plan approved; awaiting implementation.

**Output:** Validated assumptions + validation_report.json.

**Why it matters:** Gates quality of all downstream tasks.

**Effort:** 5-6 hours (Egon data model + Larry validation logic + integration).

---

### Task 5: Validation Observability Dashboard (Reporting)

**What:** Dashboard showing FermiSanityCheck pass/fail rates, trends, failure categories.

**Current state:** Not built.

**Output:** Weekly reports: "80 plans passed, 12 failed quantitative grounding, top 3 failure modes: ___".

**Why it matters:** Tracks product health; informs what assumptions users typically miss.

**Effort:** 4-6 hours (once FermiSanityCheck live).

---

### Task 6: Stakeholder Assumption Validation UI (Pre-Plan)

**What:** After MakeAssumptions, surface top 5 assumptions for stakeholder sign-off.

**Current state:** Not built.

**Output:** "Please confirm these 5 assumptions" → yes/no/discuss → re-run if rejected.

**Why it matters:** De-risks expensive full plan generation; increases stakeholder buy-in.

**Effort:** 3-4 hours.

---

### Task 7: Complexity Gating Implementation (Cost Optimization)

**What:** Route simple projects through fast-track, complex through full.

**Current state:** Not built.

**Output:** Complexity score function + route definitions (fast/standard/full).

**Why it matters:** Reduces cost 70% for 60% of plans.

**Effort:** 4-6 hours.

---

## Recommended Execution Path

### Phase 1 (This Week): Stabilization + Fermi
1. Implement FermiSanityCheck (Larry + Egon, ~5-6 hours)
2. Create Proposal Sizing Matrix (1 of us, ~3-4 hours)
3. Build Proposal Dependency Graph (1 of us, ~4-6 hours)

**Goal:** Clear picture of high-leverage vs. nice-to-have.

### Phase 2 (Next Week): Quick Wins
1. Parameterize #07 Elo-Ranking weights (~2-4 hours)
2. Merge #38 + #44 Risk terminology (~2-3 hours)
3. Document #63-66 relationships (~1 hour)

**Goal:** Deliver 80:20 impact without big rewrites.

### Phase 3 (Week After): Strategic Proposals
1. Write #55 (Execution Feedback Loop)
2. Write #57 (Stakeholder Assumption Validation)
3. Write #58 (Complexity-Gated Pipeline)

**Goal:** Fill critical gaps; unlock learning/scalability.

---

## Sign-Off

- [ ] Simon: Priorities approved?
- [ ] Simon: Any gaps or unanswered questions?
- [ ] Larry/Egon: Ready to execute Phase 1?

---

*This triage is a living document. Updates as Simon provides feedback or new context emerges.*

---

**Appendix: Full Proposal List (For Reference)**

All 54 + new proposals organized by theme:

**Core Pipeline (1-8):** 01-agent-smart-routing, 02-plans-as-LLM-templates, 03-distributed-plan-execution, 04-plan-explain-as-API-service, 05-semantic-plan-search-graph, 06-adopt-on-the-fly, 07-elo-ranking, 08-ui-for-editing-plan

**Capital/Investors (11-15):** 11-investor-thesis-matching-engine, 12-evidence-based-founder-execution-index, 13-portfolio-aware-capital-allocation, 14-confidence-weighted-funding-auctions, 15-outcome-feedback-and-model-governance

**Plugins (16-20):** 16-on-demand-plugin-synthesis-hub, 17-plugin-adaptation-lifecycle, 18-plugin-benchmarking-coverage-harness, 19-plugin-safety-governance-for-runtime-loading, 20-plugin-hub-discovery-ranking-and-reuse

**Experts/Verification (21-30):** 21-expert-discovery-and-fit-scoring, 22-multi-stage-verification-workflow, 23-expert-collaboration-marketplace-and-reputation, 24-cross-border-project-verification-framework, 25-verification-incentives-governance-and-liability, 26-news-intake-and-opportunity-sensing-grid, 27-multi-angle-topic-verification-engine, 28-autonomous-bid-factory-orchestration, 29-elo-ranked-bid-selection-and-escalation, 30-autonomous-bid-governance-risk-and-ethics

**Finance/Estimation (31-37):** 31-token-counting-and-cost-transparency, 32-gantt-parallelization-and-fast-tracking, 33-cost-breakdown-structure-cbs, 34-finance-top-down-estimation, 35-finance-bottom-up-estimation-and-reconciliation, 36-monte-carlo-plan-success-probability-engine, 37-cashflow-and-funding-stress-monte-carlo

**Risk/Analysis (38-46):** 38-risk-propagation-network-and-failure-modes, 39-frontier-research-gap-mapper-for-megaprojects, 40-three-hypotheses-engine-for-unsolved-challenges, 41-autonomous-execution-of-plan, 42-evidence-traceability-ledger, 43-assumption-drift-monitor, 44-investor-grade-audit-pack-generator, 45-counterfactual-scenario-explorer, 46-execution-readiness-scoring

**Infrastructure (47-54):** 47-openclaw-agent-skill-integration, 48-moltbook-reputation-bridge, 49-distributed-physical-task-dispatch-protocol, 50-agent-to-agent-payment-gateway, 51-decentralized-planexe-survivability, 52-mcp-oauth, 53-uuid-only-task-id, 54-agent-safety-trusted-information

**New Proposals (To Be Created):** 55-execution-feedback-loop, 56-adversarial-red-team-reality-check, 57-assumption-validation-with-stakeholders, 58-complexity-gated-pipeline, 59-software-plan-specialization, 60-plan-versioning-and-comparison, 61-validation-observability-dashboard, 62-aggregate-data-strategy, 63-competitive-positioning, 64-megaproject-scalability, 65-cost-reduction-curve
