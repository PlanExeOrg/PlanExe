# Luigi Agents Migration — `.agents/luigi/`

> **Branch:** `feature/luigi-agents-migration`  
> **PR:** [PlanExeOrg/PlanExe #54](https://github.com/PlanExeOrg/PlanExe/pull/54)  
> **Status:** 🔶 Migration in progress — logic ported, updates needed before running

---

## What is Luigi?

[Luigi](https://github.com/spotify/luigi) is an open-source Python task orchestration framework originally built at Spotify. It models work as a directed acyclic graph (DAG) of tasks, where each task declares its inputs, outputs, and dependencies. PlanExe has historically used Luigi's pipeline metaphor to structure its multi-stage planning process — a sequential flow of stages, each composed of discrete task agents.

The agent files in this directory are **TypeScript `AgentDefinition` objects** that mirror that pipeline structure. They don't run Luigi directly; rather, they define the AI agents that own each step in PlanExe's planning DAG.

---

## What's in Here

**73 files total:**

- `AGENTS_REVIEW.md` — Migration review doc (start here for the full picture)
- **11 Stage Lead agents** — Orchestrators that coordinate a stage's task agents
- **62 Task agents** — Individual task workers, one per planning step

Each `.ts` file exports a single `AgentDefinition` object with:
- `id` — Unique agent identifier (e.g. `luigi-createwbslevel1`)
- `displayName` — Human-readable label
- `model` — LLM to use (⚠️ currently hardcoded to `openai/gpt-5` / `openai/gpt-5-mini` — needs updating)
- `toolNames` — Tools available to this agent (e.g. `read_files`, `think_deeply`, `spawn_agents`)
- `instructionsPrompt` — The agent's core system prompt describing its role, inputs, outputs, and handoff
- `spawnableAgents` *(stage leads only)* — List of task agents this orchestrator can spawn
- `includeMessageHistory` — Whether the agent sees prior conversation context

---

## Directory Structure

```
.agents/luigi/
├── AGENTS_REVIEW.md                    ← Full migration analysis & recommendations
│
├── *_stage_lead.ts (11 files)          ← Orchestrators (one per pipeline stage)
│   ├── plan_foundation_stage_lead.ts
│   ├── risk_assumptions_stage_lead.ts
│   ├── strategy_stage_lead.ts
│   ├── team_stage_lead.ts
│   ├── context_stage_lead.ts
│   ├── wbs_schedule_stage_lead.ts
│   ├── analysis_stage_lead.ts
│   ├── documentation_stage_lead.ts
│   ├── expert_quality_stage_lead.ts
│   ├── governance_stage_lead.ts
│   └── reporting_stage_lead.ts
│
└── *-agent.ts (62 files)              ← Task workers (one per pipeline step)
    ├── identifypurpose-agent.ts
    ├── identifyrisks-agent.ts
    ├── createwbslevel1-agent.ts
    └── ... (59 more)
```

### File Naming Convention

| Pattern | Role | Example |
|---|---|---|
| `<stage>_stage_lead.ts` | Stage orchestrator | `wbs_schedule_stage_lead.ts` |
| `<taskname>-agent.ts` | Individual task worker | `createwbslevel1-agent.ts` |

**Note the delimiters:** stage leads use `_` (underscores), task agents use `-` (hyphens) between the task name and `-agent.ts` suffix.

---

## The 11 Pipeline Stages

| # | Stage | Lead File | Purpose |
|---|---|---|---|
| 1 | Plan Foundation | `plan_foundation_stage_lead.ts` | Convert intent → baseline project plan |
| 2 | Risk & Assumptions | `risk_assumptions_stage_lead.ts` | Identify risks & strategic assumptions |
| 3 | Strategy | `strategy_stage_lead.ts` | Develop solution options via lever analysis |
| 4 | Team | `team_stage_lead.ts` | Assemble & profile the project team |
| 5 | Context | `context_stage_lead.ts` | Ground plan in geo/market context |
| 6 | WBS & Schedule | `wbs_schedule_stage_lead.ts` | Build hierarchical work breakdown structure |
| 7 | Analysis & Gating | `analysis_stage_lead.ts` | Quality gate before execution ⚠️ |
| 8 | Documentation | `documentation_stage_lead.ts` | Generate required planning documents |
| 9 | Expert Review | `expert_quality_stage_lead.ts` | SME validation & quality checks |
| 10 | Governance | `governance_stage_lead.ts` | Define governance & decision authorities ⚠️ |
| 11 | Reporting | `reporting_stage_lead.ts` | Generate executive outputs & pitches |

---

## How to Explore

### Step 1: Read the review doc first

```bash
cat .agents/luigi/AGENTS_REVIEW.md
```

`AGENTS_REVIEW.md` contains the full migration analysis: which stages are immediately applicable, what needs refactoring, and a prioritized action list. It's the authoritative map of this directory.

### Step 2: Start with the most applicable stages

The following are **high-confidence, low-effort** stages to explore or activate first:

- **Plan Foundation** (`plan_foundation_stage_lead.ts` + 3 task agents) — Core planning logic, minimal changes needed
- **Risk & Assumptions** (`risk_assumptions_stage_lead.ts` + 4 agents) — Evergreen risk management
- **WBS & Schedule** (`wbs_schedule_stage_lead.ts` + 8 agents) — Solid decomposition logic

### Step 3: Read a stage lead, then its task agents

Stage leads list their `spawnableAgents`. Use those IDs to find the corresponding task agent files. For example:

```ts
// plan_foundation_stage_lead.ts
spawnableAgents: ['luigi-preprojectassessment', 'luigi-projectplan', 'luigi-relatedresources', ...]
```

→ Corresponding files: `preprojectassessment-agent.ts`, `projectplan-agent.ts`, `relatedresources-agent.ts`

---

## Example: What One Agent Does

**`identifypurpose-agent.ts`** — a good representative task agent:

```ts
const definition: AgentDefinition = {
  id: 'luigi-identifypurpose',
  displayName: 'Luigi Identify Purpose Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the IdentifyPurposeTask step inside the Luigi pipeline.
- Stage: Analysis & Gating
- Objective: Distill the main purpose and success criteria for the plan.
- Key inputs: Validated prompt, premise attack outcomes, stakeholder constraints.
- Expected outputs: Statement of purpose, measurable outcomes, scope boundaries.
- Handoff: Provide PlanTypeTask with the clarified mission.`,
  includeMessageHistory: false,
}
```

**What this tells you:**
- It owns exactly one task (`IdentifyPurposeTask`) — Single Responsibility Principle
- It uses lightweight tools: read files + deep thinking
- Its `instructionsPrompt` is self-contained: stage context, inputs, outputs, and handoff are all explicit
- `includeMessageHistory: false` — it gets a clean context (unlike stage leads which use `true`)

This pattern is **consistent across all 62 task agents** — the only things that change are the stage context, objective, inputs/outputs, and handoff target.

---

## What's NOT Ready Yet

These items need to be resolved before the agents can run in production:

### 🔴 Must Fix Before Running

| Issue | Detail |
|---|---|
| **Model versions hardcoded** | All agents reference `openai/gpt-5` or `openai/gpt-5-mini` — update to current available models |
| **Tool abstractions** | `read_files` needs to resolve to the actual file/storage API in PlanExe 2026 |
| **Analysis & Gating stage** | Human approval loop design not confirmed for PlanExe 2026 architecture |
| **Governance stage** | Needs alignment with PlanExe 2026 governance charter before reuse |

### 🟡 Design Decisions Pending

| Issue | Detail |
|---|---|
| **Output format strategy** | All agents output Markdown — final destination (PowerPoint, PDF, Confluence, etc.) not yet bound |
| **HRIS/ERP integrations** | Team stage assumes enterprise HRIS connections not yet wired |
| **Error handling** | No retry logic or escalation chains defined |

### 🟢 Nice to Have (Later)

| Issue | Detail |
|---|---|
| **Parallel execution** | Some stages could run concurrently (WBS + Schedule, Context + Team) |
| **Feedback loops** | Currently one-pass; iterative refinement not yet modelled |
| **Bayesian scheduling** | WBS duration estimation is deterministic; three-point estimates would improve accuracy |

See `AGENTS_REVIEW.md` for the full priority breakdown (P1/P2/P3).

---

## Overall Applicability

Per the migration review (2026-02-20):

| Category | Count | Reusable? | Effort |
|---|---|---|---|
| Stage Leads | 11 | ✅ ~90% | Medium |
| Core Planning (WBS, Schedule, Risk) | 20 | ✅ ~95% | Low–Medium |
| Team & Context | 10 | ✅ ~90% | Medium |
| Documentation & Output | 17 | ⚠️ ~60% | High |
| Governance & Gates | 8 | ⚠️ ~70% | High |
| **Total** | **73** | **~80% core logic reusable** | **Medium overall** |

---

## Contributing / Next Steps

1. **Pick a stage** from the high-applicability list above
2. **Update the model ref** (`openai/gpt-5` → your current model)
3. **Validate tool names** against PlanExe 2026's registered tool registry
4. **Test the stage lead** by running it against a sample planning brief
5. **Document what you changed** — update `AGENTS_REVIEW.md` with your findings

---

*Migrated from `82deutschmark/PlanExe:staging2` · Review completed 2026-02-20 · PR #54*
