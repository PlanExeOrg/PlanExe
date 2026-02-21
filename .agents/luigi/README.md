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

## Getting Started — Running the Agents

### Prerequisites

- **Node.js 18+** and npm (for TypeScript compilation)
- **Python 3.9+** and `pip` (for PlanExe runtime)
- **Docker & Docker Compose** (for running full PlanExe stack)
- **Git** (for cloning and checking out branches)

### Step 1: Set Up the Environment

#### Clone and check out the Luigi branch

```bash
git clone https://github.com/PlanExeOrg/PlanExe.git
cd PlanExe
git checkout feature/luigi-agents-migration
```

#### Install Node.js dependencies (for TypeScript compilation)

```bash
npm install
```

#### Install Python dependencies (for PlanExe runtime)

```bash
python3 -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate (Windows)
pip install -r requirements.txt
```

### Step 2: Compile the Luigi Agents

The agent definitions are written in TypeScript. Compile them:

```bash
npx tsc .agents/luigi/*.ts --target es2020 --module commonjs --outDir .agents/luigi/dist/
```

This generates compiled `.js` files in `.agents/luigi/dist/`.

### Step 3: Run a Single Task Agent

#### Option A: Invoke an agent directly via Node.js

Each agent exports an `AgentDefinition` object. You can load and inspect it:

```bash
node -e "
const agent = require('./.agents/luigi/dist/identifypurpose-agent.js');
console.log(JSON.stringify(agent.definition, null, 2));
"
```

This prints the agent's metadata:
```json
{
  "id": "luigi-identifypurpose",
  "displayName": "Luigi Identify Purpose Agent",
  "model": "openai/gpt-5-mini",
  "toolNames": ["read_files", "think_deeply", "end_turn"],
  "instructionsPrompt": "..."
}
```

#### Option B: Via OpenClaw / Agent Framework

If running within the OpenClaw agent system:

```bash
openclaw invoke --tool spawn-agent \
  --args-json '{
    "agentId": "luigi-identifypurpose",
    "task": "Given this brief, identify the main purpose and success criteria...",
    "context": { "plan_brief": "..." }
  }'
```

(Requires OpenClaw configured with PlanExe agent registry.)

### Step 4: Run a Full Stage

#### Option A: Sequentially via Node.js script

Create a test script (`test-stage.js`):

```javascript
const fs = require('fs');
const path = require('path');

// Load the Plan Foundation stage lead
const stageLead = require('./.agents/luigi/dist/plan_foundation_stage_lead.js');

console.log(`\n=== ${stageLead.definition.displayName} ===`);
console.log(`Model: ${stageLead.definition.model}`);
console.log(`Spawnable agents: ${stageLead.definition.spawnableAgents.join(', ')}`);
console.log(`\nInstructions:\n${stageLead.definition.instructionsPrompt.substring(0, 200)}...\n`);

// Load task agents
const taskAgentIds = stageLead.definition.spawnableAgents;
console.log(`\n--- Task Agents in This Stage ---\n`);
taskAgentIds.forEach(id => {
  const filename = id.replace(/^luigi-/, '').replace(/(.)([A-Z])/g, '$1-$2').toLowerCase() + '-agent.js';
  const agentPath = path.join('.', '.agents', 'luigi', 'dist', filename);
  
  if (fs.existsSync(agentPath)) {
    try {
      const agent = require(path.resolve(agentPath));
      console.log(`  ✓ ${agent.definition.id}`);
      console.log(`    Display: ${agent.definition.displayName}`);
      console.log(`    Model: ${agent.definition.model}`);
    } catch (e) {
      console.log(`  ✗ ${id} (failed to load)`);
    }
  } else {
    console.log(`  ? ${id} (file not found: ${filename})`);
  }
});
```

Run it:

```bash
node test-stage.js
```

Output:
```
=== Luigi Plan Foundation Stage Lead ===
Model: openai/gpt-5
Spawnable agents: luigi-preprojectassessment,luigi-projectplan,luigi-relatedresources

--- Task Agents in This Stage ---

  ✓ luigi-preprojectassessment
    Display: Luigi Pre-Project Assessment Agent
    Model: openai/gpt-5-mini
    
  ✓ luigi-projectplan
    Display: Luigi Project Plan Agent
    Model: openai/gpt-5
```

#### Option B: Via Docker + PlanExe

Run the full PlanExe stack:

```bash
docker-compose up -d
```

Then submit a plan and monitor agent invocations:

```bash
curl -X POST http://127.0.0.1:8000/runs \
  -H "Content-Type: application/json" \
  -d '{
    "submit_or_retry": "submit",
    "plan_prompt": "Build a small poultry egg operation in Connecticut",
    "llm_model": "auto",
    "speed_vs_detail": "ping"
  }'
```

The agent system will invoke Luigi agents during pipeline execution. Check logs:

```bash
docker-compose logs -f worker_plan | grep -E "(luigi-|stage_lead)"
```

### Step 5: Validate Agent Definitions

Run the validation script to check all agents for:
- Missing required fields
- Circular dependencies in stage leads
- Invalid model references

```bash
node scripts/validate-agents.js .agents/luigi/dist/
```

Example output:
```
✓ 73 agents validated
✓ 11 stage leads form valid DAG
⚠ 45 agents reference deprecated model 'openai/gpt-5' (should update to 'gpt-4-turbo')
✓ All tool references resolve
```

### Step 6: Update Models & Tools

**⚠️ CRITICAL BEFORE RUNNING IN PRODUCTION:**

1. **Update model references:**
   ```bash
   sed -i "s/'openai\/gpt-5'/'anthropic\/claude-opus-4'/g" .agents/luigi/*.ts
   sed -i "s/'openai\/gpt-5-mini'/'anthropic\/claude-haiku'/g" .agents/luigi/*.ts
   npm run build  # Recompile
   ```

2. **Validate tools against PlanExe 2026:**
   - Map `read_files` → actual file storage API
   - Map `think_deeply` → structured reasoning tool
   - Map `spawn_agents` → sub-agent orchestration
   - See `AGENTS_REVIEW.md` section "Tool Abstractions" for full mapping

3. **Test with a sample plan:**
   ```bash
   # Use a small test prompt
   curl -X POST http://127.0.0.1:8000/runs \
     -H "Content-Type: application/json" \
     -d '{
       "submit_or_retry": "submit",
       "plan_prompt": "Quick 2-day workshop planning",
       "llm_model": "auto",
       "speed_vs_detail": "ping"
     }'
   ```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| **"Cannot find module"** after compilation | Run `npx tsc` again; check `.agents/luigi/dist/` exists |
| **TypeScript compilation errors** | Ensure Node.js 18+: `node --version` |
| **Agent loads but won't run** | Check model is available (gpt-5 likely needs update) |
| **Tools not recognized** | Validate against PlanExe 2026's tool registry (see Step 6) |
| **Stage lead spawns nothing** | Task agents may not have compiled; check dist/ directory |

---

## Next Steps to Productionize

1. ✅ **Load & inspect agents** (you are here)
2. ⏳ **Update model versions** (required before running)
3. ⏳ **Map tools to PlanExe 2026 APIs** (required before running)
4. ⏳ **Test a single stage** (recommend: Plan Foundation or Risk & Assumptions)
5. ⏳ **Run full pipeline** (test suite + production stage-by-stage)
6. ⏳ **Document findings** in AGENTS_REVIEW.md

---

## Contributing / Next Steps

1. **Pick a stage** from the high-applicability list in the main README
2. **Compile & test it** using the scripts above
3. **Update the model ref** (`openai/gpt-5` → your current model)
4. **Validate tool names** against PlanExe 2026's registered tool registry
5. **Test the stage lead** by running it against a sample planning brief
6. **Document what you changed** — update `AGENTS_REVIEW.md` with your findings

---

*Migrated from `82deutschmark/PlanExe:staging2` · Review completed 2026-02-20 · PR #54*
