/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-identifytaskdependencies',
  displayName: 'Luigi Identify Dependencies Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the IdentifyTaskDependenciesTask step inside the Luigi pipeline.
- Stage: WBS & Scheduling (Decompose work, map dependencies, estimate durations, and create the timeline.)
- Objective: Determine logical dependencies across WBS tasks to inform sequencing.
- Key inputs: Master WBS dataset, risk and governance constraints.
- Expected outputs: Dependency map annotated with critical path considerations.
- Handoff: Share with EstimateTaskDurationsTask and schedule builders.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for wbs-schedule-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
