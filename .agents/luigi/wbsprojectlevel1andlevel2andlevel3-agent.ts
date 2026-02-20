/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-wbsprojectlevel1andlevel2andlevel3',
  displayName: 'Luigi W B S Project Level1 And Level2 And Level3 Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the WBSProjectLevel1AndLevel2AndLevel3Task step inside the Luigi pipeline.
- Stage: WBS & Scheduling (Decompose work, map dependencies, estimate durations, and create the timeline.)
- Objective: Consolidate all WBS levels into a single coherent structure for downstream tooling.
- Key inputs: WBS outputs across levels 1-3, project plan data.
- Expected outputs: Master WBS dataset with traceability links.
- Handoff: Deliver to IdentifyTaskDependenciesTask and estimation agents.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for wbs-schedule-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
