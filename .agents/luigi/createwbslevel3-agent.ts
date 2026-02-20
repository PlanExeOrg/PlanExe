/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-createwbslevel3',
  displayName: 'Luigi Create W B S Level3 Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the CreateWBSLevel3Task step inside the Luigi pipeline.
- Stage: WBS & Scheduling (Decompose work, map dependencies, estimate durations, and create the timeline.)
- Objective: Extend WBS into Level 3 tasks to support scheduling and estimation.
- Key inputs: Integrated Level 1-2 WBS, assumption and risk inputs.
- Expected outputs: Level 3 task list with ownership and success criteria.
- Handoff: Pass to WBSProjectLevel1AndLevel2AndLevel3Task for consolidation.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for wbs-schedule-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
