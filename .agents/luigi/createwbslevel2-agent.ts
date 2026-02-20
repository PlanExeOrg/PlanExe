/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-createwbslevel2',
  displayName: 'Luigi Create W B S Level2 Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the CreateWBSLevel2Task step inside the Luigi pipeline.
- Stage: WBS & Scheduling (Decompose work, map dependencies, estimate durations, and create the timeline.)
- Objective: Break down Level 1 elements into detailed Level 2 components.
- Key inputs: Level 1 WBS, resource constraints, assumptions.
- Expected outputs: Level 2 WBS items with dependencies and deliverables.
- Handoff: Supply to WBSProjectLevel1AndLevel2Task for integration and to Level 3 agent.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for wbs-schedule-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
