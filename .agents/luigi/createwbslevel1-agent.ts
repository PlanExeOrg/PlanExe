/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-createwbslevel1',
  displayName: 'Luigi Create W B S Level1 Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the CreateWBSLevel1Task step inside the Luigi pipeline.
- Stage: WBS & Scheduling (Decompose work, map dependencies, estimate durations, and create the timeline.)
- Objective: Define top-level WBS structure covering major workstreams.
- Key inputs: Project plan outline, scenario decisions, governance constraints.
- Expected outputs: Level 1 WBS entries with descriptions and ownership cues.
- Handoff: Provide to CreateWBSLevel2Task to extend hierarchy.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for wbs-schedule-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
