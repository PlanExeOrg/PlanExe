/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-wbsprojectlevel1andlevel2',
  displayName: 'Luigi W B S Project Level1 And Level2 Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the WBSProjectLevel1AndLevel2Task step inside the Luigi pipeline.
- Stage: WBS & Scheduling (Decompose work, map dependencies, estimate durations, and create the timeline.)
- Objective: Integrate Level 1 and Level 2 WBS data into consistent project structures.
- Key inputs: Level 1 and Level 2 WBS outputs, project plan metadata.
- Expected outputs: Combined WBS model ready for deeper decomposition.
- Handoff: Provide to CreateWBSLevel3Task and scheduling agents.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for wbs-schedule-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
