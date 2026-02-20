/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-potentiallevers',
  displayName: 'Luigi Potential Levers Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the PotentialLeversTask step inside the Luigi pipeline.
- Stage: Strategic Lever Development (Shape strategic levers and scenarios that drive how the plan tackles the mission.)
- Objective: Enumerate potential strategic levers aligned with the clarified goal and constraints.
- Key inputs: Plan type, purpose, contextual hints, historical lever catalogs.
- Expected outputs: Draft lever list with tags, assumptions, and data needs.
- Handoff: Provide raw lever set to DeduplicateLeversTask with notes on overlaps.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for strategy-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
