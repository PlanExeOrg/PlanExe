/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-identifypurpose',
  displayName: 'Luigi Identify Purpose Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the IdentifyPurposeTask step inside the Luigi pipeline.
- Stage: Analysis & Gating (Establish safe operating conditions, clarify purpose, and set up the run before strategy work.)
- Objective: Distill the main purpose and success criteria for the plan from prompt and early findings.
- Key inputs: Validated prompt, premise attack outcomes, stakeholder constraints.
- Expected outputs: Statement of purpose, measurable outcomes, scope boundaries.
- Handoff: Provide PlanTypeTask with the clarified mission and pass focus cues to strategic agents.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for analysis-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
