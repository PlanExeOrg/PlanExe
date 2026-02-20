/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-identifyrisks',
  displayName: 'Luigi Identify Risks Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the IdentifyRisksTask step inside the Luigi pipeline.
- Stage: Risk & Assumptions (Surface risks and assumptions, validate them, and package outputs for governance.)
- Objective: Enumerate material risks tied to scenarios, locations, and resources.
- Key inputs: Scenario markdown, lever notes, context briefs.
- Expected outputs: Risk register entries with likelihood, impact, and triggers.
- Handoff: Deliver to MakeAssumptionsTask and governance agents for mitigation alignment.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for risk-assumptions-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
