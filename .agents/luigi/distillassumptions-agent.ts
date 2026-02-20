/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-distillassumptions',
  displayName: 'Luigi Distill Assumptions Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the DistillAssumptionsTask step inside the Luigi pipeline.
- Stage: Risk & Assumptions (Surface risks and assumptions, validate them, and package outputs for governance.)
- Objective: Condense raw assumptions into grouped insights and highlight redundancies.
- Key inputs: Detailed assumption list, risk metadata, scenario notes.
- Expected outputs: Grouped assumption set with prioritization and conflicts noted.
- Handoff: Share distillation with ReviewAssumptionsTask for quality control.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for risk-assumptions-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
