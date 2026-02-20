/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-plantype',
  displayName: 'Luigi Plan Type Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the PlanTypeTask step inside the Luigi pipeline.
- Stage: Analysis & Gating (Establish safe operating conditions, clarify purpose, and set up the run before strategy work.)
- Objective: Categorize the plan type and maturity level to guide branching logic and resource allocation.
- Key inputs: Purpose summary, historical templates, gating diagnostics.
- Expected outputs: Plan taxonomy selection, reasoning, and implications for lever exploration.
- Handoff: Signal PotentialLeversTask about focus areas and complexity expectations.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for analysis-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
