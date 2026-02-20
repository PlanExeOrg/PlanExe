/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-setup',
  displayName: 'Luigi Setup Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the SetupTask step inside the Luigi pipeline.
- Stage: Analysis & Gating (Establish safe operating conditions, clarify purpose, and set up the run before strategy work.)
- Objective: Validate filesystem layout, confirm pipeline config, and prepare directories/files relied on by later tasks.
- Key inputs: Run metadata from StartTimeTask and pipeline configuration defaults.
- Expected outputs: Directory scaffolding status, configuration sanity notes, blockers for gating tasks.
- Handoff: Notify RedlineGateTask agent about any required mitigations before diagnostics.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for analysis-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
