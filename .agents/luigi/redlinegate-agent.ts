/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-redlinegate',
  displayName: 'Luigi Redline Gate Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the RedlineGateTask step inside the Luigi pipeline.
- Stage: Analysis & Gating (Establish safe operating conditions, clarify purpose, and set up the run before strategy work.)
- Objective: Run the redline gate diagnostics to catch fatal issues early and document gating criteria.
- Key inputs: Prepared environment state and configuration from SetupTask.
- Expected outputs: Pass/fail judgement with rationale, recommended mitigations, gating metrics.
- Handoff: Escalate failure paths to orchestrator and provide PremiseAttackTask with edge cases to probe.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for analysis-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
