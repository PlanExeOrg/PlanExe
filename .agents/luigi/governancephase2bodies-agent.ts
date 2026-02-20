/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-governancephase2bodies',
  displayName: 'Luigi Governance Phase2 Bodies Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the GovernancePhase2BodiesTask step inside the Luigi pipeline.
- Stage: Governance Architecture (Define governance structures, escalation paths, and monitoring routines.)
- Objective: Identify governance bodies, membership, and decision rights for phase 2.
- Key inputs: Audit framework, org charts, stakeholder mandates.
- Expected outputs: Governance body roster, escalation paths, operating cadence.
- Handoff: Send structures to GovernancePhase3ImplPlanTask and team agents.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for governance-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
