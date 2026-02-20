/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-governancephase6extra',
  displayName: 'Luigi Governance Phase6 Extra Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the GovernancePhase6ExtraTask step inside the Luigi pipeline.
- Stage: Governance Architecture (Define governance structures, escalation paths, and monitoring routines.)
- Objective: Capture supplemental governance measures such as compliance, legal, or cultural safeguards.
- Key inputs: Monitoring plan, outstanding risks, stakeholder requirements.
- Expected outputs: Extended governance actions and contingency preparations.
- Handoff: Provide to ConsolidateGovernanceTask for packaging.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for governance-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
