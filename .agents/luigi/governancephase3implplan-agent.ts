/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-governancephase3implplan',
  displayName: 'Luigi Governance Phase3 Impl Plan Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the GovernancePhase3ImplPlanTask step inside the Luigi pipeline.
- Stage: Governance Architecture (Define governance structures, escalation paths, and monitoring routines.)
- Objective: Outline governance implementation actions, tooling, and timelines.
- Key inputs: Bodies roster, audit requirements, project plan milestones.
- Expected outputs: Implementation tasks, tool stack recommendations, integration checkpoints.
- Handoff: Provide to later governance phases for monitoring alignment.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for governance-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
