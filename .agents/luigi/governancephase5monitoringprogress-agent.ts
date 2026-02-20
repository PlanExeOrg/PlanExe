/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-governancephase5monitoringprogress',
  displayName: 'Luigi Governance Phase5 Monitoring Progress Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the GovernancePhase5MonitoringProgressTask step inside the Luigi pipeline.
- Stage: Governance Architecture (Define governance structures, escalation paths, and monitoring routines.)
- Objective: Define monitoring routines, KPIs, and feedback loops for ongoing governance effectiveness.
- Key inputs: Escalation matrix, project plan metrics, risk signals.
- Expected outputs: Monitoring playbook with cadence, dashboards, and alert criteria.
- Handoff: Pass monitoring plan to GovernancePhase6ExtraTask and reporting agents.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for governance-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
