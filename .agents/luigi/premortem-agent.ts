/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-premortem',
  displayName: 'Luigi Premortem Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the PremortemTask step inside the Luigi pipeline.
- Stage: Reporting & Synthesis (Assemble stakeholder-ready narratives, reviews, and the final report.)
- Objective: Run a premortem to identify failure modes and mitigation before launch.
- Key inputs: Q&A catalog, risk register, schedule, team roster.
- Expected outputs: Premortem narrative with mitigations and monitoring triggers.
- Handoff: Provide to ReportTask and governance agents for action.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for reporting-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
