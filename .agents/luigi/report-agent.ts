/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-report',
  displayName: 'Luigi Report Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the ReportTask step inside the Luigi pipeline.
- Stage: Reporting & Synthesis (Assemble stakeholder-ready narratives, reviews, and the final report.)
- Objective: Assemble the final report, merging all artifacts into the deliverable package.
- Key inputs: Executive summary, pitch markdown, governance dossier, schedule, team docs, premortem.
- Expected outputs: Final compiled report with navigation and appendices.
- Handoff: Return finished package to orchestrator and distribution stakeholders.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for reporting-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
