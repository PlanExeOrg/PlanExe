/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-convertpitchtomarkdown',
  displayName: 'Luigi Convert Pitch To Markdown Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the ConvertPitchToMarkdownTask step inside the Luigi pipeline.
- Stage: Reporting & Synthesis (Assemble stakeholder-ready narratives, reviews, and the final report.)
- Objective: Convert the pitch narrative into markdown suitable for distribution and reuse.
- Key inputs: Pitch draft, formatting standards, stakeholder tone guidance.
- Expected outputs: Markdown version of the pitch with links to supporting artifacts.
- Handoff: Share with ReviewPlanTask and final report pipeline.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for reporting-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
