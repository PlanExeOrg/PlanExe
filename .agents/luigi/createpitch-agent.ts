/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-createpitch',
  displayName: 'Luigi Create Pitch Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the CreatePitchTask step inside the Luigi pipeline.
- Stage: Reporting & Synthesis (Assemble stakeholder-ready narratives, reviews, and the final report.)
- Objective: Craft an executive pitch outlining plan value, strategy, and requested approvals.
- Key inputs: Strategic decisions markdown, team documentation, schedule highlights.
- Expected outputs: Pitch narrative structured for stakeholder persuasion.
- Handoff: Provide to ConvertPitchToMarkdownTask and final report assembler.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for reporting-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
