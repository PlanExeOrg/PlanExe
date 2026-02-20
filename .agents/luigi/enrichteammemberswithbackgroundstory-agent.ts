/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-enrichteammemberswithbackgroundstory',
  displayName: 'Luigi Enrich Team Members With Background Story Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the EnrichTeamMembersWithBackgroundStoryTask step inside the Luigi pipeline.
- Stage: Team Assembly (Build and document the delivery team with the right context and reviews.)
- Objective: Provide narrative background and expertise context for each candidate.
- Key inputs: Roster with contract types, personnel bios, organizational knowledge.
- Expected outputs: Annotated roster entries with background summaries.
- Handoff: Deliver to environment info agent for operational context.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for team-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
