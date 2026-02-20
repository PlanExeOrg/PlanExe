/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-enrichteammemberswithenvironmentinfo',
  displayName: 'Luigi Enrich Team Members With Environment Info Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the EnrichTeamMembersWithEnvironmentInfoTask step inside the Luigi pipeline.
- Stage: Team Assembly (Build and document the delivery team with the right context and reviews.)
- Objective: Align each team member with environment, tooling, and logistical considerations.
- Key inputs: Narrative roster, project environment requirements, location/currency notes.
- Expected outputs: Roster with environment compatibility notes and support needs.
- Handoff: Provide to ReviewTeamTask for validation.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for team-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
