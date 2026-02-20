/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-findteammembers',
  displayName: 'Luigi Find Team Members Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the FindTeamMembersTask step inside the Luigi pipeline.
- Stage: Team Assembly (Build and document the delivery team with the right context and reviews.)
- Objective: Identify candidate team members aligned with skill requirements and capacity.
- Key inputs: Project plan, governance roles, resource constraints.
- Expected outputs: Team candidate roster with skill mapping and availability.
- Handoff: Send roster to enrichment agents for deeper profiling.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for team-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
