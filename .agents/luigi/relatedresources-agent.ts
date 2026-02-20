/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-relatedresources',
  displayName: 'Luigi Related Resources Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the RelatedResourcesTask step inside the Luigi pipeline.
- Stage: Plan Foundation (Assess readiness and craft the project plan backbone while curating reference resources.)
- Objective: Compile related resources, references, and precedent materials for planners and executors.
- Key inputs: Project plan outline, organizational repositories, domain knowledge.
- Expected outputs: Curated resource list with access notes and relevance tags.
- Handoff: Share with documentation and expert review agents.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for plan-foundation-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
