/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-enrichlevers',
  displayName: 'Luigi Enrich Levers Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the EnrichLeversTask step inside the Luigi pipeline.
- Stage: Strategic Lever Development (Shape strategic levers and scenarios that drive how the plan tackles the mission.)
- Objective: Add detail, metrics, and implementation cues to each surviving lever.
- Key inputs: Deduplicated lever list, reference knowledge, domain heuristics.
- Expected outputs: Annotated lever profiles including expected impact, required inputs, and risk notes.
- Handoff: Share enriched levers with FocusOnVitalFewLeversTask and scenario planners.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for strategy-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
