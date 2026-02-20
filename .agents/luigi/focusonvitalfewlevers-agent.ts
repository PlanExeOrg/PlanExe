/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-focusonvitalfewlevers',
  displayName: 'Luigi Focus On Vital Few Levers Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the FocusOnVitalFewLeversTask step inside the Luigi pipeline.
- Stage: Strategic Lever Development (Shape strategic levers and scenarios that drive how the plan tackles the mission.)
- Objective: Prioritize the most critical levers using impact vs. effort and dependency logic.
- Key inputs: Enriched lever profiles, capacity constraints, mission priorities.
- Expected outputs: Ranked short list of vital levers with justification and deferred candidates.
- Handoff: Inform StrategicDecisionsMarkdownTask and CandidateScenariosTask about chosen focus areas.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for strategy-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
