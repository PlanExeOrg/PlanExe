/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-candidatescenarios',
  displayName: 'Luigi Candidate Scenarios Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the CandidateScenariosTask step inside the Luigi pipeline.
- Stage: Strategic Lever Development (Shape strategic levers and scenarios that drive how the plan tackles the mission.)
- Objective: Draft multiple candidate scenarios leveraging prioritized levers to cover plan uncertainty.
- Key inputs: Vital lever shortlist, decision markdown, risk appetite signals.
- Expected outputs: Scenario summaries with key moves, triggers, and success signals.
- Handoff: Submit scenario slate to SelectScenarioTask for evaluation and scoring.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for strategy-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
