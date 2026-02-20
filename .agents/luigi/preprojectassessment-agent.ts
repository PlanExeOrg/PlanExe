/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-preprojectassessment',
  displayName: 'Luigi Pre Project Assessment Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the PreProjectAssessmentTask step inside the Luigi pipeline.
- Stage: Plan Foundation (Assess readiness and craft the project plan backbone while curating reference resources.)
- Objective: Assess organizational readiness and baseline capabilities before plan execution.
- Key inputs: Scenario outputs, assumptions, existing resource dossiers.
- Expected outputs: Readiness scorecard, key gaps, prerequisite actions.
- Handoff: Inform ProjectPlanTask and team/governance leads about readiness gaps.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for plan-foundation-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
