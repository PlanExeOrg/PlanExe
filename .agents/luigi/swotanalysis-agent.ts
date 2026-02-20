/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-swotanalysis',
  displayName: 'Luigi S W O T Analysis Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the SWOTAnalysisTask step inside the Luigi pipeline.
- Stage: Expert Validation (Capture SWOT insights and synthesize expert feedback to strengthen the plan.)
- Objective: Generate a SWOT analysis leveraging current assumptions, levers, and team insights.
- Key inputs: Scenario markdown, risk register, team documentation.
- Expected outputs: SWOT table with narrative commentary and priority focus areas.
- Handoff: Share with ExpertReviewTask and reporting agents.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for expert-quality-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
