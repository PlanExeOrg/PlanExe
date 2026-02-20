/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-scenariosmarkdown',
  displayName: 'Luigi Scenarios Markdown Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the ScenariosMarkdownTask step inside the Luigi pipeline.
- Stage: Strategic Lever Development (Shape strategic levers and scenarios that drive how the plan tackles the mission.)
- Objective: Produce polished markdown capturing the selected scenario narratives for reuse.
- Key inputs: Scenario selection results, narrative fragments, stakeholder tone guidance.
- Expected outputs: Scenario markdown document with links to lever references and implications.
- Handoff: Publish artifact to assumptions, team, and reporting stages.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for strategy-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
