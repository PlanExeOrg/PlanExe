/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-currencystrategy',
  displayName: 'Luigi Currency Strategy Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the CurrencyStrategyTask step inside the Luigi pipeline.
- Stage: Context Localization (Capture location and currency context so downstream planning respects operational realities.)
- Objective: Define currency handling, conversion assumptions, and financial localization strategy.
- Key inputs: Location roster, financial constraints, historical FX data cues.
- Expected outputs: Currency handling plan with hedging notes and accounting implications.
- Handoff: Provide outputs to assumptions and budgeting agents for alignment.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for context-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
