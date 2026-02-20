/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-physicallocations',
  displayName: 'Luigi Physical Locations Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the PhysicalLocationsTask step inside the Luigi pipeline.
- Stage: Context Localization (Capture location and currency context so downstream planning respects operational realities.)
- Objective: Map the physical or geopolitical locations relevant to plan execution and logistics.
- Key inputs: Scenario markdown, prompt constraints, known site data.
- Expected outputs: Location roster with context notes, timezone considerations, and dependencies.
- Handoff: Share with CurrencyStrategyTask and risk/assumption agents for validation.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for context-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
