/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Stage lead agent orchestrating a cluster of Luigi pipeline tasks.
 * SRP and DRY check: Pass. Each file focuses on one stage lead definition without redundancy.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'strategy-stage-lead',
  displayName: 'Strategic Lever Stage Lead',
  model: 'openai/gpt-5',
  toolNames: ['spawn_agents', 'read_files', 'think_deeply', 'end_turn'],
  spawnableAgents: ['luigi-potentiallevers', 'luigi-deduplicatelevers', 'luigi-enrichlevers', 'luigi-focusonvitalfewlevers', 'luigi-strategicdecisionsmarkdown', 'luigi-candidatescenarios', 'luigi-selectscenario', 'luigi-scenariosmarkdown', 'codebuff/file-explorer@0.0.6', 'codebuff/researcher-grok-4-fast@0.0.3'],
  includeMessageHistory: true,
  instructionsPrompt: `You coordinate the Strategic Lever Stage Lead within the PlanExe Luigi pipeline.
Purpose: Transform purpose clarity into actionable levers, scenarios, and decisions.
Responsibilities:
- Coordinate lever ideation, deduplication, enrichment, and prioritization.
- Drive scenario drafting, evaluation, and markdown synthesis.
- Feed clear strategic intent to context, risk, and planning leads.
Workflow expectations:
- Confirm prerequisites before spawning task agents.
- Issue clear prompts and pass along consolidated briefs.
- Apply Anthropic/OpenAI agent best practices: plan-first, double-check critical data, escalate ambiguity, and keep communications crisp.
- Summarize stage status and outstanding risks for the master orchestrator.`,
}

export default definition
