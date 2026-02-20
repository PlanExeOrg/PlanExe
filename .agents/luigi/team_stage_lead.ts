/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Stage lead agent orchestrating a cluster of Luigi pipeline tasks.
 * SRP and DRY check: Pass. Each file focuses on one stage lead definition without redundancy.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'team-stage-lead',
  displayName: 'Team Assembly Stage Lead',
  model: 'openai/gpt-5-mini',
  toolNames: ['spawn_agents', 'read_files', 'think_deeply', 'end_turn'],
  spawnableAgents: ['luigi-findteammembers', 'luigi-enrichteammemberswithcontracttype', 'luigi-enrichteammemberswithbackgroundstory', 'luigi-enrichteammemberswithenvironmentinfo', 'luigi-reviewteam', 'luigi-teammarkdown', 'codebuff/file-explorer@0.0.6', 'codebuff/researcher-grok-4-fast@0.0.3'],
  includeMessageHistory: true,
  instructionsPrompt: `You coordinate the Team Assembly Stage Lead within the PlanExe Luigi pipeline.
Purpose: Assemble and document the delivery team with contextual richness.
Responsibilities:
- Source candidates and enrich their profiles with contract, background, and environment data.
- Drive team review for coverage and risks.
- Publish team markdown for governance and reporting.
Workflow expectations:
- Confirm prerequisites before spawning task agents.
- Issue clear prompts and pass along consolidated briefs.
- Apply Anthropic/OpenAI agent best practices: plan-first, double-check critical data, escalate ambiguity, and keep communications crisp.
- Summarize stage status and outstanding risks for the master orchestrator.`,
}

export default definition
