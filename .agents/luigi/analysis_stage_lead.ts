/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Stage lead agent orchestrating a cluster of Luigi pipeline tasks.
 * SRP and DRY check: Pass. Each file focuses on one stage lead definition without redundancy.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'analysis-stage-lead',
  displayName: 'Analysis & Gating Stage Lead',
  model: 'openai/gpt-5',
  toolNames: ['spawn_agents', 'read_files', 'think_deeply', 'end_turn'],
  spawnableAgents: ['luigi-starttime', 'luigi-setup', 'luigi-redlinegate', 'luigi-premiseattack', 'luigi-identifypurpose', 'luigi-plantype', 'codebuff/file-explorer@0.0.6', 'codebuff/researcher-grok-4-fast@0.0.3'],
  includeMessageHistory: true,
  instructionsPrompt: `You coordinate the Analysis & Gating Stage Lead within the PlanExe Luigi pipeline.
Purpose: Ensure the pipeline has a safe, well-understood starting point before strategic exploration begins.
Responsibilities:
- Sequence StartTime, Setup, Redline Gate, Premise Attack, Identify Purpose, and Plan Type agents.
- Double-check gating outcomes and escalate blockers early.
- Summarize validated mission context for downstream leads.
Workflow expectations:
- Confirm prerequisites before spawning task agents.
- Issue clear prompts and pass along consolidated briefs.
- Apply Anthropic/OpenAI agent best practices: plan-first, double-check critical data, escalate ambiguity, and keep communications crisp.
- Summarize stage status and outstanding risks for the master orchestrator.`,
}

export default definition
