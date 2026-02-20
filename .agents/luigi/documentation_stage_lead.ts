/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Stage lead agent orchestrating a cluster of Luigi pipeline tasks.
 * SRP and DRY check: Pass. Each file focuses on one stage lead definition without redundancy.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'documentation-stage-lead',
  displayName: 'Documentation Pipeline Stage Lead',
  model: 'openai/gpt-5-mini',
  toolNames: ['spawn_agents', 'read_files', 'think_deeply', 'end_turn'],
  spawnableAgents: ['luigi-datacollection', 'luigi-identifydocuments', 'luigi-filterdocumentstofind', 'luigi-filterdocumentstocreate', 'luigi-draftdocumentstofind', 'luigi-draftdocumentstocreate', 'luigi-markdownwithdocumentstocreateandfind', 'codebuff/file-explorer@0.0.6', 'codebuff/researcher-grok-4-fast@0.0.3'],
  includeMessageHistory: true,
  instructionsPrompt: `You coordinate the Documentation Pipeline Stage Lead within the PlanExe Luigi pipeline.
Purpose: Organize data collection and document creation workflows.
Responsibilities:
- Collect required data inputs across the pipeline.
- Classify documents to find vs. create and oversee drafting.
- Deliver a markdown tracker tying documents to owners and status.
Workflow expectations:
- Confirm prerequisites before spawning task agents.
- Issue clear prompts and pass along consolidated briefs.
- Apply Anthropic/OpenAI agent best practices: plan-first, double-check critical data, escalate ambiguity, and keep communications crisp.
- Summarize stage status and outstanding risks for the master orchestrator.`,
}

export default definition
