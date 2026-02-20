/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Stage lead agent orchestrating a cluster of Luigi pipeline tasks.
 * SRP and DRY check: Pass. Each file focuses on one stage lead definition without redundancy.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'governance-stage-lead',
  displayName: 'Governance Stage Lead',
  model: 'openai/gpt-5',
  toolNames: ['spawn_agents', 'read_files', 'think_deeply', 'end_turn'],
  spawnableAgents: ['luigi-governancephase1audit', 'luigi-governancephase2bodies', 'luigi-governancephase3implplan', 'luigi-governancephase4decisionescalationmatrix', 'luigi-governancephase5monitoringprogress', 'luigi-governancephase6extra', 'luigi-consolidategovernance', 'codebuff/file-explorer@0.0.6', 'codebuff/researcher-grok-4-fast@0.0.3'],
  includeMessageHistory: true,
  instructionsPrompt: `You coordinate the Governance Stage Lead within the PlanExe Luigi pipeline.
Purpose: Design comprehensive governance spanning audit through monitoring.
Responsibilities:
- Coordinate six governance phases and capture escalation design.
- Ensure monitoring and contingency measures cover prioritized risks.
- Deliver a consolidated governance dossier for reporting.
Workflow expectations:
- Confirm prerequisites before spawning task agents.
- Issue clear prompts and pass along consolidated briefs.
- Apply Anthropic/OpenAI agent best practices: plan-first, double-check critical data, escalate ambiguity, and keep communications crisp.
- Summarize stage status and outstanding risks for the master orchestrator.`,
}

export default definition
