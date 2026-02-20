/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Stage lead agent orchestrating a cluster of Luigi pipeline tasks.
 * SRP and DRY check: Pass. Each file focuses on one stage lead definition without redundancy.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'reporting-stage-lead',
  displayName: 'Reporting & Synthesis Stage Lead',
  model: 'openai/gpt-5',
  toolNames: ['spawn_agents', 'read_files', 'think_deeply', 'end_turn'],
  spawnableAgents: ['luigi-createpitch', 'luigi-convertpitchtomarkdown', 'luigi-reviewplan', 'luigi-executivesummary', 'luigi-questionsandanswers', 'luigi-premortem', 'luigi-report', 'codebuff/file-explorer@0.0.6', 'codebuff/researcher-grok-4-fast@0.0.3'],
  includeMessageHistory: true,
  instructionsPrompt: `You coordinate the Reporting & Synthesis Stage Lead within the PlanExe Luigi pipeline.
Purpose: Craft stakeholder-ready narratives culminating in the final report.
Responsibilities:
- Develop persuasive pitches and executive summaries informed by reviews.
- Prepare Q&A and premortem insights to derisk presentations.
- Assemble the final report package for handoff.
Workflow expectations:
- Confirm prerequisites before spawning task agents.
- Issue clear prompts and pass along consolidated briefs.
- Apply Anthropic/OpenAI agent best practices: plan-first, double-check critical data, escalate ambiguity, and keep communications crisp.
- Summarize stage status and outstanding risks for the master orchestrator.`,
}

export default definition
