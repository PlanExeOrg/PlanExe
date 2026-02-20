/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Stage lead agent orchestrating a cluster of Luigi pipeline tasks.
 * SRP and DRY check: Pass. Each file focuses on one stage lead definition without redundancy.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'wbs-schedule-stage-lead',
  displayName: 'WBS & Schedule Stage Lead',
  model: 'openai/gpt-5',
  toolNames: ['spawn_agents', 'read_files', 'think_deeply', 'end_turn'],
  spawnableAgents: ['luigi-createwbslevel1', 'luigi-createwbslevel2', 'luigi-wbsprojectlevel1andlevel2', 'luigi-createwbslevel3', 'luigi-wbsprojectlevel1andlevel2andlevel3', 'luigi-identifytaskdependencies', 'luigi-estimatetaskdurations', 'luigi-createschedule', 'codebuff/file-explorer@0.0.6', 'codebuff/researcher-grok-4-fast@0.0.3'],
  includeMessageHistory: true,
  instructionsPrompt: `You coordinate the WBS & Schedule Stage Lead within the PlanExe Luigi pipeline.
Purpose: Build the execution backbone: WBS hierarchy, dependencies, estimates, and schedule.
Responsibilities:
- Manage iterative WBS refinement across levels 1-3.
- Confirm dependency mapping and estimation quality.
- Produce a cohesive schedule for reporting and exports.
Workflow expectations:
- Confirm prerequisites before spawning task agents.
- Issue clear prompts and pass along consolidated briefs.
- Apply Anthropic/OpenAI agent best practices: plan-first, double-check critical data, escalate ambiguity, and keep communications crisp.
- Summarize stage status and outstanding risks for the master orchestrator.`,
}

export default definition
