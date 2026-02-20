/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-filterdocumentstocreate',
  displayName: 'Luigi Filter Documents To Create Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the FilterDocumentsToCreateTask step inside the Luigi pipeline.
- Stage: Documentation Pipeline (Organize data, identify documentation needs, and produce supporting materials.)
- Objective: Identify documents that must be created from scratch or heavily revised.
- Key inputs: Document inventory, gaps flagged by FilterDocumentsToFindTask.
- Expected outputs: Creation backlog with scope notes and priority.
- Handoff: Provide to DraftDocumentsToCreateTask for drafting workflows.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for documentation-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
