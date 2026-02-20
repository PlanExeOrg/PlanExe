/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-draftdocumentstocreate',
  displayName: 'Luigi Draft Documents To Create Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the DraftDocumentsToCreateTask step inside the Luigi pipeline.
- Stage: Documentation Pipeline (Organize data, identify documentation needs, and produce supporting materials.)
- Objective: Draft new documents or heavy revisions aligned with plan objectives.
- Key inputs: Creation backlog, scenario outputs, governance requirements.
- Expected outputs: Draft content outlines or ready-to-use documents.
- Handoff: Provide to MarkdownWithDocumentsToCreateAndFindTask for merged narrative.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for documentation-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
