/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-markdownwithdocumentstocreateandfind',
  displayName: 'Luigi Markdown With Documents To Create And Find Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the MarkdownWithDocumentsToCreateAndFindTask step inside the Luigi pipeline.
- Stage: Documentation Pipeline (Organize data, identify documentation needs, and produce supporting materials.)
- Objective: Assemble markdown capturing both documents to create and to find, ensuring traceability.
- Key inputs: Briefs and drafts from document-specific agents.
- Expected outputs: Markdown matrix linking documents, owners, and status.
- Handoff: Share with reporting stage and orchestrator for oversight.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for documentation-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
