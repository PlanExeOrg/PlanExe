/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-datacollection',
  displayName: 'Luigi Data Collection Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the DataCollectionTask step inside the Luigi pipeline.
- Stage: Documentation Pipeline (Organize data, identify documentation needs, and produce supporting materials.)
- Objective: Collect necessary data inputs from previous tasks and repository sources for documentation steps.
- Key inputs: Artifacts across pipeline, resource lists, governance outputs.
- Expected outputs: Organized data bundle referenced by document drafting agents.
- Handoff: Distribute to IdentifyDocumentsTask to drive document coverage analysis.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for documentation-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
