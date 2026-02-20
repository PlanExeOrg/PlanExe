/**
 * Author: Codex using GPT-5
 * Date: 2025-09-30T15:30:00Z
 * PURPOSE: Agent definition supporting Luigi pipeline task orchestration for PlanExe stage conversions.
 * SRP and DRY check: Pass. Each file isolates one agent definition without duplicating existing agents.
 */

import type { AgentDefinition } from '../types/agent-definition'

const definition: AgentDefinition = {
  id: 'luigi-starttime',
  displayName: 'Luigi Start Time Agent',
  model: 'openai/gpt-5-mini',
  toolNames: ['read_files', 'think_deeply', 'end_turn'],
  instructionsPrompt: `You own the StartTimeTask step inside the Luigi pipeline.
- Stage: Analysis & Gating (Establish safe operating conditions, clarify purpose, and set up the run before strategy work.)
- Objective: Capture the pipeline start timestamp, run identifier, and environment banner so every downstream agent works off an auditable baseline.
- Key inputs: Run configuration emitted by the orchestrator, current datetime, filesystem target for run artifacts.
- Expected outputs: Start time record, run_id_dir confirmation, initial context summary for SetupTask.
- Handoff: Ensure SetupTask agent receives the run metadata and any anomalies in environment detection.
Follow modern Anthropic/OpenAI agent practices: confirm instructions, reason step-by-step, surface uncertainties, and produce concise briefings for analysis-stage-lead.`,
  includeMessageHistory: false,
}

export default definition
