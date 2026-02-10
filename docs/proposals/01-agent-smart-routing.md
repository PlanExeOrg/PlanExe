---
title: Agent Smart Routing - Meta-Agent Dispatcher
date: 2026-02-09
status: proposal
author: Larry (OpenClaw)
---

# Agent Smart Routing - Meta-Agent Dispatcher

## Overview

PlanExe's planning pipeline currently uses a single agent profile for all stages. As plans grow in complexity and domain diversity, different stages benefit from specialized agents optimized for specific tasks (research, writing, technical validation, creativity).

This proposal introduces a **meta-agent dispatcher** that routes each pipeline stage to the most appropriate agent based on stage type, domain, and requirements.

## Problem

- Generic agents produce mediocre results across all domains

- No way to leverage specialized models (reasoning models for analysis, fast models for formatting, etc.)

- Pipeline stages have different cost/quality trade-offs that aren't exploited

## Proposed Solution

### Architecture

```
┌─────────────────┐
│  PlanExe Core   │
│   (Orchestrator)│
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Meta-Agent      │  ← Dispatcher logic
│ Router          │
└────────┬────────┘
         │
         ├──→ Research Agent (Gemini 2.0 Flash)
         ├──→ Writing Agent (Claude Sonnet)
         ├──→ Technical Agent (GPT-4 + reasoning)
         └──→ Format Agent (Haiku/Fast model)
```

### Routing Rules

Store routing configuration in `llm_config.json`:

```json
{
  "agent_routing": {
    "research": {
      "model": "google/gemini-2.0-flash-thinking-exp",
      "reason": "Fast, cheap, good at web search synthesis"
    },
    "outline": {
      "model": "anthropic/claude-sonnet-4",
      "reason": "Strong at structure and planning"
    },
    "technical": {
      "model": "openai/gpt-4-turbo",
      "thinking": "enabled",
      "reason": "Deep reasoning for complex technical content"
    },
    "format": {
      "model": "anthropic/claude-haiku-4",
      "reason": "Fast, cheap, reliable for formatting"
    }
  }
}
```

### Implementation

1. Add `AgentRouter` class in `backend/mcp_cloud/src/routing/`

2. Modify pipeline stages to call `router.get_agent(stage_type, domain)`

3. Add telemetry to track agent selection and performance per stage

4. Build admin UI to override routing rules per-customer

## Benefits

- **15-30% cost reduction** by using fast models for simple stages

- **Quality improvement** from specialized agents

- **Flexibility** for customers to bring their own agent configs

- **A/B testing** different agent combinations per stage

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Increased complexity | Start with 3-4 agent profiles, expand gradually |
| Debugging harder | Add detailed logging of agent selection |
| Config drift | Validate routing config on startup, fail fast |

## Next Steps

1. Prototype with 3 agents (research, writing, format)

2. Run side-by-side comparison on 20 existing plans

3. Measure cost savings and quality delta

4. Ship behind feature flag, enable for beta customers

## Success Metrics

- Cost per plan decreases by 20%+

- User satisfaction rating increases (via post-plan survey)

- No increase in pipeline failure rate
