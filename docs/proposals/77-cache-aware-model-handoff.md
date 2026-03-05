# Proposal 77: Cache-Aware Model Handoff Architecture

**Status:** Draft  
**Date:** 27 February 2026  
**Author:** Larry (VoynichLabs / OpenClaw)  
**Depends on:** Proposal #73 (task complexity routing), Proposal #74 (model routing UX modes)  
**Related:** Claude Code Engineering Blog — [Prompt Caching at Scale](https://www.anthropic.com/engineering/prompt-caching)

---

## Summary

Proposal #73 defined *when* to switch models (complexity rubric → model tier). This proposal defines *how* to switch models without destroying the prompt cache and turning cost savings into cost increases.

The two proposals together form a complete model routing architecture for PlanExe.

---

## The Problem: Naive Model Switching Is Counter-Productive

The intuitive model routing implementation looks like this:

1. Score task complexity → tier = Haiku
2. Switch the current session from Opus to Haiku
3. Continue execution with Haiku

**This is wrong.** It costs *more* than using Opus for everything.

### Why

Prompt caches are **model-specific** and built on **prefix matching**. When a session accumulates 100K tokens with Opus, switching to Haiku does not transfer that cached context. The Haiku API call must re-process all 100K tokens from scratch to build its own cache — at Haiku's per-token rate.

At that point:
- **Opus cached cost:** ~$0.375 per 100K tokens (at $3.75/MTok cache read)
- **Haiku cold-start cost:** ~$1.00 per 100K tokens (at $10/MTok input, uncached)

The "cheaper" model becomes the expensive one because we abandoned a warm cache.

> *"If you're 100k tokens into a conversation with Opus and want to ask a question that is fairly easy to answer, it would actually be more expensive to switch to Haiku than to have Opus answer, because we would need to rebuild the prompt cache for Haiku."*  
> — Claude Code Engineering Team

This is counter-intuitive. The math is unforgiving.

---

## The Solution: Cache-Safe Subagent Handoff

The correct pattern is **never switch models mid-session**. Instead:

1. The current model (e.g., Opus) completes its work
2. Opus prepares a **structured handoff summary** — a compact, self-contained context document
3. A **new subagent** on the target model tier (e.g., Haiku) starts fresh with only the handoff summary
4. The subagent builds its own cache from zero — but with a *small* initial context, not 100K tokens

The subagent's cache builds cheaply because it starts small. The parent session's cache remains intact because nothing changed.

### Why This Works

- **Parent session:** Opus's warm cache is never invalidated; the parent can continue or terminate cleanly
- **Subagent:** Starts with a compact, curated context — fast to cache, cheap to run
- **Total cost:** Subagent cold-start is cheap when starting small; far cheaper than mid-session model swap after large context accumulation

---

## Handoff Message Format

When a PlanExe task step crosses a model tier boundary, the current model should produce a structured handoff before the subagent is spawned.

### Recommended Handoff Schema

```json
{
  "handoff_version": "1.0",
  "task_id": "<uuid>",
  "task_description": "<what needs to be done, plain language>",
  "target_model_tier": "haiku | minimax | sonnet | opus",
  "complexity_score": {
    "file_size": 2,
    "semantic_complexity": 1,
    "ambiguity": 2,
    "context_dependency": 1,
    "total": 6
  },
  "required_context": {
    "plan_summary": "<curated summary of relevant plan state>",
    "relevant_files": ["<list of files the subagent needs>"],
    "prior_decisions": ["<decisions made upstream that constrain this task>"],
    "success_criteria": "<what done looks like>",
    "constraints": ["<hard limits: time, cost, scope>"]
  },
  "handoff_instructions": "<specific instructions for the receiving model>",
  "parent_task_id": "<uuid of parent task, for result routing>"
}
```

### Handoff Principles

1. **Curate aggressively.** The handoff should contain only what the subagent needs. Every extra token in the handoff is a token that must be cached from scratch.
2. **Include success criteria explicitly.** The subagent cannot ask clarifying questions in the same way the parent could.
3. **Prior decisions are not optional.** If Opus decided X, the Haiku subagent must know X so it doesn't re-litigate it.
4. **No full history dumps.** Do not copy the entire conversation history into the handoff. Summarize; curate.

---

## Integration with Proposal #73: When to Trigger a Handoff

Using the 4-dimension rubric from Proposal #73:

| Total Score | Model Tier | Handoff Trigger |
|-------------|------------|-----------------|
| 4–7         | Minimax    | Handoff if current session is Haiku or higher |
| 8–11        | Haiku      | Handoff if current session is Sonnet or higher |
| 12–15       | Sonnet     | Handoff if current session is Opus |
| 16–20       | Opus       | No handoff needed; Opus handles directly |

**Key rule:** Only trigger a handoff when crossing *downward* in model tier (routing to a cheaper model). Routing *upward* (needing more powerful model mid-session) should use a different pattern — see §5 below.

---

## Upward Routing: When You Need More Power

The inverse case: a task starts on Haiku but mid-execution proves to be harder than the rubric scored. Two options:

1. **Abort and re-route:** The Haiku subagent recognizes it cannot complete the task, returns a structured escalation response, and PlanExe spawns a new Opus subagent with the same handoff (plus Haiku's partial findings)
2. **Ask the parent:** The subagent delivers a "blocked" response upstream; the parent Opus session picks it up and completes the work (parent cache still warm)

Option 2 is preferred when the parent session is still running. Option 1 is for asynchronous task queues.

---

## Tool Set Stability During Handoff

Proposal #73 defines routing by task type. The cache doctrine adds a constraint: **the subagent's tool set must be defined at spawn time and never changed.**

Changing tools mid-subagent-session invalidates the subagent's cache — same problem, smaller scale.

**Implementation note:** Use `defer_loading: true` stubs for rarely-used tools. Keep the declared tool set stable. Let the subagent discover full schemas when needed rather than loading all schemas upfront.

---

## What PlanExe Should NOT Do

| Anti-pattern | Why it's wrong |
|---|---|
| Switch the model parameter mid-session | Destroys the accumulated prefix cache; costs more than using original model |
| Pass full conversation history to subagent | Increases cold-start cost; defeats the purpose of routing to a cheaper model |
| Add or remove tools between handoff steps | Invalidates cache at every tool set change |
| Use different system prompts across handoffs | Each variant builds its own cache from scratch; prefer a stable system prompt |
| Skip the handoff summary for "simple" tasks | Without curated context, the subagent may re-do work or make wrong assumptions |

---

## Relationship to PlanExe's Existing Architecture

### Luigi Task Pipeline

Each Luigi task in PlanExe generates work and passes results forward. The handoff pattern maps naturally onto this: when a Luigi task needs to delegate to a cheaper model for a subtask, it generates a handoff document (structured output) rather than a continuation of the same LLM session.

### Compaction / Summarization Steps

When PlanExe summarizes completed plan phases, it should use **cache-safe forking**: same system prompt, same tool set, parent conversation history prepended, new summarization prompt appended. This reuses the parent's cached prefix rather than starting from scratch.

Reference: Claude Code Engineering Blog, *Cache-Safe Forking* section.

### MCP Server Calls

PlanExe's MCP server makes API calls for each plan step. Each of these is an independent API call with its own cache lifecycle. The system prompt and tool set for these calls should be **identical across all MCP calls** to maximize cross-call cache hits. Dynamic content (the plan request) goes last in the message sequence, not in the system prompt.

---

## Metrics to Track

Per the cache doctrine: **monitor cache hit rate like uptime**. Recommended metrics for PlanExe:

| Metric | Target | Alert Threshold |
|---|---|---|
| Cache hit rate (per model tier) | >85% | <70% |
| Handoff summary size | <8K tokens | >16K tokens |
| Subagent cold-start cost vs. direct Opus cost | <50% of Opus | >90% of Opus (handoff not helping) |
| Mid-session model switches (should be zero) | 0 | Any |

---

## Implementation Checklist (Docs-Only Phase)

This proposal is docs-only. Before implementing:

- [ ] Simon reviews and approves the handoff schema
- [ ] Agreement on which Luigi tasks should trigger downward routing
- [ ] Agreement on upward escalation protocol (abort-and-respawn vs. return-to-parent)
- [ ] Cache hit rate monitoring infrastructure scoped (separate proposal or extension of Proposal #31)

---

## References

- **Proposal #73:** Task Complexity Scoring + Model Routing Rubric (merged, PR #102)
- **Proposal #74:** Model Routing UX Modes (merged)
- **Claude Code Engineering Blog:** Prompt Caching at Scale — lessons from building Claude Code around prompt caching at production scale
- **Anthropic API Docs:** Prompt caching, `cache_control` breakpoints, model-specific cache retention

---

*The complexity rubric tells you when to switch. This proposal tells you how. Both are required for a correct implementation.*
