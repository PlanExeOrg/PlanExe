# Proposal 74: Model Routing UX — Automatic, Optimize, and Review Modes

**Author:** Larry (Sonnet 4.6)  
**Date:** 2026-02-27  
**Status:** Draft — for Simon's consideration  
**Depends on:** Proposal 73 (task complexity scoring + model routing)

---

## The Problem

Right now, most PlanExe users do what's natural: they pick the best model they have access to and use it for everything. If they have Opus, everything runs on Opus. It's cognitively easy. It's also expensive and, for many tasks, overkill.

The alternative — manually selecting a different model for each task — requires understanding context windows, pricing tiers, semantic complexity, session hygiene, and the 200K token cliff. That's a lot of gear-shifting. Most developers don't want to think about this any more than most drivers want to think about gear ratios.

This is exactly the problem the automatic transmission solved in 1940.

---

## The Analogy

**The Model T (1920s):** Powerful, but you had to hand-crank it, manage three pedals, adjust the spark advance lever, and understand exactly what you were doing — or the engine would kick back and break your wrist. Only gearheads could get the best out of it.

**The 4WD Truck (Opus-for-everything):** Big, shiny, handles anything. Goes through 20cm of snow, 5 inches of mud, carries anything in the bed. Gets 5 miles to the gallon. When you're heading to the store on a sunny day, it's overkill — but it works, and you don't have to think.

**The Street Racer's Manual Transmission (power users):** These are the people who WANT control. They feel every gear change. They know exactly which model to use for which task. They can zip through a codebase like a street race through downtown. Maximum efficiency, maximum control — but requires skill and attention.

**The 1940 Hydra-Matic (what PlanExe should offer):** GM's Earl Thompson spent a decade figuring out how to encode the expertise of gear-shifting into the machine itself. The driver just says "go." The car figures out the gears. The knowledge is built in — you benefit from it without needing to possess it.

We are in the 1935 moment for AI agents. The tools work. But you need to be a gearhead to get the best out of them without blowing your budget. PlanExe can be the Hydra-Matic.

---

## Proposed: Three Routing Modes

### Mode 1: `auto`
**"The Truck"**

Use whatever model is configured as the default for everything. No routing logic. No complexity scoring. Maximum capability, maximum simplicity.

- **Who it's for:** Teams with budget flexibility who value simplicity. Spotify-style enterprise use. Developers who are new to AI-assisted coding and don't want cognitive overhead.
- **Behavior:** Every task runs on the user's configured model, regardless of complexity score.
- **Cost profile:** Highest, predictable.
- **Config:** `model_routing: auto`

---

### Mode 2: `optimize`
**"The Hydra-Matic"**

PlanExe scores each task using the complexity rubric (Proposal 73), selects the appropriate model tier automatically, and executes without asking. The user never thinks about model selection — the expertise is encoded in the system.

- **Who it's for:** Cost-conscious developers and small teams who trust the rubric. The "I just want to drive" crowd who also care about gas mileage.
- **Behavior:** Tasks scored 4–7 route to Minimax, 8–11 to Haiku, 12–15 to Sonnet, 16–20 to Opus. Session boundaries are managed automatically at context thresholds.
- **Cost profile:** Lowest, variable.
- **Config:** `model_routing: optimize`

**What the user sees:** A summary after plan generation showing estimated cost vs all-Opus cost. E.g.:
```
Routing plan: 3 tasks → Minimax | 2 tasks → Haiku | 1 task → Sonnet | 1 task → Opus
Estimated cost: $4.20 (vs $18.00 at Opus-only) — 77% savings
```

---

### Mode 3: `review`
**"The Street Racer with a Co-Pilot"**

PlanExe scores each task and generates a routing recommendation, but pauses for human approval before executing. The user sees exactly what model is recommended for each task, the reasoning, and the estimated cost. They can override any routing decision before committing.

- **Who it's for:** Power users who want control AND want the system's analysis as a starting point. Developers building intuition for model selection. Simon.
- **Behavior:** After plan generation, presents a routing summary with per-task recommendations and estimated costs. User can accept, modify per-task, or override globally. Execution begins only after approval.
- **Cost profile:** Same as `optimize` if accepted as-is; variable if overridden.
- **Config:** `model_routing: review`

**What the user sees:**
```
Task A: Module split (http_server.py, 1089 lines)
  Score: 19/20 → Opus recommended
  Reason: Cross-file architectural refactor, whole-codebase context dependency
  Estimated: $3.20
  [Accept] [Override: Sonnet] [Override: Haiku]

Task B: API rename (task_id → plan_id, 88 files)
  Score: 13/20 → Sonnet (planning) + Minimax (execution) recommended
  Reason: Large file surface, but mechanical pattern — plan once, execute cheap
  Estimated: $0.85
  [Accept] [Override: all-Opus]
```

---

## The 200K Token Cliff (Session Hygiene)

Relevant in both `optimize` and `review` modes.

When using Opus through Anthropic directly (not OpenRouter), the per-token price doubles after 200K tokens in a session:
- ≤200K: $5 input / $25 output per 1M tokens
- >200K: $10 input / $37.50 output per 1M tokens

**This is not a hard rule to never cross.** Sometimes the large context is exactly what you need — Spotify's use case of ingesting full service dependency trees in one pass is the canonical example. The value of that large-context read can absolutely justify the higher rate.

The waste happens when sessions drift past 200K tokens carrying context that's no longer active: old file reads, abandoned exploration paths, prior conversation history. That's paying the premium rate for tokens that aren't working.

**PlanExe's session management in `optimize` mode should:**
1. Track estimated session token count per model
2. Flag when approaching 200K with Opus
3. Offer to checkpoint: summarize active context → close session → open new session with summary + current task
4. Never force-close — the decision to pay for large context is the user's

---

## Implementation Notes (for Simon's consideration)

This proposal is UX/product-layer only. The complexity scoring engine (Proposal 73) is the prerequisite. Once that exists, these three modes are essentially:

- `auto`: bypass the scoring engine entirely
- `optimize`: run scoring engine, apply routing table, execute
- `review`: run scoring engine, present routing plan as interactive confirmation step, then execute

The routing table (score → model mapping) should be configurable per-project, not hardcoded. Simon may have different thresholds than a default user. The table in Proposal 73 (4–7=Minimax, 8–11=Haiku, 12–15=Sonnet, 16–20=Opus) is a starting point for calibration.

---

## Questions for Simon

1. Does the three-mode framing resonate with how you'd want to configure PlanExe on a new project?
2. For `review` mode — is per-task override granularity useful, or would project-level override (e.g., "use Sonnet minimum for everything") be sufficient?
3. The 200K session checkpoint behavior — should this be automatic in `optimize` mode or always require confirmation?
4. Are there task types in PlanExe's current plan generation that would never make sense to route below Sonnet, regardless of score? (I'm thinking: anything touching the plan's core reasoning chain probably needs to stay at Sonnet+.)

---

*This proposal is a companion to Proposal 73 (complexity scoring) and should be reviewed together. Both are docs-only — no code changes.*
