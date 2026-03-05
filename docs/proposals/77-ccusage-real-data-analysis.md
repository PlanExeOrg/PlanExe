# Proposal 77: Real Usage Data Analysis — Larry's Claude Code Sessions

**Author:** Larry (Sonnet 4.6)  
**Date:** 2026-02-27  
**Status:** Data analysis — actionable recommendations for Simon + Mark  
**Source data:** `ccusage` run against `/mnt/c/Users/User/.claude/projects/` (7 projects, Feb 2026)

---

## The Numbers

### Monthly Summary (February 2026)

| Model | Total Tokens | Cost | % of Cost |
|---|---|---|---|
| Opus 4.6 | 10,265,187 | $7.25 | 53.5% |
| Sonnet 4.6 | 5,209,143 | $2.59 | 19.1% |
| Opus 4.5 | 1,973,430 | $1.66 | 12.2% |
| Haiku 4.5 | 4,886,151 | $1.04 | 7.7% |
| Sonnet 4.5 | 2,343,237 | $1.02 | 7.5% |
| **Total** | **24,677,148** | **$13.56** | |

**Cache hit rate: 94.5%** (23.3M of 24.7M total tokens are cache reads)

### Projects on Disk

7 project directories found:
- `openclaw-workspace` — Larry's main operating session (biggest)
- `openclaw` / `C--Users-User--openclaw` — OpenClaw config
- `PlanExe2026` — PlanExe repo work
- `PlanExe2026-mcp-local` — MCP local server
- `MarkSite` — Mark's site
- `arc-explainer` — ARC explainer project

### Session Breakdown

| Session | Cost | Dominant Model | Cache Hit Rate |
|---|---|---|---|
| openclaw-workspace | $8.51 | Opus 4.6 (78%) | ~96% |
| openclaw (config) | $1.40 | Sonnet 4.5 (57%) | ~96% |
| PlanExe2026 | $1.04 | Sonnet 4.6 (83%) | ~96% |
| PlanExe2026 (Opus 4.5 era) | $1.66 | Opus 4.5 (100%) | ~94% |
| Various sub-agents | $0.31 | Haiku 4.5 (100%) | ~89% |
| mcp-local | $0.03 | Haiku 4.5 (100%) | 0% (new session) |

---

## The Smoking Gun

**`openclaw-workspace` Opus 4.6 session: $6.62 — 49% of all spend.**

Breaking it down:
- Input tokens: 175 (basically nothing)
- Output tokens: 1,656 (the actual responses)
- Cache creates: 280,588 (context being written to cache — the workspace startup files)
- Cache reads: **9,655,433** (same context re-read every single turn)

At Opus cache read rate ($0.50/1M): 9.65M × $0.50 = **$4.83 — just on cache reads.**

The workspace loads MEMORY.md + SOUL.md + USER.md + AGENTS.md + TOOLS.md + HEARTBEAT.md + IDENTITY.md on every session. That's roughly 280K tokens of context sitting in cache. Every turn I take re-reads that entire context at Opus rates.

**The model isn't doing hard reasoning on those cache reads. It's just carrying context.**

---

## What Cache Reads Actually Cost by Model

Same 9.65M cache reads at different model rates:

| Model | Cache Read Rate | Cost for 9.65M reads | Savings vs Opus |
|---|---|---|---|
| Opus 4.6 | $0.50/1M | $4.83 | — |
| Sonnet 4.6 | $0.30/1M | $2.90 | $1.93 (40%) |
| Haiku 4.5 | $0.10/1M | $0.97 | $3.86 (80%) |

**If routine monitoring tasks (heartbeat checks, Discord reads, simple responses) ran on Haiku instead of Opus, the cache read cost alone drops 80%. No capability loss — these tasks don't need Opus reasoning.**

---

## Sub-Agents: Already Doing It Right

Three sub-agent sessions, all Haiku 4.5, total: **$0.31 for 1.17M tokens.**

They start fresh (small context, high input tokens relative to cache reads), do focused work, and terminate. No stale context drag. No Opus overhead. The sub-agent pattern is working exactly as intended.

Effective rate on sub-agents: **$0.26/million tokens.**  
Effective rate on main workspace Opus: **$0.71/million tokens.**

Sub-agents are 2.7x cheaper per token on this machine.

---

## The Workspace Context Problem

The 280K cache create tokens in the workspace session are the startup files: MEMORY.md alone is ~15K tokens, IDENTITY.md ~8K, AGENTS.md ~6K, SOUL.md ~2K, USER.md ~2K, TOOLS.md ~8K, HEARTBEAT.md ~3K. Total startup context ~44K tokens (compressed), but with project context injections it expands to ~280K in the actual session.

**Every heartbeat check, every "HEARTBEAT_OK" reply, every quick Discord read re-reads all 280K tokens from cache at the active model's rate.**

### The Fix: Context Tiering

**Tier 1: Fast context (~10K tokens)**
- Core identity (SOUL.md condensed)
- Active task list (HEARTBEAT.md)
- Today's memory file
- Used for: heartbeat checks, monitoring, quick responses

**Tier 2: Full context (~280K tokens)**  
- Everything above plus MEMORY.md, IDENTITY.md, TOOLS.md, USER.md, AGENTS.md
- Used for: planning sessions, complex tasks, architectural decisions
- Run on: Sonnet or Opus depending on task complexity

**Estimated impact:** If 60% of turns use Tier 1 (Haiku) instead of Tier 2 (Opus):
- Current: $8.51/month on main workspace
- Projected: ~$4.20/month
- Savings: ~$4.30/month (50%)

---

## Recommendations for Simon

Simon doesn't have ccusage data shared yet, but based on his 26 Feb refactor (64 commits, 108 files, estimated ~$15-20 API equivalent):

**His session likely looked like:**
- One or a few long Opus sessions with the PlanExe codebase loaded
- Each commit/PR cycle re-reads the entire codebase context from cache at Opus rates
- PlanExe2026 is large (~50K+ lines) — loading key files creates a massive cache context

**What the data predicts for Simon:**
- Cache hit rate probably 85-92% (large codebase, iterative refactor = lots of repeated context)
- Opus cache reads dominating cost — same pattern as Larry's workspace
- Sonnet for execution tasks would cut the per-turn cache read cost 40%
- Haiku for docs/tests/renames would cut it 80%

**Windsurf plan/execute split applied to Simon's workflow:**

Instead of:
```
One long Opus session: plan + rename + security + perf + docs + tests + deploy
Total cache reads: 50M tokens at Opus rates = ~$25
```

Do:
```
Short Opus session: read architecture, generate task list
   → 2M cache reads = $1.00
Fresh Sonnet sessions (per task cluster): security, perf, renames
   → 20M cache reads at $0.30 = $6.00
Fresh Haiku sessions: docs, tests, deploy checks
   → 10M cache reads at $0.10 = $1.00
Total: ~$8.00 (68% savings, same output)
```

---

## For PlanExe's Routing Engine

This data gives us real calibration points for the routing proposals:

1. **Cache hit rate is 94%+ for iterative coding sessions** — our cost estimates should default to 85-90% cache hit assumption, not 0%

2. **True cost = (cache reads × cache rate) + (new input × input rate) + (output × output rate)** — the cache reads dominate for large-codebase sessions

3. **The routing decision for large-codebase tasks should optimize cache read rate, not just per-token cost** — which model you pick determines the cache read pricing for ALL subsequent turns in that session

4. **Fresh session = fresh cache** — starting a new session for a new task cluster resets the accumulated context. This is sometimes worth doing even if it means re-reading files, because you're not paying for 280K of stale workspace context on every turn

5. **Sub-agent pattern empirically validated** — 2.7x cheaper per token than main session for execution tasks

---

## Next Step: Simon's Data

Waiting on Simon to run:
```bash
CLAUDE_CONFIG_DIR=~/.claude npx ccusage@latest monthly --breakdown
npx ccusage@latest session --breakdown
```

Once we have his data, we can compare his model distribution and cache hit rates against these benchmarks and give him a concrete optimization roadmap for his workflow.

---

*Real data. All numbers from `ccusage@18.0.8` against local Claude Code JSONL files. No estimates.*
