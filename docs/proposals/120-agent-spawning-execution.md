# Proposal 120: Agent-Spawning Execution — Plans That Boot Their Own Runtime

**Author:** Larry (OpenClaw Agent)  
**Status:** Proposal  
**Date:** March 19, 2026  
**Discussion:** Simon (neoneye) — Discord direct

---

## The Gap: Plans Without Executors

PlanExe generates **detailed plans with fictional team rosters**. Each plan includes:
- Structured task graph (dependencies, phases, milestones)
- Assigned "team members" with roles, expertise, and responsibilities
- Lever system mapping decision authority and resource control

But there's a problem: **the team members are text. Nobody actually executes the plan.**

The fictional project manager, designer, engineer, QA specialist—they exist in the document, but not in reality. A human reader must:
1. Parse the plan manually
2. Assign real people
3. Coordinate execution outside PlanExe
4. Hope something happens

Result: Plans become read-only artifacts. The computational effort to generate them is wasted.

---

## The Idea: Fictional Teams Become Real Agents

What if PlanExe's `ReviewTeamTask` output wasn't just team bios—but **agent manifests**?

Instead of:
```
**Project Manager (Jane):** Experienced in scope management, stakeholder communication...
```

PlanExe generates:
```yaml
# jane.soul.md
name: Jane
expertise: [scope-management, stakeholder-communication, risk-mitigation]
decision_authority: [sprint-scope, timeline-negotiation]
communication_channel: #jane-pm
```

That manifest becomes a **real OpenClaw agent**:
- `SOUL.md` — expertise, personality, decision-making rules
- `IDENTITY.md` — hardware constraints, available tools, local identity
- Channel configuration — where they post status, receive work, escalate blockers

The agent reads the plan, identifies tasks assigned to them, picks them up, executes them, and reports back.

**Three lobsters prove this works:** Larry, Egon, and Bubba are OpenClaw agents with SOUL.md files. In 47 days, they shipped **308 PRs** across multiple repos. They operated with minimal human supervision, coordinated via Discord, and self-escalated when blocked.

---

## How It Works: The Execution Dance

### 1. Plan Graph = Task Dependency Graph
PlanExe already generates a detailed task graph. Each node is:
- A concrete work item
- Assigned to one or more team members
- Blocked by upstream tasks
- Blocking downstream tasks

OpenClaw agents consume this graph as a feed. They poll or subscribe to their assigned work.

### 2. Lever System = Decision Authority Map
Plans specify who decides what:
```
Scope decisions → PM (Jane)
Technical architecture → Lead Engineer (Bob)
Quality gates → QA (Sara)
```

When an agent hits a decision point, they check the lever map. If they own that lever, they decide. If not, they escalate to whoever does.

Example:
- Agent: "Should we add feature X?"
- Lever map: Feature decisions owned by Product Manager
- Agent escalates to PM, receives decision, continues work

### 3. Each Agent Owns Their Task Queue
When a plan is instantiated:
1. PlanExe outputs agent manifests (SOUL.md + IDENTITY.md per team member)
2. Each agent's identity includes a task channel (Discord, email, webhook, etc.)
3. The plan is posted as a series of tasks to each agent's channel
4. Each agent reads their queue, prioritizes, executes

Agents self-coordinate:
- "I'm blocked on Jane's API design" → escalates to Jane
- "I need @qa review before shipping" → posts for review
- "This task is outside my expertise" → escalates to project manager

### 4. Agent Highway Patrol: Validation Pipeline
Before work ships, a **critic agent** (like a Highway Patrol officer) validates:
- Does this PR pass tests?
- Does it match the plan requirements?
- Are there security/quality issues?
- Are there open feedback comments?

If validation fails, the work bounces back to the author. If it passes, it gets merged.

This prevents broken code from shipping while keeping humans out of the approval loop for routine work.

### 5. Humans Supervise the Swarm, Not Individual Tasks
A human project lead doesn't micro-manage every subtask. Instead:
- They set plan parameters and constraints
- They respond to escalations (decisions the agents can't make)
- They monitor progress (agent status reports)
- They intervene if the swarm goes off track

This is orders of magnitude cheaper than waterfall PM oversight.

---

## Proof of Concept: 3 Lobsters, 308 PRs, 47 Days

We have working proof that agent-with-SOUL execution works:

**The Agents:**
- **Larry** (OpenClaw assistant) — generalist executor, planner, coordinator
- **Egon** (OpenClaw helper) — specialist in certain domains, collaborator
- **Bubba** (Discord bot + remote agent) — hardware specialist, integrations

**The Results:**
- 308 PRs merged across VoynichLabs repos in 47 days
- Multiple concurrent projects (website, swarm-coordination, PlanExe forks)
- Agents self-coordinated via Discord, Git, and plan files
- Minimal human intervention (mostly decisions and approvals)

**What Made It Work:**
- Clear SOUL.md for each agent (expertise, rules, decision authority)
- Shared Discord channels for coordination and escalation
- Task-oriented (pull from plan/queue, execute, report, move to next)
- Humans supervise the swarm, not individual PRs

---

## Open Questions for Simon

Before full implementation, we need to answer:

### 1. Agent Lifecycle Management
- When a plan starts, do agents boot fresh? Or reuse existing agents?
- Do agents persist across multiple plans?
- How do we version/rollback an agent's SOUL.md if their decision-making needs adjustment mid-plan?

### 2. Partial Execution Failure
- If Agent A completes 80% of their tasks, then gets reassigned, what happens to the remaining 20%?
- Can agents hand off incomplete work? To whom?
- How does PlanExe track "partial completion" in the plan status?

### 3. Tool Provisioning Per Specialist
- A design agent needs Figma. An engineer needs GitHub. An ML specialist needs Jupyter.
- Does PlanExe describe tool requirements per team member?
- How does OpenClaw provision those tools when the agent boots?

---

## Next Steps

If this resonates:

1. **Spec the agent manifest format** — formalize SOUL.md + IDENTITY.md schema for PlanExe output
2. **Wire up `ReviewTeamTask`** — modify it to generate agent config, not just text bios
3. **Build an executor** — OpenClaw service that polls plans and dispatches work to agents
4. **Test on a real plan** — generate a small plan, boot agents, let them execute
5. **Iterate on coordination** — refine escalation, decision authority, status reporting

---

## Why This Matters

**For PlanExe:**
- Plans become *actionable*, not just documents
- Generated artifacts (code, designs, reports) are real, not hypothetical
- Users see immediate ROI: "I ran a plan and got a website"

**For OpenClaw:**
- Agents scale from 3 to N without human overhead
- Humans move from "task manager" to "swarm supervisor"
- Plans are the language agents use to coordinate

**For the broader vision:**
- Machines can plan AND execute
- Humans provide direction; machines provide labor
- The gap between "what should happen" and "what actually happens" closes

---

## Appendix: References

- **Lobster Swarm Proof of Concept:** 3 agents, 308 PRs, 47 days (Feb-Mar 2026)
- **OpenClaw Architecture:** Agent orchestration platform for multi-agent execution
- **PlanExe ReviewTeamTask:** Generates fictional team rosters with expertise/roles
