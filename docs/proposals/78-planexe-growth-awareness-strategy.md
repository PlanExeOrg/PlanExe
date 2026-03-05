# Proposal 78: PlanExe Growth & Awareness Strategy

**Author:** Larry (VoynichLabs)  
**Date:** 28 February 2026  
**Status:** Draft — for Simon's review and team execution  
**Relates to:** Proposals 73–77 (technical foundation complete)

---

## Executive Summary

PlanExe's technical foundation is solid (complexity routing, cache-aware handoff, cost-aware execution). The missing piece is *visibility*. This proposal outlines a systematic growth strategy across five channels: GitHub metrics, AI agent adoption, human user growth, Discord community, and social media presence.

The goal: transform PlanExe from an internal planning tool into the standard planning layer for agent-driven systems.

---

## Current State (Feb 2026)

**GitHub:**
- ~150 stars (estimate)
- 8+ merged proposals (PRs #102–#106, etc.)
- Active development on PlanExeOrg/PlanExe upstream
- VoynichLabs fork (PlanExe2026) with experimental features

**Community:**
- Discord: minimal activity (~5 members, core team only)
- Twitter: no active account
- Reddit: no presence
- HackerNews: no post history
- Product Hunt: not launched

**Agent Adoption:**
- Zero documented integration examples
- No clear "how to integrate PlanExe" onboarding
- No agent SDK or library (MCP available but undocumented externally)

**Human Users:**
- Direct users: Simon + internal team (3–5 people)
- Enterprise/commercial: none
- Academic interest: unknown

---

## Goals (90-day horizon)

| Metric | Current | Target | 90-day |
|--------|---------|--------|--------|
| GitHub stars | ~150 | 500+ | 300–400 |
| GitHub forks | ~20 | 100+ | 50–75 |
| Open PRs from external | 0 | 5+ | 2–3 |
| Agent integration examples | 0 | 5+ | 3–4 |
| Discord members | 5 | 50+ | 25–30 |
| Monthly website visits | ~0 | 500+ | 200–300 |
| Social media followers | 0 | 100+ | 30–50 |

---

## Strategy 1: GitHub Visibility & Contribution Onboarding

### Current Problem
PlanExe is technically excellent but looks dormant to outsiders. Proposal docs are deep, but there's no "start here" landing page that explains what PlanExe *is* for someone encountering it for the first time.

### Actions

**A. README Overhaul**
- Add a one-sentence value prop: *"PlanExe: Multi-agent planning with automatic cost-aware model routing and execution auditing."*
- Add a "Getting Started in 5 Minutes" section (install, run hello-world plan, view cost breakdown)
- Link to live demo or demo video
- Add badges (stars, forks, build status, license)

**B. Contribution Guide**
- Create `CONTRIBUTING.md` with clear categories:
  - Bug reports (template)
  - Feature requests (template)
  - Documentation improvements (point to `/docs/proposals`)
  - Code contributions (fork → PR workflow)
- Document the proposal-first workflow (write proposal, get feedback, then implement)

**C. Examples & Integrations Hub**
- Create `/examples` directory with 3–5 documented, runnable examples:
  - "Plan a 3-step DevOps workflow" (show cost routing)
  - "Multi-agent conversation with PlanExe auditing"
  - "Integrate with CrewAI orchestrator"
  - "Run locally with Ollama + PlanExe"
  - "Cost-aware autonomous agent loop"
- Each example: README, source code, expected output, cost breakdown

**D. Badges & Social Proof**
- Add GitHub star badge to README (auto-updates)
- Add "Used by [Agent names]" section (once we have adoption)
- Add testimonial quotes (from Simon, Egon, Bubba, early users)

### Owner
**Egon** (documentation + examples)

### Timeline
1–2 weeks

---

## Strategy 2: Agent SDK & Integration Library

### Current Problem
There's no clear way for external agents (AutoGen, CrewAI, Anthropic agents, OpenClaw) to use PlanExe. The MCP protocol exists but is underdocumented.

### Actions

**A. Agent Integration Guide**
- Write `/docs/agent-integration.md`:
  - MCP protocol overview
  - JSON-RPC endpoint format
  - PlanExe capabilities (plan, estimate_cost, execute_audit)
  - Required authentication
  - Error handling & retries
  - Code samples in 3 languages (Python, JavaScript, Go)

**B. Python SDK (pip-installable)**
- Minimal wrapper around MCP protocol
- `pip install planexe`
- Simple API:
  ```python
  from planexe import PlanExe
  client = PlanExe(host="http://localhost:8001")
  plan = client.generate("write a blog post about agents")
  cost = client.estimate_cost(plan)
  result = client.execute(plan)
  print(f"Executed for ${cost.total_usd}")
  ```
- Publish to PyPI
- Update README with quick-start

**C. Worked Examples per Framework**
- AutoGen + PlanExe example (notebook)
- CrewAI + PlanExe example (repository)
- OpenClaw + PlanExe example (repository, may already exist)
- LangChain agent + PlanExe (repository)
- Anthropic SDK + PlanExe (code sample)

### Owner
**Bubba** (SDK + core examples)

### Timeline
2–3 weeks

---

## Strategy 3: AI Agent Adoption Targets

### Current Problem
Zero documented integrations. External agents don't know PlanExe exists.

### Actions

**A. Outreach to Agent Frameworks**
- AutoGen (Microsoft) — submit PR with PlanExe integration example
- CrewAI — reach out with integration code + blog post offer
- LangChain — submit LangChain tool for PlanExe
- Anthropic SDK — showcase PlanExe in agent loop examples
- OpenClaw — document PlanExe as MCP available by default

**B. Agent-to-Agent Capability Advertising**
- Publish `/docs/a2a-capability.json` (from Proposal 76)
- Link from main README: "PlanExe advertises itself to agents via the A2A protocol"
- Include x402 payment capability advertisement (even if not yet implemented)

**C. Case Study: Autonomous Agent Loop**
- Write blog post: *"Building a Self-Auditing Autonomous Agent with PlanExe Cost Routing"*
- Show:
  - Agent generates plan
  - PlanExe routes by complexity
  - Cost breakdown per step
  - Agent learns which models are cost-efficient
- Code: `/examples/autonomous-loop-with-auditing`
- Publish to dev.to, HackerNews, Reddit

**D. Research Papers & Conferences**
- Propose talk at AI agent conferences (e.g., Hugging Face, OpenAI DevDay adjacent, SmallModels)
- Write technical paper: *"Cost-Aware Routing in Multi-Model Agent Systems"*
- Target venues: arXiv, agent-focused workshops

### Owner
**Egon** (outreach + examples)

### Timeline
3–4 weeks (some parallel)

---

## Strategy 4: Human User Acquisition

### Current Problem
No path for human users (product managers, ML engineers, DevOps teams) to discover PlanExe.

### Actions

**A. Product Website**
- Simple marketing site (Astro or similar):
  - Hero: *"Plan once. Route intelligently. Track costs."*
  - Features section: complexity routing, cost auditing, agent-ready
  - Screenshot/screencast: run a plan, see cost breakdown
  - Pricing: (free for self-hosted, freemium for cloud later)
  - CTA: "Get Started" → link to GitHub + docs

**B. Blog & Content Marketing**
- Monthly blog posts (on website + cross-post to dev.to, Medium):
  - *"How Much Does It Cost to Run Your AI Workflow?"* (cost analysis)
  - *"When to Use Haiku vs. Opus: The PlanExe Complexity Rubric"* (explainer)
  - *"Autonomous Agents Need Auditing"* (thought leadership)
  - *"Building Multi-Model Systems Without Breaking the Bank"* (case study)

**C. Email Newsletter (optional)**
- Monthly digest: new proposals, user stories, cost trends
- Link at bottom of GitHub README: "Subscribe to PlanExe updates"

**D. Webinar / Demo**
- 30-minute live demo:
  - Run a plan in real-time
  - Show cost breakdown
  - Q&A with Simon
  - Record for YouTube

### Owner
**Bubba** (website) + **Egon** (blog + outreach)

### Timeline
2–3 weeks for MVP site, ongoing for blog

---

## Strategy 5: Community & Social Media

### Current Problem
Zero online community. No social media presence.

### Actions

**A. Discord Community**
- Invite 20–30 interested developers (from GitHub stars, agent framework communities)
- Create channels:
  - `#announcements` — new features, proposals, releases
  - `#showcase` — user projects, integrations, case studies
  - `#help` — troubleshooting, questions
  - `#proposals` — discussion of pending proposals before/after merge
  - `#jobs` — PlanExe-related opportunities
- Weekly "What are you building?" thread

**B. Twitter / X**
- Create account: @PlanExeOrg (or similar)
- Tweet schedule (2–3x/week):
  - New proposals (with 1-paragraph summary)
  - Agent integrations
  - Cost benchmarks (*"Running this 5-step plan with Sonnet vs. Haiku routing saved $0.47"*)
  - Links to blog posts
  - Retweet agent framework updates

**C. HackerNews / Reddit**
- Post 2–3 Show HN threads over 90 days:
  - Launch: *"Show HN: PlanExe — Cost-Aware Planning Layer for AI Agents"*
  - Follow-up: *"PlanExe Now Has Agent SDKs"*
  - Blog post links in comments
- Monitor r/MachineLearning, r/LocalLLM for PlanExe-relevant threads; answer questions

**D. Podcast / Talk Circuit**
- Pitch Simon for interviews:
  - AI Breakfast podcast
  - Gradient Descent (if agent-focused)
  - Local LLM / Small Models podcasts
- Speaking proposals to conferences (3–6 months out)

### Owner
**Larry** (social media + outreach) + **Simon** (key interviews)

### Timeline
1 week to set up, ongoing

---

## Execution Plan (90 Days)

### Week 1–2 (Now – March 15)
- [ ] README overhaul (Egon)
- [ ] CONTRIBUTING.md (Egon)
- [ ] Twitter account setup (Larry)
- [ ] Discord server setup (Bubba)
- [ ] Python SDK skeleton (Bubba)

### Week 3–4 (March 15–29)
- [ ] Agent integration guide (Bubba)
- [ ] First 2 agent examples (Bubba + Egon)
- [ ] Website MVP (Bubba)
- [ ] First blog post (Egon)
- [ ] Twitter content calendar planned (Larry)

### Week 5–8 (March 29 – April 26)
- [ ] AutoGen + CrewAI integration PRs submitted (Egon)
- [ ] Python SDK released to PyPI (Bubba)
- [ ] 3–4 blog posts published (Egon)
- [ ] Case study example complete (Egon + Bubba)
- [ ] Discord invites sent, community bootstrapped (Larry)
- [ ] HackerNews post (Simon + Larry)

### Week 9–12 (April 26 – May 24)
- [ ] Conference talk proposals submitted (Simon)
- [ ] Podcast interview recorded (Simon + 1 lobster)
- [ ] User feedback incorporated
- [ ] Website metrics reviewed
- [ ] Social media engagement reviewed

---

## Success Metrics

**GitHub:**
- 300–400 new stars (total 450–550)
- 2–3 PRs from external contributors
- 1–2 issues from external users

**Agent Adoption:**
- 3–4 worked integration examples published
- 2–3 agent frameworks documenting PlanExe in their ecosystem
- 10+ developers mentioning PlanExe in agent projects (Twitter, GitHub, forums)

**Human Users:**
- 200+ website visits/month
- 2–3 blog posts reaching 100+ readers each
- 1–2 users in Discord from external sources

**Community:**
- 25–30 Discord members
- 30–50 Twitter followers
- 1 HackerNews post with 100+ upvotes
- 1 podcast interview

---

## Dependencies & Assumptions

- **Assumption:** Simon is available for 2–3 key interviews/talks (120 min total)
- **Assumption:** External frameworks (AutoGen, CrewAI) are willing to accept integration PRs
- **Dependency:** Website infrastructure (Vercel, GitHub Pages, or similar)
- **Dependency:** Twitter/social media continuity (Larry or designee owns daily updates)

---

## Risks & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Low external interest in adoption | Medium | High | Start with agent framework maintainers who are already interested; showcase internal use case (OpenClaw) first |
| Contributors don't follow proposal-first workflow | Medium | Medium | Clear CONTRIBUTING.md + examples; pair review with external PRs |
| Social media/blog burnout | Medium | Low | Rotate authors (Egon, Larry, Bubba); repurpose proposal docs into blog posts |
| No budget for ads | High | Low | Organic growth is slower but sustainable; focus on free channels (Twitter, Reddit, HackerNews) |

---

## Conclusion

PlanExe is ready to be discovered. This strategy provides a map to turn technical excellence into community momentum and market adoption.

The next 90 days should result in:
1. **Infrastructure:** website, SDK, integration examples, community platform
2. **Visibility:** 3–5x GitHub stars, social media presence, blog authority
3. **Adoption:** first external agent integrations, user feedback, case studies

Success looks like: *"PlanExe is the default planning layer for cost-aware AI agents."*
