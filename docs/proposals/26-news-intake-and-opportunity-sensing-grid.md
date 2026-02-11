---
title: News Intake + Opportunity Sensing Grid for Autonomous Bidding
date: 2026-02-10
status: proposal
author: Larry the Laptop Lobster
---

# News Intake + Opportunity Sensing Grid for Autonomous Bidding

## Pitch
Build a continuous news-intake grid that detects project opportunities (bridge, IT infrastructure, utilities, public procurement) and turns them into structured planning prompts at scale. The grid should convert weak signals into structured opportunities, rank them by urgency and bidability, and feed a planning engine with the right context for fast, defensible responses.

## Why
If an autonomous AI organization generates ~1000 plans/day, the bottleneck is not planning - it is **finding high-value opportunities early** and classifying them correctly.

## Goals

- Detect real opportunities before the average bidder.
- Convert noisy, unstructured announcements into a consistent `opportunity_event`.
- Score urgency, bidability, strategic fit, and evidence quality.
- Generate ready-to-plan prompts with no missing critical inputs.
- Maintain auditability so humans can trust automated detection.

## Proposal
Implement a multi-source intake pipeline:

1. Ingest signals from procurement feeds, industry media, government notices, and infrastructure newsletters.
2. Normalize each item to an `opportunity_event` schema.
3. Score urgency + bidability + strategic fit.
4. Auto-generate candidate prompts for plan creation.

## Source Categories To Monitor

- Public procurement portals (national + regional)
- Government transport/infrastructure bulletins
- Utility/telecom modernization notices
- Construction/engineering trade publications
- Press wires (major project announcements)
- Local/regional news for early non-centralized opportunities

## System Architecture

```text
Signal Ingestion
  -> Feeds, portals, news
  -> Alerts, newsletters
  -> Press releases

Parsing + Normalization
  -> Language detection
  -> Entity extraction
  -> Standardized schema

Opportunity Scoring
  -> Urgency
  -> Bidability
  -> Strategic fit
  -> Evidence quality

Prompt Generator
  -> PlanExe prompt draft
  -> Missing info checklist
  -> Suggested next actions

Review + Dispatch
  -> Human-in-the-loop
  -> Auto-plan threshold
  -> CRM / bidding workflow
```

## Core Schema

```json
{
  "event_id": "...",
  "source": "...",
  "domain": "bridge|it_infra|energy|...",
  "region": "...",
  "estimated_budget": "...",
  "deadline_hint": "...",
  "procurement_stage": "pre_notice|rfp|tender|award",
  "buyer_type": "government|sovereign|enterprise|ngo",
  "contract_type": "fixed|cost_plus|ppp|concession",
  "language": "da|en|pt|...",
  "confidence": 0.0,
  "evidence_quality": "weak|medium|strong",
  "source_freshness_hours": 0,
  "signals": ["..."],
  "raw_text": "..."
}
```

## Opportunity Scoring Model

The grid should compute a composite `OpportunityScore` for each event, making sure each sub-score is explainable:

- **Urgency (0-100):** deadline proximity, scarcity of time to respond, and stage (RFP vs pre-notice).
- **Bidability (0-100):** contract clarity, budget signal, likely fit to internal capabilities, and compliance feasibility.
- **Strategic Fit (0-100):** overlap with thesis, geography, portfolio gaps, and margin potential.
- **Evidence Quality (0-100):** source credibility, corroboration, and clarity of requirements.

**Example composite formula:**

```
OpportunityScore =
  0.35*Urgency +
  0.30*Bidability +
  0.25*StrategicFit +
  0.10*EvidenceQuality
```

Also compute a **Missing Info Penalty** that flags items requiring clarification before a plan can be generated.

## Ingestion Rules

- Prefer authoritative sources (procurement portals, official notices) over reprints.
- Apply deduplication using `event_id` + fuzzy similarity on title/location/budget.
- Track `source_freshness_hours` to avoid stale opportunities.
- Capture original text for auditability.

## Prompt Generation Strategy

For each qualified event:

1. Generate a **PlanExe prompt** with minimal rework needed.
2. Attach a **missing-info checklist** with deadlines and dependencies.
3. Attach **recommended next actions** (e.g., request tender docs, schedule site visit).

The prompt should include structured facts and explicit unknowns. This prevents hallucinated assumptions from contaminating the plan.

## Human-in-the-Loop Thresholds

Define three levels:

- **Auto-Plan:** high score + strong evidence + clear requirements.
- **Review Required:** medium score or incomplete data.
- **Discard:** low score or weak evidence signal.

This allows the system to scale while avoiding wasted planning cycles.

## Example Scenarios

### A) Denmark Government Project Announcement (Time-Boxed Bid)

**Signal:** Danish government announces a cross-border infrastructure project. Bidders have `X` weeks to respond.

**Sensing grid outcome:**

- Detects an official notice (authoritative source).
- Assigns high urgency due to strict deadline.
- Identifies buyer as government with procurement compliance requirements.
- Generates a PlanExe prompt with a procurement checklist and translation note.

**Prompt output excerpt (conceptual):**

- Domain: transport infrastructure
- Region: Denmark + neighboring country
- Deadline: `X weeks` from notice date
- Contract: likely PPP or fixed-price
- Missing info: tender docs, pre-qualification criteria, environmental review status

### B) Company Layoffs Indicate Distress and Need for Help

**Signal:** News reports a company has laid off a large percentage of staff.

**Sensing grid outcome:**

- Detects layoffs + revenue pressure + restructuring language.
- Flags opportunity for turnaround services or partnership.
- Classifies as enterprise-private sector (non-procurement).
- Assigns medium urgency (short window to engage before competitors).

**Prompt output excerpt (conceptual):**

- Domain: operational turnaround / cost reduction
- Region: company HQ + key operational sites
- Evidence: news sources only (weak to medium)
- Missing info: financials, contractability, decision makers

### C) Researcher Whitepaper With Potential Productization

**Signal:** A researcher publishes a whitepaper and invites collaboration.

**Sensing grid outcome:**

- Classifies as early-stage, pre-commercial.
- Scores strategic fit based on domain match and novelty.
- Low urgency but high potential value.
- Generates a PlanExe prompt focused on proof-of-concept and commercialization.

**Prompt output excerpt (conceptual):**

- Domain: deep tech / research commercialization
- Region: researcher's institution
- Evidence: paper + citations (medium evidence)
- Missing info: IP ownership, licensing terms, target market

## Success Metrics

- Opportunity recall vs known project announcements
- Time-to-detection after first public signal
- % opportunities converted to high-quality planning prompts
- Precision@N: % of top-ranked items that lead to viable plans
- Time saved per bid cycle vs manual sourcing
- Conversion rate from opportunity to funded project

## Risks

- **False positives:** wasted planning cycles. Mitigate with evidence scoring and review gates.
- **False negatives:** missed high-value opportunities. Mitigate by widening sources and alert thresholds.
- **Source bias:** over-reliance on English or major outlets. Mitigate with multilingual ingestion.
- **Gaming or PR spin:** misleading announcements. Mitigate via cross-source verification.

## Implementation Roadmap

### Phase 1: Ingestion + Schema

- Build connectors for procurement feeds and major news sources.
- Implement entity extraction and schema normalization.
- Basic scoring heuristics and deduplication.

### Phase 2: Scoring + Prompting

- Train scoring logic on historical outcomes.
- Add missing-info checklist generation.
- Integrate with PlanExe prompt creation.

### Phase 3: Operational Integration

- Human-in-the-loop review interface.
- CRM and bidding workflow dispatch.
- Feedback loop from bid outcomes to scoring.
