# Proposals Authoring Guide

This folder contains product and research proposals that render under `/proposals/` on docs. The best proposals in this folder share a few consistent traits: they are precise, actionable, and anchored in PlanExe’s existing pipeline.

Below is the distilled guidance based on the current proposals in this folder.

## What Makes a Proposal Good (Observed Patterns)
- **Clear pitch + why now**: A short, specific pitch followed by a concrete “why” (the bottleneck, failure mode, or opportunity).
- **Concrete artifacts**: The best proposals list tangible outputs (schemas, APIs, workflow artifacts, rank formulas, decision classes).
- **Integration points**: They explain where the change fits (e.g., `run_plan_pipeline.py`, routing config, queue, admin UI, MCP).
- **Phased implementation**: They sequence the work in small, verifiable phases.
- **Measurable success**: They define metrics with directionality or target ranges.
- **Risks with mitigations**: They name real failure modes and how to reduce them.
- **Examples or diagrams**: When relevant, they include a snippet, architecture diagram, or formula.

## Naming and Title
- **Filename**: keep the numeric prefix for ordering, e.g. `27-multi-angle-topic-verification-engine.md`.
- **Title**: do **not** include the number in the H1.
  - Good: `# Multi-Angle Topic Verification Engine Before Bidding`
  - Avoid: `# 27) Multi-Angle Topic Verification Engine Before Bidding`

## Metadata Block (Required)
Place directly under the H1. Example:

```
**Author:** PlanExe Team  
**Date:** 2026-02-10  
**Status:** Proposal  
**Tags:** `investors`, `matching`, `roi`, `ranking`, `marketplace`
```

Notes:
- Use backticks for each tag so MkDocs renders them cleanly.
- Keep tags short and searchable.

## Front Matter (Required)
All proposals must include YAML front matter (`---` blocks with `title`, `date`, `status`, `author`). Keep it consistent:
- The front matter `title` must match the H1 (no numeric prefix).
- Don’t rely on the filename for display titles.

## Required Sections
Every proposal should include at least:
- **Pitch**: one short paragraph stating the idea.
- **Problem**: why this matters now.
- **Proposal / Solution**: what we intend to build.
- **Success metrics**: how we will measure outcomes.
- **Risks**: key risks and mitigations.

Optional but recommended:
- **Architecture** or **Workflow**
- **Phases** or **Implementation**
- **Data model / API / formula** when relevant
- **Integration** (where it plugs into current PlanExe systems)

## Markdown Formatting Rules (MkDocs Material)
MkDocs is strict about lists. To avoid lists rendering as a single paragraph:
- **Always add a blank line before numbered or bulleted lists.**
- Keep list items on their own lines.

Correct:

```
## Proposal
Define verification stages:

1. **Stage A: Triage Review (fast)** — identify critical flaws and missing evidence.
2. **Stage B: Domain Review (deep)** — engineering/legal/environmental/financial domain checks.
3. **Stage C: Integration Review** — reconcile cross-domain conflicts.
4. **Stage D: Final Verification Report** — signed conclusions + conditions.
```

Avoid:

```
## Proposal
Define verification stages:
1. **Stage A: Triage Review (fast)** — identify critical flaws and missing evidence.
```

## Suggested Template

```
# Title (no number)

**Author:** PlanExe Team  
**Date:** YYYY-MM-DD  
**Status:** Proposal  
**Tags:** `tag1`, `tag2`, `tag3`

---

## Pitch
One paragraph.

## Problem
Why this matters.

## Proposal
What we plan to build.

## Implementation (optional)
Phases or architecture.

## Integration (optional)
Where it plugs into PlanExe.

## Success Metrics
- Metric 1
- Metric 2

## Risks
- Risk 1
- Risk 2
```
