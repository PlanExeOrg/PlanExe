# Coding Standards

This document summarizes the generally applicable engineering expectations for this repository.

## Communication Style

- Keep responses tight and non-jargony; do not dump chain-of-thought.
- Ask only essential questions after consulting docs first.
- Mention when a web search could surface important, up-to-date information.
- Call out when docs/plans are unclear (and what you checked).
- Pause on errors, think, then request input if truly needed.
- End completed tasks with "done" (or "next" if awaiting instructions).

## Non-Negotiables

- **No guessing:** for unfamiliar or recently changed libraries/frameworks, locate and read docs before coding.
- **Quality over speed:** slow down, think, and get a plan approved before implementation.
- **Production-only:** no mocks, stubs, placeholders, fake data, or simulated logic in final code.
- **SRP/DRY:** enforce single responsibility and avoid duplication; search for existing utilities before adding new ones.
- **Real integration:** assume env vars/secrets/external APIs are healthy; if something breaks, treat it as a bug to fix.
- **Real data only:** never estimate, simulate, or guess at metrics. Pull real data from logs/APIs. No exceptions.

## Workflow

1. **Deep analysis:** understand existing architecture and reuse opportunities before touching code.
2. **Plan architecture:** define responsibilities and reuse decisions before implementation.
3. **Implement modularly:** build small, focused modules and compose from existing patterns.
4. **Verify integration:** validate with real services and real flows (no scaffolding).

## Plans (Required Before Substantive Work)

- Create a plan doc: `docs/{DD-MON-YYYY}-{goal}-plan.md`
- Plan must include:
  - **Scope:** what is in and out.
  - **Architecture:** responsibilities, modules to reuse, where new code will live.
  - **TODOs:** ordered steps including verification steps.
  - **Docs/Changelog touchpoints:** what will be updated if behavior changes.
- Seek approval on the plan before implementing.

## File Headers (Required for TS/JS/Py)

Every TypeScript, JavaScript, or Python file created or edited must start with:

```
Author: {Model Name}
Date: {timestamp}
PURPOSE: Verbose details about functionality, integration points, dependencies
SRP/DRY check: Pass/Fail - did you verify existing functionality?
```

- Update header metadata when touching a file.
- Do NOT add to JSON, SQL migrations, or file types that cannot support comments.

## Code Quality

- **Naming:** meaningful names; avoid one-letter variables except in tight loops.
- **Error handling:** exhaustive, user-safe errors; handle failure modes explicitly.
- **Comments:** explain non-obvious logic and integration boundaries inline.
- **Reuse:** prefer shared helpers and shadcn/ui components over custom one-offs.
- **Architecture:** prefer repositories/services patterns over raw SQL.
- **Pragmatism:** fix root causes; avoid unrelated refactors and avoid over/under-engineering.

## UI/UX Expectations

- State transitions must be clear: collapse/disable prior controls when an action starts.
- Avoid clutter: do not render huge static lists or "everything at once" views.
- Streaming: keep streams visible until the user confirms they have read them.
- Design: avoid "AI slop" (default fonts, random gradients, over-rounding). Make deliberate choices.

## Docs, Changelog, and Version Control

- Any behavior change requires updating relevant docs and CHANGELOG.md (SemVer; what/why/how; include author/model name).
- Do not commit unless explicitly requested; when asked, use descriptive commit messages.
- Keep technical depth in docs/changelog rather than dumping it into chat.

## Platform & Environment

- Host OS: Windows 11. Docker, Git, and Node.js installed. WSL2 available.
- Shell: bash in WSL2 Ubuntu.

## Prohibited Habits

- No time estimates.
- No premature celebration. Nothing is completed until the user tests it.
- No shortcuts that compromise code quality.
- No overly technical explanations.
- No engagement-baiting questions ("Want me to?" / "Should I?").
