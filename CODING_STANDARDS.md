# Coding Standards (Egon-Friendly)

This document summarizes the generally applicable engineering expectations for PlanExe work from Egon’s Linux workspace. It mirrors the same spirit as the existing instructions (especially those captured in AGENTS.md) but strips Windows-specific references so it’s accurate for a Linux-first context.

## Communication Style

- Keep responses tight and non-jargony; do not dump chain-of-thought.
- Ask only essential questions after consulting docs first.
- Mention when a web search could surface important, up-to-date information.
- Call out unclear docs/plans (and what you checked).
- Pause on errors, think, then request input if truly needed.
- End completed tasks with “done” (or “next” if awaiting instructions).
- Reference AGENTS.md/IDENTITY.md context before referencing other agents or tooling.

## Non-Negotiables

- **No guessing:** when encountering unfamiliar/recently changed libraries or frameworks, locate and read authoritative docs before coding.
- **Quality over speed:** slow down, think, and get a plan approved before implementation.
- **Production-only:** no mocks, stubs, placeholders, fake data, or simulated logic in final code.
- **SRP/DRY:** enforce single responsibility and avoid duplication; search for existing utilities before adding new ones.
- **Real integration:** assume env vars/secrets/external APIs are healthy; if something breaks, treat it as a bug and fix it.
- **Real data only:** never estimate, simulate, or guess metrics. Pull real data from logs/APIs.

## Workflow

1. **Deep analysis:** understand architecture and reuse opportunities before touching code.
2. **Plan architecture:** define responsibilities and reuse decisions before implementation.
3. **Implement modularly:** build small, focused modules and compose from existing patterns.
4. **Verify integration:** validate with real services and flows (no scaffolding).

## Plans (Required Before Substantive Work)

- Draft a plan doc under `docs/{DD-MON-YYYY}-{goal}-plan.md`.
- Plans must include:
  - **Scope:** what is in/out.
  - **Architecture:** responsibilities, reuse choices, module locations.
  - **TODOs:** ordered steps (include verification steps).
  - **Docs/Changelog touchpoints:** list what updates when behavior changes.
- Seek approval on the plan before implementing.

## File Headers (TS/JS/Py edits)

Every TypeScript, JavaScript, or Python file created/edited must start with:

```
Author: {Model Name}
Date: {timestamp}
PURPOSE: Detailed description of functionality, integration points, dependencies.
SRP/DRY check: Pass/Fail – did you verify existing functionality?
```

- Update header metadata when touching a file.
- Skip JSON, SQL migrations, or file types that lack comments.

## Code Quality

- **Naming:** meaningful names; avoid single-letter variables except in tight loops.
- **Error handling:** exhaustive, user-safe errors; handle failure modes explicitly.
- **Comments:** explain non-obvious logic and integration boundaries inline.
- **Reuse:** prefer shared helpers/components over custom one-offs.
- **Architecture:** prefer repositories/services patterns over raw SQL.
- **Pragmatism:** fix root causes; avoid unrelated refactors or over/under-engineering.

## UI/UX Expectations

- State transitions must be clear: collapse/disable prior controls when an action starts.
- Avoid clutter: do not render huge static lists or everything at once.
- Streaming: keep streams visible until the user confirms they have read them.
- Design: avoid default "AI slop" (generic fonts, random gradients, over-rounding). Make deliberate choices.

## Docs, Changelog, and Version Control

- Any behavior change requires updating relevant docs and CHANGELOG.md (SemVer; include what/why/how and author/model name).
- Do not commit unless explicitly requested; when asked, use descriptive commit messages.
- Keep technical depth in docs/changelog rather than dumping it into chat.

## Platform & Environment

- Host OS: Ubuntu 24.04 (Linode) or similar Debian-based Linux.
- Shell: bash/zsh (the default OpenClaw workspace shell).
- Tools: Git, Python 3.12+, `uv`, Node.js (via package manager), Docker where needed.
- Refer to TOOLS.md for machine-specific notes (e.g., SSH, cameras, TTS voices).
- This document assumes you are not on Windows/WSL; ignore the Windows-specific sections from the original version.

## Agent Continuity Notes

- AGENTS.md, SOUL.md, USER.md, and MEMORY.md define your persona/rules. Review them before making behavior-affecting changes.
- Keep `memory/YYYY-MM-DD.md` and `MEMORY.md` updated per guidance; updating these files changes your working memory.
- The PlanExe workflow prefers docs-first proposals—write the plan doc before coding and reference the relevant doc sections in your final notes.

## Prohibited Habits

- No time estimates.
- No premature celebration. Nothing is complete until the user tests it.
- No shortcuts that compromise code quality.
- No overly technical explanations.
- No engagement-baiting questions ("Want me to?" / "Should I?").
