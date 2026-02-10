# UI for Editing Plans

## Status
Draft

## Context
The production site at [home.planexe.org](https://home.planexe.org/) currently does not provide a user-facing UI for creating plans. Users can sign in and manage accounts, but there is no end-user workflow for creating, revisiting, or editing plans in the browser.

Today there are two ways to create plans, but neither is suitable as the long-term end-user experience.

### MCP Interface
The MCP interface can create plans and store them in the database. It also uses `example_prompts`, which helps users land on a reasonable starting prompt instead of a blank textarea.

Limitations:

- It is an expert-user-facing interface, not a friendly beginner UI.

- There is no editing workflow for existing plans.

### Gradio UI (`frontend_single_user`)
The `frontend_single_user` UI is a Gradio interface intended for local or developer use, not for end users.

What works well:

- It supports `Retry`, which re-runs the Luigi pipeline where it left off. This allows manual plan editing by deleting files and regenerating downstream content.

Limitations:

- It does not use the database, so created plans are not persisted and users cannot browse past plans.

- It does not know credit balances. Creating a plan costs tokens, and if the user has insufficient funds, the UI should refuse creation.

- The prompt input is a plain textarea. Users often omit critical constraints (for example, no location or unrealistic budgets). This leads to weak plans or incorrect assumptions, such as the system guessing locations when the user intended a specific geography.

## Goals

- Provide a user-facing plan creation UI on [home.planexe.org](https://home.planexe.org/) and when running locally via docker.

- Ensure plans are persisted and can be revisited.

- Enforce credit checks before plan creation.

- Keep the frontend implementation simple and fully under our control.

## Non-Goals

- Building a React-based frontend. React is controlled by Meta and is not desired.

## Architecture Direction

- Backend: Flask.

- Frontend: handwritten HTML, CSS, and JavaScript.

## Phases
### Phase 1: UI for Creating Plans

- Provide the same benefit as MCP `example_prompts` to help users start with a strong initial prompt.

- Let users submit a plan request through a dedicated form.

- Validate credits and refuse creation when funds are insufficient.

- Persist created plans and allow users to browse past plans.

### Phase 2: UI for Editing Plans

- Display plan parts in topological ordering, because the Luigi pipeline is a DAG of tasks.

- When a part is edited, regenerate downstream parts that depend on it.

### Phase 3: UI for Executing Plans

- As execution reveals surprises, incorporate them into the existing plan.

- Maintain topological ordering so downstream parts update correctly.
