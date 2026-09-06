---
title: Welcome to PlanExe
---

# Welcome!

PlanExe turns a single plain-English goal into a ~40-page [strategic plan](https://en.wikipedia.org/wiki/Strategic_planning) in ~15 minutes using local or cloud LLMs. It’s an accelerator for first drafts — not a replacement for human refinement. PlanExe removes most of the labor for the planning scaffold; the final 10–30% that makes a plan credible and defensible remains human work.

**Try it first, then decide if you want to run it locally.**

- **Open a full sample report (HTML)**: [Minecraft Escape sample report](https://planexe.org/20251016_minecraft_escape_report.html)
- **Try PlanExe in your browser (no installation)**: [planexe.org](https://planexe.org/)
- **See more examples**: [planexe.org/examples](https://planexe.org/examples/)

---

## Start here (pick your path)

Use the short decision guide: [Start here](start_here.md)

---

## Core guides

- [Prompt writing guide](prompt_writing_guide.md)
- [Plan output anatomy](plan_output_anatomy.md)
- [Costs and models](costs_and_models.md)
- [Business model (developer)](business_model.md)

---

## What you get

PlanExe generates a **single HTML report** (a self-contained artifact you can open in a browser). See the sample report here:
[Minecraft Escape sample report](https://planexe.org/20251016_minecraft_escape_report.html)

---

## 2-minute tour

Open the sample report and do this:

1. Read **Executive Summary** to see the top-level deliverables, budget, risks, and next steps.
2. Jump to **Gantt** to see how the goal gets broken down into many concrete tasks.
3. Open **Premortem** to see what could go wrong and what to do about it.

---

## Get help

If you run into issues, **join the PlanExe Discord** — the community and maintainers are there to help. When you ask, include what you tried, your setup (OS, Docker vs local, model/LLM), and any error output so others can help you quickly.

**[Join the PlanExe Discord →](https://planexe.org/discord)**

---

## Links

- Website: [planexe.org](https://planexe.org/)
- GitHub: [PlanExeOrg/PlanExe](https://github.com/PlanExeOrg/PlanExe)
- Discord: [planexe.org/discord](https://planexe.org/discord)

---

## Model Context Protocol (MCP)

PlanExe provides an MCP server for AI agents.

The tool workflow is:

1. Call `example_plans` to preview the output. This step is optional.
2. Call `example_prompts`.
3. Call `model_profiles` to choose a model profile. This step is optional.
4. Draft and approve a detailed prompt.
5. Call `plan_create`.
6. Poll `plan_status` about every five minutes until the plan is complete.
7. Call `plan_retry` if generation fails and should restart.
8. Download the report or source bundle with `plan_file_info`.

Each `plan_create` call returns a new `plan_id`. Clients are responsible for
tracking plans they create in parallel.

### Connect to the hosted MCP server

You need:

- An account at [home.planexe.org](https://home.planexe.org)
- Enough account credit to generate a plan
- A PlanExe API key beginning with `pex_`

Add this configuration to an MCP-compatible client:

```json
{
  "mcpServers": {
    "planexe": {
      "url": "https://mcp.planexe.org/mcp",
      "headers": {
        "X-API-Key": "pex_your_api_key_here"
      }
    }
  }
}
```

PlanExe has setup guides for:

- [Claude](mcp/claude.md)
- [Codex](mcp/codex.md)
- [Cursor](mcp/cursor.md)
- [LM Studio](mcp/lm_studio.md)
- [Windsurf](mcp/windsurf.md)
- [Antigravity](mcp/antigravity.md)

See [MCP setup](mcp/mcp_setup.md), [tool details](mcp/mcp_details.md), and the
[PlanExe MCP interface](mcp/planexe_mcp_interface.md) for the full reference.

### Run the MCP server locally

You need Docker and an LLM provider configuration. Start the stack:

```bash
docker compose up --build
```

Confirm that the web interface can create a plan, then connect the MCP client
to:

```text
http://localhost:8001/mcp
```

Authentication is disabled by default in the local Docker configuration.

## Run the pipeline from the command line

Run these commands from the root of a PlanExe checkout:

```bash
# For a substantial project, put the full brief in a file
./planexe create_plan \
    --plan-file regional_clean_water_brief.txt \
    --output-dir ./planexe-outputs/1984-12-31/RegionalCleanWater_v1

# A shorter prompt can also be supplied directly
./planexe create_plan \
    --plan-text "Plan a phased ERP migration across five countries without interrupting manufacturing" \
    --output-dir ./planexe-outputs/1984-12-31/ManufacturingERP_v1
```

The command creates the output directory and writes:

| File | Contents |
|---|---|
| `start_time.json` | Start time in UTC |
| `plan.txt` | The submitted project brief |

## Use PlanExe from an AI agent

An AI agent can create a plan through the MCP server and use the report or
source bundle in later work.

- Service metadata: [mcp.planexe.org/llms.txt](https://mcp.planexe.org/llms.txt)
- MCP endpoint: `https://mcp.planexe.org/mcp`
- Prompt format: call `example_prompts`, then draft 300–800 words of flowing
  prose covering the objective, constraints, timeline, stakeholders, resources,
  and success criteria
- Agent workflow: [Autonomous agent guide](mcp/autonomous_agent_guide.md)

The downloadable source bundle includes Markdown, JSON, and CSV artifacts such
as the work breakdown structure and pre-project assessment.

## Run PlanExe locally with Docker

Docker Compose runs the web interface, plan worker, and supporting services.

### 1. Clone the repository

```bash
git clone https://github.com/PlanExeOrg/PlanExe.git
cd PlanExe
```

### 2. Configure a model provider

Copy `.env.docker-example` to `.env`, then add the required credentials for the
provider you selected. For example, an OpenRouter setup uses
`OPENROUTER_API_KEY`.

The containers mount `.env` and `llm_config/`. To use Ollama on the host, select
the relevant Ollama profile and make sure it is listening on
`http://host.docker.internal:11434`.

### 3. Start PlanExe

```bash
docker compose up worker_plan frontend_multi_user
```

The web interface becomes available at `http://localhost:5001` after its
dependencies pass their health checks. Generated plans are written to `run/`.

To watch the worker:

```bash
docker compose logs -f worker_plan
```

Stop the services with `Ctrl+C`.

### 4. Rebuild after dependency changes

```bash
docker compose build --no-cache worker_plan frontend_multi_user
```

See [Docker setup](docker.md) for ports, configuration, and troubleshooting.

## Model configuration

- For cloud models, see [OpenRouter](ai_providers/openrouter.md).
- For local models, see [Ollama](ai_providers/ollama.md) or
  [LM Studio](ai_providers/lm_studio.md).

Cloud models are generally simpler to configure. Local models give you more
control over the model and where project data is processed.

## Next steps

- [Choose how to run PlanExe](start_here.md)
- [Write a project brief](prompt_writing_guide.md)
- [Understand the generated files](plan_output_anatomy.md)
- [Review costs and model choices](costs_and_models.md)
- [Get help on Discord](https://planexe.org/discord)
