[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/planexeorg-planexe-badge.png)](https://mseep.ai/app/planexeorg-planexe)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)"
            srcset="docs/hero/planexe-hero-v1-grid-dark.svg">
    <img src="docs/hero/planexe-hero-v1-grid-light.svg"
         alt="The PlanExe icon is the P character and E character"
         width="100%">
  </picture>
</p>

<p align="center">
  <strong>Turn your idea into a comprehensive plan in minutes, not months.</strong>
</p>

<p align="center">
  <a href="https://app.mach-ai.com/planexe_early_access">
    <img src="https://img.shields.io/badge/%F0%9F%9A%80%20Try%20PlanExe%20in%20your%20browser-Generate%20a%20free%20plan-2ea44f?style=for-the-badge" alt="Try PlanExe in your browser — generate a free plan" height="48">
  </a>
</p>

<p align="center">
  Describe your idea, hit submit, and PlanExe returns a ~40-page plan in about 15 minutes.
</p>

<p align="center">
  <a href="https://home.planexe.org/"><strong>Create an account</strong></a> &nbsp;|&nbsp;
  <a href="https://planexe.org/examples/"><strong>See example plans</strong></a> &nbsp;|&nbsp;
  <a href="https://docs.planexe.org/getting_started/"><strong>Getting started guide</strong></a>
</p>

---

## Example plans generated with PlanExe

- A business plan for a [Minecraft-themed escape room](https://planexe.org/20251016_minecraft_escape_report.html).
- A business plan for a [Faraday cage manufacturing company](https://planexe.org/20250720_faraday_enclosure_report.html).
- A pilot project for a [Human as-a Service](https://planexe.org/20251012_human_as_a_service_protocol_report.html).
- See more [examples here](https://planexe.org/examples/).

## What is PlanExe?

PlanExe is an open-source tool and the premier planning tool for AI agents. It turns a single plain-english goal statement into a 40-page, strategic plan in ~15 minutes using local or cloud models. It's an accelerator for outlines, but no silver bullet for polished plans.

Typical output contains:

- Executive summary
- Gantt chart
- Governance structure
- Role descriptions
- Stakeholder maps
- Risk registers
- SWOT analyses

PlanExe produces well-structured, domain-aware output: correct terminology, logical task sequencing, and coherent sections. For technical topics (engineering programs, regulated industries), it often gets the vocabulary and structure right. Think of it as a first-draft scaffold that gives you something concrete to critique and refine.

However, the output has consistent weaknesses that matter: budgets are assumed rather than derived, timeline estimates are not grounded in real resource constraints, risk mitigations tend toward generic advice, and legal/regulatory details are plausible-sounding but unverified. The output should be treated as a structured starting point, not a deliverable. How much work it saves depends heavily on the project. For brainstorming or a first outline, it can save hours. For a client-ready plan, expect significant rework on every number, timeline, and risk section.

---

## Model Context Protocol (MCP)

PlanExe exposes an MCP server for AI agents at [https://mcp.planexe.org/](https://mcp.planexe.org/)

Assuming you have an MCP-compatible client ([Claude](https://docs.planexe.org/mcp/claude/), [Cursor](https://docs.planexe.org/mcp/cursor/), [Codex](https://docs.planexe.org/mcp/codex/), [LM Studio](https://docs.planexe.org/mcp/lm_studio/), [Windsurf](https://docs.planexe.org/mcp/windsurf/), OpenClaw, [Antigravity](https://docs.planexe.org/mcp/antigravity/)).

The Tool workflow

1. `example_plans` (optional, preview what PlanExe output looks like)
2. `example_prompts`
3. `model_profiles` (optional, helps choose `model_profile`)
4. non-tool step: draft/approve prompt
5. `plan_create`
6. `plan_status` (poll every 5 minutes until done)
7. optional if failed: `plan_retry`
8. download the result via `plan_file_info`

Concurrency note: each `plan_create` call returns a new `plan_id`; server-side global per-client concurrency is not capped, so clients should track their own parallel plans.

### Option A: Remote MCP (fastest path)

#### Prerequisites

- An account at [https://home.planexe.org](https://home.planexe.org).
- Sufficient funds to create plans.
- A PlanExe API key (`pex_...`) from your account

Use this endpoint directly in your MCP client:

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

### Option B: Run MCP server locally with Docker

#### Prerequisites

- Docker
- OpenRouter account
- Create a PlanExe `.env` file with `OPENROUTER_API_KEY`.

Start the full stack:

```bash
docker compose up --build
```

Make sure that you can create plans in the web interface, before proceeding to MCP.

Then connect your client to:

- `http://localhost:8001/mcp`

For local docker defaults, auth is disabled in `docker-compose.yml`.

### MCP docs

- Setup overview: [https://docs.planexe.org/mcp/mcp_setup/](https://docs.planexe.org/mcp/mcp_setup/)
- Tool details and flow: [https://docs.planexe.org/mcp/mcp_details/](https://docs.planexe.org/mcp/mcp_details/)
- Claude: [https://docs.planexe.org/mcp/claude/](https://docs.planexe.org/mcp/claude/)
- Cursor: [https://docs.planexe.org/mcp/cursor/](https://docs.planexe.org/mcp/cursor/)
- Codex: [https://docs.planexe.org/mcp/codex/](https://docs.planexe.org/mcp/codex/)
- PlanExe MCP interface: [https://docs.planexe.org/mcp/planexe_mcp_interface/](https://docs.planexe.org/mcp/planexe_mcp_interface/)
- MCP Registry publishing metadata (`server.json`): `mcp_cloud/server.json`
- `llms.txt`: [https://mcp.planexe.org/llms.txt](https://mcp.planexe.org/llms.txt)

---

## Running the Pipeline via CLI (no frontend)

If you have a local Python environment set up and want to invoke the pipeline
directly — without the Flask UI — use the `planexe` CLI script at
the repo root.

```bash
# Create a plan from a text prompt
./planexe create_plan \
    --plan-text "Small coffee shop in Copenhagen, Denmark" \
    --output-dir ./planexe-outputs/1984-12-31/MyCoffeeShop_v1

# Or: read the plan prompt from a file
./planexe create_plan \
    --plan-file my_plan.txt \
    --output-dir ./planexe-outputs/1984-12-31/MyCoffeeShop_v1
```

The script creates the run directory if it does not exist and writes:

| File | Contents |
|---|---|
| `start_time.json` | `{"server_iso_utc": "<current UTC ISO timestamp>"}` |
| `plan.txt` | The plain-text plan prompt |

---

## For AI Agents

PlanExe is designed as infrastructure for AI agents. If you are an AI agent reading this:

- **Discovery:** Read [`public/llms.txt`](public/llms.txt) for service metadata, MCP tools, and endpoints.
- **MCP endpoint:** `https://mcp.planexe.org/mcp` — connect with an API key from [home.planexe.org](https://home.planexe.org/).
- **Workflow:** Call `example_prompts` to learn the expected prompt format, draft a detailed prompt (~300-800 words of flowing prose), then call `plan_create`.
- **Agent guide:** See [`docs/mcp/autonomous_agent_guide.md`](docs/mcp/autonomous_agent_guide.md) for the complete autonomous workflow.
- **Key outputs in zip:** `wbs_level1.json` (work packages), `wbs_level2.json` (tasks), `pre_project_assessment.json` (feasibility).

---

<details>
<summary><strong> Run locally with Docker (Click to expand)</strong></summary>

<br>

**Prerequisite:** Docker with Docker Compose installed; you only need basic [Docker](https://en.wikipedia.org/wiki/Docker_(software)) knowledge. No local Python setup is required because everything runs in containers.

### Follow these steps

1. Clone the repo and enter it:

```bash
git clone https://github.com/PlanExeOrg/PlanExe.git
cd PlanExe
```

2. Provide an LLM provider. Copy `.env.docker-example` to `.env` and fill in `OPENROUTER_API_KEY` with your key from [OpenRouter](https://openrouter.ai/). The containers mount `.env` and `llm_config/`; pick a model profile there. For host-side Ollama, use the `docker-ollama-llama3.1` entry and ensure Ollama is listening on `http://host.docker.internal:11434`.

3. Start the stack (first run builds the images):

```bash
docker compose up worker_plan frontend_multi_user
```

   The worker listens on http://localhost:8000 and the UI comes up on http://localhost:5001 after the Postgres and worker healthchecks pass.

4. Open http://localhost:5001 in your browser, create an account (or log in with the admin credentials from `.env`), enter your idea, and watch progress with:

```bash
docker compose logs -f worker_plan
```

   Outputs are written to `run/` on the host (mounted into both containers).

5. Stop with `Ctrl+C` (or `docker compose down`). Rebuild after code/dependency changes:

```bash
docker compose build --no-cache worker_plan frontend_multi_user
```

For compose tips, alternate ports, or troubleshooting, see `docs/docker.md` or `docker-compose.md`.

### Configuration

**Config A:** Run a model in the cloud using a paid provider. Follow the instructions in [OpenRouter](https://docs.planexe.org/ai_providers/openrouter/).

**Config B:** Run models locally on a high-end computer. Follow the instructions for either [Ollama](https://docs.planexe.org/ai_providers/ollama/) or [LM Studio](https://docs.planexe.org/ai_providers/lm_studio/). When using host-side tools with Docker, point the model URL at the host (for example `http://host.docker.internal:11434` for Ollama).

Recommendation: I recommend **Config A** as it offers the most straightforward path to getting PlanExe working reliably.

</details>

---

<details>
<summary><strong> Help (Click to expand)</strong></summary>

<br>

For help or feedback.

Join the [PlanExe Discord](https://planexe.org/discord).

</details>
