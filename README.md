# PlanExe

<p align="center">
  <img src="docs/planexe-humanoid-factory.gif?raw=true" alt="PlanExe - Turn your idea into a comprehensive plan in minutes, not months." width="700">
</p>

<p align="center">
  <strong>Turn your idea into a comprehensive plan in minutes, not months.</strong>
</p>

---

## Example plans generated with PlanExe

- A business plan for a [Minecraft-themed escape room](https://planexe.org/20251016_minecraft_escape_report.html).
- A business plan for a [Faraday cage manufacturing company](https://planexe.org/20250720_faraday_enclosure_report.html).
- A pilot project for a [Human as-a Service](https://planexe.org/20251012_human_as_a_service_protocol_report.html).
- See more [examples here](https://planexe.org/examples/).

## What is PlanExe?

PlanExe is an open-source tool, that turns a single plain-english goal statement into a 40-page, strategic plan in ~15 minutes using a local or cloud models. It's an accelerator for outlines, but no silver bullet for polished plans.

Typical output contains:
- Executive summary
- Gantt chart
- Governance structure
- Role descriptions
- Stakeholder maps
- Risk registers
- SWOT analyses

The technical quality of structure, formatting, and coherence is consistently excellent—often superior to human junior/mid-tier consulting drafts. However, budgets remain headline-only, timelines contain errors, metrics are usually vague, and legal/operational realism is weak on high-stakes topics. A usable, client-ready version still requires weeks to months of skilled human refinement.

PlanExe removes 70–90 % of the labor for the planning scaffold on any topic, but the final 10–30 % that separates a polished document from a credible, defensible plan remains human-only work.

---

New to PlanExe? Follow the [Getting Started](docs/getting_started.md) guide.

<details>
<summary><strong> Try it out now (Click to expand)</strong></summary>
<br>

You can generate 1 plan for free.

[Try it here →](https://app.mach-ai.com/planexe_early_access)

</details>

---

<details>
<summary><strong> Run locally with Docker (Click to expand)</strong></summary>

<br>

**Prerequisite:** Docker with Docker Compose installed; you only need basic [Docker](https://en.wikipedia.org/wiki/Docker_(software)) knowledge. No local Python setup is required because everything runs in containers.

### Quickstart: single-user UI + worker (frontend_single_user + worker_plan)

1. Clone the repo and enter it:

```bash
git clone https://github.com/PlanExeOrg/PlanExe.git
cd PlanExe
```

2. Provide an LLM provider. Copy `.env.docker-example` to `.env` and fill in `OPENROUTER_API_KEY` with your key from [OpenRouter](https://openrouter.ai/). The containers mount `.env` and `llm_config.json`; pick a model profile there. For host-side Ollama, use the `docker-ollama-llama3.1` entry and ensure Ollama is listening on `http://host.docker.internal:11434`.

3. Start the stack (first run builds the images):

```bash
docker compose up worker_plan frontend_single_user
```

   The worker listens on http://localhost:8000 and the UI comes up on http://localhost:7860 after the worker healthcheck passes.

4. Open http://localhost:7860 in your browser. Optional: set `PLANEXE_PASSWORD` in `.env` to require a password. Enter your idea, click the generate button, and watch progress with:

```bash
docker compose logs -f worker_plan
```

   Outputs are written to `run/` on the host (mounted into both containers).

5. Stop with `Ctrl+C` (or `docker compose down`). Rebuild after code/dependency changes:

```bash
docker compose build --no-cache worker_plan frontend_single_user
```

For compose tips, alternate ports, or troubleshooting, see `docs/docker.md` or `docker-compose.md`.

### Configuration

**Config A:** Run a model in the cloud using a paid provider. Follow the instructions in [OpenRouter](docs/openrouter.md).

**Config B:** Run models locally on a high-end computer. Follow the instructions for either [Ollama](docs/ollama.md) or [LM Studio](docs/lm_studio.md). When using host-side tools with Docker, point the model URL at the host (for example `http://host.docker.internal:11434` for Ollama).

Recommendation: I recommend **Config A** as it offers the most straightforward path to getting PlanExe working reliably.

</details>

---

<details>
<summary><strong> Screenshots (Click to expand)</strong></summary>

<br>

You input a vague description of what you want and PlanExe outputs a plan.

[YouTube video: Using PlanExe to plan a lunar base](https://www.youtube.com/watch?v=7AM2F1C4CGI)

![Screenshot of PlanExe](/docs/planexe-humanoid-factory.jpg?raw=true "Screenshot of PlanExe")

</details>

---

<details>
<summary><strong> Help (Click to expand)</strong></summary>

<br>

For help or feedback.

Join the [PlanExe Discord](https://planexe.org/discord).

</details>
