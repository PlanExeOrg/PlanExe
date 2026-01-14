# Getting started with PlanExe

This guide shows new users how to launch the `frontend_single_user` UI with Docker using OpenRouter as the LLM provider. No local Python or pip setup is needed.

## 1. Prerequisites

Install [Docker](https://www.docker.com/).

Create an account on [OpenRouter](https://openrouter.ai/) and top up around 5 USD in credits (paid models works, the free models are unreliable).
It cost around 0.1 USD to generate a plan, when using PlanExe's default settings.

## 2. Clone the repo
```bash
git clone https://github.com/neoneye/PlanExe.git
cd PlanExe
```

## 3. Configure secrets
Copy `.env.docker-example` to `.env`.

Add your OpenRouter key:
```bash
OPENROUTER_API_KEY='sk-or-v1-your-key'
```

## 4. Start the single-user stack
```bash
docker compose up worker_plan frontend_single_user
```

Wait for http://localhost:7860 to become available.

Stop with `Ctrl+C`.

## 5. Use the UI
Open http://localhost:7860 in your browser. 

You can now submit your prompt.

The generated plans are written to `run/<timestamped-output-dir>`.

![Screenshot of PlanExe](planexe-humanoid-factory.jpg?raw=true "Screenshot of PlanExe")

## Troubleshooting and next steps
- For Docker tips, see [docker.md](docker.md).
- For OpenRouter-specific notes, see [openrouter.md](openrouter.md).
- If the UI fails to load or plans don’t start, check worker logs: `docker compose logs -f worker_plan`.

## Community
Need help? Join the [PlanExe Discord](https://neoneye.github.io/PlanExe-web/discord).
