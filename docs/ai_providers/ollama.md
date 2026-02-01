---
title: Ollama - AI provider
---

# Using PlanExe with Ollama

This is for **advanced users** that already have PlanExe working. If you're new to PlanExe, start with [OpenRouter](openrouter.md) instead.

[Ollama](https://ollama.com/) is an open source app for macOS/Windows/Linux for running LLMs on your own computer (or on a remote computer).

PlanExe processes more text than regular chat. You will need expensive hardware to run a LLM at a reasonable speed.

## Quickstart (Docker)
1. Install Ollama on your host and pull a small model: `ollama run llama3.1` (downloads ~4.9 GB and proves the host service works).  
2. Copy `.env.docker-example` to `.env` (even if you leave keys empty for Ollama) and pick the Docker entry in `llm_config.json` (snippet below) so `base_url` points to `http://host.docker.internal:11434` (Docker Desktop) or your Linux bridge IP.  
3. Start PlanExe: `docker compose up worker_plan frontend_single_user`. Open http://localhost:7860, submit a prompt, and watch `docker compose logs -f worker_plan` for progress.

### Host-only (no Docker) — for advanced users
- Use the host entry (e.g., `"ollama-llama3.1"`) in `llm_config.json` so `base_url` stays on `http://localhost:11434`.
- Start your preferred PlanExe runner (e.g., a local Python environment) and ensure Ollama is already running on the host before you submit jobs.

## Configuration

In the `llm_config.json` find a config that starts with `ollama-` such as `"ollama-llama3.1"` (host) or `"docker-ollama-llama3.1"` (Docker). Use the `docker-` entry when PlanExe runs in Docker so requests reach the host.

On the [Ollama Search Models](https://ollama.com/search) website. Find the corresponding model. Go to the info page for the model:
[ollama/library/llama3.1](https://ollama.com/library/llama3.1). The info page shows how to install the model on your computer, in this case `ollama run llama3.1`. To get started, go for a `8b` model that is `4.9GB`.

### Minimum viable setup
- Start with an 8B model (≈5 GB download). Expect workable speeds on a 16 GB RAM laptop or a GPU with ≥8 GB VRAM; larger models slow sharply without more hardware.
- If you need faster responses, move to a bigger GPU box or use a cloud model via OpenRouter instead of upsizing Ollama locally.

### Run Ollama locally with Docker

- Make sure the container can reach Ollama on the host. On macOS/Windows (Docker Desktop) use the preconfigured entry in `llm_config.json` (snippet below) with `base_url` pointing to `http://host.docker.internal:11434`. On Linux, use your Docker bridge IP (often `http://172.17.0.1:11434`) and, if needed, add `extra_hosts: ["host.docker.internal:host-gateway"]` under `worker_plan` in `docker-compose.yml`.
- Find your bridge IP on Linux:

```bash
ip addr show docker0 | awk '/inet /{print $2}'
```

- If `docker0` is missing (alternate bridge names, Podman, etc.), inspect the default bridge gateway instead:

```bash
docker network inspect bridge | awk -F'"' '/Gateway/{print $4}'
```

- Example `llm_config.json` entry:

```json
"docker-ollama-llama3.1": {
    "comment": "This runs on your own computer. It's free. Requires Ollama to be installed. PlanExe runs in a Docker container, and ollama is installed on the host the computer.",
    "class": "Ollama",
    "arguments": {
        "model": "llama3.1:latest",
        "base_url": "http://host.docker.internal:11434",
        "temperature": 0.5,
        "request_timeout": 120.0,
        "is_function_calling_model": false
    }
}
```

- Restart or rebuild the worker/frontends after updating `llm_config.json`: `docker compose up worker_plan frontend_single_user` (add `--build` or run `docker compose build worker_plan frontend_single_user` if the image needs the new config baked in).

## Troubleshooting

Use the command line to compare Ollama's list of installed models with the configurations in your `llm_config.json` file. Run:

```bash
PROMPT> ollama list
NAME                                             ID              SIZE      MODIFIED       
hf.co/unsloth/Llama-3.1-Tulu-3-8B-GGUF:Q4_K_M    08fe35cc5878    4.9 GB    19 minutes ago    
phi4:latest                                      ac896e5b8b34    9.1 GB    6 weeks ago       
qwen2.5-coder:latest                             2b0496514337    4.7 GB    2 months ago      
llama3.1:latest                                  42182419e950    4.7 GB    5 months ago      
```

Inside PlanExe, when clicking `Submit`, a new `Output Dir` should be created containing a `log.txt`. Open that file and scroll to the bottom, see if there are any error messages about what is wrong.

Report your issue on [Discord](https://planexe.org/discord). Please include info about your system, such as: "I'm on macOS with M1 Max with 64 GB.".

Where to look for logs:
- Host filesystem: `run/<timestamped-output-dir>/log.txt` (mounted from the container).
- Container logs: `docker compose logs -f worker_plan` (watch for connection errors to Ollama).
- Structured-output failures: if you see JSON/parse errors or malformed outputs in `log.txt`, try a different Ollama model or quantization; not all models return structured output cleanly.

## How to add a new Ollama model to `llm_config.json`

You can find models and installation instructions here:
- [Ollama](https://ollama.com/search) – Overview of popular models, curated by the Ollama team.
- [Hugging Face](https://huggingface.co/docs/hub/ollama) – A vast collection of GGUF models.

For a model to work with PlanExe, it must meet the following criteria:

- Minimum 8192 output tokens.
- Support structured output. Not every model does this reliably; you may need to try a few nearby models (or quantizations) before finding one that cleanly returns the structured responses PlanExe expects.
- Reliable. Avoid fragile setups where it works one day, but not the next day. If it's a beta version, be aware that it may stop working.
- Low latency.

Steps to add a model:

1. Follow the instructions on Ollama or Hugging Face to install the model.
1. Copy the model id from the `ollama list` command, such as `llama3.1:latest`
2. Paste the model id into the `llm_config.json`.
3. Restart PlanExe to apply the changes.

## Run Ollama on a remote computer

In `llm_config.json`, insert `base_url` with the url to run on. Prefer a secure tunnel (example below) or a firewall-restricted host—avoid exposing Ollama publicly.

SSH tunnel example from your local machine:

```bash
ssh -N -L 11434:localhost:11434 user@remote-host
```

Then set `base_url` to `http://localhost:11434` while the tunnel is running.

```json
"ollama-llama3.1": {
    "comment": "This runs on on a remote computer. Requires Ollama to be installed.",
    "class": "Ollama",
    "arguments": {
        "model": "llama3.1:latest",
        "base_url": "http://example.com:11434",
        "temperature": 0.5,
        "request_timeout": 120.0,
        "is_function_calling_model": false
    }
}
```

---

## Next steps

- Learn prompt quality: [Prompt writing guide](../prompt_writing_guide.md)
- Understand output sections: [Plan output anatomy](../plan_output_anatomy.md)
