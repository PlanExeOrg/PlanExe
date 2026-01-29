# Using PlanExe with LM Studio

[LM Studio](https://lmstudio.ai/) is an open source app for macOS/Windows/Linux for running LLMs on your own computer. It's great for troubleshooting.

PlanExe processes more text than regular chat. You will need expensive hardware to run a LLM at a reasonable speed.

## Quickstart (Docker)
1) Install LM Studio on your host and download a small model inside LM Studio (e.g., `Qwen2.5-7B-Instruct-1M`, ~4.5 GB).  
2) Copy `.env.docker-example` to `.env` (even if you leave keys empty for LM Studio) and use the `lmstudio-...` entry in `llm_config.json`, setting `base_url` to `http://host.docker.internal:1234` (Docker Desktop) or your Linux bridge IP.  
3) Start PlanExe: `docker compose up worker_plan frontend_single_user`. Open http://localhost:7860, submit a prompt, and watch `docker compose logs -f worker_plan` for progress.

### Host-only (no Docker) — for advanced users
- Use the host entry (e.g., `"lmstudio-qwen2.5-7b-instruct-1m"`) in `llm_config.json` so `base_url` stays on `http://127.0.0.1:1234`.
- Start your preferred PlanExe runner (e.g., a local Python environment) and make sure the LM Studio server is running before you submit jobs.

## Configuration

In the `llm_config.json` find a config that starts with `lmstudio-` such as `"lmstudio-qwen2.5-7b-instruct-1m"`. Inside LM Studio, find the model with that exact id and download it. Here is the Qwen model on
[huggingface](https://huggingface.co/lmstudio-community/Qwen2.5-7B-Instruct-1M-GGUF) (~4.5 GB).

Inside LM Studio, go to the `Developer` page (CMD+2 or Windows+2 or Ctrl+2). Start the server.
The UI should show `Status: Running [x]` and `Reachable at: http://127.0.0.1:1234`.

### Minimum viable setup
- Start with a ~7B model (≈5 GB download). Expect workable speeds on a 16 GB RAM laptop or a GPU with ≥8 GB VRAM; larger models slow sharply without more hardware.
- Structured output matters: not all models return clean structured output. If you see malformed/JSON errors, try a nearby model or quantization.

### Run LM Studio locally with Docker

- Containers cannot reach `127.0.0.1` on your host. Set `base_url` in `llm_config.json` to `http://host.docker.internal:1234` (Docker Desktop) or your Docker bridge IP on Linux (often `http://172.17.0.1:1234`). Add `extra_hosts: ["host.docker.internal:host-gateway"]` under `worker_plan` in `docker-compose.yml` if that hostname is missing on Linux.
- Find your bridge IP on Linux:

```bash
ip addr show docker0 | awk '/inet /{print $2}'
```

- If `docker0` is missing (alternate bridge names, Podman, etc.), inspect the default bridge gateway instead:

```bash
docker network inspect bridge | awk -F'"' '/Gateway/{print $4}'
```

- Example `llm_config.json` entry (add `base_url` when using Docker):

```json
"lmstudio-qwen2.5-7b-instruct-1m": {
    "comment": "Runs via LM Studio on the host; PlanExe in Docker points to the host LM Studio server.",
    "class": "LMStudio",
    "arguments": {
        "model_name": "qwen2.5-7b-instruct-1m",
        "base_url": "http://host.docker.internal:1234/v1",
        "temperature": 0.2,
        "request_timeout": 120.0,
        "is_function_calling_model": false
    }
}
```

- After editing `llm_config.json`, rebuild or restart the worker/frontends: `docker compose up worker_plan frontend_single_user` (add `--build` or run `docker compose build worker_plan frontend_single_user` if the image needs the new config baked in).

## Troubleshooting

Inside PlanExe, when clicking `Submit`, a new `Output Dir` should be created containing a `log.txt`. Open that file and scroll to the bottom, see if there are any error messages about what is wrong.

Report your issue on [Discord](https://planexe.org/discord). Please include info about your system, such as: "I'm on macOS with M1 Max with 64 GB.".

Where to look for logs:
- Host filesystem: `run/<timestamped-output-dir>/log.txt` (mounted from the container).
- Container logs: `docker compose logs -f worker_plan` (watch for connection errors to LM Studio).
- Structured-output failures: if you see JSON/parse errors or malformed outputs in `log.txt`, try a different model or quantization; not all models return structured output cleanly.

## Run LM Studio on a remote computer

Use a secure tunnel instead of exposing the server directly. Example from your local machine:

```bash
ssh -N -L 1234:localhost:1234 user@remote-host
```

Then set `base_url` to `http://localhost:1234` while the tunnel is running.
