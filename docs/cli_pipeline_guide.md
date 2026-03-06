# PlanExe CLI Pipeline Guide

Run PlanExe pipelines from the command line without the web frontend. This guide covers environment setup, configuration, pipeline execution, and troubleshooting.

---

## 1. Prerequisites

### Python & Virtual Environment

PlanExe requires **Python 3.13+** with a virtual environment:

```bash
# Install Python 3.13+ (on macOS with Homebrew)
brew install python@3.13

# Create a virtual environment in the worker_plan directory
cd /path/to/planexe/worker_plan
python3.13 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Install Dependencies

From the `worker_plan` directory with the venv activated:

```bash
# Install PlanExe and all dependencies
pip install -e .
```

This reads `pyproject.toml` and installs required packages including:
- **LLM Framework:** llama-index (core, OpenRouter, OpenAI, LM Studio, Ollama integrations)
- **Task Runner:** luigi (DAG-based pipeline orchestration)
- **Data Processing:** pandas, numpy, pydantic, SQLAlchemy
- **Web/API:** fastapi, uvicorn, aiohttp
- **Utilities:** python-dotenv, typer, rich, requests

### Verify Installation

```bash
# Test that worker_plan_internal is importable
python -c "from worker_plan_internal.plan.run_plan_pipeline import *; print('✓ PlanExe installed')"

# Check Python version
python --version
```

### API Keys

Set up your LLM provider API key in a `.env` file at the **PlanExe root** (where `llm_config/` lives):

```bash
# Create/edit .env
cat > .env << 'EOF'
OPENROUTER_API_KEY=your_openrouter_api_key_here
# or
OPENAI_API_KEY=your_openai_api_key_here
EOF
```

**Supported providers:**
- OpenRouter (recommended) — set `OPENROUTER_API_KEY`
- OpenAI — set `OPENAI_API_KEY`
- LM Studio (local) — no key needed, but service must be running
- Ollama (local) — no key needed, but service must be running

---

## 2. Environment Variables

Set these **before** running the pipeline. All paths should be **absolute**.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `RUN_ID_DIR` | **Yes** | None | Absolute path to run output directory (where seed files live). Must exist or be creatable. |
| `PLANEXE_CONFIG_PATH` | **Yes** | None | Absolute path to PlanExe root (where `llm_config/` directory lives). Usually `/path/to/planexe`. |
| `PYTHONPATH` | **Yes** | None | Must include `/path/to/planexe/worker_plan` (the Python package). |
| `PLANEXE_MODEL_PROFILE` | No | `baseline` | Model profile to use: `baseline`, `premium`, `frontier`, or `custom`. |
| `PLANEXE_LLM_CONFIG_CUSTOM_FILENAME` | No | `custom.json` | When profile is `custom`, loads `llm_config/<filename>` instead of `llm_config/baseline.json`. |
| `SPEED_VS_DETAIL` | No | `ALL_DETAILS_BUT_SLOW` | Trade-off: `FAST_BUT_SKIP_DETAILS` or `ALL_DETAILS_BUT_SLOW`. |
| `LLM_MODEL` | No | None | Explicit model override (e.g., `openrouter-google/gemini-3.1-flash-lite-preview`). Overrides profile. |

### Example: Set All Variables

```bash
#!/bin/bash

# Define paths
export RUN_ID_DIR="/absolute/path/to/my_run_20260305"
export PLANEXE_CONFIG_PATH="/absolute/path/to/planexe"
export PYTHONPATH="${PLANEXE_CONFIG_PATH}/worker_plan:${PYTHONPATH}"

# LLM configuration
export PLANEXE_MODEL_PROFILE="baseline"
export SPEED_VS_DETAIL="ALL_DETAILS_BUT_SLOW"

# API key (from .env or manually)
export OPENROUTER_API_KEY="sk-or-..."

echo "✓ Environment ready"
```

---

## 3. Required Seed Files

Before launching the pipeline, **two files must exist** in `$RUN_ID_DIR`:

### `001-1-start_time.json`

A JSON file capturing the server's local timezone and UTC time when the pipeline starts. This becomes the project start date for Gantt charts.

**File location:** `$RUN_ID_DIR/001-1-start_time.json`

**Format:**
```json
{
  "server_iso_utc": "2026-03-05T23:43:48Z",
  "server_iso_local": "2026-03-05T18:43:48-05:00",
  "server_timezone_name": "America/New_York"
}
```

**Fields:**
- `server_iso_utc` — ISO 8601 UTC time with `Z` suffix (not `+00:00`)
- `server_iso_local` — ISO 8601 local time with timezone offset
- `server_timezone_name` — IANA timezone name (e.g., `America/New_York`, `Europe/London`, `UTC`)

**Generate programmatically:**

```python
from datetime import datetime
from zoneinfo import ZoneInfo
from worker_plan_api.start_time import StartTime
from pathlib import Path

# Create StartTime using local time
local_tz = ZoneInfo("America/New_York")
local_time = datetime.now(tz=local_tz)
start_time = StartTime.create(local_time)

# Save to file
output_dir = Path("/absolute/path/to/my_run_20260305")
output_dir.mkdir(parents=True, exist_ok=True)
start_time.save(output_dir / "001-1-start_time.json")
```

### `001-2-plan.txt`

A plain text file containing the plan prompt or seed. This is the user's initial request/idea that PlanExe will expand into a full plan.

**File location:** `$RUN_ID_DIR/001-2-plan.txt`

**Format:** Plain text with optional structured hints.

**Example:**
```
Plan:
Design an American 1950s-style rock-and-roll diner experience in Tehran, Iran. 
Include concept and theme, interior decor and design, menu with American classics 
adapted for local tastes and halal requirements, music programming and live 
entertainment, staffing plan, cultural considerations and sensitivities, legal and 
regulatory challenges of operating such a business in Iran, and a full business plan 
with financials including startup costs, revenue projections, and risk mitigation 
strategies.

Today's date:
2026-Mar-03

Project start ASAP
```

**Minimal example:**
```
Build a mobile app for tracking personal finance in 3 months.
```

---

## 4. Quick Start Example

Here's a complete bash script to run a pipeline from scratch:

```bash
#!/bin/bash
set -euo pipefail

# ============================================================================
# PlanExe CLI Pipeline Runner
# ============================================================================

PLANEXE_ROOT="/Users/macmini/planexe"  # Change this to your PlanExe path
RUN_DATE=$(date +%Y%m%d_%H%M%S)
RUN_ID_DIR="${PLANEXE_ROOT}/runs/PlanExe_${RUN_DATE}"

# Create run directory
mkdir -p "${RUN_ID_DIR}"
echo "📁 Run directory: ${RUN_ID_DIR}"

# ============================================================================
# Step 1: Generate start_time.json
# ============================================================================
python3 << 'PYTHON_EOF'
from datetime import datetime
from zoneinfo import ZoneInfo
from worker_plan_api.start_time import StartTime
from pathlib import Path
import sys

# Get timezone from environment or use local
tz_name = "America/New_York"  # Change as needed
local_tz = ZoneInfo(tz_name)
local_time = datetime.now(tz=local_tz)
start_time = StartTime.create(local_time)

# Save to file
output_dir = Path(sys.argv[1])
start_time.save(output_dir / "001-1-start_time.json")
print(f"✓ Created {output_dir / '001-1-start_time.json'}")
PYTHON_EOF
"${RUN_ID_DIR}"

# ============================================================================
# Step 2: Create plan.txt (user input)
# ============================================================================
cat > "${RUN_ID_DIR}/001-2-plan.txt" << 'PLAN_EOF'
Plan:
Design a sustainable urban farming community in Portland, Oregon.
Include crop selection, water management, soil preparation, 
community engagement strategy, business model, and a 12-month 
implementation roadmap.

Today's date:
$(date +%Y-%b-%d)

Project start ASAP
PLAN_EOF
echo "✓ Created plan.txt"

# ============================================================================
# Step 3: Set environment variables
# ============================================================================
export RUN_ID_DIR="${RUN_ID_DIR}"
export PLANEXE_CONFIG_PATH="${PLANEXE_ROOT}"
export PYTHONPATH="${PLANEXE_ROOT}/worker_plan:${PYTHONPATH:-}"
export PLANEXE_MODEL_PROFILE="baseline"
export SPEED_VS_DETAIL="ALL_DETAILS_BUT_SLOW"

# Load .env if it exists (for API keys)
if [ -f "${PLANEXE_ROOT}/.env" ]; then
  set -a
  source "${PLANEXE_ROOT}/.env"
  set +a
fi

# Verify required variables
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "❌ Error: OPENROUTER_API_KEY or OPENAI_API_KEY not set"
  exit 1
fi

echo "✓ Environment variables set"

# ============================================================================
# Step 4: Activate venv and run pipeline
# ============================================================================
cd "${PLANEXE_ROOT}/worker_plan"
source .venv/bin/activate

# Run the pipeline
echo "🚀 Starting PlanExe pipeline..."
python -m worker_plan_internal.plan.run_plan_pipeline

echo "✅ Pipeline complete!"
echo "📊 Results: ${RUN_ID_DIR}"
```

**Usage:**

```bash
# Make it executable and run
chmod +x run_pipeline.sh
./run_pipeline.sh
```

---

## 5. Available Model Profiles

Model profiles define which LLM(s) to use for different pipeline steps. Located in `llm_config/*.json`.

### Built-in Profiles

| Profile | File | Purpose | Cost |
|---------|------|---------|------|
| `baseline` | `llm_config/baseline.json` | Default production models. Balanced cost/quality. | ~$5–15 per run |
| `premium` | `llm_config/premium.json` | Higher-tier models. Better outputs, higher cost. | ~$20–50 per run |
| `frontier` | `llm_config/frontier.json` | Experimental/unreleased models. Cutting-edge. | $50+ per run |
| `custom` | `llm_config/custom.json` | User-defined. Set `PLANEXE_LLM_CONFIG_CUSTOM_FILENAME`. | Varies |

### Using a Profile

```bash
export PLANEXE_MODEL_PROFILE="premium"
python -m worker_plan_internal.plan.run_plan_pipeline
```

### Using a Custom Profile

1. Create or edit a JSON config file:
   ```bash
   cat > llm_config/my_config.json << 'EOF'
   {
     "openrouter-google/gemini-3.1-flash-lite-preview": {
       "comment": "Fast, cheap, good",
       "luigi_workers": 4,
       "class": "OpenRouter",
       "arguments": {
         "model": "google/gemini-3.1-flash-lite-preview",
         "api_key": "${OPENROUTER_API_KEY}",
         "temperature": 0.7,
         "timeout": 60.0,
         "max_tokens": 8192
       },
       "pricing_kind": "paid"
     }
   }
   EOF
   ```

2. Set environment variables:
   ```bash
   export PLANEXE_MODEL_PROFILE="custom"
   export PLANEXE_LLM_CONFIG_CUSTOM_FILENAME="my_config.json"
   python -m worker_plan_internal.plan.run_plan_pipeline
   ```

### Override Model Explicitly

For debugging or one-off runs, override the profile completely:

```bash
export LLM_MODEL="openrouter-google/gemini-3.1-flash-lite-preview"
python -m worker_plan_internal.plan.run_plan_pipeline
```

---

## 6. Troubleshooting

### KeyError: 'server_iso_utc'

**Symptom:**
```
KeyError: 'server_iso_utc'
```

**Cause:** The `001-1-start_time.json` file is missing or malformed.

**Solution:**
1. Verify file exists: `ls -la ${RUN_ID_DIR}/001-1-start_time.json`
2. Regenerate it using the Python code in [Section 3](#required-seed-files)
3. Ensure all three fields are present: `server_iso_utc`, `server_iso_local`, `server_timezone_name`

### FileNotFoundError: '001-2-plan.txt'

**Symptom:**
```
FileNotFoundError: [Errno 2] No such file or directory: '001-2-plan.txt'
```

**Cause:** The `001-2-plan.txt` file does not exist in `$RUN_ID_DIR`.

**Solution:**
```bash
echo "Your plan here" > ${RUN_ID_DIR}/001-2-plan.txt
```

### PYTHONPATH Not Set

**Symptom:**
```
ModuleNotFoundError: No module named 'worker_plan_internal'
```

**Solution:**
```bash
export PYTHONPATH="/path/to/planexe/worker_plan:${PYTHONPATH}"
python -m worker_plan_internal.plan.run_plan_pipeline
```

### API Key Not Found

**Symptom:**
```
Error: OPENROUTER_API_KEY not set
```

**Solution:**
```bash
# Option 1: Export directly
export OPENROUTER_API_KEY="sk-or-..."

# Option 2: Create .env at PlanExe root
cat > /path/to/planexe/.env << 'EOF'
OPENROUTER_API_KEY=sk-or-...
EOF

# Option 3: Load .env before running
set -a
source /path/to/planexe/.env
set +a
```

### Out of Memory

**Symptom:**
```
MemoryError: ... cannot allocate ...
```

**Cause:** Large models or text processing exhausting RAM.

**Solution:**
- Use a lighter model profile: `export PLANEXE_MODEL_PROFILE="baseline"`
- Set speed preference: `export SPEED_VS_DETAIL="FAST_BUT_SKIP_DETAILS"`
- Reduce context window in custom LLM config

### Luigi Worker Hang

**Symptom:** Pipeline appears stuck; no new output for 10+ minutes.

**Cause:** LLM API timeout or slow model.

**Solution:**
1. Check API status (OpenRouter/OpenAI dashboard)
2. Try a faster model: `export LLM_MODEL="openrouter-google/gemini-3.1-flash-lite-preview"`
3. Increase timeout in LLM config: `"timeout": 120.0`
4. Cancel and resume:
   ```bash
   # Kill the pipeline (Ctrl+C)
   # Remove the completion marker
   rm ${RUN_ID_DIR}/999-pipeline_complete.txt
   # Restart
   python -m worker_plan_internal.plan.run_plan_pipeline
   ```

### Disk Space Exhausted

**Symptom:**
```
IOError: [Errno 28] No space left on device
```

**Solution:**
- Check: `df -h ${RUN_ID_DIR}`
- Clean old runs: `rm -rf /path/to/planexe/runs/PlanExe_*` (backup first!)
- Reduce detail: `export SPEED_VS_DETAIL="FAST_BUT_SKIP_DETAILS"`

---

## 7. Understanding the Output

### Run Directory Structure

```
/absolute/path/to/my_run_20260305/
├── 001-1-start_time.json         ← Seed (generated)
├── 001-2-plan.txt                ← Seed (user input)
├── 002-1-redline_gate.json        ← Analysis phase
├── 002-2-redline_gate.md          ← Human-readable
├── 003-...                         ← Assumptions phase
├── 004-...                         ← Strategy phase
├── ...                             ← More pipeline outputs
└── 999-pipeline_complete.txt      ← Marker (pipeline done)
```

### Key Output Files

- `*.json` — Raw LLM responses and structured data
- `*.md` — Human-readable markdown summaries
- `*.csv` — Tabular data (Gantt charts, schedules)
- `*.html` — Interactive visualizations (Gantt viewer, reports)

---

## 8. Resume an Interrupted Run

To continue a partially completed run:

```bash
export RUN_ID_DIR="/path/to/existing/run"
export PLANEXE_CONFIG_PATH="/path/to/planexe"
export PYTHONPATH="/path/to/planexe/worker_plan:${PYTHONPATH}"

# If already complete, remove the marker
rm ${RUN_ID_DIR}/999-pipeline_complete.txt

# Resume
python -m worker_plan_internal.plan.run_plan_pipeline
```

---

## 9. Monitor Progress

### View Real-Time Logs

```bash
# In a separate terminal
tail -f ${RUN_ID_DIR}/*.log  # If logs are created
```

### Check Output Files

```bash
# See what's been generated so far
ls -1t ${RUN_ID_DIR}/ | head -20
```

### Estimate Time

- **baseline profile:** 20–40 minutes
- **premium profile:** 30–60 minutes
- **frontier profile:** 45–120 minutes

Depends on LLM provider speed and API load.

---

## 10. Integration with Agents

### For Bubba (OpenClaw Bot)

```bash
#!/bin/bash
source ~/.zshrc  # Load API keys

export RUN_ID_DIR="/tmp/planexe_run_$(date +%s)"
export PLANEXE_CONFIG_PATH="/Users/macmini/planexe"
export PYTHONPATH="${PLANEXE_CONFIG_PATH}/worker_plan"

cd "${PLANEXE_CONFIG_PATH}/worker_plan"
source .venv/bin/activate

python -m worker_plan_internal.plan.run_plan_pipeline

echo "Plan completed in ${RUN_ID_DIR}"
```

### For CI/CD (GitHub Actions)

```yaml
name: PlanExe Pipeline
on: workflow_dispatch
jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.13"
      - name: Install dependencies
        run: |
          cd worker_plan
          python -m pip install -e .
      - name: Run pipeline
        env:
          RUN_ID_DIR: /tmp/run
          PLANEXE_CONFIG_PATH: ${{ github.workspace }}
          PYTHONPATH: ${{ github.workspace }}/worker_plan
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
        run: |
          mkdir -p ${RUN_ID_DIR}
          python -m worker_plan_internal.plan.run_plan_pipeline
```

---

## Additional Resources

- **LLM Config:** See `docs/llm_config.md` for detailed LLM setup
- **PlanExe Architecture:** See `docs/plan.md`
- **Output Anatomy:** See `docs/plan_output_anatomy.md`
- **Model Pricing:** See `docs/costs_and_models.md`
