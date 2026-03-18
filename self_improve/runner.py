"""
Run the IdentifyPotentialLevers pipeline step with a candidate system prompt
against baseline training data and capture the output.

Usage:
    python -m self_improve.runner \
        --system-prompt-file candidate.txt \
        --baseline-dir /path/to/baseline/train \
        --prompt-lab-dir /path/to/PlanExe-prompt-lab \
        --model ollama-llama3.1
"""
import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

# Add worker_plan/ to sys.path so worker_plan_internal imports work.
_worker_plan_dir = str(Path(__file__).resolve().parent.parent / "worker_plan")
if _worker_plan_dir not in sys.path:
    sys.path.insert(0, _worker_plan_dir)

from llama_index.core.instrumentation import get_dispatcher
from worker_plan_internal.lever.identify_potential_levers import IdentifyPotentialLevers
from worker_plan_internal.llm_util.llm_executor import LLMExecutor, LLMModelFromName
from worker_plan_internal.llm_util.track_activity import TrackActivity
from worker_plan_internal.llm_util.usage_metrics import set_usage_metrics_path, record_usage_metric

logger = logging.getLogger(__name__)



# Lock for thread-safe writes to shared files and global state
_file_lock = threading.Lock()

INPUT_FILES = [
    "001-2-plan.txt",
    "002-6-identify_purpose.md",
    "002-8-plan_type.md",
]

FILE_LABELS = [
    "plan.txt",
    "purpose.md",
    "plan_type.md",
]


def load_user_prompt(plan_dir: Path) -> str:
    """
    Read the 3 input files from a plan directory and concatenate them
    exactly as PotentialLeversTask.run_inner() does.
    """
    parts = []
    for filename, label in zip(INPUT_FILES, FILE_LABELS):
        file_path = plan_dir / filename
        content = file_path.read_text()
        parts.append(f"File '{label}':\n{content}")
    return "\n\n".join(parts)


@dataclass
class PlanResult:
    name: str
    status: str
    duration_seconds: float
    error: str | None = None
    calls_succeeded: int | None = None


def run_single_plan(
    plan_dir: Path,
    output_dir: Path,
    model_names: list[str],
) -> PlanResult:
    """
    Run IdentifyPotentialLevers for one plan directory. Writes the raw and
    clean JSON outputs into output_dir/<plan_name>/.

    Creates its own LLMExecutor so it's safe to call from multiple threads.
    """
    plan_name = plan_dir.name
    plan_output_dir = output_dir / plan_name
    plan_output_dir.mkdir(parents=True, exist_ok=True)

    llm_models = LLMModelFromName.from_names(model_names)
    llm_executor = LLMExecutor(llm_models=llm_models)

    # Set up per-plan usage tracking.
    # set_usage_metrics_path uses thread-local storage, but we still hold
    # _file_lock while configuring it alongside the dispatcher to keep the
    # setup/teardown atomic.
    track_activity_path = plan_output_dir / "track_activity.jsonl"
    track_activity = TrackActivity(
        jsonl_file_path=track_activity_path,
        write_to_logger=False,
    )
    dispatcher = get_dispatcher()

    with _file_lock:
        set_usage_metrics_path(plan_output_dir / "usage_metrics.jsonl")
        dispatcher.add_event_handler(track_activity)

    t0 = time.monotonic()
    try:
        user_prompt = load_user_prompt(plan_dir)
        result = IdentifyPotentialLevers.execute(llm_executor, user_prompt)

        raw_path = plan_output_dir / "002-9-potential_levers_raw.json"
        clean_path = plan_output_dir / "002-10-potential_levers.json"
        result.save_raw(str(raw_path))
        result.save_clean(str(clean_path))

        duration = time.monotonic() - t0

        expected_calls = 3
        actual_calls = len(result.responses)
        if actual_calls < expected_calls:
            logger.warning(
                f"{plan_name}: partial recovery — {actual_calls}/{expected_calls} calls succeeded"
            )

        logger.info(f"{plan_name}: {len(result.levers)} levers in {duration:.1f}s")
        return PlanResult(
            name=plan_name,
            status="ok",
            duration_seconds=round(duration, 2),
            calls_succeeded=actual_calls,
        )
    except Exception as e:
        duration = time.monotonic() - t0
        logger.error(f"{plan_name}: failed after {duration:.1f}s — {e}")
        return PlanResult(
            name=plan_name,
            status="error",
            duration_seconds=round(duration, 2),
            error=str(e),
        )
    finally:
        with _file_lock:
            set_usage_metrics_path(None)
            dispatcher.event_handlers.remove(track_activity)
        track_activity_path.unlink(missing_ok=True)
        _maybe_generate_activity_overview(plan_output_dir)


def _maybe_generate_activity_overview(plan_output_dir: Path) -> None:
    """Generate activity_overview.json from usage_metrics.jsonl if missing.

    This covers backends (e.g. Anthropic) where LlamaIndex instrumentation
    events don't fire, so TrackActivity never writes activity_overview.json,
    but the Anthropic httpx hook has written token counts to usage_metrics.jsonl.
    """
    overview_path = plan_output_dir / "activity_overview.json"
    if overview_path.exists():
        return

    metrics_path = plan_output_dir / "usage_metrics.jsonl"
    if not metrics_path.exists():
        return

    models: dict[str, dict] = {}
    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if not row.get("success"):
            continue
        # Only generate if we have token counts (otherwise nothing useful to add)
        input_tokens = row.get("input_tokens")
        output_tokens = row.get("output_tokens")
        if input_tokens is None and output_tokens is None:
            continue

        model_name = row.get("model", "unknown")
        stats = models.setdefault(model_name, {
            "total_cost": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        })
        inp = int(input_tokens or 0)
        out = int(output_tokens or 0)
        think = int(row.get("thinking_tokens") or 0)
        cost = float(row.get("cost_usd") or 0.0)
        stats["input_tokens"] += inp
        stats["output_tokens"] += out
        stats["thinking_tokens"] += think
        stats["total_tokens"] += inp + out + think
        stats["total_cost"] += cost
        stats["calls"] += 1

    if not models:
        return

    overview = {
        "last_updated": datetime.now().isoformat(),
        "models": models,
        "total_cost": sum(m["total_cost"] for m in models.values()),
        "total_input_tokens": sum(m["input_tokens"] for m in models.values()),
        "total_output_tokens": sum(m["output_tokens"] for m in models.values()),
        "total_thinking_tokens": sum(m["thinking_tokens"] for m in models.values()),
        "total_tokens": sum(m["total_tokens"] for m in models.values()),
    }
    try:
        overview_path.write_text(json.dumps(overview, indent=2, sort_keys=True))
        logger.info("Generated %s from usage_metrics.jsonl", overview_path)
    except Exception as exc:
        logger.warning("Failed to generate activity_overview.json: %s", exc)


def _run_cmd(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _collect_system_info() -> dict:
    info: dict = {
        "os": platform.system(),
        "os_version": platform.platform(),
        "arch": platform.machine(),
        "cpu_count": os.cpu_count(),
    }

    system = platform.system()
    if system == "Darwin":
        mac_ver = platform.mac_ver()[0]
        if mac_ver:
            info["os_version"] = f"macOS {mac_ver}"
        info["cpu_model"] = _run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
        memsize = _run_cmd(["sysctl", "-n", "hw.memsize"])
        if memsize:
            info["memory_gb"] = round(int(memsize) / (1024 ** 3), 1)
        gpu_info = _run_cmd(["system_profiler", "SPDisplaysDataType", "-json"])
        if gpu_info:
            try:
                displays = json.loads(gpu_info).get("SPDisplaysDataType", [])
                gpus = []
                for d in displays:
                    gpu_entry: dict = {"name": d.get("sppci_model", "unknown")}
                    vram = d.get("sppci_vram")
                    if vram:
                        gpu_entry["vram"] = vram
                    elif d.get("sppci_vram_shared"):
                        gpu_entry["vram_shared"] = d["sppci_vram_shared"]
                    gpus.append(gpu_entry)
                if gpus:
                    info["gpu"] = gpus
            except (json.JSONDecodeError, KeyError):
                pass
    elif system == "Linux":
        info["cpu_model"] = _run_cmd(["bash", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"])
        meminfo = _run_cmd(["bash", "-c", "grep MemTotal /proc/meminfo | awk '{print $2}'"])
        if meminfo:
            info["memory_gb"] = round(int(meminfo) / (1024 ** 2), 1)
        nvidia = _run_cmd(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        if nvidia:
            gpus = []
            for line in nvidia.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2:
                    gpus.append({"name": parts[0], "vram_mb": int(parts[1])})
            if gpus:
                info["gpu"] = gpus

    return info


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_jsonl(path: Path, obj: dict) -> None:
    with _file_lock:
        with open(path, "a") as f:
            f.write(json.dumps(obj) + "\n")


def _emit_event(events_path: Path, event: str, **kwargs) -> None:
    entry = {"timestamp": _timestamp(), "event": event, **kwargs}
    _append_jsonl(events_path, entry)
    logger.info(f"event: {event} {kwargs}")


STEP_NAME = "identify_potential_levers"


def _resolve_workers(model_names: list[str]) -> int:
    """Look up luigi_workers from llm_config/ JSON files for the given models."""
    llm_config_dir = Path(__file__).resolve().parent.parent / "llm_config"
    if not llm_config_dir.is_dir():
        return 1

    # Merge all config files into one dict
    all_configs: dict = {}
    for json_file in llm_config_dir.glob("*.json"):
        try:
            with open(json_file) as f:
                all_configs.update(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue

    workers_candidates: list[int] = []
    for name in model_names:
        config = all_configs.get(name)
        if not isinstance(config, dict):
            continue
        value = config.get("luigi_workers")
        if value is None:
            continue
        try:
            w = int(value)
        except (TypeError, ValueError):
            continue
        if w >= 1:
            workers_candidates.append(w)

    return min(workers_candidates) if workers_candidates else 1


def _next_history_counter(history_dir: Path) -> int:
    """Scan history/ for the highest existing run number and return +1."""
    max_counter = -1
    if not history_dir.exists():
        return 0
    for bucket in history_dir.iterdir():
        if not bucket.is_dir() or not bucket.name.isdigit():
            continue
        bucket_base = int(bucket.name) * 100
        for run_dir in bucket.iterdir():
            if not run_dir.is_dir():
                continue
            try:
                idx = int(run_dir.name.split("_")[0])
                max_counter = max(max_counter, bucket_base + idx)
            except (IndexError, ValueError):
                pass
    return max_counter + 1


def _history_run_dir(prompt_lab_dir: Path, step_name: str) -> Path:
    """Create and return the next history run directory.

    Uses mkdir without exist_ok so that parallel processes that race on the
    same counter will fail and retry with the next number instead of silently
    sharing a directory.
    """
    history_dir = prompt_lab_dir / "history"
    counter = _next_history_counter(history_dir)
    for _ in range(50):
        bucket = str(counter // 100)
        entry = f"{counter % 100:02d}_{step_name}"
        run_dir = history_dir / bucket / entry
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            counter += 1
    raise RuntimeError(f"Could not allocate history run dir after 50 attempts (last: {run_dir})")


class _ThreadFilter(logging.Filter):
    """Only accept log records from a specific thread."""

    def __init__(self, thread_id: int):
        super().__init__()
        self.thread_id = thread_id

    def filter(self, record: logging.LogRecord) -> bool:
        return record.thread == self.thread_id


def _run_plan_task(
    plan_dir: Path,
    output_dir: Path,
    model_names: list[str],
    events_path: Path,
    outputs_path: Path,
) -> PlanResult:
    """Run a single plan and record events/output. Thread-safe."""
    plan_name = plan_dir.name

    # Set up per-plan log file at outputs/<plan_name>/log.txt.
    plan_log_dir = output_dir / plan_name
    plan_log_dir.mkdir(parents=True, exist_ok=True)
    plan_log_path = plan_log_dir / "log.txt"
    file_handler = logging.FileHandler(plan_log_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    thread_filter = _ThreadFilter(threading.current_thread().ident)
    file_handler.addFilter(thread_filter)
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    try:
        _emit_event(events_path, "run_single_plan_start", plan_name=plan_name)

        pr = run_single_plan(plan_dir, output_dir, model_names)

        if pr.status == "ok":
            _emit_event(events_path, "run_single_plan_complete",
                        plan_name=plan_name,
                        duration_seconds=pr.duration_seconds)
        else:
            _emit_event(events_path, "run_single_plan_error",
                        plan_name=plan_name, error=pr.error,
                        duration_seconds=pr.duration_seconds)

        if pr.calls_succeeded is not None and pr.calls_succeeded < 3:
            _emit_event(events_path, "partial_recovery",
                        plan_name=plan_name,
                        calls_succeeded=pr.calls_succeeded,
                        expected_calls=3)

        _append_jsonl(outputs_path, asdict(pr))
        return pr
    finally:
        root_logger.removeHandler(file_handler)
        file_handler.close()


def run(
    plan_dirs: list[Path],
    output_dir: Path,
    model_names: list[str],
) -> None:
    """
    Iterate over plan directories, run the lever step for each.

    Uses luigi_workers from the LLM config to parallelize when > 1.

    Writes immediately:
      - meta.json       (one level above output_dir) — run metadata, written at start
      - outputs.jsonl   (one level above output_dir) — one row per completed plan
      - events.jsonl    (one level above output_dir) — significant events as they happen
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = output_dir.parent

    # Load already-completed plans from outputs.jsonl (for resume)
    completed: set[str] = set()
    outputs_path = run_dir / "outputs.jsonl"
    if outputs_path.exists():
        for line in outputs_path.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                if entry.get("status") == "ok":
                    completed.add(entry["name"])
        if completed:
            logger.info(f"Resuming: {len(completed)} plan(s) already completed, skipping them")

    workers = _resolve_workers(model_names)

    # Write meta.json (overwrite on resume is fine — same content)
    model_info: dict = {"primary": model_names[0]}
    if len(model_names) > 1:
        model_info["fallbacks"] = model_names[1:]
    meta = {
        "step": "identify_potential_levers",
        "model": model_info,
        "workers": workers,
        "system": _collect_system_info(),
    }
    meta_path = run_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info(f"Wrote {meta_path}")
    logger.info(f"Workers: {workers}")

    events_path = run_dir / "events.jsonl"

    # Filter to plans that still need processing
    pending_dirs = [d for d in plan_dirs if d.name not in completed]
    for d in plan_dirs:
        if d.name in completed:
            logger.info(f"Skipping {d.name} (already completed)")

    if workers <= 1:
        for plan_dir in pending_dirs:
            _run_plan_task(plan_dir, output_dir,
                           model_names, events_path, outputs_path)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _run_plan_task, plan_dir, output_dir,
                    model_names, events_path, outputs_path,
                ): plan_dir
                for plan_dir in pending_dirs
            }
            for future in as_completed(futures):
                future.result()  # propagate exceptions


def main():
    parser = argparse.ArgumentParser(
        description="Run IdentifyPotentialLevers against baseline training data."
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        help="Directory containing plan subdirectories (process all).",
    )
    parser.add_argument(
        "--plan-dir",
        type=Path,
        help="Single plan directory to process (overrides --baseline-dir).",
    )
    parser.add_argument(
        "--prompt-lab-dir",
        type=Path,
        help="Path to PlanExe-prompt-lab repo. Auto-creates history/{counter}/{nn}_{step}/outputs/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Manual output directory (alternative to --prompt-lab-dir).",
    )
    parser.add_argument(
        "--model",
        required=True,
        action="append",
        dest="models",
        help="LLM model name. First is primary; additional are fallbacks.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.plan_dir:
        plan_dirs = [args.plan_dir]
    elif args.baseline_dir:
        plan_dirs = sorted(
            p for p in args.baseline_dir.iterdir() if p.is_dir()
        )
    else:
        parser.error("Either --plan-dir or --baseline-dir is required.")

    if not plan_dirs:
        parser.error("No plan directories found.")

    if args.prompt_lab_dir:
        run_dir = _history_run_dir(args.prompt_lab_dir, STEP_NAME)
        output_dir = run_dir / "outputs"
        print(f"History run: {run_dir}")
    elif args.output_dir:
        output_dir = args.output_dir
    else:
        parser.error("Either --prompt-lab-dir or --output-dir is required.")

    run(plan_dirs, output_dir, args.models)

    # Summarize from outputs.jsonl
    outputs_path = output_dir.parent / "outputs.jsonl"
    if outputs_path.exists():
        plans = [json.loads(line) for line in outputs_path.read_text().splitlines() if line.strip()]
        ok = sum(1 for p in plans if p["status"] == "ok")
        total = len(plans)
        total_duration = sum(p["duration_seconds"] for p in plans)
        print(f"\nDone: {ok}/{total} plans succeeded in {total_duration:.1f}s")


if __name__ == "__main__":
    main()
