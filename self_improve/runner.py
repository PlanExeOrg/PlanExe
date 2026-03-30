"""
Run a pipeline step against baseline training data and capture the output.

Supports multiple steps via --step flag (default: identify_potential_levers).

Usage:
    python -m self_improve.runner \
        --baseline-dir /path/to/baseline/train \
        --prompt-lab-dir /path/to/PlanExe-prompt-lab \
        --model ollama-llama3.1

    python -m self_improve.runner \
        --step identify_documents \
        --baseline-dir /path/to/baseline/train \
        --prompt-lab-dir /path/to/PlanExe-prompt-lab \
        --model anthropic-claude-haiku-4-5-pinned
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
from worker_plan_api.filenames import FilenameEnum
from worker_plan_internal.lever.identify_potential_levers import IdentifyPotentialLevers
from worker_plan_internal.lever.deduplicate_levers import DeduplicateLevers
from worker_plan_internal.lever.enrich_potential_levers import EnrichPotentialLevers
from worker_plan_internal.document.identify_documents import IdentifyDocuments
from worker_plan_internal.llm_util.llm_executor import LLMExecutor, LLMModelFromName
from worker_plan_internal.llm_util.track_activity import TrackActivity
from worker_plan_internal.llm_util.usage_metrics import set_usage_metrics_path, record_usage_metric

logger = logging.getLogger(__name__)


# Lock for thread-safe writes to shared files and global state
_file_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Step configurations
# ---------------------------------------------------------------------------

# identify_potential_levers — original step
_LEVERS_INPUT_FILES = [
    (FilenameEnum.INITIAL_PLAN, "plan.txt"),
    (FilenameEnum.IDENTIFY_PURPOSE_MARKDOWN, "purpose.md"),
    (FilenameEnum.PLAN_TYPE_MARKDOWN, "plan_type.md"),
]

# identify_documents — needs many upstream files
_DOCUMENTS_INPUT_FILES = [
    (FilenameEnum.STRATEGIC_DECISIONS_MARKDOWN, "strategic_decisions.md"),
    (FilenameEnum.SCENARIOS_MARKDOWN, "scenarios.md"),
    (FilenameEnum.CONSOLIDATE_ASSUMPTIONS_SHORT_MARKDOWN, "assumptions.md"),
    (FilenameEnum.PROJECT_PLAN_MARKDOWN, "project-plan.md"),
    (FilenameEnum.RELATED_RESOURCES_MARKDOWN, "related-resources.md"),
    (FilenameEnum.SWOT_MARKDOWN, "swot-analysis.md"),
    (FilenameEnum.TEAM_MARKDOWN, "team.md"),
    (FilenameEnum.EXPERT_CRITICISM_MARKDOWN, "expert-review.md"),
]

# deduplicate_levers — same context files as levers, plus the clean levers JSON
_DEDUPLICATE_INPUT_FILES = _LEVERS_INPUT_FILES
_DEDUPLICATE_LEVERS_FILE = FilenameEnum.POTENTIAL_LEVERS_CLEAN

# enrich_potential_levers — same context files as levers, plus the deduplicated levers raw JSON
_ENRICH_INPUT_FILES = _LEVERS_INPUT_FILES
_ENRICH_DEDUPLICATED_LEVERS_FILE = FilenameEnum.DEDUPLICATED_LEVERS_RAW

# Separate file for identify_purpose_dict (loaded as JSON, not concatenated)
_DOCUMENTS_PURPOSE_FILE = FilenameEnum.IDENTIFY_PURPOSE_RAW

SUPPORTED_STEPS = ["identify_potential_levers", "deduplicate_levers", "enrich_potential_levers", "identify_documents"]

# Default wall-clock timeout per plan (seconds).  Prevents a single stuck LLM
# call from blocking the entire run.  The Anthropic SDK may retry internally
# (max_retries × timeout), and LLMExecutor retries on top of that.  600s is
# generous enough for normal operation but catches true hangs.
DEFAULT_PLAN_TIMEOUT = 600


def _load_user_prompt(plan_dir: Path, input_files: list[tuple[str, str]]) -> str:
    """Read input files from a plan directory and concatenate them."""
    parts = []
    for filename, label in input_files:
        file_path = plan_dir / filename.value
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


def _run_levers(plan_dir: Path, plan_output_dir: Path, llm_executor: LLMExecutor) -> PlanResult:
    """Execute the identify_potential_levers step."""
    plan_name = plan_dir.name
    user_prompt = _load_user_prompt(plan_dir, _LEVERS_INPUT_FILES)
    result = IdentifyPotentialLevers.execute(llm_executor, user_prompt)

    raw_path = plan_output_dir / FilenameEnum.POTENTIAL_LEVERS_RAW.value
    clean_path = plan_output_dir / FilenameEnum.POTENTIAL_LEVERS_CLEAN.value
    result.save_raw(str(raw_path))
    result.save_clean(str(clean_path))

    actual_calls = len(result.responses)
    # The adaptive loop uses min_levers=15, max_calls=5 and stops early
    # when enough levers are accumulated. A 2-call success is normal for
    # models that produce 8+ levers per call. Only warn if we got fewer
    # responses than expected for 15 levers (~3 calls at 5-7 levers each).
    if actual_calls < 3:
        logger.warning(
            f"{plan_name}: partial recovery — {actual_calls} calls succeeded"
        )
    return PlanResult(
        name=plan_name,
        status="ok",
        duration_seconds=0,  # filled by caller
        calls_succeeded=actual_calls,
    )


def _run_deduplicate(plan_dir: Path, plan_output_dir: Path, llm_executor: LLMExecutor) -> PlanResult:
    """Execute the deduplicate_levers step."""
    plan_name = plan_dir.name
    project_context = _load_user_prompt(plan_dir, _DEDUPLICATE_INPUT_FILES)

    levers_path = plan_dir / _DEDUPLICATE_LEVERS_FILE.value
    with open(levers_path) as f:
        raw_levers_list = json.load(f)

    result = DeduplicateLevers.execute(llm_executor, project_context=project_context, raw_levers_list=raw_levers_list)

    raw_path = plan_output_dir / FilenameEnum.DEDUPLICATED_LEVERS_RAW.value
    result.save_raw(str(raw_path))

    return PlanResult(
        name=plan_name,
        status="ok",
        duration_seconds=0,  # filled by caller
        calls_succeeded=1,  # single batch call
    )


def _run_enrich(plan_dir: Path, plan_output_dir: Path, llm_executor: LLMExecutor) -> PlanResult:
    """Execute the enrich_potential_levers step."""
    plan_name = plan_dir.name
    project_context = _load_user_prompt(plan_dir, _ENRICH_INPUT_FILES)

    dedup_path = plan_dir / _ENRICH_DEDUPLICATED_LEVERS_FILE.value
    with open(dedup_path) as f:
        json_dict = json.load(f)
        lever_item_list = json_dict["deduplicated_levers"]

    result = EnrichPotentialLevers.execute(llm_executor, project_context=project_context, raw_levers_list=lever_item_list)

    raw_path = plan_output_dir / FilenameEnum.ENRICHED_LEVERS_RAW.value
    result.save_raw(str(raw_path))

    return PlanResult(
        name=plan_name,
        status="ok",
        duration_seconds=0,  # filled by caller
        calls_succeeded=result.batches_succeeded,
    )


def _run_documents(plan_dir: Path, plan_output_dir: Path, llm_executor: LLMExecutor) -> PlanResult:
    """Execute the identify_documents step."""
    plan_name = plan_dir.name
    user_prompt = _load_user_prompt(plan_dir, _DOCUMENTS_INPUT_FILES)

    # Load identify_purpose_dict separately (needed by IdentifyDocuments).
    purpose_path = plan_dir / _DOCUMENTS_PURPOSE_FILE.value
    with open(purpose_path) as f:
        identify_purpose_dict = json.load(f)

    # IdentifyDocuments.execute() takes a raw LLM, not an LLMExecutor.
    # Wrap it in llm_executor.run() to get retry/fallback behaviour.
    def execute_fn(llm):
        return IdentifyDocuments.execute(llm, user_prompt, identify_purpose_dict)

    result = llm_executor.run(execute_fn)

    result.save_raw(str(plan_output_dir / FilenameEnum.IDENTIFIED_DOCUMENTS_RAW.value))
    result.save_markdown(str(plan_output_dir / FilenameEnum.IDENTIFIED_DOCUMENTS_MARKDOWN.value))
    result.save_json_documents_to_find(str(plan_output_dir / FilenameEnum.IDENTIFIED_DOCUMENTS_TO_FIND_JSON.value))
    result.save_json_documents_to_create(str(plan_output_dir / FilenameEnum.IDENTIFIED_DOCUMENTS_TO_CREATE_JSON.value))

    return PlanResult(
        name=plan_name,
        status="ok",
        duration_seconds=0,  # filled by caller
        calls_succeeded=1,
    )


def run_single_plan(
    plan_dir: Path,
    output_dir: Path,
    model_names: list[str],
    step: str = "identify_potential_levers",
) -> PlanResult:
    """
    Run a pipeline step for one plan directory. Writes outputs into
    output_dir/<plan_name>/.

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
        if step == "identify_potential_levers":
            pr = _run_levers(plan_dir, plan_output_dir, llm_executor)
        elif step == "deduplicate_levers":
            pr = _run_deduplicate(plan_dir, plan_output_dir, llm_executor)
        elif step == "enrich_potential_levers":
            pr = _run_enrich(plan_dir, plan_output_dir, llm_executor)
        elif step == "identify_documents":
            pr = _run_documents(plan_dir, plan_output_dir, llm_executor)
        else:
            raise ValueError(f"Unknown step: {step}")

        duration = time.monotonic() - t0
        pr.duration_seconds = round(duration, 2)
        logger.info(f"{plan_name}: completed in {duration:.1f}s")
        return pr

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


DEFAULT_STEP = "identify_potential_levers"


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
    step: str = "identify_potential_levers",
    plan_timeout: int = DEFAULT_PLAN_TIMEOUT,
) -> PlanResult:
    """Run a single plan and record events/output. Thread-safe.

    Enforces *plan_timeout* as a wall-clock ceiling.  If the plan doesn't
    complete in time, it is recorded as an error (the underlying thread may
    still be running but results are discarded).
    """
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

        # Enforce wall-clock timeout so a stuck LLM call doesn't block forever.
        from concurrent.futures import ThreadPoolExecutor as _TPE, TimeoutError as _TE
        with _TPE(max_workers=1) as executor:
            future = executor.submit(run_single_plan, plan_dir, output_dir, model_names, step)
            try:
                pr = future.result(timeout=plan_timeout)
            except _TE:
                logger.error(f"{plan_name}: killed after {plan_timeout}s (plan timeout)")
                pr = PlanResult(
                    name=plan_name,
                    status="error",
                    duration_seconds=float(plan_timeout),
                    error=f"plan timeout after {plan_timeout}s",
                )

        if pr.status == "ok":
            _emit_event(events_path, "run_single_plan_complete",
                        plan_name=plan_name,
                        duration_seconds=pr.duration_seconds)
        else:
            _emit_event(events_path, "run_single_plan_error",
                        plan_name=plan_name, error=pr.error,
                        duration_seconds=pr.duration_seconds)

        if (step == "identify_potential_levers"
                and pr.calls_succeeded is not None
                and pr.calls_succeeded < 3):
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
    step: str = "identify_potential_levers",
    plan_timeout: int = DEFAULT_PLAN_TIMEOUT,
) -> None:
    """
    Iterate over plan directories, run the specified step for each.

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
        "step": step,
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
                           model_names, events_path, outputs_path,
                           step=step, plan_timeout=plan_timeout)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _run_plan_task, plan_dir, output_dir,
                    model_names, events_path, outputs_path,
                    step, plan_timeout,
                ): plan_dir
                for plan_dir in pending_dirs
            }
            for future in as_completed(futures):
                future.result()  # propagate exceptions


def main():
    parser = argparse.ArgumentParser(
        description="Run a pipeline step against baseline training data."
    )
    parser.add_argument(
        "--step",
        default=DEFAULT_STEP,
        choices=SUPPORTED_STEPS,
        help=f"Pipeline step to run (default: {DEFAULT_STEP}).",
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
    parser.add_argument(
        "--plan-timeout",
        type=int,
        default=DEFAULT_PLAN_TIMEOUT,
        help=f"Wall-clock timeout per plan in seconds (default: {DEFAULT_PLAN_TIMEOUT}).",
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
        run_dir = _history_run_dir(args.prompt_lab_dir, args.step)
        output_dir = run_dir / "outputs"
        print(f"History run: {run_dir}")
    elif args.output_dir:
        output_dir = args.output_dir
    else:
        parser.error("Either --prompt-lab-dir or --output-dir is required.")

    run(plan_dirs, output_dir, args.models, step=args.step,
        plan_timeout=args.plan_timeout)

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
