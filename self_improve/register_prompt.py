"""
Register the current system prompt for a pipeline step into the prompt-lab repo.

Usage:
    python -m self_improve.register_prompt \
        --step identify_potential_levers \
        --prompt-lab-dir /path/to/PlanExe-prompt-lab
"""
import argparse
import hashlib
import sys
from pathlib import Path

# Add worker_plan/ to sys.path so worker_plan_internal imports work.
_worker_plan_dir = str(Path(__file__).resolve().parent.parent / "worker_plan")
if _worker_plan_dir not in sys.path:
    sys.path.insert(0, _worker_plan_dir)

STEP_PROMPTS = {
    "identify_potential_levers": (
        "worker_plan_internal.lever.identify_potential_levers",
        "IDENTIFY_POTENTIAL_LEVERS_SYSTEM_PROMPT",
    ),
    "identify_documents": (
        "worker_plan_internal.document.identify_documents",
        "IDENTIFY_DOCUMENTS_BUSINESS_SYSTEM_PROMPT",
    ),
}


def _load_prompt(step: str) -> str:
    if step not in STEP_PROMPTS:
        raise ValueError(f"Unknown step: {step!r}. Available: {list(STEP_PROMPTS.keys())}")
    module_path, attr_name = STEP_PROMPTS[step]
    module = __import__(module_path, fromlist=[attr_name])
    return getattr(module, attr_name).strip()


def _next_index(prompts_dir: Path) -> int:
    """Find the highest existing prompt index and return index + 1."""
    max_index = -1
    if prompts_dir.exists():
        for f in prompts_dir.iterdir():
            if f.name.startswith("prompt_") and f.suffix == ".txt":
                try:
                    idx = int(f.name.split("_")[1])
                    max_index = max(max_index, idx)
                except (IndexError, ValueError):
                    pass
    return max_index + 1


def register(step: str, prompt_lab_dir: Path) -> Path | None:
    prompt_text = _load_prompt(step)
    sha256 = hashlib.sha256(prompt_text.encode()).hexdigest()

    prompts_dir = prompt_lab_dir / "prompts" / step
    prompts_dir.mkdir(parents=True, exist_ok=True)

    # Check if this prompt already exists (by SHA)
    for f in prompts_dir.iterdir():
        if sha256 in f.name:
            print(f"Prompt already registered: {f.name}")
            return None

    index = _next_index(prompts_dir)
    filename = f"prompt_{index}_{sha256}.txt"
    file_path = prompts_dir / filename
    file_path.write_text(prompt_text)
    print(f"Registered: {file_path}")
    return file_path


def main():
    parser = argparse.ArgumentParser(
        description="Register the current system prompt for a pipeline step."
    )
    parser.add_argument(
        "--step",
        required=True,
        choices=list(STEP_PROMPTS.keys()),
        help="Pipeline step name.",
    )
    parser.add_argument(
        "--prompt-lab-dir",
        required=True,
        type=Path,
        help="Path to the PlanExe-prompt-lab repo.",
    )
    args = parser.parse_args()

    register(args.step, args.prompt_lab_dir)


if __name__ == "__main__":
    main()
