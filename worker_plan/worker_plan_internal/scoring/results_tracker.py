"""
JSONL-based results tracker for experiment outcomes.

Follows the track_activity.jsonl convention: one JSON object per line,
append-only, human-readable.

PROMPT> python -m worker_plan_internal.scoring.results_tracker
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ResultsEntry:
    """A single experiment result entry for the JSONL log."""
    experiment_id: str
    timestamp: str
    task_name: str
    baseline_avg_score: float
    candidate_avg_score: float
    delta: float
    status: str  # "keep", "discard", "inconclusive"
    candidate_description: str
    duration_seconds: int = 0

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "task_name": self.task_name,
            "baseline_avg_score": round(self.baseline_avg_score, 2),
            "candidate_avg_score": round(self.candidate_avg_score, 2),
            "delta": round(self.delta, 2),
            "status": self.status,
            "candidate_description": self.candidate_description,
            "duration_seconds": self.duration_seconds,
        }


class ResultsTracker:
    """Append-only JSONL tracker for experiment results."""

    def __init__(self, results_file: Path):
        self.results_file = results_file

    def append(self, entry: ResultsEntry) -> None:
        """Append a single results entry to the JSONL file."""
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")
        logger.info(f"Appended result for experiment {entry.experiment_id}")

    def load_all(self) -> list[ResultsEntry]:
        """Load all results entries from the JSONL file."""
        if not self.results_file.exists():
            return []
        entries = []
        with open(self.results_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    entries.append(ResultsEntry(**d))
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Skipping malformed line {line_num}: {e}")
        return entries

    def recent(self, n: int = 10) -> list[ResultsEntry]:
        """Return the N most recent entries."""
        all_entries = self.load_all()
        return all_entries[-n:]

    def summary_for_task(self, task_name: str) -> dict:
        """Summarize results for a specific task."""
        entries = [e for e in self.load_all() if e.task_name == task_name]
        if not entries:
            return {"task_name": task_name, "total_experiments": 0}

        kept = [e for e in entries if e.status == "keep"]
        discarded = [e for e in entries if e.status == "discard"]
        inconclusive = [e for e in entries if e.status == "inconclusive"]
        avg_delta = sum(e.delta for e in entries) / len(entries)

        return {
            "task_name": task_name,
            "total_experiments": len(entries),
            "kept": len(kept),
            "discarded": len(discarded),
            "inconclusive": len(inconclusive),
            "avg_delta": round(avg_delta, 2),
        }


if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO)

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ResultsTracker(Path(tmpdir) / "experiment_results.jsonl")

        # Add sample entries
        entry1 = ResultsEntry(
            experiment_id="test-001",
            timestamp="2026-03-09T10:00:00Z",
            task_name="swot_business",
            baseline_avg_score=6.2,
            candidate_avg_score=7.1,
            delta=0.9,
            status="keep",
            candidate_description="Added experience qualifier",
            duration_seconds=120,
        )
        entry2 = ResultsEntry(
            experiment_id="test-002",
            timestamp="2026-03-09T11:00:00Z",
            task_name="swot_business",
            baseline_avg_score=6.5,
            candidate_avg_score=6.3,
            delta=-0.2,
            status="inconclusive",
            candidate_description="Removed bullet formatting",
            duration_seconds=95,
        )
        tracker.append(entry1)
        tracker.append(entry2)

        print("All entries:")
        for entry in tracker.load_all():
            print(f"  {entry.experiment_id}: {entry.status} (delta={entry.delta:+.1f})")

        print(f"\nRecent 1: {tracker.recent(1)[0].experiment_id}")

        summary = tracker.summary_for_task("swot_business")
        print(f"\nSummary: {json.dumps(summary, indent=2)}")
        print("\nPASSED")
