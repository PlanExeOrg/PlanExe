"""
Token usage metrics for plan executions.

Tracks input tokens, output tokens, and thinking tokens for each LLM call
during a plan execution, supporting multiple provider types.
"""
import logging
from typing import Optional
from datetime import datetime, UTC
from database_api.planexe_db_singleton import db
from sqlalchemy import JSON, Integer, String, Float

logger = logging.getLogger(__name__)


class TokenMetrics(db.Model):
    """Stores token usage metrics for a single LLM invocation during plan execution."""
    __tablename__ = 'token_metrics'

    # Unique identifier for this token metric record
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    # When was this metric recorded
    timestamp = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(UTC), index=True)

    # The run ID from the plan execution
    run_id = db.Column(String(255), nullable=False, index=True)

    # The LLM model name that was used
    llm_model = db.Column(String(255), nullable=False, index=True)

    # The task/stage name where the LLM was called (e.g., "IdentifyPurpose", "ReviewPlan")
    task_name = db.Column(String(255), nullable=True, index=True)

    # Number of tokens in the prompt/input
    input_tokens = db.Column(Integer, nullable=True)

    # Number of tokens in the generated output
    output_tokens = db.Column(Integer, nullable=True)

    # Number of tokens used for thinking/reasoning (for providers that support it, e.g., o1, o3)
    thinking_tokens = db.Column(Integer, nullable=True)

    # Duration of the LLM call in seconds
    duration_seconds = db.Column(Float, nullable=True)

    # Whether the call succeeded
    success = db.Column(db.Boolean, nullable=False, default=False)

    # Error message if the call failed
    error_message = db.Column(db.Text, nullable=True)

    # Provider-specific raw usage data (for debugging/transparency)
    raw_usage_data = db.Column(JSON, nullable=True)

    def __repr__(self):
        total = (self.input_tokens or 0) + (self.output_tokens or 0) + (self.thinking_tokens or 0)
        return (f"<TokenMetrics(run_id='{self.run_id}', model='{self.llm_model}', "
                f"task='{self.task_name}', total_tokens={total}, success={self.success})>")

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens used in this invocation."""
        return (self.input_tokens or 0) + (self.output_tokens or 0) + (self.thinking_tokens or 0)

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'run_id': self.run_id,
            'llm_model': self.llm_model,
            'task_name': self.task_name,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'thinking_tokens': self.thinking_tokens,
            'total_tokens': self.total_tokens,
            'duration_seconds': self.duration_seconds,
            'success': self.success,
            'error_message': self.error_message,
        }


class TokenMetricsSummary:
    """Aggregated token metrics for a plan execution."""

    def __init__(self, run_id: str, metrics: list[TokenMetrics]):
        self.run_id = run_id
        self.metrics = metrics

    @property
    def total_input_tokens(self) -> int:
        """Sum of all input tokens."""
        return sum(m.input_tokens or 0 for m in self.metrics)

    @property
    def total_output_tokens(self) -> int:
        """Sum of all output tokens."""
        return sum(m.output_tokens or 0 for m in self.metrics)

    @property
    def total_thinking_tokens(self) -> int:
        """Sum of all thinking tokens."""
        return sum(m.thinking_tokens or 0 for m in self.metrics)

    @property
    def total_tokens(self) -> int:
        """Sum of all tokens across all categories."""
        return self.total_input_tokens + self.total_output_tokens + self.total_thinking_tokens

    @property
    def total_duration_seconds(self) -> float:
        """Sum of all LLM call durations."""
        return sum(m.duration_seconds or 0 for m in self.metrics)

    @property
    def total_calls(self) -> int:
        """Total number of LLM calls."""
        return len(self.metrics)

    @property
    def successful_calls(self) -> int:
        """Number of successful calls."""
        return sum(1 for m in self.metrics if m.success)

    @property
    def failed_calls(self) -> int:
        """Number of failed calls."""
        return sum(1 for m in self.metrics if not m.success)

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            'run_id': self.run_id,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_thinking_tokens': self.total_thinking_tokens,
            'total_tokens': self.total_tokens,
            'total_duration_seconds': self.total_duration_seconds,
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'metrics': [m.to_dict() for m in self.metrics],
        }
