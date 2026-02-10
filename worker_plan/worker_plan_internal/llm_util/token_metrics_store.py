"""
Store and retrieve token metrics from the database.

This module handles all database operations for token metrics,
providing a clean interface for the LLM pipeline to record token usage.
"""
import logging
from typing import Optional, List
from datetime import datetime, UTC

logger = logging.getLogger(__name__)

__all__ = ["TokenMetricsStore", "get_token_metrics_store"]

# Global instance (lazy-loaded)
_token_metrics_store: Optional['TokenMetricsStore'] = None


def get_token_metrics_store() -> 'TokenMetricsStore':
    """Get or create the global TokenMetricsStore instance."""
    global _token_metrics_store
    if _token_metrics_store is None:
        _token_metrics_store = TokenMetricsStore()
    return _token_metrics_store


class TokenMetricsStore:
    """
    Store and retrieve token metrics from the database.
    
    This class provides methods to record token usage for individual LLM calls
    and retrieve aggregated metrics for plan executions.
    """

    def __init__(self):
        """Initialize the store. Database connection is lazy-loaded."""
        self.db = None
        self.TokenMetrics = None
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        """Lazily initialize database connection."""
        if self._initialized:
            return True

        try:
            # Lazy import to avoid circular dependencies
            from database_api.planexe_db_singleton import db
            from database_api.model_token_metrics import TokenMetrics

            self.db = db
            self.TokenMetrics = TokenMetrics
            self._initialized = True
            return True
        except ImportError as e:
            logger.warning(f"Could not initialize TokenMetricsStore: {e}. Token metrics will not be stored.")
            return False
        except Exception as e:
            logger.error(f"Unexpected error initializing TokenMetricsStore: {e}")
            return False

    def record_token_usage(
        self,
        run_id: str,
        llm_model: str,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        thinking_tokens: Optional[int] = None,
        duration_seconds: Optional[float] = None,
        task_name: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        raw_usage_data: Optional[dict] = None,
    ) -> bool:
        """
        Record token usage for an LLM call.

        Args:
            run_id: The plan execution run ID
            llm_model: The LLM model name (e.g., "gpt-4", "ollama-llama3.1")
            input_tokens: Number of input tokens (optional)
            output_tokens: Number of output tokens (optional)
            thinking_tokens: Number of thinking/reasoning tokens (optional)
            duration_seconds: Duration of the LLM call in seconds (optional)
            task_name: Name of the task/stage calling the LLM (optional)
            success: Whether the call succeeded
            error_message: Error message if the call failed
            raw_usage_data: Provider-specific usage data for debugging

        Returns:
            True if the metric was recorded successfully, False otherwise
        """
        if not self._ensure_initialized():
            return False

        try:
            metric = self.TokenMetrics(
                run_id=run_id,
                llm_model=llm_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                thinking_tokens=thinking_tokens,
                duration_seconds=duration_seconds,
                task_name=task_name,
                success=success,
                error_message=error_message,
                raw_usage_data=raw_usage_data,
            )
            self.db.session.add(metric)
            self.db.session.commit()
            logger.debug(
                f"Recorded token usage: run_id={run_id}, model={llm_model}, "
                f"input={input_tokens}, output={output_tokens}, thinking={thinking_tokens}"
            )
            return True
        except Exception as e:
            logger.error(f"Error recording token metrics: {e}", exc_info=True)
            try:
                self.db.session.rollback()
            except Exception:
                pass
            return False

    def get_metrics_for_run(self, run_id: str) -> List:
        """
        Get all token metrics for a specific plan execution.

        Args:
            run_id: The plan execution run ID

        Returns:
            List of TokenMetrics objects for this run
        """
        if not self._ensure_initialized():
            return []

        try:
            metrics = self.TokenMetrics.query.filter_by(run_id=run_id).all()
            return metrics or []
        except Exception as e:
            logger.error(f"Error retrieving token metrics for run {run_id}: {e}")
            return []

    def get_summary_for_run(self, run_id: str) -> Optional[dict]:
        """
        Get aggregated token metrics summary for a plan execution.

        Args:
            run_id: The plan execution run ID

        Returns:
            Dictionary with aggregated metrics, or None if there's an error
        """
        if not self._ensure_initialized():
            return None

        try:
            metrics = self.get_metrics_for_run(run_id)
            if not metrics:
                return {
                    'run_id': run_id,
                    'total_input_tokens': 0,
                    'total_output_tokens': 0,
                    'total_thinking_tokens': 0,
                    'total_tokens': 0,
                    'total_duration_seconds': 0,
                    'total_calls': 0,
                    'successful_calls': 0,
                    'failed_calls': 0,
                    'metrics': [],
                }

            return {
                'run_id': run_id,
                'total_input_tokens': sum(m.input_tokens or 0 for m in metrics),
                'total_output_tokens': sum(m.output_tokens or 0 for m in metrics),
                'total_thinking_tokens': sum(m.thinking_tokens or 0 for m in metrics),
                'total_tokens': sum((m.input_tokens or 0) + (m.output_tokens or 0) + (m.thinking_tokens or 0) for m in metrics),
                'total_duration_seconds': sum(m.duration_seconds or 0 for m in metrics),
                'total_calls': len(metrics),
                'successful_calls': sum(1 for m in metrics if m.success),
                'failed_calls': sum(1 for m in metrics if not m.success),
                'metrics': [m.to_dict() for m in metrics],
            }
        except Exception as e:
            logger.error(f"Error generating summary for run {run_id}: {e}")
            return None

    def delete_metrics_for_run(self, run_id: str) -> bool:
        """
        Delete all token metrics for a specific plan execution.

        Args:
            run_id: The plan execution run ID

        Returns:
            True if deletion was successful, False otherwise
        """
        if not self._ensure_initialized():
            return False

        try:
            self.TokenMetrics.query.filter_by(run_id=run_id).delete()
            self.db.session.commit()
            logger.info(f"Deleted token metrics for run {run_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting token metrics for run {run_id}: {e}")
            try:
                self.db.session.rollback()
            except Exception:
                pass
            return False
