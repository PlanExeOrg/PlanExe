"""Typed exceptions for LLM call failures."""

import uuid


class LLMChatError(ValueError):
    """Raised when an LLM chat interaction fails.

    Carries the root-cause exception and a unique ``error_id`` for
    cross-referencing log entries with usage_metrics.jsonl rows.

    Extends ``ValueError`` for backward compatibility with existing
    ``except ValueError`` catch sites.
    """

    def __init__(self, cause: Exception, error_id: str | None = None, message: str | None = None):
        self.cause = cause
        self.error_id = error_id or uuid.uuid4().hex[:12]
        self.message = message or "LLM chat interaction failed"
        super().__init__(f"{self.message} [{self.error_id}]: {cause}")
