# Token Counting Implementation for PlanExe

This document describes the token counting feature that tracks LLM API usage across plan executions.

## Overview

The token counting system automatically captures and stores token metrics from all LLM calls made during plan execution. This includes:

- **Input tokens**: Tokens in the prompt/query
- **Output tokens**: Tokens in the generated response
- **Thinking tokens**: Tokens used for reasoning/internal computation (for providers that support it, e.g., o1, o3)
- **Call duration**: Time taken for each LLM invocation
- **Success/failure**: Whether the call succeeded or failed

## Architecture

### Components

1. **Database Model** (`database_api/model_token_metrics.py`)
   - `TokenMetrics`: Stores individual LLM invocation metrics
   - `TokenMetricsSummary`: Provides aggregated statistics

2. **Token Extraction** (`worker_plan/worker_plan_internal/llm_util/token_counter.py`)
   - `TokenCount`: Container for token count data
   - `extract_token_count()`: Extracts tokens from various provider response types
   - Supports: OpenAI, OpenRouter, Anthropic, Ollama, and other LLamaIndex-compatible providers

3. **Metrics Storage** (`worker_plan/worker_plan_internal/llm_util/token_metrics_store.py`)
   - `TokenMetricsStore`: Handles all database operations
   - Lazy-loads database connection to avoid import cycles
   - Methods for recording, retrieving, and aggregating metrics

4. **Pipeline Integration** (`worker_plan/worker_plan_internal/llm_util/token_instrumentation.py`)
   - `set_current_run_id()`: Initializes tracking for a plan execution
   - `record_llm_tokens()`: Decorator for automatic token capture
   - `record_attempt_tokens()`: Direct recording of attempt-level metrics

5. **API Endpoints** (`worker_plan/app.py`)
   - `GET /runs/{run_id}/token-metrics`: Aggregated metrics summary
   - `GET /runs/{run_id}/token-metrics/detailed`: Detailed per-call metrics

## Database Schema

### token_metrics Table

```sql
CREATE TABLE token_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    run_id VARCHAR(255) NOT NULL,
    llm_model VARCHAR(255) NOT NULL,
    task_name VARCHAR(255),
    input_tokens INTEGER,
    output_tokens INTEGER,
    thinking_tokens INTEGER,
    duration_seconds FLOAT,
    success BOOLEAN NOT NULL DEFAULT FALSE,
    error_message TEXT,
    raw_usage_data JSON,
    INDEX idx_run_id (run_id),
    INDEX idx_llm_model (llm_model),
    INDEX idx_task_name (task_name),
    INDEX idx_timestamp (timestamp)
);
```

## Migration Guide

### For Existing Installations

The token metrics table is created automatically when the Flask application initializes (`db.create_all()`). No manual migration is required.

If you need to create the table manually on an existing database:

```python
from database_api.planexe_db_singleton import db
from database_api.model_token_metrics import TokenMetrics

db.create_all()
```

### Docker Environments

The table is automatically created when the Flask container starts. No additional steps needed.

## Usage

### Automatic Token Tracking

Token tracking is automatically initialized for each plan execution:

1. The pipeline sets the run ID when starting
2. Each LLM call is tracked automatically
3. Token counts are extracted from provider responses
4. Metrics are stored in the database

### Retrieving Metrics

**Aggregated Summary:**
```bash
curl http://localhost:8000/runs/PlanExe_20250210_120000/token-metrics
```

Response:
```json
{
  "run_id": "PlanExe_20250210_120000",
  "total_input_tokens": 45231,
  "total_output_tokens": 12450,
  "total_thinking_tokens": 0,
  "total_tokens": 57681,
  "total_duration_seconds": 234.5,
  "total_calls": 42,
  "successful_calls": 41,
  "failed_calls": 1,
  "metrics": [...]
}
```

**Detailed Per-Call Metrics:**
```bash
curl http://localhost:8000/runs/PlanExe_20250210_120000/token-metrics/detailed
```

Response:
```json
{
  "run_id": "PlanExe_20250210_120000",
  "count": 42,
  "metrics": [
    {
      "id": 1,
      "timestamp": "2025-02-10T12:00:15.123456",
      "llm_model": "gpt-4-turbo",
      "task_name": "IdentifyPurpose",
      "input_tokens": 1234,
      "output_tokens": 567,
      "thinking_tokens": 0,
      "total_tokens": 1801,
      "duration_seconds": 5.2,
      "success": true,
      "error_message": null
    },
    ...
  ]
}
```

### Custom Instrumentation

To manually record token metrics:

```python
from worker_plan_internal.llm_util.token_instrumentation import set_current_run_id
from worker_plan_internal.llm_util.token_metrics_store import get_token_metrics_store

# Set run ID for tracking
set_current_run_id("PlanExe_20250210_120000")

# Record metrics
store = get_token_metrics_store()
store.record_token_usage(
    run_id="PlanExe_20250210_120000",
    llm_model="gpt-4",
    input_tokens=1000,
    output_tokens=500,
    duration_seconds=3.5,
    task_name="MyTask",
    success=True,
)
```

## Token Provider Support

### Supported Providers

- **OpenAI** (GPT-4, GPT-3.5-turbo, etc.)
- **OpenRouter** (access to multiple models)
- **Anthropic** (Claude, with cache_usage support)
- **Ollama** (local models)
- **Groq**
- **LM Studio**
- **Custom OpenAI-compatible endpoints**

### Response Structure Support

The token counter automatically handles:

1. **llama_index ChatResponse** (most common)
   - Extracts usage from `response.raw['usage']` or `response.message.usage`

2. **OpenAI Usage Objects**
   - Looks for `prompt_tokens`, `completion_tokens`, `reasoning_tokens`

3. **Dictionary Responses**
   - Supports both nested (`usage.prompt_tokens`) and flat formats

4. **Anthropic Responses with Cache**
   - Extracts `cache_creation_input_tokens` as thinking tokens

## Performance Considerations

### Database

- Token metrics are stored asynchronously with minimal impact on pipeline performance
- Indices on `run_id`, `llm_model`, and `timestamp` enable fast queries
- Old metrics can be deleted manually if storage becomes an issue:

```python
from worker_plan_internal.llm_util.token_metrics_store import get_token_metrics_store

store = get_token_metrics_store()
store.delete_metrics_for_run("PlanExe_20250210_120000")
```

### Import Impact

- Token tracking modules use lazy loading
- No database connection established until metrics are recorded
- Negligible overhead if database is unavailable

## Error Handling

### Database Unavailable

If the database is unavailable:
- Token extraction still works (logs warning)
- Pipeline execution continues normally
- Metrics are not persisted

### Provider-Specific Issues

Some providers may not include token usage in responses:
- Metrics are recorded with `None` values for unavailable fields
- The system handles partial information gracefully
- Raw provider response is stored for debugging

## Future Enhancements

Potential improvements for future versions:

1. **Cost Calculation**: Calculate API costs based on token usage and pricing tiers
2. **Rate Limiting**: Implement budget-based limits on token usage
3. **Metrics Visualization**: Dashboard showing token usage over time
4. **Provider Optimization**: Recommend optimal provider/model based on token efficiency
5. **Cache Metrics**: Track and report on cache hits (for Anthropic, etc.)
6. **Batch Processing**: Aggregate metrics across multiple runs for analysis

## Troubleshooting

### Metrics Not Being Recorded

1. Check that `RUN_ID_DIR` environment variable is set
2. Verify database is accessible
3. Check logs for errors: `grep "token" application.log`

### Missing Token Counts

Some issues that may result in `None` token counts:

1. Provider doesn't include usage in response (check provider API)
2. Response structure differs from expected format
3. Custom LLM wrapper doesn't expose usage properly

To debug:

```python
from worker_plan_internal.llm_util.token_counter import extract_token_count

# Test extraction with actual response
token_count = extract_token_count(your_response)
print(token_count)
```

### Database Errors

If you see `database locked` errors:

- Ensure only one pipeline instance is running per database
- For multi-process setups, use proper connection pooling
- Check Flask database configuration

## API Integration Example

Example Python script to fetch token metrics:

```python
import requests
import json

# Get aggregated metrics
response = requests.get(
    "http://localhost:8000/runs/PlanExe_20250210_120000/token-metrics"
)
summary = response.json()

print(f"Total tokens: {summary['total_tokens']}")
print(f"Successful calls: {summary['successful_calls']}")
print(f"Total duration: {summary['total_duration_seconds']}s")

# Analyze costs (example for GPT-4 pricing)
input_cost = summary['total_input_tokens'] * 0.00003  # $0.03 per 1M input tokens
output_cost = summary['total_output_tokens'] * 0.0006  # $0.06 per 1M output tokens
total_cost = input_cost + output_cost

print(f"Estimated cost: ${total_cost:.4f}")
```

## References

- [OpenAI Token Counting](https://platform.openai.com/docs/guides/tokens)
- [Anthropic API Documentation](https://docs.anthropic.com/)
- [OpenRouter API Reference](https://openrouter.ai/docs/api-reference)
- [LLamaIndex Documentation](https://docs.llamaindex.ai/)
