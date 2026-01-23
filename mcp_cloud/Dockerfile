FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PLANEXE_CONFIG_PATH=/app \
    PLANEXE_RUN_DIR=/app/run \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1 \
    PYTHONPATH=/app:/app/mcp_cloud:/app/database_api

WORKDIR /app

# Bring in the shared database models and this MCP server implementation.
COPY database_api /app/database_api
COPY mcp_cloud /app/mcp_cloud

# Install dependencies
RUN set -eux; \
    pip install --no-cache-dir --upgrade pip; \
    pip install --no-cache-dir --prefer-binary -r /app/mcp_cloud/requirements.txt

# Ensure output directory exists (mounted in docker-compose).
RUN mkdir -p /app/run

# Default to HTTP server mode (can be overridden with PLANEXE_MCP_MODE env var)
CMD ["python", "-m", "mcp_cloud.http_server"]
