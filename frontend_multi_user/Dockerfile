FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PLANEXE_CONFIG_PATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1 \
    PYTHONPATH=/app:/app/frontend_multi_user:/app/frontend_multi_user/src

WORKDIR /app

# Copy application code and supporting files
COPY worker_plan/worker_plan_api /app/worker_plan_api
COPY database_api /app/database_api
COPY frontend_multi_user /app/frontend_multi_user
COPY llm_config /app/llm_config
COPY public/llms.txt /app/public/llms.txt

# Install dependencies from frontend_multi_user pyproject
RUN set -eux; \
    pip install --no-cache-dir "pip==24.3.1"; \
    pip install --no-cache-dir --prefer-binary /app/frontend_multi_user

# Default location for generated plans
RUN mkdir -p /app/run

WORKDIR /app/frontend_multi_user

EXPOSE 5000

ENV PLANEXE_FRONTEND_MULTIUSER_WORKERS=4

CMD gunicorn wsgi:app \
    --bind 0.0.0.0:${PLANEXE_FRONTEND_MULTIUSER_PORT:-5000} \
    --workers ${PLANEXE_FRONTEND_MULTIUSER_WORKERS} \
    --timeout 60 \
    --chdir /app/frontend_multi_user/src
