"""
PlanExe MCP Cloud — HTTP server (backward-compatibility re-export shim)

All implementation has moved to focused modules:
- server_boot.py     — config, MCP/FastAPI creation, lifespan, entry point
- middleware.py       — CORS, auth, rate limiting, body size, enforce_api_key
- tool_http_bridge.py — request/response models, result normalization, tool wrappers
- route_registration.py — MCP tool registration, MCP prompts, route handlers

This module re-exports public symbols so that existing tests and the
Dockerfile entry point (``python -m mcp_cloud.http_server``) continue
to work without modification.
"""

# --- server_boot exports (config, servers, lifespan) ----------------------
from mcp_cloud.server_boot import (  # noqa: F401
    AUTH_REQUIRED,
    CORS_ORIGINS,
    DOWNLOAD_RATE_LIMIT_REQUESTS,
    DOWNLOAD_RATE_LIMIT_WINDOW_SECONDS,
    GLAMA_MAINTAINER_EMAIL,
    HTTP_HOST,
    HTTP_PORT,
    MAX_BODY_BYTES,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    REQUIRED_API_KEY,
    SERVER_VERSION,
    _parse_bool_env,
    _split_csv_env,
    _startup_log,
    app,
    mcp_server,
)

# --- middleware exports (auth, CORS, rate limiting) -----------------------
from mcp_cloud.middleware import (  # noqa: F401
    PUBLIC_JSONRPC_METHODS_NO_AUTH,
    PUBLIC_TOOL_CALLS_NO_AUTH,
    _NormalizeMcpPath,
    _allowed_cors_origin,
    _append_cors_headers,
    _authenticated_user_api_key_ctx,
    _client_identifier,
    _download_rate_buckets,
    _enforce_body_size,
    _enforce_download_rate_limit,
    _enforce_rate_limit,
    _extract_api_key,
    _extract_jsonrpc_methods_from_payload,
    _extract_jsonrpc_methods_from_request,
    _extract_jsonrpc_tools_call_names,
    _extract_rest_tools_call_name,
    _get_authenticated_user_api_key,
    _has_valid_download_token,
    _is_public_mcp_request_without_auth,
    _log_auth_rejection,
    _make_jsonrpc_auth_error,
    _normalize_api_key_value,
    _rate_buckets,
    _rate_lock,
    _request_origin,
    _sweep_rate_buckets,
    _validate_api_key,
    enforce_api_key,
)

# --- tool_http_bridge exports (models, wrappers, normalization) -----------
from mcp_cloud.tool_http_bridge import (  # noqa: F401
    MCPToolCallRequest,
    MCPToolCallResponse,
    ModelProfileInput,
    ResultArtifactInput,
    _normalize_tool_result,
    _parse_error_from_text,
    call_tool_via_registry,
    example_plans,
    example_prompts,
    extract_text_content,
    model_profiles,
    plan_create,
    plan_file_info,
    plan_list,
    plan_resume,
    plan_retry,
    plan_status,
    plan_stop,
    send_feedback,
)

# --- route_registration exports (for direct access) ----------------------
from mcp_cloud.route_registration import (  # noqa: F401
    options_mcp,
    head_mcp_trailing_slash,
    register_routes,
    register_tools_and_prompts,
)

# --- Entry point (preserves `python -m mcp_cloud.http_server`) -----------
if __name__ == "__main__":
    from mcp_cloud.server_boot import HTTP_HOST, HTTP_PORT, AUTH_REQUIRED
    import logging
    import uvicorn

    logger = logging.getLogger(__name__)
    logger.info(f"Starting PlanExe MCP Cloud server on {HTTP_HOST}:{HTTP_PORT}")
    if AUTH_REQUIRED:
        logger.info(
            "Authentication required: UserApiKey (from home.planexe.org) or PLANEXE_MCP_API_KEY"
        )
    else:
        logger.warning("Authentication disabled via PLANEXE_MCP_REQUIRE_AUTH=false")

    uvicorn.run("mcp_cloud.server_boot:app", host=HTTP_HOST, port=HTTP_PORT, reload=False)
