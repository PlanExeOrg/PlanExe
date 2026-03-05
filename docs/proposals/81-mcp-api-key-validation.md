# Proposal 81 — MCP API Key Validation

## Status

Implemented (2026-03-05)

## Date

2026-03-05

## Problem

The MCP server (`mcp_cloud`) accepts junk API keys and lets callers create plans,
retry plans, and consume resources without ever verifying the key is real.

Today's auth logic is a binary toggle controlled by `PLANEXE_MCP_REQUIRE_AUTH`:

| Value   | Behavior |
|---------|----------|
| `true`  | Reject missing/invalid keys with 401/403. |
| `false` | Accept **anything** — junk keys, empty keys, no key at all. |

The problem is that `false` is the default for local Docker deployments, where
there is only an admin user and no OAuth. In that mode, if someone supplies
`X-API-Key: garbage`, the server silently accepts it, creates plans attributed
to no real user, and provides no feedback that the key is wrong.

This leads to confusion: a user can copy-paste a stale or mistyped key into
their MCP client config, believe everything is fine, and later discover that
billing, per-key stats, and "Last Used" timestamps never worked.

### Current code (`http_server.py`, `_validate_api_key`)

```python
if not AUTH_REQUIRED:
    # Auth disabled — still resolve the key for attribution but never reject.
    if provided_key:
        user = await asyncio.to_thread(_resolve_user_from_api_key, provided_key)
        if user:
            _authenticated_user_api_key_ctx.set(provided_key)
    return None          # ← always allows the request, even with junk key
```

## Goals

1. **Reject invalid keys** — when a caller explicitly provides an `X-API-Key`
   that does not match any active key in the database, return an error with a
   clear message, regardless of deployment mode.
2. **Allow keyless access on localhost** — when no key is provided and the server
   is in local/admin-only mode, continue allowing requests (current behavior).
3. **Clear error messages** — tell the user exactly what went wrong and where to
   get a valid key.

## Non-Goals

- Implementing OAuth on the MCP server (see Proposal 52).
- Rate limiting or abuse prevention (separate concern).
- Changing the `PLANEXE_MCP_REQUIRE_AUTH=true` code path (already correct).

## Design

### New behavior matrix

| `REQUIRE_AUTH` | Key provided? | Key valid? | Result |
|----------------|---------------|------------|--------|
| `true`         | No            | —          | **401** Missing API key |
| `true`         | Yes           | No         | **403** Invalid API key |
| `true`         | Yes           | Yes        | Allow (authenticated) |
| `false`        | No            | —          | **Allow** (anonymous/admin) |
| `false`        | Yes           | No         | **403** Invalid API key |
| `false`        | Yes           | Yes        | Allow (authenticated + attribution) |

The only change from today is row 3 of the `false` block: when auth is
disabled but a key **is** provided and does **not** resolve, the server now
rejects instead of silently ignoring.

The reasoning: if the caller went through the trouble of setting `X-API-Key`,
they clearly intend to authenticate. Silently accepting a bad key is worse than
telling them it's wrong.

### Error response

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32001,
    "message": "Invalid API key. Check your key or create a new one at https://home.planexe.org/"
  }
}
```

HTTP status: **403 Forbidden**.

For the `REQUIRE_AUTH=false` case, the error message should also hint that
running without a key is fine for local use:

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32001,
    "message": "Invalid API key. Remove the X-API-Key header for local access, or get a valid key at https://home.planexe.org/"
  }
}
```

### Code changes

Two changes in **`mcp_cloud/http_server.py`**:

#### 1. `_validate_api_key` — reject invalid keys in local mode

Previously the `AUTH_REQUIRED=false` branch silently accepted any key.
Now, if a key is provided but does not resolve to a real user, the request
is rejected with 403:

```python
if not AUTH_REQUIRED:
    if provided_key:
        user = await asyncio.to_thread(_resolve_user_from_api_key, provided_key)
        if user:
            _authenticated_user_api_key_ctx.set(provided_key)
        else:
            await _log_auth_rejection(request, reason="invalid_api_key_local")
            return JSONResponse(
                status_code=403,
                content={
                    "detail": (
                        "Invalid API key. "
                        "Remove the X-API-Key header for local access, "
                        "or get a valid key at https://home.planexe.org/"
                    )
                },
            )
    return None  # No key provided, allow anonymous/admin access
```

#### 2. `enforce_api_key` middleware — validate at connection time

Previously, the `initialize` and other handshake/discovery JSON-RPC methods
were in `PUBLIC_JSONRPC_METHODS_NO_AUTH` and skipped validation entirely.
A junk key sailed through the MCP handshake unchallenged; the user only
discovered the problem on the first paid tool call.

Now, even for public methods, if a key **is** provided, it is validated
immediately. This means a bad key is rejected on the very first
`initialize` request — the MCP connection fails before the client sees any
tools:

```python
if request.method != "OPTIONS" and (
    request.url.path.startswith("/mcp") or request.url.path.startswith("/download")
):
    is_public = await _is_public_mcp_request_without_auth(request)

    # Even for public/discovery methods (initialize, tools/list, etc.),
    # validate the API key if one was provided.
    if is_public and _extract_api_key(request):
        error_response = await _validate_api_key(request)
        if error_response:
            return _append_cors_headers(request, error_response)

    if not is_public:
        is_tokenized_download = (
            request.url.path.startswith("/download")
            and _has_valid_download_token(request)
        )
        if not is_tokenized_download:
            error_response = await _validate_api_key(request)
            if error_response:
                return _append_cors_headers(request, error_response)
```

Keyless requests still pass through to public methods as before.

### Interaction with `PLANEXE_MCP_API_KEY` (shared secret)

The shared-secret check (`REQUIRED_API_KEY`) only runs when `AUTH_REQUIRED=true`.
Local mode (`AUTH_REQUIRED=false`) does not use shared secrets, so no change
is needed there.

## Backward Compatibility

- **Local users who never set `X-API-Key`**: no change, requests still allowed.
- **Local users with a valid key**: no change, key resolves and is attributed.
- **Local users with a junk key**: **breaking change** — previously silently
  accepted, now rejected with 403. This is intentional and desirable. The fix
  is to either remove the key or replace it with a valid one.
- **Production (`REQUIRE_AUTH=true`)**: no change at all.

## Verification

1. Start MCP server with `PLANEXE_MCP_REQUIRE_AUTH=false`.
2. Connect via MCP Inspector **without** `X-API-Key` → should work (anonymous).
3. Connect with a **valid** `pex_...` key → should work (authenticated, stats tracked).
4. Connect with `X-API-Key: junk` → connection should **fail immediately**
   during the `initialize` handshake with 403 and a clear error message.
   The client should never see the tool list.
5. Start MCP server with `PLANEXE_MCP_REQUIRE_AUTH=true`.
6. Connect without key → 401 at connection time.
7. Connect with junk key → 403 at connection time.
8. Connect with valid key → works.
