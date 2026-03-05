import unittest
from unittest.mock import patch
import asyncio

import mcp_cloud.http_server as http_server


class _RequestStub:
    def __init__(
        self,
        headers: dict[str, str],
        method: str = "GET",
        path: str = "/",
        body: bytes | None = None,
    ):
        self.headers = headers
        self.method = method
        self.url = type("URL", (), {"path": path})()
        self._body = body or b""

    async def body(self) -> bytes:
        return self._body


class _ResponseStub:
    def __init__(self):
        self.headers: dict[str, str] = {}


class TestCorsEnvParsing(unittest.TestCase):
    def test_split_csv_env_strips_quotes(self):
        parsed = http_server._split_csv_env('"https://a.example.com", "https://b.example.com"')
        self.assertEqual(parsed, ["https://a.example.com", "https://b.example.com"])

    def test_split_csv_env_parses_json_array(self):
        parsed = http_server._split_csv_env('["https://a.example.com", "https://b.example.com"]')
        self.assertEqual(parsed, ["https://a.example.com", "https://b.example.com"])

    def test_split_csv_env_empty(self):
        self.assertEqual(http_server._split_csv_env(""), [])


class TestCorsHeaderAppending(unittest.TestCase):
    def test_append_cors_headers_with_wildcard(self):
        request = _RequestStub(headers={"origin": "http://localhost:6274"})
        response = _ResponseStub()
        with patch.object(http_server, "CORS_ORIGINS", ["*"]):
            updated = http_server._append_cors_headers(request, response)
        self.assertEqual(updated.headers.get("Access-Control-Allow-Origin"), "*")

    def test_append_cors_headers_with_allowed_specific_origin(self):
        origin = "http://localhost:6274"
        request = _RequestStub(
            headers={
                "origin": origin,
                "access-control-request-headers": "x-api-key,content-type",
            }
        )
        response = _ResponseStub()
        with patch.object(http_server, "CORS_ORIGINS", [origin]):
            updated = http_server._append_cors_headers(request, response)
        self.assertEqual(updated.headers.get("Access-Control-Allow-Origin"), origin)
        self.assertEqual(
            updated.headers.get("Access-Control-Allow-Headers"),
            "x-api-key,content-type",
        )
        self.assertEqual(updated.headers.get("Vary"), "Origin")

    def test_append_cors_headers_not_added_for_disallowed_origin(self):
        request = _RequestStub(headers={"origin": "https://unknown.example.com"})
        response = _ResponseStub()
        with patch.object(http_server, "CORS_ORIGINS", ["https://allowed.example.com"]):
            updated = http_server._append_cors_headers(request, response)
        self.assertNotIn("Access-Control-Allow-Origin", updated.headers)


class TestPublicMcpNoAuthRules(unittest.TestCase):
    def test_public_tools_listing_route(self):
        request = _RequestStub(headers={}, method="GET", path="/mcp/tools")
        result = asyncio.run(http_server._is_public_mcp_request_without_auth(request))
        self.assertTrue(result)

    def test_public_mcp_get_for_redirect_probe(self):
        request = _RequestStub(headers={}, method="GET", path="/mcp")
        result = asyncio.run(http_server._is_public_mcp_request_without_auth(request))
        self.assertTrue(result)

    def test_public_mcp_head_for_redirect_probe(self):
        request = _RequestStub(headers={}, method="HEAD", path="/mcp")
        result = asyncio.run(http_server._is_public_mcp_request_without_auth(request))
        self.assertTrue(result)

    def test_public_mcp_post_for_redirect_probe(self):
        request = _RequestStub(
            headers={},
            method="POST",
            path="/mcp",
            body=b'{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"plan_create","arguments":{"prompt":"x"}}}',
        )
        result = asyncio.run(http_server._is_public_mcp_request_without_auth(request))
        self.assertTrue(result)

    def test_public_streamable_initialize(self):
        request = _RequestStub(
            headers={},
            method="POST",
            path="/mcp",
            body=b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}',
        )
        result = asyncio.run(http_server._is_public_mcp_request_without_auth(request))
        self.assertTrue(result)

    def test_public_streamable_tools_list(self):
        request = _RequestStub(
            headers={},
            method="POST",
            path="/mcp/",
            body=b'{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}',
        )
        result = asyncio.run(http_server._is_public_mcp_request_without_auth(request))
        self.assertTrue(result)

    def test_public_streamable_prompts_list(self):
        request = _RequestStub(
            headers={},
            method="POST",
            path="/mcp/",
            body=b'{"jsonrpc":"2.0","id":4,"method":"prompts/list","params":{}}',
        )
        result = asyncio.run(http_server._is_public_mcp_request_without_auth(request))
        self.assertTrue(result)

    def test_public_streamable_resources_list(self):
        request = _RequestStub(
            headers={},
            method="POST",
            path="/mcp/",
            body=b'{"jsonrpc":"2.0","id":5,"method":"resources/list","params":{}}',
        )
        result = asyncio.run(http_server._is_public_mcp_request_without_auth(request))
        self.assertTrue(result)

    def test_public_streamable_resource_templates_list(self):
        request = _RequestStub(
            headers={},
            method="POST",
            path="/mcp/",
            body=b'{"jsonrpc":"2.0","id":6,"method":"resources/templates/list","params":{}}',
        )
        result = asyncio.run(http_server._is_public_mcp_request_without_auth(request))
        self.assertTrue(result)

    def test_non_public_streamable_tools_call(self):
        request = _RequestStub(
            headers={},
            method="POST",
            path="/mcp/",
            body=b'{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"plan_create","arguments":{"prompt":"x"}}}',
        )
        result = asyncio.run(http_server._is_public_mcp_request_without_auth(request))
        self.assertFalse(result)

    def test_public_streamable_tools_call_model_profiles(self):
        request = _RequestStub(
            headers={},
            method="POST",
            path="/mcp/",
            body=b'{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"model_profiles","arguments":{}}}',
        )
        result = asyncio.run(http_server._is_public_mcp_request_without_auth(request))
        self.assertTrue(result)

    def test_public_streamable_tools_call_example_prompts(self):
        request = _RequestStub(
            headers={},
            method="POST",
            path="/mcp/",
            body=b'{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"example_prompts","arguments":{}}}',
        )
        result = asyncio.run(http_server._is_public_mcp_request_without_auth(request))
        self.assertTrue(result)

    def test_public_rest_tools_call_model_profiles(self):
        request = _RequestStub(
            headers={},
            method="POST",
            path="/mcp/tools/call",
            body=b'{"tool":"model_profiles","arguments":{}}',
        )
        result = asyncio.run(http_server._is_public_mcp_request_without_auth(request))
        self.assertTrue(result)

    def test_non_public_rest_tools_call_plan_create(self):
        request = _RequestStub(
            headers={},
            method="POST",
            path="/mcp/tools/call",
            body=b'{"tool":"plan_create","arguments":{"prompt":"x"}}',
        )
        result = asyncio.run(http_server._is_public_mcp_request_without_auth(request))
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
