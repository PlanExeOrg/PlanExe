import asyncio
import unittest

import mcp_cloud.http_server as http_server


class TestHttpServerRouting(unittest.TestCase):
    def test_normalize_mcp_path_rewrites_slash(self):
        """_NormalizeMcpPath rewrites /mcp to /mcp/ at the ASGI scope level."""
        captured_path = None

        async def dummy_app(scope, receive, send):
            nonlocal captured_path
            captured_path = scope["path"]

        middleware = http_server._NormalizeMcpPath(dummy_app)
        scope = {"type": "http", "path": "/mcp"}
        asyncio.run(middleware(scope, None, None))
        self.assertEqual(captured_path, "/mcp/")

    def test_normalize_mcp_path_preserves_trailing_slash(self):
        """_NormalizeMcpPath does not alter /mcp/ (already correct)."""
        captured_path = None

        async def dummy_app(scope, receive, send):
            nonlocal captured_path
            captured_path = scope["path"]

        middleware = http_server._NormalizeMcpPath(dummy_app)
        scope = {"type": "http", "path": "/mcp/"}
        asyncio.run(middleware(scope, None, None))
        self.assertEqual(captured_path, "/mcp/")

    def test_options_mcp_returns_ok(self):
        response = asyncio.run(http_server.options_mcp())
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
