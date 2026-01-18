import unittest

from mcp_server.http_utils import strip_redundant_content


class TestHttpUtils(unittest.TestCase):
    def test_strip_redundant_content_only_when_structured_present(self):
        payload = {"content": [{"text": "hi"}]}
        stripped, changed = strip_redundant_content(payload)
        self.assertFalse(changed)
        self.assertEqual(stripped, payload)

    def test_strip_redundant_content_removes_content(self):
        payload = {"content": [{"text": "hi"}], "structuredContent": {"result": []}}
        stripped, changed = strip_redundant_content(payload)
        self.assertTrue(changed)
        self.assertNotIn("content", stripped)
        self.assertIn("structuredContent", stripped)

    def test_strip_redundant_content_non_dict(self):
        payload = ["content"]
        stripped, changed = strip_redundant_content(payload)
        self.assertFalse(changed)
        self.assertEqual(stripped, payload)

if __name__ == "__main__":
    unittest.main()
