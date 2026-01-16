"""Helpers for loading .env files in the MCP server."""
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def load_planexe_dotenv(module_dir: Optional[Path] = None) -> tuple[bool, list[Path]]:
    """Load .env from mcp_server/.env, falling back to repo root."""
    base_dir = module_dir or Path(__file__).parent
    paths = [base_dir / ".env", base_dir.parent / ".env"]
    loaded = False
    for path in paths:
        if load_dotenv(path):
            loaded = True
            break
    return loaded, paths
