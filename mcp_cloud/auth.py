"""PlanExe MCP Cloud – API-key hashing and user resolution."""
import hashlib
import logging
import os
from datetime import UTC, datetime
from typing import Any, Optional

from mcp_cloud.db_setup import app, db, UserApiKey, UserAccount

logger = logging.getLogger(__name__)


def validate_api_key_secret() -> None:
    """Raise if PLANEXE_API_KEY_SECRET is not set.

    Call at startup when authentication is required so the server
    fails hard instead of silently falling back to a dev secret.
    """
    if not os.environ.get("PLANEXE_API_KEY_SECRET"):
        raise RuntimeError(
            "PLANEXE_API_KEY_SECRET is not set. "
            "Set this environment variable or disable auth with PLANEXE_MCP_REQUIRE_AUTH=false."
        )


def _hash_user_api_key(raw_key: str) -> str:
    secret = os.environ.get("PLANEXE_API_KEY_SECRET", "dev-api-key-secret")
    if secret == "dev-api-key-secret":
        logger.warning("PLANEXE_API_KEY_SECRET not set. Using dev secret for API key hashing.")
    return hashlib.sha256(f"{secret}:{raw_key}".encode("utf-8")).hexdigest()

def _resolve_user_from_api_key(raw_key: str) -> Optional[dict[str, Any]]:
    if not raw_key:
        return None
    key_hash = _hash_user_api_key(raw_key)
    with app.app_context():
        api_key = UserApiKey.query.filter_by(key_hash=key_hash, revoked_at=None).first()
        if not api_key:
            return None
        user = db.session.get(UserAccount, api_key.user_id)
        if not user:
            return None

        user_context = {
            "user_id": str(user.id),
            "credits_balance": float(user.credits_balance or 0),
            "api_key_id": str(api_key.id),
        }
        api_key.last_used_at = datetime.now(UTC)
        db.session.commit()
        return user_context
