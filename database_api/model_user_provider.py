import uuid
from datetime import datetime, UTC
from database_api.planexe_db_singleton import db
from sqlalchemy_utils import UUIDType
from sqlalchemy import JSON


class UserProvider(db.Model):
    # A unique identifier for this provider link.
    id = db.Column(UUIDType(binary=False), default=uuid.uuid4, primary_key=True)
    # Owning user account.
    user_id = db.Column(UUIDType(binary=False), nullable=False, index=True)

    # Provider name and provider-specific user id.
    provider = db.Column(db.String(32), nullable=False, index=True)
    provider_user_id = db.Column(db.String(256), nullable=False, index=True)
    # Email as reported by the provider (may be null).
    email = db.Column(db.String(256), nullable=True)

    # Full provider profile payload for debugging and future fields.
    raw_profile = db.Column(JSON, nullable=True)
    # When the link was created and last used.
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(UTC))
    last_login_at = db.Column(db.DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"UserProvider(provider={self.provider!r}, provider_user_id={self.provider_user_id!r})"
