import uuid
from datetime import datetime, UTC
from database_api.planexe_db_singleton import db
from sqlalchemy_utils import UUIDType


class UserApiKey(db.Model):
    # A unique identifier for the API key record.
    id = db.Column(UUIDType(binary=False), default=uuid.uuid4, primary_key=True)
    # Owning user account.
    user_id = db.Column(UUIDType(binary=False), nullable=False, index=True)
    # Hash of the API key (never store the raw key).
    key_hash = db.Column(db.String(128), nullable=False, unique=True)
    # Short prefix for display and audit logs.
    key_prefix = db.Column(db.String(16), nullable=False, index=True)
    # Key lifecycle timestamps.
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(UTC))
    last_used_at = db.Column(db.DateTime, nullable=True)
    revoked_at = db.Column(db.DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"UserApiKey(user_id={self.user_id}, prefix={self.key_prefix!r})"
