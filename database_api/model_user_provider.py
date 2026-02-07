import uuid
from datetime import datetime, UTC
from database_api.planexe_db_singleton import db
from sqlalchemy_utils import UUIDType
from sqlalchemy import JSON


class UserProvider(db.Model):
    id = db.Column(UUIDType(binary=False), default=uuid.uuid4, primary_key=True)
    user_id = db.Column(UUIDType(binary=False), nullable=False, index=True)

    provider = db.Column(db.String(32), nullable=False, index=True)
    provider_user_id = db.Column(db.String(256), nullable=False, index=True)
    email = db.Column(db.String(256), nullable=True)

    raw_profile = db.Column(JSON, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(UTC))
    last_login_at = db.Column(db.DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"UserProvider(provider={self.provider!r}, provider_user_id={self.provider_user_id!r})"
