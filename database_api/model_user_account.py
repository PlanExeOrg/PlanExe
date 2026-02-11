import uuid
from datetime import datetime, UTC
from database_api.planexe_db_singleton import db
from sqlalchemy_utils import UUIDType


class UserAccount(db.Model):
    # A unique identifier for the user.
    id = db.Column(UUIDType(binary=False), default=uuid.uuid4, primary_key=True)
    # When was the account created and last updated.
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(UTC))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))

    # Primary email for the user (may be null if provider doesn't share it).
    email = db.Column(db.String(256), nullable=True, index=True)
    # Display name and optional split name fields.
    name = db.Column(db.String(256), nullable=True)
    given_name = db.Column(db.String(128), nullable=True)
    family_name = db.Column(db.String(128), nullable=True)
    # Locale and avatar for UI display.
    locale = db.Column(db.String(64), nullable=True)
    avatar_url = db.Column(db.String(512), nullable=True)

    # Admin flag for UI access.
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    # Tracks the one-time free plan.
    free_plan_used = db.Column(db.Boolean, default=False, nullable=False)
    # Current credit balance (integer credits).
    credits_balance = db.Column(db.Integer, default=0, nullable=False)

    # Last time the user logged in via OAuth.
    last_login_at = db.Column(db.DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"UserAccount(id={self.id}, email={self.email!r}, credits={self.credits_balance})"
