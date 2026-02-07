import uuid
from datetime import datetime, UTC
from database_api.planexe_db_singleton import db
from sqlalchemy_utils import UUIDType


class UserAccount(db.Model):
    id = db.Column(UUIDType(binary=False), default=uuid.uuid4, primary_key=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(UTC))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC))

    email = db.Column(db.String(256), nullable=True, index=True)
    name = db.Column(db.String(256), nullable=True)
    given_name = db.Column(db.String(128), nullable=True)
    family_name = db.Column(db.String(128), nullable=True)
    locale = db.Column(db.String(64), nullable=True)
    avatar_url = db.Column(db.String(512), nullable=True)

    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    free_plan_used = db.Column(db.Boolean, default=False, nullable=False)
    credits_balance = db.Column(db.Integer, default=0, nullable=False)

    last_login_at = db.Column(db.DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"UserAccount(id={self.id}, email={self.email!r}, credits={self.credits_balance})"
