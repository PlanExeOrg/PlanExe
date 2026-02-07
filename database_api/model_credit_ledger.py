import uuid
from datetime import datetime, UTC
from database_api.planexe_db_singleton import db
from sqlalchemy_utils import UUIDType


class CreditLedger(db.Model):
    id = db.Column(UUIDType(binary=False), default=uuid.uuid4, primary_key=True)
    user_id = db.Column(UUIDType(binary=False), nullable=False, index=True)
    delta = db.Column(db.Integer, nullable=False)
    reason = db.Column(db.String(128), nullable=False)
    source = db.Column(db.String(32), nullable=False)
    external_id = db.Column(db.String(256), nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(UTC))

    def __repr__(self) -> str:
        return f"CreditLedger(user_id={self.user_id}, delta={self.delta}, source={self.source!r})"
