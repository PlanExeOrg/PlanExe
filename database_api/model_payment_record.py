import uuid
from datetime import datetime, UTC
from database_api.planexe_db_singleton import db
from sqlalchemy_utils import UUIDType
from sqlalchemy import JSON


class PaymentRecord(db.Model):
    id = db.Column(UUIDType(binary=False), default=uuid.uuid4, primary_key=True)
    user_id = db.Column(UUIDType(binary=False), nullable=False, index=True)
    provider = db.Column(db.String(32), nullable=False, index=True)
    provider_payment_id = db.Column(db.String(256), nullable=False, index=True)
    credits = db.Column(db.Integer, nullable=False)
    amount = db.Column(db.Integer, nullable=False)  # minor currency units (e.g., cents)
    currency = db.Column(db.String(16), nullable=False)
    status = db.Column(db.String(32), nullable=False)
    raw_payload = db.Column(JSON, nullable=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(UTC))

    def __repr__(self) -> str:
        return f"PaymentRecord(provider={self.provider!r}, payment_id={self.provider_payment_id!r})"
