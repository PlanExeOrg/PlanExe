import uuid
from datetime import datetime, UTC
from database_api.planexe_db_singleton import db
from sqlalchemy_utils import UUIDType


class CreditHistory(db.Model):
    """
    Append-only ledger of credit changes.

    What it represents:
    - Each row is a single credit change: +credits for purchases/promos, -credits for plan creation.
    - The UserAccount.balance is the current state; the ledger is the history that produced it.

    Why it exists:
    - Audit trail for payments, free grants, and usage.
    - Ability to reconcile Stripe/Telegram records or investigate disputes.
    - Ability to rebuild a user’s balance from history if needed.

    When to write rows:
    - Normal flow: create rows only via application logic (payment webhooks, plan creation, admin grant).
    - Manual edits should be rare and only for corrections (e.g., refund, mistaken charge).
    - Do not delete rows; add a compensating entry instead.
    
    Important:
    - Ledger entries do NOT move money. Refunds must be issued through Stripe/Telegram.
    """
    # A unique identifier for the ledger entry.
    id = db.Column(UUIDType(binary=False), default=uuid.uuid4, primary_key=True)
    # Owning user account.
    user_id = db.Column(UUIDType(binary=False), nullable=False, index=True)
    # Credit delta (positive for purchase, negative for usage).
    delta = db.Column(db.Integer, nullable=False)
    # Short reason and source for audit (e.g., plan_created, stripe).
    reason = db.Column(db.String(128), nullable=False)
    source = db.Column(db.String(32), nullable=False)
    # Optional external payment or invoice id.
    external_id = db.Column(db.String(256), nullable=True)
    # When the ledger entry was created.
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(UTC))

    def __repr__(self) -> str:
        return f"CreditHistory(user_id={self.user_id}, delta={self.delta}, source={self.source!r})"
