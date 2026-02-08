# Database API

Shared database models used by multiple PlanExe services (e.g., `frontend_multi_user`, future `worker_plan_database`). Models live here to keep them out of `worker_plan_api`, which stays worker-focused and lightweight.

## Contents
- `model_event.py`: `EventType` enum and `EventItem` SQLAlchemy model.
- `model_taskitem.py`: `TaskState` enum and `TaskItem` SQLAlchemy model.
- `model_worker.py`: `WorkerItem` SQLAlchemy model for worker heartbeats.
- `model_nonce.py`: `NonceItem` SQLAlchemy model for nonce tracking.
- `model_user_account.py`: `UserAccount` for OAuth users, credits, and profile data.
- `model_user_provider.py`: `UserProvider` links OAuth providers to users.
- `model_user_api_key.py`: `UserApiKey` for MCP credits and attribution.
- `model_credit_history.py`: `CreditHistory` append-only credit ledger.
- `model_payment_record.py`: `PaymentRecord` for Stripe/Telegram payments.

## How to import
Add the repo root (containing `database_api/`) to `PYTHONPATH`, then:
```python
from database_api.planexe_db_singleton import db
from database_api.model_event import EventType, EventItem
from database_api.model_taskitem import TaskItem, TaskState
from database_api.model_worker import WorkerItem
from database_api.model_nonce import NonceItem
from database_api.model_user_account import UserAccount
from database_api.model_user_provider import UserProvider
from database_api.model_user_api_key import UserApiKey
from database_api.model_credit_history import CreditHistory
from database_api.model_payment_record import PaymentRecord
```

Each model expects a `db` instance to be available in the module namespace (e.g., via `from database_api.planexe_db_singleton import db` in your service). Keep the models as-is to avoid divergence across services.

## Why this package depends on Flask
- I dislike that `database_api` has a dependency on `Flask`. Ideally it should only be `SQLAlchemy` and no flask. However I have tried getting rid of `Flask+flask_sqlalchemy` and make it only depend on `SQLAlchemy`, but it complicate things, and I don't want to maintain complicated code. So I choose the simple way, accepting that `Flask` is a dependency.
- The shared `db` object is a `flask_sqlalchemy.SQLAlchemy` instance; models subclass `db.Model`, and both `frontend_multi_user` (admin UI) and `worker_plan_database` expect to initialize it via `db.init_app(app)`.
- Flask’s app/request contexts drive session lifecycle (create/teardown, rollback on errors) and the `.query` helper that the models and admin views rely on.
- The admin UI registers views using `db.session`; switching to a plain `SQLAlchemy` base would break those until session wiring and teardown hooks were rebuilt.
- Even the UI-free workers spin up a minimal Flask app purely to bootstrap `db` and context handling; a new `Base` would need equivalent scoped-session setup and teardown hooks.
- Removing `Flask` would require defining a new declarative base, custom scoped session management, and updating every service (workers + frontend) to use the new base consistently—risky until all consumers are migrated together.
