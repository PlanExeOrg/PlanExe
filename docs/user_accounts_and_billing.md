---
title: User accounts and billing (database)
---

# User accounts and billing (database)

These tables support OAuth logins, API keys, and credit‑based billing.

---

## UserAccount

Represents a user in PlanExe.

Fields:
- `id` (UUID, primary key)
- `email`, `name`, `given_name`, `family_name`
- `locale`, `avatar_url`
- `is_admin` (bool)
- `free_plan_used` (bool)
- `credits_balance` (int)
- `last_login_at`, `created_at`, `updated_at`

---

## UserProvider

Links a user to an OAuth provider identity.

Fields:
- `id` (UUID, primary key)
- `user_id` (UUID, foreign key to UserAccount)
- `provider` (string, e.g. google/github/discord)
- `provider_user_id` (string)
- `email` (string)
- `raw_profile` (JSON)
- `created_at`, `last_login_at`

---

## UserApiKey

API key record for MCP usage.

Fields:
- `id` (UUID, primary key)
- `user_id` (UUID, foreign key)
- `key_hash` (sha256 hash)
- `key_prefix` (short prefix for display)
- `created_at`, `last_used_at`, `revoked_at`

Notes:
- Only the hash is stored. The full key is shown once at creation.

---

## CreditLedger

Append‑only ledger of credit changes.

Fields:
- `id` (UUID, primary key)
- `user_id` (UUID, foreign key)
- `delta` (int, positive or negative)
- `reason` (string)
- `source` (string, e.g. stripe/telegram/mcp/web)
- `external_id` (string)
- `created_at`

---

## PaymentRecord

Stores completed payment details.

Fields:
- `id` (UUID, primary key)
- `user_id` (UUID, foreign key)
- `provider` (string, stripe/telegram)
- `provider_payment_id` (string)
- `credits` (int)
- `amount` (int, minor units)
- `currency` (string)
- `status` (string)
- `raw_payload` (JSON)
- `created_at`

---

## Payment and refund flows

### Buy credits (Stripe)
1. User opens **Account** and chooses credits.
2. Stripe Checkout is created.
3. Stripe sends `checkout.session.completed` webhook.
4. App creates a `PaymentRecord` and a **CreditLedger** entry (+credits).

### Buy credits (Telegram Stars)
1. User opens **Account** and chooses credits.
2. App creates an invoice link via Telegram.
3. Telegram sends `successful_payment` webhook.
4. App creates a `PaymentRecord` and a **CreditLedger** entry (+credits).

### Spend credits (create a plan)
1. User submits a plan.
2. App deducts 1 credit.
3. A **CreditLedger** entry is created (delta = -1, reason = plan_created).

### Close account (user wants money back)
Typical approach:
- If credits are unused, issue a refund in Stripe/Telegram.
- Add a **CreditLedger** entry to remove credits (negative delta) or to zero the balance.
- Keep the ledger history intact (do not delete rows).

### Refund / correction
If something went wrong:
- Process the refund with the payment provider (Stripe/Telegram).
- Add a **CreditLedger** entry that reverses the original credit grant.
- Optionally update `PaymentRecord.status` (e.g., refunded).
