# Proposal: Lobster Museum Donation Routing via PlanExe Stripe

**Date:** 26 February 2026  
**Author:** Larry the Laptop Lobster  
**Type:** Docs-only proposal (no runtime behavior changes in this PR)

## Why this proposal exists

Voynich website now shows a Lobster Museum support block with:
- Public crypto receive addresses (ETH/EVM + SOL)
- Stripe donation CTA

To keep billing sane, Stripe should stay centralized in PlanExe (same Stripe account + same ledger path), while still letting us distinguish museum donations from normal PlanExe credit top-ups.

## Current state (already deployed outside this repo)

- Voynich frontend donation link now appends routing metadata by default:
  - `source=lobster_museum`
  - `tier=lobby`
- Voynich deploy remains static; no billing backend there.

## Existing PlanExe touchpoints

In `frontend_multi_user/src/app.py`:
- `stripe_checkout()` route around line ~2340 (`/billing/stripe/checkout`, login required)
- `stripe_webhook()` route around line ~2420 (`/billing/stripe/webhook`)
- `_apply_payment_credits(...)` helper around line ~902
- `_finalize_stripe_checkout_session(...)` helper around line ~966

In `database_api`:
- `model_payment_record.py` (`PaymentRecord`)
- `model_credit_history.py` (`CreditHistory`)

## Goals

- Keep Stripe handling in PlanExe only.
- Accept museum routing metadata from inbound checkout initiation.
- Persist enough metadata so we can separate museum donations in reporting.
- Preserve current PlanExe top-up behavior for existing users.

## Non-goals

- No Voynich billing backend.
- No multi-service payment split.
- No schema-heavy donor CRM in phase 1.

## Recommended rollout

### Phase 1 (fast path)

Use existing checkout endpoint and preserve login-gated flow.

1. Parse optional request metadata in `stripe_checkout()`:
   - `source` (default: `planexe`)
   - `tier` (default: `standard`)
   - optional `campaign`

2. Add metadata into `stripe.checkout.Session.create(...metadata=...)`:
   - existing keys (`user_id`, `credits`)
   - plus `source`, `tier`, `campaign`

3. In `stripe_webhook()` on `checkout.session.completed`, read metadata and include it in:
   - event context logs
   - stored `raw_payload` (already present)
   - credit history `reason` convention (e.g. `stripe_topup_lobster_museum` when source matches)

4. Reporting:
   - initial split via query/filter on metadata in `raw_payload` and event context

### Phase 2 (small schema hardening)

Add optional normalized columns to `PaymentRecord`:
- `source` (string, indexed)
- `tier` (string, nullable)
- `campaign` (string, nullable)

Keep `raw_payload` as source-of-truth fallback.

### Phase 3 (public donation lane, optional)

If we want truly public non-login donations:
- Add dedicated endpoint (e.g. `/billing/stripe/donate`) with anti-abuse controls
- Decide accounting model:
  - map to a service `UserAccount`, or
  - allow nullable accounting path via dedicated donation table

This should be a separate approved proposal before implementation.

## Suggested acceptance criteria for Phase 1 implementation PR

- Existing PlanExe top-up flow remains unchanged when no metadata provided.
- Museum CTA path carries `source=lobster_museum` through Stripe session metadata.
- Webhook logs and stored payload show `source` and `tier` for museum-originated payments.
- At least one documented query/runbook for extracting museum donations.

## Backward compatibility

- Fully backward compatible in Phase 1.
- No breaking API changes.
- Metadata defaults preserve legacy behavior.

## Security and abuse notes

- Continue using current Stripe signature verification (`PLANEXE_STRIPE_WEBHOOK_SECRET`).
- For public endpoint work (Phase 3), require explicit anti-abuse design (rate limits + amount caps + origin checks).

## Open questions for Simon review

1. Should museum donations continue to mint credits in Phase 1, or should they be tracked as donation-only entries?
2. Is metadata-in-payload sufficient short-term, or do we require normalized columns immediately?
3. Should `source` enum values be formalized now (`planexe`, `lobster_museum`, etc.)?

## Proposed next PR sequence

1. **This PR (docs-only):** approval on routing approach and phase order.
2. **Implementation PR A:** Phase 1 metadata pass-through + webhook/context logging.
3. **Implementation PR B (optional):** normalized columns + reporting helpers.
4. **Implementation PR C (optional):** public donation endpoint.
