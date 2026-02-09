---
title: Stripe (credits and local testing)
---

# Stripe (credits and local testing)

PlanExe uses Stripe Checkout for buying credits. This page explains how the flow works, why credits may not update when running locally, and how to test without real money.

---

## How credits are applied

1. User clicks "Pay with Stripe" on the Account page and completes checkout on Stripe’s site.
2. Stripe redirects the user back to your app (e.g. `/account?stripe=success`).
3. **Credits are added only when Stripe sends a webhook.** Stripe calls your app at `/billing/stripe/webhook` with a `checkout.session.completed` event; the app then creates a `PaymentRecord` and a `CreditHistory` entry and updates `UserAccount.credits_balance`.

So the redirect back to `/account` does **not** by itself add credits. The webhook does.

---

## Why credits stay 0 on localhost

When the app runs on `localhost` (e.g. `http://localhost:5001`), Stripe’s servers cannot reach your machine. They need to POST to your webhook URL; `localhost` is only reachable from your own computer. So the `checkout.session.completed` webhook never hits your app, and credits are never applied.

**Fix:** use the Stripe CLI to forward webhooks from Stripe to your local server.

---

## Stripe CLI (forward webhooks to localhost)

The Stripe CLI is a separate developer tool (not listed with the [Stripe SDKs](https://docs.stripe.com/sdks)). It can tunnel webhook events to your local app.

### Where to find it

- **Install:** [Install the Stripe CLI](https://docs.stripe.com/stripe-cli/install)
- **Overview:** [Stripe CLI](https://stripe.com/docs/stripe-cli)

### Install (macOS, Homebrew)

```bash
brew install stripe/stripe-cli/stripe
```

Other platforms: see the [install guide](https://docs.stripe.com/stripe-cli/install).

### Use it for webhooks

1. Log in (opens browser with a pairing code):

   ```bash
   stripe login
   ```

2. Start forwarding webhooks to your app (adjust port if needed):

   ```bash
   stripe listen --forward-to localhost:5001/billing/stripe/webhook
   ```

3. The CLI prints a **webhook signing secret** (`whsec_...`). Add it to your environment:

   ```env
   PLANEXE_STRIPE_WEBHOOK_SECRET='whsec_xxxxx'
   ```

4. Restart the PlanExe frontend so it loads the new secret. Keep `stripe listen` running while you test payments.

Events sent to the CLI are forwarded to your local `/billing/stripe/webhook` and signed with the secret the CLI showed you. Your app can then verify the signature and apply credits.

---

## Testing without real money (test mode)

Use Stripe **test mode** so no real charges are made.

### 1. Use test API keys

In the [Stripe Dashboard](https://dashboard.stripe.com), turn on **Test mode** (toggle top right).

- Go to **Developers → API keys** ([dashboard.stripe.com/test/apikeys](https://dashboard.stripe.com/test/apikeys)).
- Use the **Secret key** that starts with `sk_test_...` (not `sk_live_...`).

In your `.env` (or environment) for local/dev:

```env
PLANEXE_STRIPE_SECRET_KEY='sk_test_...'
```

Use the test key only for development; keep the live key for production.

### 2. Test card numbers

At checkout, use Stripe’s [test card numbers](https://docs.stripe.com/testing#cards). No real payment is processed.

| Result        | Card number             |
|---------------|-------------------------|
| Success       | `4242 4242 4242 4242`   |
| Card declined | `4000 0000 0000 0002`   |
| Requires auth | `4000 0025 0000 3155`   |

- **Expiry:** any future date (e.g. `12/34`).
- **CVC:** any 3 digits (e.g. `123`).
- **ZIP:** any value (e.g. `12345`).

### 3. Webhook secret when using the CLI

When you run `stripe listen`, the signing secret it prints is for **test** events. Put that value in `PLANEXE_STRIPE_WEBHOOK_SECRET`. In production you will configure a separate webhook endpoint in the Stripe Dashboard and use that endpoint’s secret.

---

## Environment variables

| Variable | Purpose |
|----------|---------|
| `PLANEXE_STRIPE_SECRET_KEY` | Stripe secret key (`sk_test_...` or `sk_live_...`). Required for checkout and webhooks. |
| `PLANEXE_STRIPE_WEBHOOK_SECRET` | Webhook signing secret (`whsec_...`). Required to verify that webhook requests come from Stripe. For local dev, use the secret from `stripe listen`. |
| `PLANEXE_STRIPE_CURRENCY` | Currency for Checkout (default: `usd`). |
| `PLANEXE_CREDIT_PRICE_CENTS` | Price per credit in cents (default: `100`). |
| `PLANEXE_FRONTEND_MULTIUSER_PUBLIC_URL` | Public base URL used for Stripe success/cancel redirects (e.g. `http://localhost:5001` or your production URL). |

---

## See also

- [User accounts and billing (database)](user_accounts_and_billing.md) — tables and flows for credits, payments, and refunds.
