"""Billing blueprint — Stripe and Telegram payment routes."""
import json
import logging
import os
import uuid
from decimal import Decimal
from typing import Any, Optional

import requests
import stripe
from flask import Blueprint, abort, current_app, jsonify, redirect, request, url_for
from flask_login import current_user, login_required

from database_api.model_credit_history import CreditHistory
from database_api.model_event import EventItem, EventType
from database_api.model_payment_record import PaymentRecord
from database_api.model_user_account import UserAccount
from database_api.planexe_db_singleton import db

from src.utils import to_credit_decimal

logger = logging.getLogger(__name__)

billing_bp = Blueprint("billing", __name__, url_prefix="/billing")


def _new_model(model_cls: Any, **kwargs: Any) -> Any:
    from typing import cast
    return cast(Any, model_cls)(**kwargs)


def _record_event(event_type: EventType, message: str, context: Optional[dict[str, Any]] = None) -> None:
    try:
        event = _new_model(EventItem, event_type=event_type, message=message, context=context)
        db.session.add(event)
        db.session.commit()
    except Exception as exc:
        logger.error("Failed to persist event item. message=%s error=%s", message, exc, exc_info=True)
        db.session.rollback()


def _apply_credit_delta(user: UserAccount, delta: Decimal, reason: str, source: str, external_id: Optional[str] = None) -> None:
    from src.utils import CREDIT_SCALE
    current_balance = to_credit_decimal(user.credits_balance)
    next_balance = current_balance + to_credit_decimal(delta)
    user.credits_balance = max(Decimal("0"), next_balance).quantize(CREDIT_SCALE)
    ledger = _new_model(
        CreditHistory,
        user_id=user.id,
        delta=to_credit_decimal(delta),
        reason=reason,
        source=source,
        external_id=external_id,
    )
    db.session.add(ledger)
    db.session.commit()


def _apply_payment_credits(
    user_id: str,
    provider: str,
    provider_payment_id: str,
    credits: Decimal,
    amount: int,
    currency: str,
    raw_payload: dict[str, Any],
) -> str:
    try:
        user_uuid = uuid.UUID(str(user_id))
    except ValueError:
        logger.error("Invalid user_id in payment payload: %s", user_id)
        return "invalid_user_id"
    user = db.session.get(UserAccount, user_uuid)
    if not user:
        logger.error("Payment user not found: %s", user_id)
        return "user_not_found"
    existing = PaymentRecord.query.filter_by(
        provider=provider,
        provider_payment_id=provider_payment_id,
    ).first()
    if existing:
        return "duplicate_payment"
    credit_amount = to_credit_decimal(credits)
    payment = _new_model(
        PaymentRecord,
        user_id=user.id,
        provider=provider,
        provider_payment_id=provider_payment_id,
        credits=credit_amount,
        amount=amount,
        currency=currency,
        status="completed",
        raw_payload=raw_payload,
    )
    db.session.add(payment)
    db.session.commit()
    _apply_credit_delta(
        user,
        delta=credit_amount,
        reason="credits_purchased",
        source=provider,
        external_id=provider_payment_id,
    )
    return "applied"


def _finalize_stripe_checkout_session(user: UserAccount, checkout_session_id: str) -> str:
    stripe_secret = os.environ.get("PLANEXE_STRIPE_SECRET_KEY")
    if not stripe_secret:
        _record_event(
            EventType.GENERIC_ERROR,
            "Stripe success return ignored (Stripe not configured)",
            context={"user_id": str(user.id), "checkout_session_id": checkout_session_id},
        )
        return "stripe_not_configured"

    stripe.api_key = stripe_secret
    try:
        session_obj: Any = stripe.checkout.Session.retrieve(checkout_session_id)
    except Exception as exc:
        _record_event(
            EventType.GENERIC_ERROR,
            "Stripe session retrieval failed",
            context={"user_id": str(user.id), "checkout_session_id": checkout_session_id, "error": str(exc)},
        )
        return "session_retrieve_failed"

    metadata = session_obj.get("metadata") or {}
    metadata_user_id = str(metadata.get("user_id") or "")
    metadata_credits = to_credit_decimal(metadata.get("credits", "0") or 0)
    payment_status = session_obj.get("payment_status") or ""

    if metadata_user_id != str(user.id):
        _record_event(
            EventType.GENERIC_ERROR,
            "Stripe session user mismatch",
            context={
                "user_id": str(user.id),
                "metadata_user_id": metadata_user_id,
                "checkout_session_id": checkout_session_id,
            },
        )
        return "user_mismatch"

    if payment_status != "paid":
        _record_event(
            EventType.GENERIC_ERROR,
            "Stripe session not paid",
            context={
                "user_id": str(user.id),
                "checkout_session_id": checkout_session_id,
                "payment_status": payment_status,
            },
        )
        return "not_paid"

    if metadata_credits <= 0:
        _record_event(
            EventType.GENERIC_ERROR,
            "Stripe session missing credits metadata",
            context={
                "user_id": str(user.id),
                "checkout_session_id": checkout_session_id,
                "metadata": metadata,
            },
        )
        return "missing_credits"

    status = _apply_payment_credits(
        user_id=str(user.id),
        provider="stripe",
        provider_payment_id=session_obj.get("id", ""),
        credits=metadata_credits,
        amount=session_obj.get("amount_total") or 0,
        currency=session_obj.get("currency") or "usd",
        raw_payload=session_obj,
    )
    _record_event(
        EventType.GENERIC_EVENT if status in ("applied", "duplicate_payment") else EventType.GENERIC_ERROR,
        "Stripe success return processed",
        context={
            "user_id": str(user.id),
            "checkout_session_id": checkout_session_id,
            "payment_status": payment_status,
            "credits": str(metadata_credits),
            "amount_minor": session_obj.get("amount_total") or 0,
            "amount_major": (session_obj.get("amount_total") or 0) / 100.0,
            "currency": session_obj.get("currency") or "usd",
            "status": status,
        },
    )
    return status


@billing_bp.route("/stripe/checkout", methods=["POST"])
@login_required
def stripe_checkout():
    if current_user.is_admin:
        abort(403)
    stripe_secret = os.environ.get("PLANEXE_STRIPE_SECRET_KEY")
    if not stripe_secret:
        return jsonify({"error": "Stripe not configured"}), 400
    stripe.api_key = stripe_secret
    credits = int(request.form.get("credits", "1"))
    if credits < 1:
        return jsonify({"error": "credits must be >= 1"}), 400
    price_per_credit = int(os.environ.get("PLANEXE_CREDIT_PRICE_CENTS", "100"))
    amount = credits * price_per_credit
    currency = os.environ.get("PLANEXE_STRIPE_CURRENCY", "usd")
    public_base_url = current_app.config.get("PUBLIC_BASE_URL", "")
    success_url = (
        f"{public_base_url}/account?stripe=success&session_id={{CHECKOUT_SESSION_ID}}"
        if public_base_url
        else url_for("account", _external=True)
    )
    cancel_url = f"{public_base_url}/account?stripe=cancel" if public_base_url else url_for("account", _external=True)
    _record_event(
        EventType.GENERIC_EVENT,
        "Stripe checkout requested",
        context={
            "user_id": str(current_user.id),
            "credits": credits,
            "amount_minor": amount,
            "amount_major": amount / 100.0,
            "currency": currency,
        },
    )
    try:
        session_obj: Any = stripe.checkout.Session.create(
            mode="payment",
            success_url=success_url,
            cancel_url=cancel_url,
            line_items=[{
                "price_data": {
                    "currency": currency,
                    "product_data": {"name": "PlanExe credits"},
                    "unit_amount": amount,
                },
                "quantity": 1,
            }],
            metadata={
                "user_id": str(current_user.id),
                "credits": str(credits),
            },
        )
    except Exception as exc:
        _record_event(
            EventType.GENERIC_ERROR,
            "Stripe checkout creation failed",
            context={
                "user_id": str(current_user.id),
                "credits": credits,
                "amount_minor": amount,
                "currency": currency,
                "error": str(exc),
            },
        )
        return jsonify({"error": "stripe checkout failed"}), 400
    _record_event(
        EventType.GENERIC_EVENT,
        "Stripe checkout session created",
        context={
            "user_id": str(current_user.id),
            "credits": credits,
            "amount_minor": amount,
            "amount_major": amount / 100.0,
            "currency": currency,
            "checkout_session_id": session_obj.get("id"),
            "checkout_payment_status": session_obj.get("payment_status"),
        },
    )
    session_url = session_obj.url
    if not isinstance(session_url, str) or not session_url:
        return jsonify({"error": "stripe checkout missing redirect url"}), 502
    return redirect(session_url)


@billing_bp.route("/stripe/webhook", methods=["POST"])
def stripe_webhook():
    stripe_secret = os.environ.get("PLANEXE_STRIPE_SECRET_KEY")
    webhook_secret = os.environ.get("PLANEXE_STRIPE_WEBHOOK_SECRET")
    if not stripe_secret:
        return jsonify({"error": "Stripe not configured"}), 400
    stripe.api_key = stripe_secret
    payload = request.get_data()
    sig_header = request.headers.get("Stripe-Signature")
    event_id = None
    event_type = None
    event: Any = None
    try:
        if webhook_secret:
            if sig_header is None:
                raise ValueError("Missing Stripe-Signature header")
            event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
        else:
            event = json.loads(payload)
        event_id = event.get("id")
        event_type = event.get("type")
    except Exception as exc:
        logger.error("Stripe webhook error: %s", exc)
        _record_event(
            EventType.GENERIC_ERROR,
            "Stripe webhook rejected",
            context={
                "error": str(exc),
                "has_signature_header": bool(sig_header),
                "webhook_secret_configured": bool(webhook_secret),
            },
        )
        return jsonify({"error": "invalid payload"}), 400

    _record_event(
        EventType.GENERIC_EVENT,
        "Stripe webhook received",
        context={
            "stripe_event_id": event_id,
            "stripe_event_type": event_type,
        },
    )

    if event_type == "checkout.session.completed":
        session_obj = event["data"]["object"]
        metadata = session_obj.get("metadata") or {}
        user_id = metadata.get("user_id")
        credits = to_credit_decimal(metadata.get("credits", "0") or 0)
        if user_id and credits > 0:
            status = _apply_payment_credits(
                user_id=user_id,
                provider="stripe",
                provider_payment_id=session_obj.get("id", ""),
                credits=credits,
                amount=session_obj.get("amount_total") or 0,
                currency=session_obj.get("currency") or "usd",
                raw_payload=session_obj,
            )
            _record_event(
                EventType.GENERIC_EVENT if status in ("applied", "duplicate_payment") else EventType.GENERIC_ERROR,
                "Stripe payment completion processed",
                context={
                    "stripe_event_id": event_id,
                    "stripe_event_type": event_type,
                    "user_id": user_id,
                    "credits": str(credits),
                    "amount_minor": session_obj.get("amount_total") or 0,
                    "amount_major": (session_obj.get("amount_total") or 0) / 100.0,
                    "currency": session_obj.get("currency") or "usd",
                    "checkout_session_id": session_obj.get("id", ""),
                    "checkout_payment_status": session_obj.get("payment_status"),
                    "status": status,
                },
            )
        else:
            _record_event(
                EventType.GENERIC_ERROR,
                "Stripe completed event missing billing metadata",
                context={
                    "stripe_event_id": event_id,
                    "stripe_event_type": event_type,
                    "checkout_session_id": session_obj.get("id", ""),
                    "metadata_present": bool(metadata),
                    "user_id": user_id,
                    "credits": str(credits),
                },
            )
    elif event_type in ("checkout.session.async_payment_failed", "payment_intent.payment_failed", "checkout.session.expired"):
        event_object = event.get("data", {}).get("object", {}) if isinstance(event.get("data"), dict) else {}
        metadata = event_object.get("metadata") or {}
        _record_event(
            EventType.GENERIC_ERROR,
            "Stripe payment failed",
            context={
                "stripe_event_id": event_id,
                "stripe_event_type": event_type,
                "user_id": metadata.get("user_id"),
                "credits": metadata.get("credits"),
                "amount_minor": event_object.get("amount_total") or event_object.get("amount"),
                "amount_major": ((event_object.get("amount_total") or event_object.get("amount") or 0) / 100.0),
                "currency": event_object.get("currency"),
                "checkout_session_id": event_object.get("id"),
                "payment_intent_id": event_object.get("payment_intent"),
                "payment_status": event_object.get("payment_status"),
                "failure_message": event_object.get("last_payment_error", {}).get("message") if isinstance(event_object.get("last_payment_error"), dict) else None,
            },
        )
    return jsonify({"status": "ok"})


@billing_bp.route("/telegram/invoice", methods=["POST"])
@login_required
def telegram_invoice():
    if current_user.is_admin:
        abort(403)
    bot_token = os.environ.get("PLANEXE_TELEGRAM_BOT_TOKEN")
    if not bot_token:
        return jsonify({"error": "Telegram not configured"}), 400
    credits = int(request.form.get("credits", "1"))
    if credits < 1:
        return jsonify({"error": "credits must be >= 1"}), 400
    price_per_credit = int(os.environ.get("PLANEXE_TELEGRAM_STARS_PER_CREDIT", "100"))
    payload = f"planexe:{current_user.id}:{credits}:{uuid.uuid4()}"
    url = f"https://api.telegram.org/bot{bot_token}/createInvoiceLink"
    response = requests.post(url, json={
        "title": "PlanExe credits",
        "description": f"{credits} credit(s) for PlanExe",
        "payload": payload,
        "currency": "XTR",
        "prices": [{"label": "PlanExe credits", "amount": credits * price_per_credit}],
    }, timeout=10)
    if response.status_code != 200:
        return jsonify({"error": "telegram error", "details": response.text}), 400
    data = response.json()
    if not data.get("ok"):
        return jsonify({"error": "telegram error", "details": data}), 400
    return redirect(data["result"])


@billing_bp.route("/telegram/webhook", methods=["POST"])
def telegram_webhook():
    bot_token = os.environ.get("PLANEXE_TELEGRAM_BOT_TOKEN")
    if not bot_token:
        return jsonify({"error": "Telegram not configured"}), 400
    update = request.get_json(silent=True) or {}
    if "pre_checkout_query" in update:
        query_id = update["pre_checkout_query"]["id"]
        requests.post(
            f"https://api.telegram.org/bot{bot_token}/answerPreCheckoutQuery",
            json={"pre_checkout_query_id": query_id, "ok": True},
            timeout=5,
        )
        return jsonify({"status": "ok"})
    message = update.get("message") or {}
    payment = message.get("successful_payment")
    if payment:
        payload = payment.get("invoice_payload", "")
        try:
            _, user_id, credits, _nonce = payload.split(":", 3)
            credits_decimal = to_credit_decimal(credits)
        except Exception:
            return jsonify({"status": "ignored"})
        _apply_payment_credits(
            user_id=user_id,
            provider="telegram",
            provider_payment_id=payment.get("telegram_payment_charge_id", ""),
            credits=credits_decimal,
            amount=payment.get("total_amount") or 0,
            currency=payment.get("currency") or "XTR",
            raw_payload=payment,
        )
    return jsonify({"status": "ok"})
