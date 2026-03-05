"""
Gunicorn entry point for the multi-user frontend.

Usage:
    gunicorn src.wsgi:app --bind 0.0.0.0:5000 --workers 4
"""
import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s'
)

logger = logging.getLogger(__name__)

try:
    from app import MyFlaskApp  # type: ignore[reportAttributeAccessIssue]
    flask_app_instance = MyFlaskApp()
    app = flask_app_instance.app
except ValueError as exc:
    logger.critical("Configuration error – service will not start: %s", exc)
    sys.exit(1)
