"""
Email utilities for authentication.
"""

from flask import render_template, current_app
from flask_mail import Message
from app import mail
import logging

logger = logging.getLogger(__name__)


def send_email(subject, sender, recipients, text_body, html_body):
    """Send email with error handling."""
    try:
        msg = Message(subject, sender=sender, recipients=recipients)
        msg.body = text_body
        msg.html = html_body
        mail.send(msg)
        logger.info(f"Email sent successfully to {recipients}")
    except Exception as e:
        logger.error(f"Failed to send email to {recipients}: {e}")


def send_password_reset_email(user):
    """Send password reset email."""
    token = user.get_reset_password_token()
    send_email(
        'Reset Your Password - Stock Analysis',
        sender=current_app.config['MAIL_DEFAULT_SENDER'],
        recipients=[user.email],
        text_body=render_template('email/reset_password.txt', user=user, token=token),
        html_body=render_template('email/reset_password.html', user=user, token=token)
    )