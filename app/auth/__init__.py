"""
Authentication module for the stock analysis application.
"""

from flask import Blueprint

auth = Blueprint('auth', __name__)

from app.auth import routes