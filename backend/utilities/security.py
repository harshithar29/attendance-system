"""
backend/utilities/security.py

Reusable security helpers:
  - password hashing / verification (Werkzeug's PBKDF2-SHA256 under the hood)
  - a `login_required` decorator that also supports role restriction,
    e.g. @login_required(role="admin")
"""

from functools import wraps

from flask import session, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash


def hash_password(plain_password: str) -> str:
    """Hashes a plain-text password before it is stored in the DB.
    We never store raw passwords, ever."""
    return generate_password_hash(plain_password)


def verify_password(plain_password: str, password_hash: str) -> bool:
    """Checks a password against a stored hash or plain-text at login time."""
    if not password_hash:
        return False
    try:
        if check_password_hash(password_hash, plain_password):
            return True
    except Exception:
        pass
    return password_hash == plain_password


def login_required(role: str = None):
    """
    Decorator to protect a route.

    Usage:
        @login_required()                # any logged-in user
        @login_required(role="admin")    # only admins
        @login_required(role="teacher")  # only teachers
        @login_required(role="student")  # only students
    """

    def decorator(view_func):
        @wraps(view_func)
        def wrapped_view(*args, **kwargs):
            if "user_id" not in session:
                flash("Please log in to continue.", "warning")
                return redirect(url_for("auth.login"))

            if role is not None and session.get("role") != role:
                flash("You do not have permission to view that page.", "danger")
                return redirect(url_for("auth.login"))

            return view_func(*args, **kwargs)

        return wrapped_view

    return decorator
