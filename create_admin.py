"""
create_admin.py

Run this once, after the database schema has been created, to bootstrap
the very first Admin account. Passwords must be hashed in Python (the
DB schema only stores hashes), so this can't be done via a plain
INSERT statement in schema.sql.

Usage:
    python create_admin.py
"""

import os

from dotenv import load_dotenv

load_dotenv()

from backend.models.admin import Admin
from backend.utilities.security import hash_password
from backend.services.db_service import init_db_pool


def main():
    init_db_pool()

    name = os.environ.get("DEFAULT_ADMIN_NAME", "Super Admin")
    email = os.environ.get("DEFAULT_ADMIN_EMAIL", "admin@college.edu")
    username = os.environ.get("DEFAULT_ADMIN_USERNAME", "admin")
    password = os.environ.get("DEFAULT_ADMIN_PASSWORD", "Admin@123")

    existing = Admin.find_by_username(username)
    if existing:
        print(f"An admin with username '{username}' already exists. Nothing to do.")
        return

    password_hash = hash_password(password)
    Admin.create(name=name, email=email, username=username, password_hash=password_hash)

    print("Admin account created successfully:")
    print(f"  Username: {username}")
    print(f"  Password: {password}")
    print("  (log in at /login, then change this password later)")


if __name__ == "__main__":
    main()
