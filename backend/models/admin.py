"""
backend/models/admin.py

Data-access layer for the `admin` table.
Models only know how to talk to their own table -- they never contain
route logic (that belongs in controllers) or raw Flask code.
"""

from backend.services.db_service import run_query


class Admin:
    @staticmethod
    def find_by_username(username):
        query = "SELECT * FROM admin WHERE username = %s LIMIT 1"
        return run_query(query, (username,), fetch_one=True)

    @staticmethod
    def find_by_id(admin_id):
        query = "SELECT * FROM admin WHERE admin_id = %s LIMIT 1"
        return run_query(query, (admin_id,), fetch_one=True)

    @staticmethod
    def create(name, email, username, password_hash):
        query = """
            INSERT INTO admin (name, email, username, password_hash)
            VALUES (%s, %s, %s, %s)
        """
        return run_query(query, (name, email, username, password_hash), commit=True)

    @staticmethod
    def count_all():
        query = "SELECT COUNT(*) AS total FROM admin"
        row = run_query(query, fetch_one=True)
        return row["total"] if row else 0
