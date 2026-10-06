"""
backend/models/teacher.py

Data-access layer for the teachers table.
"""

from backend.services.db_service import run_query


class Teacher:

    @staticmethod
    def find_by_username(username):
        query = """
            SELECT *
            FROM teachers
            WHERE username = %s
            LIMIT 1
        """
        return run_query(query, (username,), fetch_one=True)

    @staticmethod
    def find_by_id(teacher_id):
        query = """
            SELECT *
            FROM teachers
            WHERE teacher_id = %s
            LIMIT 1
        """
        return run_query(query, (teacher_id,), fetch_one=True)

    @staticmethod
    def create(name, email, username, password_hash, phone=None):
        query = """
            INSERT INTO teachers
                (name, email, username, password_hash, phone)
            VALUES
                (%s, %s, %s, %s, %s)
        """

        return run_query(
            query,
            (
                name,
                email,
                username,
                password_hash,
                phone
            ),
            commit=True
        )

    @staticmethod
    def get_all():
        query = """
            SELECT
                teacher_id,
                name,
                email,
                username,
                phone,
                created_at
            FROM teachers
            ORDER BY name
        """

        return run_query(
            query,
            fetch_all=True
        )

    @staticmethod
    def update(
        teacher_id,
        name,
        email,
        username,
        phone=None
    ):
        query = """
            UPDATE teachers
            SET
                name = %s,
                email = %s,
                username = %s,
                phone = %s
            WHERE teacher_id = %s
        """

        return run_query(
            query,
            (
                name,
                email,
                username,
                phone,
                teacher_id
            ),
            commit=True
        )

    @staticmethod
    def update_password(teacher_id, password_hash):
        query = """
            UPDATE teachers
            SET password_hash = %s
            WHERE teacher_id = %s
        """

        return run_query(
            query,
            (
                password_hash,
                teacher_id
            ),
            commit=True
        )

    @staticmethod
    def delete(teacher_id):
        query = """
            DELETE FROM teachers
            WHERE teacher_id = %s
        """

        return run_query(
            query,
            (teacher_id,),
            commit=True
        )

    @staticmethod
    def count_all():
        query = """
            SELECT COUNT(*) AS total
            FROM teachers
        """

        row = run_query(
            query,
            fetch_one=True
        )

        return row["total"] if row else 0