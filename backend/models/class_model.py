"""
backend/models/class_model.py

Data-access layer for the classes table.
"""

from backend.services.db_service import run_query


class ClassModel:

    @staticmethod
    def create(class_name, section=None):
        query = """
            INSERT INTO classes
                (class_name, section)
            VALUES
                (%s, %s)
        """

        return run_query(
            query,
            (class_name, section),
            commit=True
        )

    @staticmethod
    def get_all():
        query = """
            SELECT
                class_id,
                class_name,
                section,
                created_at
            FROM classes
            ORDER BY class_name, section
        """

        return run_query(
            query,
            fetch_all=True
        )

    @staticmethod
    def find_by_id(class_id):
        query = """
            SELECT
                class_id,
                class_name,
                section,
                created_at
            FROM classes
            WHERE class_id = %s
            LIMIT 1
        """

        return run_query(
            query,
            (class_id,),
            fetch_one=True
        )

    @staticmethod
    def update(class_id, class_name, section=None):
        query = """
            UPDATE classes
            SET
                class_name = %s,
                section = %s
            WHERE class_id = %s
        """

        return run_query(
            query,
            (
                class_name,
                section,
                class_id
            ),
            commit=True
        )

    @staticmethod
    def delete(class_id):
        query = """
            DELETE FROM classes
            WHERE class_id = %s
        """

        return run_query(
            query,
            (class_id,),
            commit=True
        )

    @staticmethod
    def count_all():
        query = """
            SELECT COUNT(*) AS total
            FROM classes
        """

        row = run_query(
            query,
            fetch_one=True
        )

        return row["total"] if row else 0