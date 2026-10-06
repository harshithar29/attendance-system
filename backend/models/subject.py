"""
backend/models/subject.py

Data-access layer for the subjects table.
"""

from backend.services.db_service import run_query


class Subject:

    @staticmethod
    def create(subject_name, subject_code, class_id=None):
        query = """
            INSERT INTO subjects
                (subject_name, subject_code, class_id)
            VALUES
                (%s, %s, %s)
        """

        return run_query(
            query,
            (
                subject_name,
                subject_code,
                class_id
            ),
            commit=True
        )

    @staticmethod
    def get_all():
        query = """
            SELECT
                s.subject_id,
                s.subject_name,
                s.subject_code,
                s.class_id,
                c.class_name,
                c.semester,
                c.branch,
                c.section,
                s.created_at
            FROM subjects s
            LEFT JOIN classes c
                ON s.class_id = c.class_id
            ORDER BY s.subject_name
        """

        return run_query(
            query,
            fetch_all=True
        )

    @staticmethod
    def find_by_id(subject_id):
        query = """
            SELECT
                subject_id,
                subject_name,
                subject_code,
                class_id,
                created_at
            FROM subjects
            WHERE subject_id = %s
            LIMIT 1
        """

        return run_query(
            query,
            (subject_id,),
            fetch_one=True
        )

    @staticmethod
    def find_by_code(subject_code):
        query = """
            SELECT
                subject_id,
                subject_name,
                subject_code,
                class_id,
                created_at
            FROM subjects
            WHERE subject_code = %s
            LIMIT 1
        """

        return run_query(
            query,
            (subject_code,),
            fetch_one=True
        )

    @staticmethod
    def update(
        subject_id,
        subject_name,
        subject_code,
        class_id=None
    ):
        query = """
            UPDATE subjects
            SET
                subject_name = %s,
                subject_code = %s,
                class_id = %s
            WHERE subject_id = %s
        """

        return run_query(
            query,
            (
                subject_name,
                subject_code,
                class_id,
                subject_id
            ),
            commit=True
        )

    @staticmethod
    def delete(subject_id):
        query = """
            DELETE FROM subjects
            WHERE subject_id = %s
        """

        return run_query(
            query,
            (subject_id,),
            commit=True
        )

    @staticmethod
    def count_all():
        query = """
            SELECT COUNT(*) AS total
            FROM subjects
        """

        row = run_query(
            query,
            fetch_one=True
        )

        return row["total"] if row else 0