"""
backend/models/student.py

Data-access layer for the students table.
"""

from backend.services.db_service import run_query


class Student:

    @staticmethod
    def find_by_username(username):
        username_clean = str(username).strip()
        # 1. Match by student_number (USN) or email (always available)
        try:
            query = """
                SELECT *
                FROM students
                WHERE UPPER(student_number) = UPPER(%s)
                   OR LOWER(email) = LOWER(%s)
                LIMIT 1
            """
            res = run_query(query, (username_clean, username_clean), fetch_one=True)
            if res:
                return res
        except Exception:
            pass

        # 2. Fallback to username column if present
        try:
            query = "SELECT * FROM students WHERE username = %s LIMIT 1"
            return run_query(query, (username_clean,), fetch_one=True)
        except Exception:
            return None

    @staticmethod
    def find_by_id(student_id):
        query = """
            SELECT
                s.student_id,
                s.name,
                s.student_number,
                s.student_number AS usn,
                s.email,
                '' AS phone,
                s.username,
                s.password_hash,
                s.face_encoding,
                s.created_at,
                c.class_id,
                c.semester,
                c.branch,
                c.section
            FROM students s
            LEFT JOIN class_students cs ON s.student_id = cs.student_id
            LEFT JOIN classes c ON cs.class_id = c.class_id
            WHERE s.student_id = %s
            LIMIT 1
        """
        return run_query(query, (student_id,), fetch_one=True)

    @staticmethod
    def create(
        name,
        usn,
        email,
        username,
        password_hash,
        phone=None,
        semester=None,
        branch=None,
        section=None
    ):
        query = """
            INSERT INTO students
            (
                name,
                student_number,
                email,
                username,
                password_hash
            )
            VALUES (%s, %s, %s, %s, %s)
        """

        new_id = run_query(
            query,
            (
                name,
                usn,
                email,
                username,
                password_hash
            ),
            commit=True
        )

        if new_id and semester and branch:
            class_row = run_query(
                "SELECT class_id FROM classes WHERE semester = %s AND branch = %s LIMIT 1",
                (semester, branch),
                fetch_one=True
            )
            if class_row:
                cid = class_row["class_id"]
                run_query(
                    "INSERT INTO class_students (class_id, student_id) VALUES (%s, %s)",
                    (cid, new_id),
                    commit=True
                )

        return new_id

    @staticmethod
    def get_all():
        query = """
            SELECT
                s.student_id,
                s.name,
                s.student_number,
                s.student_number AS usn,
                s.email,
                '' AS phone,
                c.semester,
                c.branch,
                c.section,
                s.username,
                s.created_at
            FROM students s
            LEFT JOIN class_students cs ON s.student_id = cs.student_id
            LEFT JOIN classes c ON cs.class_id = c.class_id
            ORDER BY s.name
        """

        return run_query(query, fetch_all=True)

    @staticmethod
    def update(
        student_id,
        name,
        usn,
        email,
        phone=None,
        semester=None,
        branch=None,
        section=None,
        username=None
    ):
        query = """
            UPDATE students
            SET
                name = %s,
                student_number = %s,
                email = %s,
                username = %s
            WHERE student_id = %s
        """

        res = run_query(
            query,
            (
                name,
                usn,
                email,
                username,
                student_id
            ),
            commit=True
        )

        if semester is not None and branch is not None:
            class_row = run_query(
                "SELECT class_id FROM classes WHERE semester = %s AND branch = %s LIMIT 1",
                (semester, branch),
                fetch_one=True
            )
            if class_row:
                cid = class_row["class_id"]
                run_query("DELETE FROM class_students WHERE student_id = %s", (student_id,), commit=True)
                run_query(
                    "INSERT INTO class_students (class_id, student_id) VALUES (%s, %s)",
                    (cid, student_id),
                    commit=True
                )

        return res

    @staticmethod
    def update_password(student_id, password_hash):
        query = """
            UPDATE students
            SET password_hash = %s
            WHERE student_id = %s
        """

        return run_query(
            query,
            (password_hash, student_id),
            commit=True
        )

    @staticmethod
    def delete(student_id):
        run_query("DELETE FROM class_students WHERE student_id = %s", (student_id,), commit=True)
        run_query("DELETE FROM attendance WHERE student_id = %s", (student_id,), commit=True)
        query = """
            DELETE FROM students
            WHERE student_id = %s
        """

        return run_query(
            query,
            (student_id,),
            commit=True
        )

    @staticmethod
    def count_all():
        query = """
            SELECT COUNT(*) AS total
            FROM students
        """

        row = run_query(query, fetch_one=True)

        return row["total"] if row else 0