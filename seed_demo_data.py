"""
seed_demo_data.py

Optional helper: creates one demo Teacher and one demo Student so you
can immediately test all three login roles without building the
Admin's "Add Teacher / Add Student" screens (those arrive in Phase 2).

Usage:
    python seed_demo_data.py
"""

from dotenv import load_dotenv

load_dotenv()

from backend.models.teacher import Teacher
from backend.models.student import Student
from backend.utilities.security import hash_password
from backend.services.db_service import init_db_pool


def main():
    init_db_pool()

    # --- Demo Teacher ---
    if not Teacher.find_by_username("teacher1"):
        Teacher.create(
            name="Priya Sharma",
            email="priya.sharma@college.edu",
            username="teacher1",
            password_hash=hash_password("Teacher@123"),
            phone="9876543210",
        )
        print("Created demo teacher -> username: teacher1 / password: Teacher@123")
    else:
        print("Demo teacher already exists.")

    # --- Demo Student ---
    if not Student.find_by_username("student1"):
        Student.create(
            name="Rohan Verma",
            email="rohan.verma@college.edu",
            username="student1",
            password_hash=hash_password("Student@123"),
            roll_no="CS2023-001",
            class_id=None,
            phone="9123456780",
        )
        print("Created demo student -> username: student1 / password: Student@123")
    else:
        print("Demo student already exists.")


if __name__ == "__main__":
    main()
