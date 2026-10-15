"""
backend/controllers/admin_controller.py

Admin routes:
- Admin Dashboard
- Teacher Management
- Student Management
- Class Management
- Subject Management
"""

import os
import re
import json
from datetime import datetime

def get_subject_semester(sub_code, class_sem=None):
    m = re.search(r'\d', str(sub_code or ''))
    if m:
        val = int(m.group(0))
        if 1 <= val <= 8:
            return val
    if class_sem and str(class_sem).isdigit() and 1 <= int(class_sem) <= 8:
        return int(class_sem)
    return 7

from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    Response,
    jsonify,
)

from backend.models.teacher import Teacher
from backend.models.student import Student
from backend.models.class_model import ClassModel
from backend.models.subject import Subject

from backend.services.db_service import run_query
from backend.utilities.security import (
    login_required,
    hash_password,
)


# ==========================================================
# ADMIN BLUEPRINT
# ==========================================================

admin_bp = Blueprint(
    "admin",
    __name__,
    url_prefix="/admin"
)


# ==========================================================
# ADMIN DASHBOARD
# ==========================================================

@admin_bp.route("/dashboard")
@login_required(role="admin")
def dashboard():
    from config.config import Config

    # 1. Teachers
    teacher_list = Teacher.get_all() or []

    # 2. Students & Face Registration status
    students_query = """
        SELECT
            s.student_id,
            s.name,
            s.student_number,
            s.email,
            c.class_name,
            c.semester,
            c.branch,
            c.section
        FROM students s
        LEFT JOIN class_students cs ON s.student_id = cs.student_id
        LEFT JOIN classes c ON cs.class_id = c.class_id
        ORDER BY s.student_id ASC
    """
    students_raw = run_query(students_query, fetch_all=True) or []

    emb_counts_query = """
        SELECT student_id, COUNT(*) AS emb_count
        FROM face_embeddings
        GROUP BY student_id
    """
    emb_counts = {
        row["student_id"]: row["emb_count"]
        for row in (run_query(emb_counts_query, fetch_all=True) or [])
    }

    dataset_dir = os.path.join(Config.BASE_DIR, "uploads", "dataset")
    valid_exts = (".jpg", ".jpeg", ".png", ".webp", ".jfif", ".bmp")

    enrolled_count = 0
    total_images_all = 0

    student_face_data = []
    for st in students_raw:
        usn = (st.get("student_number") or "").strip()
        st_id = st.get("student_id")

        photo_count = 0
        if usn and os.path.isdir(dataset_dir):
            usn_dir = os.path.join(dataset_dir, usn)
            if os.path.isdir(usn_dir):
                try:
                    photos = [f for f in os.listdir(usn_dir) if f.lower().endswith(valid_exts)]
                    photo_count = len(photos)
                except Exception:
                    photo_count = 0

        total_images_all += photo_count
        has_embedding = (emb_counts.get(st_id, 0) > 0)
        if photo_count > 0 or has_embedding:
            enrolled_count += 1
            status = "Registered"
            badge_class = "success"
        else:
            status = "Pending Photos"
            badge_class = "warning"

        semester = st.get("semester") or 7
        branch = (st.get("branch") or "AIML").strip().upper()
        year = (int(semester) + 1) // 2 if str(semester).isdigit() else 4

        student_face_data.append({
            "student_id": st_id,
            "name": st.get("name") or "N/A",
            "usn": usn or "N/A",
            "class_name": st.get("class_name") or f"Sem {semester} - {branch}",
            "semester": semester,
            "branch": branch,
            "year": year,
            "photo_count": photo_count,
            "embedding_count": emb_counts.get(st_id, 0),
            "status": status,
            "badge_class": badge_class
        })

    # 3. Classes
    class_list = ClassModel.get_all() or []

    # 4. Subjects
    raw_subjects = Subject.get_all() or []
    subject_list = []
    for sub in raw_subjects:
        s_code = sub.get("subject_code") or ""
        s_sem = get_subject_semester(s_code, sub.get("semester"))
        s_branch = sub.get("branch") or "AIML"
        s_year = (int(s_sem) + 1) // 2 if str(s_sem).isdigit() else 4
        s_dict = dict(sub)
        s_dict["semester"] = s_sem
        s_dict["branch"] = s_branch
        s_dict["year"] = s_year
        subject_list.append(s_dict)

    # 4b. Teacher Assignments
    assignments_query = """
        SELECT
            ta.assignment_id,
            ta.teacher_id,
            t.name AS teacher_name,
            t.email AS teacher_email,
            ta.branch,
            ta.year,
            ta.semester,
            ta.subject_id,
            sub.subject_code,
            sub.subject_name
        FROM teacher_assignments ta
        JOIN teachers t ON ta.teacher_id = t.teacher_id
        JOIN subjects sub ON ta.subject_id = sub.subject_id
        ORDER BY ta.assignment_id DESC
    """
    teacher_assignments = run_query(assignments_query, fetch_all=True) or []

    # 5. Attendance Records (all recent logs with full branch & sem resolution)
    att_query = """
        SELECT
            a.attendance_id,
            a.status,
            a.check_in_time,
            s.name AS student_name,
            s.student_number AS usn,
            sub.subject_name,
            sub.subject_code,
            COALESCE(c.class_name, c_sub.class_name, c_st.class_name) AS class_name,
            COALESCE(c.semester, c_sub.semester, c_st.semester, 7) AS semester,
            COALESCE(c.branch, c_sub.branch, c_st.branch, 'AIML') AS branch,
            COALESCE(c.section, c_sub.section, c_st.section, 'A') AS section,
            a.video_id
        FROM attendance a
        JOIN students s ON a.student_id = s.student_id
        LEFT JOIN subjects sub ON a.subject_id = sub.subject_id
        LEFT JOIN classes c ON a.class_id = c.class_id
        LEFT JOIN classes c_sub ON sub.class_id = c_sub.class_id
        LEFT JOIN class_students cs ON s.student_id = cs.student_id
        LEFT JOIN classes c_st ON cs.class_id = c_st.class_id
        ORDER BY a.check_in_time DESC
        LIMIT 1500
    """
    attendance_records = run_query(att_query, fetch_all=True) or []

    # Map multiple periods on the same day for a subject to Class 1, Class 2, etc.
    sessions_by_day_sub = {}
    for r in attendance_records:
        dt = r.get("check_in_time")
        d_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)[:10]
        sub_name = (r.get("subject_name") or "Subject").strip().lower()
        vid = r.get("video_id") or (dt.strftime("%H:%M") if hasattr(dt, "strftime") else "")
        key = (d_str, sub_name)
        if key not in sessions_by_day_sub:
            sessions_by_day_sub[key] = []
        if vid not in sessions_by_day_sub[key]:
            sessions_by_day_sub[key].append(vid)

    for r in attendance_records:
        dt = r.get("check_in_time")
        d_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)[:10]
        sub_name = (r.get("subject_name") or "Subject").strip().lower()
        vid = r.get("video_id") or (dt.strftime("%H:%M") if hasattr(dt, "strftime") else "")
        vids = sessions_by_day_sub.get((d_str, sub_name), [])
        session_num = (vids.index(vid) + 1) if vid in vids else 1
        r["session_label"] = f"Class {session_num}"
        sem = r.get("semester") or 7
        r["year"] = (int(sem) + 1) // 2 if str(sem).isdigit() else 4

    # Attendance overall statistics
    stats_query = """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN LOWER(status) = 'present' THEN 1 ELSE 0 END) AS present_count,
            SUM(CASE WHEN LOWER(status) = 'absent' THEN 1 ELSE 0 END) AS absent_count
        FROM attendance
    """
    att_stats = run_query(stats_query, fetch_one=True) or {}
    tot_att = int(att_stats.get("total") or 0)
    pres_att = int(att_stats.get("present_count") or 0)
    abs_att = int(att_stats.get("absent_count") or 0)
    att_rate = round((pres_att / tot_att * 100), 1) if tot_att > 0 else 0.0

    # 6. Reports & Analytics
    sub_rep_query = """
        SELECT
            s.subject_id,
            s.subject_name,
            s.subject_code,
            COUNT(a.attendance_id) AS total_count,
            SUM(CASE WHEN LOWER(a.status) = 'present' THEN 1 ELSE 0 END) AS present_count
        FROM subjects s
        LEFT JOIN attendance a ON s.subject_id = a.subject_id
        GROUP BY s.subject_id, s.subject_name, s.subject_code
        ORDER BY s.subject_name
    """
    sub_rows = run_query(sub_rep_query, fetch_all=True) or []
    subject_reports = []
    for r in sub_rows:
        tot = int(r.get("total_count") or 0)
        p = int(r.get("present_count") or 0)
        pct = round((p / tot * 100), 1) if tot > 0 else 0.0
        subject_reports.append({
            "subject_name": r.get("subject_name"),
            "subject_code": r.get("subject_code"),
            "total": tot,
            "present": p,
            "percentage": pct,
            "badge_class": "success" if pct >= 75.0 else ("warning" if pct >= 65.0 else "danger")
        })

    class_rep_query = """
        SELECT
            c.class_id,
            c.class_name,
            c.semester,
            c.branch,
            c.section,
            COUNT(a.attendance_id) AS total_count,
            SUM(CASE WHEN LOWER(a.status) = 'present' THEN 1 ELSE 0 END) AS present_count
        FROM classes c
        LEFT JOIN attendance a ON c.class_id = a.class_id
        GROUP BY c.class_id, c.class_name, c.semester, c.branch, c.section
        ORDER BY c.class_name
    """
    class_rows = run_query(class_rep_query, fetch_all=True) or []
    class_reports = []
    for r in class_rows:
        tot = int(r.get("total_count") or 0)
        p = int(r.get("present_count") or 0)
        pct = round((p / tot * 100), 1) if tot > 0 else 0.0
        class_reports.append({
            "class_name": r.get("class_name") or f"Sem {r.get('semester')} - {r.get('branch')} ({r.get('section')})",
            "total": tot,
            "present": p,
            "percentage": pct,
            "badge_class": "success" if pct >= 75.0 else "danger"
        })

    defaulters_query = """
        SELECT
            s.student_id,
            s.name,
            s.student_number AS usn,
            COUNT(a.attendance_id) AS total_classes,
            SUM(CASE WHEN LOWER(a.status) = 'present' THEN 1 ELSE 0 END) AS attended_classes
        FROM students s
        JOIN attendance a ON s.student_id = a.student_id
        GROUP BY s.student_id, s.name, s.student_number
        HAVING total_classes > 0 AND (SUM(CASE WHEN LOWER(a.status) = 'present' THEN 1 ELSE 0 END) / COUNT(a.attendance_id)) < 0.75
        ORDER BY (SUM(CASE WHEN LOWER(a.status) = 'present' THEN 1 ELSE 0 END) / COUNT(a.attendance_id)) ASC
    """
    defaulters_rows = run_query(defaulters_query, fetch_all=True) or []
    defaulters = []
    for r in defaulters_rows:
        tot = int(r.get("total_classes") or 0)
        p = int(r.get("attended_classes") or 0)
        pct = round((p / tot * 100), 1) if tot > 0 else 0.0
        defaulters.append({
            "name": r.get("name"),
            "usn": r.get("usn"),
            "total": tot,
            "attended": p,
            "percentage": pct,
            "status": "Critical (<65%)" if pct < 65.0 else "Warning (65-75%)",
            "badge_class": "danger" if pct < 65.0 else "warning"
        })

    # 7. Video Processing Logs
    vid_query = """
        SELECT
            v.video_id,
            v.video_path,
            v.status,
            v.uploaded_at,
            v.processed_at,
            v.face_count,
            v.attendance_count,
            t.name AS teacher_name,
            c.class_name,
            sub.subject_name
        FROM video_uploads v
        LEFT JOIN teachers t ON v.teacher_id = t.teacher_id
        LEFT JOIN classes c ON v.class_id = c.class_id
        LEFT JOIN subjects sub ON v.subject_id = sub.subject_id
        ORDER BY v.uploaded_at DESC
        LIMIT 50
    """
    videos = run_query(vid_query, fetch_all=True) or []

    # Combined stats
    stats = {
        "total_teachers": len(teacher_list),
        "total_students": len(students_raw),
        "total_classes": len(class_list),
        "total_subjects": len(subject_list),
        "total_attendance": tot_att,
        "present_attendance": pres_att,
        "absent_attendance": abs_att,
        "attendance_rate": att_rate,
        "enrolled_faces": enrolled_count,
        "pending_faces": max(0, len(students_raw) - enrolled_count),
        "total_face_images": total_images_all,
        "total_videos": len(videos),
        "processed_videos": sum(1 for v in videos if (v.get("status") or "").lower() == "processed"),
        "defaulters_count": len(defaulters)
    }

    initial_view = request.args.get("view", "overview").strip()

    admin_attendance_list = []
    for a in attendance_records:
        b_raw = str(a.get("branch") or "AIML").strip().upper()
        b_disp = "CSE (AIML)" if b_raw == "AIML" else b_raw
        sub_code = a.get("subject_code") or ""
        s_sem = get_subject_semester(sub_code, a.get("semester"))
        try:
            s_sem = int(s_sem)
        except Exception:
            s_sem = 7
        s_year = (s_sem + 1) // 2
        admin_attendance_list.append({
            "usn": a.get("usn") or "",
            "student_name": a.get("student_name") or "",
            "branch": b_disp,
            "semester": s_sem,
            "year": s_year,
            "subject_name": a.get("subject_name") or "Subject",
            "subject_code": sub_code,
            "status": str(a.get("status") or "present").lower(),
            "check_in_time": str(a.get("check_in_time") or "")
        })
    admin_attendance_json = json.dumps(admin_attendance_list)

    return render_template(
        "admin/dashboard.html",
        name=session.get("name"),
        stats=stats,
        teachers=teacher_list,
        students=student_face_data,
        classes=class_list,
        subjects=subject_list,
        attendance_records=attendance_records,
        admin_attendance_json=admin_attendance_json,
        subject_reports=subject_reports,
        class_reports=class_reports,
        defaulters=defaulters,
        videos=videos,
        teacher_assignments=teacher_assignments,
        initial_view=initial_view
    )


# ==========================================================
# TEACHER MANAGEMENT
# ==========================================================

@admin_bp.route("/teachers")
@login_required(role="admin")
def teachers():

    teacher_list = Teacher.get_all()

    return render_template(
        "admin/teachers.html",
        teachers=teacher_list
    )


# ==========================================================
# ADD TEACHER
# ==========================================================

@admin_bp.route(
    "/teachers/add",
    methods=["GET", "POST"]
)
@login_required(role="admin")
def add_teacher():

    if request.method == "GET":

        return render_template(
            "admin/add_teacher.html"
        )

    name = request.form.get(
        "name",
        ""
    ).strip()

    email = request.form.get(
        "email",
        ""
    ).strip()

    username = request.form.get(
        "username",
        ""
    ).strip()

    password = request.form.get(
        "password",
        ""
    )

    phone = request.form.get(
        "phone",
        ""
    ).strip()

    if (
        not name
        or not email
        or not username
        or not password
    ):

        flash(
            "Please fill in all required fields.",
            "danger"
        )

        return redirect(
            url_for("admin.add_teacher")
        )

    existing_teacher = Teacher.find_by_username(
        username
    )

    if existing_teacher:

        flash(
            "Username already exists.",
            "danger"
        )

        return redirect(
            url_for("admin.add_teacher")
        )

    password_hash = hash_password(
        password
    )

    Teacher.create(
        name=name,
        email=email,
        username=username,
        password_hash=password_hash,
        phone=phone or None
    )

    flash(
        "Teacher added successfully.",
        "success"
    )

    return redirect(
        url_for("admin.teachers")
    )


# ==========================================================
# EDIT TEACHER
# ==========================================================

@admin_bp.route(
    "/teachers/edit/<int:teacher_id>",
    methods=["GET", "POST"]
)
@login_required(role="admin")
def edit_teacher(teacher_id):

    teacher = Teacher.find_by_id(
        teacher_id
    )

    if not teacher:

        flash(
            "Teacher not found.",
            "danger"
        )

        return redirect(
            url_for("admin.teachers")
        )

    if request.method == "GET":

        return render_template(
            "admin/edit_teacher.html",
            teacher=teacher
        )

    name = request.form.get(
        "name",
        ""
    ).strip()

    email = request.form.get(
        "email",
        ""
    ).strip()

    username = request.form.get(
        "username",
        ""
    ).strip()

    phone = request.form.get(
        "phone",
        ""
    ).strip()

    password = request.form.get(
        "password",
        ""
    )

    if (
        not name
        or not email
        or not username
    ):

        flash(
            "Please fill in all required fields.",
            "danger"
        )

        return redirect(
            url_for(
                "admin.edit_teacher",
                teacher_id=teacher_id
            )
        )

    existing_teacher = Teacher.find_by_username(
        username
    )

    if (
        existing_teacher
        and existing_teacher["teacher_id"] != teacher_id
    ):

        flash(
            "Username already exists.",
            "danger"
        )

        return redirect(
            url_for(
                "admin.edit_teacher",
                teacher_id=teacher_id
            )
        )

    Teacher.update(
        teacher_id,
        name,
        email,
        username,
        phone or None
    )

    if password:

        password_hash = hash_password(
            password
        )

        Teacher.update_password(
            teacher_id,
            password_hash
        )

    flash(
        "Teacher updated successfully.",
        "success"
    )

    return redirect(
        url_for("admin.teachers")
    )


# ==========================================================
# DELETE TEACHER
# ==========================================================

@admin_bp.route(
    "/teachers/delete/<int:teacher_id>",
    methods=["POST"]
)
@login_required(role="admin")
def delete_teacher(teacher_id):

    teacher = Teacher.find_by_id(
        teacher_id
    )

    if not teacher:

        flash(
            "Teacher not found.",
            "danger"
        )

        return redirect(
            url_for("admin.teachers")
        )

    Teacher.delete(
        teacher_id
    )

    flash(
        "Teacher deleted successfully.",
        "success"
    )

    return redirect(
        url_for("admin.teachers")
    )


# ==========================================================
# TEACHER ASSIGNMENT ROUTES
# ==========================================================

@admin_bp.route("/assignments/assign", methods=["POST"])
@login_required(role="admin")
def assign_teacher_subject():
    teacher_id = request.form.get("teacher_id")
    branch = (request.form.get("branch") or "").strip().upper()
    year = request.form.get("year")
    semester = request.form.get("semester")
    subject_id = request.form.get("subject_id")

    if not teacher_id or not subject_id:
        flash("Please select both a Teacher and a Subject.", "warning")
        return redirect(url_for("admin.dashboard") + "?view=assignments")

    try:
        t_id = int(teacher_id)
        s_id = int(subject_id)
        sem_val = int(semester) if semester and str(semester).isdigit() else 7
        y_val = int(year) if year and str(year).isdigit() else ((sem_val + 1) // 2)
        b_val = branch or "AIML"

        # Check existing assignment
        existing = run_query(
            "SELECT assignment_id FROM teacher_assignments WHERE teacher_id = %s AND subject_id = %s",
            (t_id, s_id),
            fetch_one=True
        )
        if existing:
            flash("This subject is already assigned to this teacher.", "info")
        else:
            run_query(
                """
                INSERT INTO teacher_assignments (teacher_id, branch, year, semester, subject_id)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (t_id, b_val, y_val, sem_val, s_id),
                commit=True
            )
            flash("Teacher assignment added successfully!", "success")
    except Exception as e:
        flash(f"Error saving assignment: {e}", "danger")

    return redirect(url_for("admin.dashboard") + "?view=assignments")


@admin_bp.route("/assignments/delete/<int:assignment_id>", methods=["POST"])
@login_required(role="admin")
def delete_teacher_assignment(assignment_id):
    try:
        run_query("DELETE FROM teacher_assignments WHERE assignment_id = %s", (assignment_id,), commit=True)
        flash("Teacher assignment removed successfully.", "success")
    except Exception as e:
        flash(f"Error deleting assignment: {e}", "danger")

    return redirect(url_for("admin.dashboard") + "?view=assignments")


# ==========================================================
# STUDENT MANAGEMENT
# ==========================================================

@admin_bp.route("/students")
@login_required(role="admin")
def students():

    student_list = Student.get_all()

    return render_template(
        "admin/students.html",
        students=student_list
    )


# ==========================================================
# ADD STUDENT
# ==========================================================

@admin_bp.route(
    "/students/add",
    methods=["GET", "POST"]
)
@login_required(role="admin")
def add_student():

    if request.method == "GET":

        return render_template(
            "admin/add_student.html"
        )

    name = request.form.get(
        "name",
        ""
    ).strip()

    usn = request.form.get(
        "usn",
        ""
    ).strip()

    email = request.form.get(
        "email",
        ""
    ).strip()

    phone = request.form.get(
        "phone",
        ""
    ).strip()

    semester = request.form.get(
        "semester",
        ""
    ).strip()

    branch = request.form.get(
        "branch",
        ""
    ).strip()

    section = request.form.get(
        "section",
        ""
    ).strip()

    username = request.form.get(
        "username",
        ""
    ).strip()

    password = request.form.get(
        "password",
        ""
    )

    # ------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------

    if (
        not name
        or not usn
        or not email
        or not semester
        or not branch
        or not section
        or not username
        or not password
    ):

        flash(
            "Please fill in all required student details.",
            "danger"
        )

        return redirect(
            url_for("admin.add_student")
        )

    # ------------------------------------------------------
    # SEMESTER
    # ------------------------------------------------------

    try:

        semester = int(semester)

        if semester < 1 or semester > 8:
            raise ValueError

    except ValueError:

        flash(
            "Semester must be between 1 and 8.",
            "danger"
        )

        return redirect(
            url_for("admin.add_student")
        )

    # ------------------------------------------------------
    # CHECK USERNAME
    # ------------------------------------------------------

    existing_student = Student.find_by_username(
        username
    )

    if existing_student:

        flash(
            "Username already exists.",
            "danger"
        )

        return redirect(
            url_for("admin.add_student")
        )

    # ------------------------------------------------------
    # CHECK USN
    # ------------------------------------------------------

    student_list = Student.get_all()

    for student in student_list:

        if student["usn"] == usn:

            flash(
                "USN already exists.",
                "danger"
            )

            return redirect(
                url_for("admin.add_student")
            )

    # ------------------------------------------------------
    # PASSWORD
    # ------------------------------------------------------

    password_hash = hash_password(
        password
    )

    # ------------------------------------------------------
    # CREATE STUDENT
    # ------------------------------------------------------

    Student.create(
        name=name,
        usn=usn,
        email=email,
        phone=phone or None,
        semester=semester,
        branch=branch,
        section=section,
        username=username,
        password_hash=password_hash
    )

    flash(
        "Student added successfully.",
        "success"
    )

    return redirect(
        url_for("admin.students")
    )


# ==========================================================
# EDIT STUDENT
# ==========================================================

@admin_bp.route(
    "/students/edit/<int:student_id>",
    methods=["GET", "POST"]
)
@login_required(role="admin")
def edit_student(student_id):

    student = Student.find_by_id(
        student_id
    )

    if not student:

        flash(
            "Student not found.",
            "danger"
        )

        return redirect(
            url_for("admin.students")
        )

    # ------------------------------------------------------
    # GET
    # ------------------------------------------------------

    if request.method == "GET":

        return render_template(
            "admin/edit_student.html",
            student=student
        )

    # ------------------------------------------------------
    # FORM DATA
    # ------------------------------------------------------

    name = request.form.get(
        "name",
        ""
    ).strip()

    usn = request.form.get(
        "usn",
        ""
    ).strip()

    email = request.form.get(
        "email",
        ""
    ).strip()

    phone = request.form.get(
        "phone",
        ""
    ).strip()

    semester = request.form.get(
        "semester",
        ""
    ).strip()

    branch = request.form.get(
        "branch",
        ""
    ).strip()

    section = request.form.get(
        "section",
        ""
    ).strip()

    username = request.form.get(
        "username",
        ""
    ).strip()

    password = request.form.get(
        "password",
        ""
    )

    # ------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------

    if (
        not name
        or not usn
        or not email
        or not semester
        or not branch
        or not section
        or not username
    ):

        flash(
            "Please fill in all required student details.",
            "danger"
        )

        return redirect(
            url_for(
                "admin.edit_student",
                student_id=student_id
            )
        )

    # ------------------------------------------------------
    # SEMESTER
    # ------------------------------------------------------

    try:

        semester = int(semester)

        if semester < 1 or semester > 8:
            raise ValueError

    except ValueError:

        flash(
            "Semester must be between 1 and 8.",
            "danger"
        )

        return redirect(
            url_for(
                "admin.edit_student",
                student_id=student_id
            )
        )

    # ------------------------------------------------------
    # CHECK USERNAME
    # ------------------------------------------------------

    existing_student = Student.find_by_username(
        username
    )

    if (
        existing_student
        and existing_student["student_id"] != student_id
    ):

        flash(
            "Username already exists.",
            "danger"
        )

        return redirect(
            url_for(
                "admin.edit_student",
                student_id=student_id
            )
        )

    # ------------------------------------------------------
    # CHECK USN
    # ------------------------------------------------------

    student_list = Student.get_all()

    for existing in student_list:

        if (
            existing["usn"] == usn
            and existing["student_id"] != student_id
        ):

            flash(
                "USN already exists.",
                "danger"
            )

            return redirect(
                url_for(
                    "admin.edit_student",
                    student_id=student_id
                )
            )

    # ------------------------------------------------------
    # UPDATE STUDENT
    # ------------------------------------------------------

    Student.update(
        student_id,
        name,
        usn,
        email,
        phone or None,
        semester,
        branch,
        section,
        username
    )

    # ------------------------------------------------------
    # UPDATE PASSWORD
    # ------------------------------------------------------

    if password:

        password_hash = hash_password(
            password
        )

        Student.update_password(
            student_id,
            password_hash
        )

    flash(
        "Student updated successfully.",
        "success"
    )

    return redirect(
        url_for("admin.students")
    )


# ==========================================================
# DELETE STUDENT
# ==========================================================

@admin_bp.route(
    "/students/delete/<int:student_id>",
    methods=["POST"]
)
@login_required(role="admin")
def delete_student(student_id):

    student = Student.find_by_id(
        student_id
    )

    if not student:

        flash(
            "Student not found.",
            "danger"
        )

        return redirect(
            url_for("admin.students")
        )

    Student.delete(
        student_id
    )

    flash(
        "Student deleted successfully.",
        "success"
    )

    return redirect(
        url_for("admin.students")
    )


# ==========================================================
# CLASS MANAGEMENT
# ==========================================================

@admin_bp.route(
    "/classes",
    methods=["GET", "POST"]
)
@login_required(role="admin")
def classes():

    if request.method == "POST":

        class_id = request.form.get(
            "class_id",
            ""
        ).strip()

        class_name = request.form.get(
            "class_name",
            ""
        ).strip()

        semester = request.form.get(
            "semester",
            ""
        ).strip()

        branch = request.form.get(
            "branch",
            ""
        ).strip()

        section = request.form.get(
            "section",
            ""
        ).strip()

        # --------------------------------------------------
        # VALIDATION
        # --------------------------------------------------

        if (
            not class_name
            or not semester
            or not branch
            or not section
        ):

            flash(
                "Please fill in all class details.",
                "danger"
            )

            return redirect(
                url_for("admin.classes")
            )

        # --------------------------------------------------
        # SEMESTER
        # --------------------------------------------------

        try:

            semester = int(semester)

            if semester < 1 or semester > 8:
                raise ValueError

        except ValueError:

            flash(
                "Semester must be between 1 and 8.",
                "danger"
            )

            return redirect(
                url_for("admin.classes")
            )

        # --------------------------------------------------
        # UPDATE
        # --------------------------------------------------

        if class_id:

            try:

                class_id = int(class_id)

            except ValueError:

                flash(
                    "Invalid class ID.",
                    "danger"
                )

                return redirect(
                    url_for("admin.classes")
                )

            existing_class = ClassModel.find_by_id(
                class_id
            )

            if not existing_class:

                flash(
                    "Class not found.",
                    "danger"
                )

                return redirect(
                    url_for("admin.classes")
                )

            ClassModel.update(
                class_id,
                class_name,
                semester,
                branch,
                section
            )

            flash(
                "Class updated successfully.",
                "success"
            )

        # --------------------------------------------------
        # CREATE
        # --------------------------------------------------

        else:

            ClassModel.create(
                class_name,
                semester,
                branch,
                section
            )

            flash(
                "Class added successfully.",
                "success"
            )

        return redirect(
            url_for("admin.classes")
        )

    # ------------------------------------------------------
    # GET CLASSES
    # ------------------------------------------------------

    class_list = ClassModel.get_all()

    edit_id = request.args.get(
        "edit"
    )

    edit_class = None

    if edit_id:

        try:

            edit_class = ClassModel.find_by_id(
                int(edit_id)
            )

        except (ValueError, TypeError):

            edit_class = None

    return render_template(
        "admin/classes.html",
        classes=class_list,
        edit_class=edit_class
    )


# ==========================================================
# DELETE CLASS
# ==========================================================

@admin_bp.route(
    "/classes/delete/<int:class_id>",
    methods=["POST"]
)
@login_required(role="admin")
def delete_class(class_id):

    existing_class = ClassModel.find_by_id(
        class_id
    )

    if not existing_class:

        flash(
            "Class not found.",
            "danger"
        )

        return redirect(
            url_for("admin.classes")
        )

    ClassModel.delete(
        class_id
    )

    flash(
        "Class deleted successfully.",
        "success"
    )

    return redirect(
        url_for("admin.classes")
    )


# ==========================================================
# SUBJECT MANAGEMENT
# ADD + VIEW + EDIT IN ONE PAGE
# ==========================================================

@admin_bp.route(
    "/subjects",
    methods=["GET", "POST"]
)
@login_required(role="admin")
def subjects():

    # ------------------------------------------------------
    # POST
    # ------------------------------------------------------

    if request.method == "POST":

        subject_id = request.form.get(
            "subject_id",
            ""
        ).strip()

        subject_name = request.form.get(
            "subject_name",
            ""
        ).strip()

        subject_code = request.form.get(
            "subject_code",
            ""
        ).strip().upper()

        class_id = request.form.get(
            "class_id",
            ""
        ).strip()

        # --------------------------------------------------
        # VALIDATION
        # --------------------------------------------------

        if (
            not subject_name
            or not subject_code
        ):

            flash(
                "Please enter subject name and subject code.",
                "danger"
            )

            return redirect(
                url_for("admin.subjects")
            )

        # --------------------------------------------------
        # CLASS ID
        # --------------------------------------------------

        if class_id:

            try:

                class_id = int(class_id)

            except ValueError:

                flash(
                    "Invalid class selected.",
                    "danger"
                )

                return redirect(
                    url_for("admin.subjects")
                )

        else:

            class_id = None

        # --------------------------------------------------
        # UPDATE
        # --------------------------------------------------

        if subject_id:

            try:

                subject_id = int(subject_id)

            except ValueError:

                flash(
                    "Invalid subject ID.",
                    "danger"
                )

                return redirect(
                    url_for("admin.subjects")
                )

            existing_subject = Subject.find_by_id(
                subject_id
            )

            if not existing_subject:

                flash(
                    "Subject not found.",
                    "danger"
                )

                return redirect(
                    url_for("admin.subjects")
                )

            existing_code = Subject.find_by_code(
                subject_code
            )

            if (
                existing_code
                and existing_code["subject_id"] != subject_id
            ):

                flash(
                    "Subject code already exists.",
                    "danger"
                )

                return redirect(
                    url_for(
                        "admin.subjects",
                        edit=subject_id
                    )
                )

            Subject.update(
                subject_id,
                subject_name,
                subject_code,
                class_id
            )

            flash(
                "Subject updated successfully.",
                "success"
            )

        # --------------------------------------------------
        # CREATE
        # --------------------------------------------------

        else:

            existing_code = Subject.find_by_code(
                subject_code
            )

            if existing_code:

                flash(
                    "Subject code already exists.",
                    "danger"
                )

                return redirect(
                    url_for("admin.subjects")
                )

            Subject.create(
                subject_name,
                subject_code,
                class_id
            )

            flash(
                "Subject added successfully.",
                "success"
            )

        return redirect(
            url_for("admin.subjects")
        )

    # ------------------------------------------------------
    # GET SUBJECTS
    # ------------------------------------------------------

    subject_list = Subject.get_all()

    class_list = ClassModel.get_all()

    # ------------------------------------------------------
    # EDIT SUBJECT
    # ------------------------------------------------------

    edit_id = request.args.get(
        "edit"
    )

    edit_subject = None

    if edit_id:

        try:

            edit_subject = Subject.find_by_id(
                int(edit_id)
            )

        except (ValueError, TypeError):

            edit_subject = None

    return render_template(
        "admin/subjects.html",
        subjects=subject_list,
        classes=class_list,
        edit_subject=edit_subject
    )


# ==========================================================
# DELETE SUBJECT
# ==========================================================

@admin_bp.route(
    "/subjects/delete/<int:subject_id>",
    methods=["POST"]
)
@login_required(role="admin")
def delete_subject(subject_id):

    existing_subject = Subject.find_by_id(
        subject_id
    )

    if not existing_subject:

        flash(
            "Subject not found.",
            "danger"
        )

        return redirect(
            url_for("admin.subjects")
        )

    Subject.delete(
        subject_id
    )

    flash(
        "Subject deleted successfully.",
        "success"
    )

    return redirect(
        url_for("admin.subjects")
    )


# ==========================================================
# FACE REGISTRATION ACTIONS & REDIRECT
# ==========================================================

@admin_bp.route("/face-registration", methods=["GET"])
@login_required(role="admin")
def face_registration():
    return redirect(url_for("admin.dashboard", view="face-registration"))


@admin_bp.route("/face-registration/upload/<int:student_id>", methods=["POST"])
@login_required(role="admin")
def upload_student_photos(student_id):
    from config.config import Config
    from backend.services.face_service import sync_student_embeddings
    from werkzeug.utils import secure_filename

    student = run_query(
        "SELECT student_id, student_number, name FROM students WHERE student_id = %s",
        (student_id,),
        fetch_one=True
    )
    if not student:
        flash("Student not found.", "danger")
        return redirect(url_for("admin.dashboard", view="face-registration"))

    usn = (student.get("student_number") or "").strip().upper()
    if not usn:
        flash("Student does not have a valid USN.", "danger")
        return redirect(url_for("admin.dashboard", view="face-registration"))

    files = request.files.getlist("photos")
    if not files or all(f.filename == "" for f in files):
        flash("Please select at least one photo to upload.", "warning")
        return redirect(url_for("admin.dashboard", view="face-registration"))

    target_dir = os.path.join(Config.BASE_DIR, "uploads", "dataset", usn)
    os.makedirs(target_dir, exist_ok=True)

    saved = 0
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".jfif", ".bmp"}
    for f in files:
        if not f.filename:
            continue
        ext = os.path.splitext(f.filename)[1].lower()
        if ext in valid_exts:
            fname = secure_filename(f"{usn}_{int(datetime.now().timestamp())}_{saved+1}{ext}")
            f.save(os.path.join(target_dir, fname))
            saved += 1

    if saved > 0:
        try:
            sync_student_embeddings()
        except Exception:
            pass
        flash(f"Successfully uploaded {saved} photo(s) for {student.get('name')} ({usn}) and updated face recognition embeddings!", "success")
    else:
        flash("No valid image files (JPG, PNG, WEBP) were uploaded.", "warning")

    return redirect(url_for("admin.dashboard", view="face-registration"))


@admin_bp.route("/face-registration/sync", methods=["POST"])
@login_required(role="admin")
def sync_face_embeddings():
    from backend.services.face_service import sync_student_embeddings
    try:
        sync_student_embeddings()
        flash("Face recognition embeddings synchronized successfully with student dataset photos!", "success")
    except Exception as e:
        flash(f"Embeddings synchronization notice: {e}", "info")
    return redirect(url_for("admin.dashboard", view="face-registration"))


# ==========================================================
# ATTENDANCE REDIRECT
# ==========================================================

@admin_bp.route("/attendance", methods=["GET"])
@login_required(role="admin")
def attendance():
    return redirect(url_for("admin.dashboard", view="attendance"))


# ==========================================================
# REPORTS REDIRECT & EXPORT CSV
# ==========================================================

@admin_bp.route("/reports", methods=["GET"])
@login_required(role="admin")
def reports():
    return redirect(url_for("admin.dashboard", view="reports"))


def _fetch_filtered_report_data(branch="", year="", semester="", subject=""):
    query = """
        SELECT
            a.attendance_id,
            DATE(a.check_in_time) AS att_date,
            TIME(a.check_in_time) AS att_time,
            s.name AS student_name,
            s.student_number AS usn,
            COALESCE(c.branch, c_sub.branch, c_st.branch, 'AIML') AS branch,
            COALESCE(c.semester, c_sub.semester, c_st.semester, 7) AS semester,
            COALESCE(sub.subject_name, 'General') AS subject_name,
            sub.subject_code,
            a.status,
            a.video_id,
            a.check_in_time
        FROM attendance a
        JOIN students s ON a.student_id = s.student_id
        LEFT JOIN subjects sub ON a.subject_id = sub.subject_id
        LEFT JOIN classes c ON a.class_id = c.class_id
        LEFT JOIN classes c_sub ON sub.class_id = c_sub.class_id
        LEFT JOIN class_students cs ON s.student_id = cs.student_id
        LEFT JOIN classes c_st ON cs.class_id = c_st.class_id
        WHERE 1=1
    """
    params = []
    if branch:
        query += " AND UPPER(COALESCE(c.branch, c_sub.branch, c_st.branch, 'AIML')) = %s"
        params.append(branch.upper())

    if semester:
        try:
            query += " AND COALESCE(c.semester, c_sub.semester, c_st.semester, 7) = %s"
            params.append(int(semester))
        except Exception:
            pass
    elif year:
        try:
            y = int(year)
            sems = [y * 2 - 1, y * 2]
            query += " AND COALESCE(c.semester, c_sub.semester, c_st.semester, 7) IN (%s, %s)"
            params.extend(sems)
        except Exception:
            pass

    if subject and subject.lower() != "all":
        query += " AND (LOWER(sub.subject_name) LIKE %s OR LOWER(sub.subject_code) LIKE %s)"
        params.extend([f"%{subject.lower()}%", f"%{subject.lower()}%"])

    query += " ORDER BY a.check_in_time DESC, s.student_number ASC"
    records = run_query(query, params=params, fetch_all=True) or []

    # Map multiple periods on the same day for a subject to Class 1, Class 2, etc.
    sessions_by_day_sub = {}
    for r in records:
        dt = r.get("check_in_time")
        d_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)[:10]
        sub_name = (r.get("subject_name") or "Subject").strip().lower()
        vid = r.get("video_id") or (dt.strftime("%H:%M") if hasattr(dt, "strftime") else "")
        key = (d_str, sub_name)
        if key not in sessions_by_day_sub:
            sessions_by_day_sub[key] = []
        if vid not in sessions_by_day_sub[key]:
            sessions_by_day_sub[key].append(vid)

    for r in records:
        dt = r.get("check_in_time")
        d_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)[:10]
        sub_name = (r.get("subject_name") or "Subject").strip().lower()
        vid = r.get("video_id") or (dt.strftime("%H:%M") if hasattr(dt, "strftime") else "")
        vids = sessions_by_day_sub.get((d_str, sub_name), [])
        session_num = (vids.index(vid) + 1) if vid in vids else 1
        r["session_label"] = f"Class {session_num}"

    return records


@admin_bp.route("/reports/export-excel")
@login_required(role="admin")
def export_reports_excel():
    import io
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    branch = (request.args.get("branch") or "").strip()
    year = (request.args.get("year") or "").strip()
    semester = (request.args.get("semester") or "").strip()
    subject = (request.args.get("subject") or "").strip()

    records = _fetch_filtered_report_data(branch, year, semester, subject)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Attendance Report"

    headers = ["USN", "NAME", "BRANCH", "SEM", "SUBJECT", "DATE", "SESSION", "STATUS"]
    ws.append(headers)

    header_font = Font(name="Calibri", size=11, bold=True, color="000000")
    header_fill = PatternFill(start_color="F1F5F9", end_color="F1F5F9", fill_type="solid")
    thin_border = Border(
        bottom=Side(style="thin", color="CBD5E1"),
        top=Side(style="thin", color="CBD5E1"),
        left=Side(style="thin", color="CBD5E1"),
        right=Side(style="thin", color="CBD5E1")
    )

    for col_idx in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = header_font
        cell.fill = header_fill
        cell.border = thin_border
        cell.alignment = Alignment(horizontal="center", vertical="center")

    for r in records:
        status_str = "Present" if (r.get("status") or "").lower() == "present" else "Absent"
        row_data = [
            r.get("usn") or "",
            r.get("student_name") or "",
            r.get("branch") or "",
            f"Sem {r.get('semester')}",
            r.get("subject_name") or "",
            str(r.get("att_date") or ""),
            r.get("session_label") or "Class 1",
            status_str
        ]
        ws.append(row_data)
        current_row = ws.max_row
        for col_idx in range(1, len(headers) + 1):
            cell = ws.cell(row=current_row, column=col_idx)
            cell.border = thin_border
            cell.alignment = Alignment(vertical="center")

    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        col_letter = col[0].column_letter
        ws.column_dimensions[col_letter].width = max(max_len + 3, 12)

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    filename = f"Attendance_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return Response(
        buf.getvalue(),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment;filename={filename}"}
    )


@admin_bp.route("/reports/export-csv")
@login_required(role="admin")
def export_reports_csv():
    import io
    import csv

    branch = (request.args.get("branch") or "").strip()
    year = (request.args.get("year") or "").strip()
    semester = (request.args.get("semester") or "").strip()
    subject = (request.args.get("subject") or "").strip()

    records = _fetch_filtered_report_data(branch, year, semester, subject)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["USN", "NAME", "BRANCH", "SEM", "SUBJECT", "DATE", "SESSION", "STATUS"])

    for r in records:
        status_str = "Present" if (r.get("status") or "").lower() == "present" else "Absent"
        writer.writerow([
            r.get("usn") or "",
            r.get("student_name") or "",
            r.get("branch") or "",
            f"Sem {r.get('semester')}",
            r.get("subject_name") or "",
            str(r.get("att_date") or ""),
            r.get("session_label") or "Class 1",
            status_str
        ])

    filename = f"Attendance_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename={filename}"}
    )


# ==========================================================
# SETTINGS & VIDEO REDIRECTS
# ==========================================================

@admin_bp.route("/settings", methods=["GET"])
@login_required(role="admin")
def settings():
    return redirect(url_for("admin.dashboard", view="settings"))


@admin_bp.route("/videos", methods=["GET"])
@login_required(role="admin")
def video_processing():
    return redirect(url_for("admin.dashboard", view="overview"))


# ==========================================================
# ADMIN REAL-TIME AI CHATBOT ASSISTANT
# ==========================================================

@admin_bp.route("/chat", methods=["POST"])
@login_required(role="admin")
def admin_chat_api():
    """
    Real-time Admin Institutional AI Assistant grounded in live database metrics.
    Answers queries regarding:
    - Overall attendance percentage and turnout
    - Total students, teachers, classes, subjects
    - Defaulters / Attendance shortage (<75%)
    - Today's attendance summary & present/absent students
    - Branch-wise attendance and student distribution
    - Teacher assignments and workloads
    """
    data = request.get_json(silent=True) or {}
    user_query = data.get("message", "").strip()

    if not user_query:
        return jsonify({
            "success": False,
            "reply": "Please ask a question about institutional attendance, faculty, or students.",
            "plain_text": "Please ask a question about institutional attendance, faculty, or students."
        }), 400

    q = user_query.lower()

    # 1. Fetch core institutional numbers
    st_cnt = run_query("SELECT COUNT(*) AS total FROM students", fetch_one=True) or {}
    total_students = int(st_cnt.get("total", 0) or 0)

    tch_cnt = run_query("SELECT COUNT(*) AS total FROM teachers", fetch_one=True) or {}
    total_teachers = int(tch_cnt.get("total", 0) or 0)

    sub_cnt = run_query("SELECT COUNT(*) AS total FROM subjects", fetch_one=True) or {}
    total_subjects = int(sub_cnt.get("total", 0) or 0)

    cls_cnt = run_query("SELECT COUNT(*) AS total FROM classes", fetch_one=True) or {}
    total_classes = int(cls_cnt.get("total", 0) or 0)

    # Attendance overall
    att_stats = run_query("""
        SELECT 
            COUNT(*) AS total_marked,
            SUM(CASE WHEN LOWER(status) = 'present' THEN 1 ELSE 0 END) AS present_count,
            SUM(CASE WHEN LOWER(status) = 'absent' THEN 1 ELSE 0 END) AS absent_count
        FROM attendance
    """, fetch_one=True) or {}
    total_marked = int(att_stats.get("total_marked", 0) or 0)
    pres_total = int(att_stats.get("present_count", 0) or 0)
    abs_total = int(att_stats.get("absent_count", 0) or 0)
    overall_pct = round((pres_total / total_marked * 100), 1) if total_marked > 0 else 0.0

    today_str = datetime.now().strftime("%Y-%m-%d")
    latest_date_row = run_query("SELECT MAX(DATE(check_in_time)) as max_date FROM attendance", fetch_one=True) or {}
    latest_att_date = str(latest_date_row.get("max_date") or today_str)

    # INTENT 1: Defaulters / Shortage (<75%)
    if any(k in q for k in ["shortage", "defaulter", "defaulters", "below 75", "<75", "< 75", "low attendance", "at risk"]):
        defaulters_q = """
            SELECT 
                s.student_id,
                s.name,
                s.student_number AS usn,
                COUNT(a.attendance_id) AS total_sessions,
                SUM(CASE WHEN LOWER(a.status) = 'present' THEN 1 ELSE 0 END) AS attended_sessions
            FROM students s
            JOIN attendance a ON s.student_id = a.student_id
            GROUP BY s.student_id, s.name, s.student_number
            HAVING total_sessions > 0 AND (attended_sessions * 100.0 / total_sessions) < 75.0
            ORDER BY (attended_sessions * 100.0 / total_sessions) ASC
        """
        def_rows = run_query(defaulters_q, fetch_all=True) or []
        def_count = len(def_rows)

        if def_count > 0:
            sample_list = "".join([
                f"<li><strong>{r['name']}</strong> ({r['usn']}) — {round((int(r['attended_sessions'] or 0)/int(r['total_sessions'] or 1))*100, 1)}% ({r['attended_sessions']}/{r['total_sessions']} classes)</li>"
                for r in def_rows[:5]
            ])
            reply_html = f"""
            <div><strong class="text-dark"><i class="bi bi-info-circle me-1 text-primary"></i> Attendance Shortage (&lt;75% Attendance):</strong></div>
            <div class="mt-1">There are currently <strong class="text-dark">{def_count} student(s)</strong> with attendance below the 75% threshold.</div>
            <ul class="ps-3 mt-2 mb-1 small">{sample_list}</ul>
            {f'<div class="text-muted small mt-1">+ {def_count - 5} more students listed in Attendance Reports.</div>' if def_count > 5 else ''}
            """
            plain = f"There are {def_count} students with attendance below 75%. The lowest attendees include {def_rows[0]['name']} at {round((int(def_rows[0]['attended_sessions'] or 0)/int(def_rows[0]['total_sessions'] or 1))*100, 1)}%."
        else:
            reply_html = """
            <div class="text-dark"><i class="bi bi-shield-check me-1 text-primary"></i> <strong>Attendance Status Normal</strong></div>
            <div class="mt-1 small text-muted">No students currently have attendance below the 75% requirement across all logged sessions.</div>
            """
            plain = "No students currently have attendance below 75% across the institution."

        return jsonify({"success": True, "reply": reply_html, "plain_text": plain})

    # INTENT 2: Today's / Latest Session Attendance & Present / Absent
    if any(k in q for k in ["today", "present", "absent", "turnout", "latest session", "who is present", "who is absent"]):
        target_date = today_str if ("today" in q or latest_att_date == today_str) else latest_att_date
        
        day_att = run_query("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN LOWER(status) = 'present' THEN 1 ELSE 0 END) as pres,
                SUM(CASE WHEN LOWER(status) = 'absent' THEN 1 ELSE 0 END) as abs_cnt
            FROM attendance
            WHERE DATE(check_in_time) = %s
        """, (target_date,), fetch_one=True) or {}
        
        d_tot = int(day_att.get("total", 0) or 0)
        d_pres = int(day_att.get("pres", 0) or 0)
        d_abs = int(day_att.get("abs_cnt", 0) or 0)
        d_pct = round((d_pres / d_tot * 100), 1) if d_tot > 0 else 0.0

        date_lbl = "Today" if target_date == today_str else f"on {target_date}"
        reply_html = f"""
        <div><strong><i class="bi bi-calendar-event me-1 text-primary"></i> Attendance Status ({date_lbl}):</strong></div>
        <ul class="ps-3 mt-2 mb-1">
            <li>Marked Records: <strong>{d_tot}</strong></li>
            <li>Present: <strong class="text-dark">{d_pres}</strong> ({d_pct}%)</li>
            <li>Absent: <strong class="text-dark">{d_abs}</strong></li>
        </ul>
        <div class="small text-muted mt-1">Institutional turnout is at {d_pct}%.</div>
        """
        plain = f"Attendance status for {date_lbl}: {d_pres} students present, {d_abs} students absent, representing {d_pct}% turnout across {d_tot} marked records."
        return jsonify({"success": True, "reply": reply_html, "plain_text": plain})

    # INTENT 3: Overall Institution Stats / Total Students / Faculty / Subjects
    if any(k in q for k in ["total", "how many", "institution", "students", "teachers", "faculty", "overview", "count", "stats"]):
        reply_html = f"""
        <div><strong><i class="bi bi-buildings-fill me-1"></i> Institutional Overview:</strong></div>
        <ul class="ps-3 mt-2 mb-1">
            <li>Registered Students: <strong>{total_students}</strong></li>
            <li>Faculty / Teachers: <strong>{total_teachers}</strong></li>
            <li>Academic Classes: <strong>{total_classes}</strong></li>
            <li>Courses / Subjects: <strong>{total_subjects}</strong></li>
            <li>Overall Attendance: <strong class="text-primary">{overall_pct}%</strong> ({pres_total} present / {total_marked} total logs)</li>
        </ul>
        """
        plain = f"The institution currently has {total_students} registered students, {total_teachers} faculty members, {total_classes} classes, and an overall attendance rate of {overall_pct}%."
        return jsonify({"success": True, "reply": reply_html, "plain_text": plain})

    # INTENT 4: Teacher Assignments
    if any(k in q for k in ["assignment", "assigned", "assign", "workload"]):
        ta_cnt = run_query("SELECT COUNT(*) AS total FROM teacher_assignments", fetch_one=True) or {}
        ta_total = int(ta_cnt.get("total", 0) or 0)
        
        reply_html = f"""
        <div><strong><i class="bi bi-person-lines-fill me-1"></i> Teacher Assignments:</strong></div>
        <div class="mt-1 small">There are currently <strong>{ta_total} active subject assignments</strong> configured across faculty members.</div>
        <div class="small text-muted mt-1">Teachers only see their assigned subjects and classes on their dashboards.</div>
        """
        plain = f"There are {ta_total} active teacher assignments configured across the institution."
        return jsonify({"success": True, "reply": reply_html, "plain_text": plain})

    # DEFAULT / FALLBACK
    reply_html = f"""
    <div><strong>Admin AI Assistant</strong> grounded in live database metrics:</div>
    <ul class="ps-3 mt-2 mb-1 small">
        <li><strong>Enrolled Students:</strong> {total_students}</li>
        <li><strong>Faculty Members:</strong> {total_teachers}</li>
        <li><strong>Overall Attendance:</strong> {overall_pct}%</li>
    </ul>
    <div class="small text-muted mt-2">You can ask: <em>"Overall attendance"</em>, <em>"Who is absent today?"</em>, <em>"List defaulters"</em>, or <em>"Show teacher assignments"</em>.</div>
    """
    plain = f"Institutional overview: {total_students} students, {total_teachers} faculty, and {overall_pct}% overall attendance. How can I assist you with academic records?"
    return jsonify({"success": True, "reply": reply_html, "plain_text": plain})