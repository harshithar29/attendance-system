"""
backend/controllers/student_controller.py

Routes and dynamic data aggregation for the Student Portal:
- Compact Student Profile Header with photo/avatar
- Today's Attendance status & percentage (dynamic)
- Subject-wise attendance calculation with progress bars & status tiers
- Attendance warnings for subjects below minimum requirement (75%)
- Full monthly calendar data with attendance indicators
- AI Personalized Study Plan based on actual attendance deficiencies
"""

import os
import math
import json
import base64
import logging
from datetime import datetime, date, timedelta

from flask import Blueprint, render_template, session, jsonify, request, redirect, url_for, flash

from config.config import Config
from backend.models.student import Student
from backend.services.db_service import run_query
from backend.utilities.security import login_required


logger = logging.getLogger(__name__)

student_bp = Blueprint(
    "student",
    __name__,
    url_prefix="/student"
)


def get_student_avatar_b64(usn):
    """
    Searches uploads/dataset/<USN>/ for student reference photos.
    Returns a base64 encoded data URI string, or None if not found.
    """
    if not usn:
        return None

    usn_clean = str(usn).strip().upper()
    dataset_dir = os.path.join(Config.BASE_DIR, "uploads", "dataset")
    if not os.path.isdir(dataset_dir):
        return None

    target_dir = None
    try:
        for entry in os.listdir(dataset_dir):
            if entry.strip().upper() == usn_clean:
                candidate = os.path.join(dataset_dir, entry)
                if os.path.isdir(candidate):
                    target_dir = candidate
                    break
    except Exception:
        return None

    if not target_dir:
        return None

    exts = (".jpg", ".jpeg", ".png", ".webp", ".jfif", ".bmp")
    try:
        for f in os.listdir(target_dir):
            if f.lower().endswith(exts):
                fpath = os.path.join(target_dir, f)
                with open(fpath, "rb") as img_file:
                    encoded = base64.b64encode(img_file.read()).decode("utf-8")
                    mime = "image/png" if f.lower().endswith(".png") else "image/jpeg"
                    return f"data:{mime};base64,{encoded}"
    except Exception as e:
        logger.debug("Could not read student avatar photo: %s", e)

    return None


@student_bp.route("/dashboard")
@login_required(role="student")
def dashboard():
    student_id = session.get("user_id")

    # ============================================================
    # 1. STUDENT PROFILE
    # ============================================================
    student = Student.find_by_id(student_id)
    if not student:
        student = run_query(
            "SELECT * FROM students WHERE student_id = %s LIMIT 1",
            (student_id,),
            fetch_one=True
        ) or {}

    usn = (
        student.get("student_number")
        or student.get("usn")
        or student.get("roll_no")
        or "N/A"
    ).strip().upper()

    student_name = student.get("name") or session.get("name") or "Student"
    student_email = student.get("email") or session.get("email") or f"{usn.lower()}@student.local"

    # Get class and section
    class_info = run_query(
        """
        SELECT c.class_id, c.class_name, c.section, c.semester, c.branch
        FROM class_students cs
        JOIN classes c ON cs.class_id = c.class_id
        WHERE cs.student_id = %s
        LIMIT 1
        """,
        (student_id,),
        fetch_one=True
    )

    if class_info:
        sem = class_info.get("semester") or student.get("semester") or 7
        br = class_info.get("branch") or student.get("branch") or "CSE (AIML)"
        sec = class_info.get("section") or student.get("section") or "A"
        class_display = f"{sem}th Semester • {br} • Sec {sec}"
        student_class_id = class_info.get("class_id")
    else:
        sem = student.get("semester") or 7
        br = student.get("branch") or "CSE (AIML)"
        sec = student.get("section") or "A"
        class_display = f"{sem}th Semester • {br} • Sec {sec}"
        student_class_id = 8  # default class_id in schema

    # Student Avatar
    student_photo = get_student_avatar_b64(usn)
    initials = "".join([part[0] for part in student_name.split() if part][:2]).upper() or "ST"

    # ============================================================
    # 2. TODAY'S ATTENDANCE
    # ============================================================
    today_records = run_query(
        """
        SELECT
            a.attendance_id,
            a.status,
            a.check_in_time,
            s.subject_name,
            s.subject_code
        FROM attendance a
        LEFT JOIN subjects s ON a.subject_id = s.subject_id
        WHERE a.student_id = %s
          AND DATE(a.check_in_time) = CURDATE()
        ORDER BY a.check_in_time DESC
        """,
        (student_id,),
        fetch_all=True
    ) or []

    if today_records:
        today_present_count = sum(
            1 for r in today_records if (r.get("status") or "").lower() == "present"
        )
        today_total_count = len(today_records)
        today_percentage = round((today_present_count / today_total_count) * 100) if today_total_count > 0 else 0
        today_status = "Present" if today_present_count > 0 else "Absent"
        today_marked = True
        first_checkin = today_records[0].get("check_in_time")
        today_time_str = first_checkin.strftime("%I:%M %p") if first_checkin else "Today"
    else:
        today_status = "Not Marked"
        today_percentage = None
        today_present_count = 0
        today_total_count = 0
        today_marked = False
        today_time_str = "No class records yet"

    # ============================================================
    # 3. SUBJECT-WISE ATTENDANCE & ATTENDANCE WARNINGS
    # ============================================================
    # Fetch subjects associated with student's class
    subjects = []
    if student_class_id:
        subjects = run_query(
            """
            SELECT subject_id, subject_name, subject_code
            FROM subjects
            WHERE class_id = %s
            ORDER BY subject_name ASC
            """,
            (student_class_id,),
            fetch_all=True
        ) or []

    if not subjects:
        subjects = run_query(
            """
            SELECT subject_id, subject_name, subject_code
            FROM subjects
            ORDER BY subject_name ASC
            """,
            fetch_all=True
        ) or []

    subject_attendance = []
    low_attendance_subjects = []
    total_attended_overall = 0
    total_conducted_overall = 0

    for sub in subjects:
        sub_id = sub["subject_id"]
        sub_name = sub.get("subject_name") or "Subject"
        sub_code = sub.get("subject_code") or ""

        # Total sessions held for this subject
        sessions_held_row = run_query(
            """
            SELECT COUNT(DISTINCT video_id) AS total
            FROM attendance
            WHERE subject_id = %s
            """,
            (sub_id,),
            fetch_one=True
        )
        class_sessions = int(sessions_held_row.get("total") or 0) if sessions_held_row else 0

        # Student's attendance records for this subject
        student_records = run_query(
            """
            SELECT status
            FROM attendance
            WHERE student_id = %s
              AND subject_id = %s
            """,
            (student_id, sub_id),
            fetch_all=True
        ) or []

        student_marked = len(student_records)
        attended_count = sum(
            1 for r in student_records if (r.get("status") or "").lower() == "present"
        )

        total_classes = max(class_sessions, student_marked)
        total_attended_overall += attended_count
        total_conducted_overall += total_classes

        if total_classes == 0:
            pct = 100.0
            pct_display = "100%"
            color_class = "success"
            status_tier = "no_data"
            classes_needed = 0
        else:
            pct = round((attended_count / total_classes) * 100, 1)
            pct_display = f"{pct}%"

            if pct < 75.0:
                color_class = "danger" if pct < 65.0 else "warning"
                status_tier = "low"
                # Minimum consecutive classes needed to reach 75%:
                # (attended + x) / (total + x) >= 0.75 => x >= 3*total - 4*attended
                needed = int(math.ceil(3 * total_classes - 4 * attended_count))
                classes_needed = max(1, needed)
                low_attendance_subjects.append({
                    "subject_id": sub_id,
                    "subject_name": sub_name,
                    "subject_code": sub_code,
                    "percentage": pct,
                    "attended": attended_count,
                    "total": total_classes,
                    "needed": classes_needed
                })
            else:
                color_class = "success"
                status_tier = "healthy"
                classes_needed = 0

        subject_attendance.append({
            "subject_id": sub_id,
            "subject_name": sub_name,
            "subject_code": sub_code,
            "attended": attended_count,
            "total": total_classes,
            "percentage": pct,
            "pct_display": pct_display,
            "color_class": color_class,
            "status_tier": status_tier,
            "classes_needed": classes_needed
        })

    # Overall Attendance Statistics
    overall_percentage = (
        round((total_attended_overall / total_conducted_overall) * 100, 1)
        if total_conducted_overall > 0 else 100.0
    )

    # 4. ATTENDANCE WARNINGS
    is_healthy = len(low_attendance_subjects) == 0
    attendance_warnings = []
    if not is_healthy:
        for item in low_attendance_subjects:
            attendance_warnings.append({
                "subject": item["subject_name"],
                "percentage": item["percentage"],
                "message": f"Attendance is low in {item['subject_name']} ({item['percentage']}%). You need {item['needed']} more consecutive classes to reach 75%.",
                "needed": item["needed"]
            })

    # Query full catalog of subjects for cascading selectors (Branch -> Year -> Semester -> Subject)
    all_subjects_catalog = run_query(
        """
        SELECT s.subject_id, s.subject_name, s.subject_code, c.branch, c.semester,
               CASE 
                   WHEN c.semester IN (1, 2) THEN 1
                   WHEN c.semester IN (3, 4) THEN 2
                   WHEN c.semester IN (5, 6) THEN 3
                   WHEN c.semester IN (7, 8) THEN 4
                   ELSE 4
               END AS year
        FROM subjects s
        LEFT JOIN classes c ON s.class_id = c.class_id
        ORDER BY s.subject_name ASC
        """,
        fetch_all=True
    ) or []

    # ============================================================
    # 5. ATTENDANCE HISTORY RECORDS
    # ============================================================
    all_attendance_query = """
        SELECT
            a.attendance_id,
            DATE(a.check_in_time) AS attendance_date,
            a.check_in_time,
            a.status,
            s.subject_id,
            s.subject_name,
            s.subject_code,
            c.branch,
            c.semester,
            CASE 
               WHEN c.semester IN (1, 2) THEN 1
               WHEN c.semester IN (3, 4) THEN 2
               WHEN c.semester IN (5, 6) THEN 3
               WHEN c.semester IN (7, 8) THEN 4
               ELSE 4
            END AS year,
            t.teacher_id,
            t.name AS teacher_name,
            t.email AS teacher_email
        FROM attendance a
        LEFT JOIN subjects s ON a.subject_id = s.subject_id
        LEFT JOIN classes c ON s.class_id = c.class_id
        LEFT JOIN video_uploads v ON a.video_id = v.video_id
        LEFT JOIN teachers t ON v.teacher_id = t.teacher_id
        WHERE a.student_id = %s
        ORDER BY a.check_in_time DESC
    """
    all_attendance_records = run_query(
        all_attendance_query,
        (student_id,),
        fetch_all=True
    ) or []

    calendar_data = {}
    for r in all_attendance_records:
        adate = r.get("attendance_date")
        if not adate:
            continue
        date_str = adate.strftime("%Y-%m-%d") if hasattr(adate, "strftime") else str(adate)[:10]
        stat = (r.get("status") or "Present").strip()

        if date_str not in calendar_data:
            calendar_data[date_str] = {
                "date": date_str,
                "status": "Present" if stat.lower() == "present" else "Absent",
                "sessions": []
            }
        elif stat.lower() == "present":
            calendar_data[date_str]["status"] = "Present"

        t_str = ""
        if r.get("check_in_time"):
            try:
                t_str = r["check_in_time"].strftime("%I:%M %p")
            except Exception:
                t_str = str(r["check_in_time"])

        calendar_data[date_str]["sessions"].append({
            "subject": r.get("subject_name") or "Class Lecture",
            "code": r.get("subject_code") or "",
            "status": stat,
            "time": t_str
        })

    # ============================================================
    # 6. AI PERSONALIZED STUDY PLAN
    # ============================================================
    ai_priorities = []
    ai_tasks = []
    ai_weekly_goals = []

    if low_attendance_subjects:
        low_sorted = sorted(low_attendance_subjects, key=lambda x: x["percentage"])
        for idx, s in enumerate(low_sorted):
            urgency = "High" if s["percentage"] < 65.0 else "Medium"
            badge_class = "danger" if s["percentage"] < 65.0 else "warning"
            ai_priorities.append({
                "subject": s["subject_name"],
                "code": s["subject_code"],
                "percentage": s["percentage"],
                "urgency": urgency,
                "badge_class": badge_class,
                "needed": s["needed"]
            })

        # Calculate daily study time based on deficiencies
        daily_mins = min(120, 45 + (len(low_sorted) * 20))
        daily_study_time = f"{daily_mins} minutes"

        lead_sub = low_sorted[0]["subject_name"]
        second_sub = low_sorted[1]["subject_name"] if len(low_sorted) > 1 else None

        if second_sub:
            ai_recommendation_summary = f"Focus more on {lead_sub} and {second_sub} this week."
        else:
            ai_recommendation_summary = f"Focus more on {lead_sub} this week."

        # Generate action-oriented AI study tasks
        for idx, s in enumerate(low_sorted[:3]):
            sname = s["subject_name"]
            ai_tasks.append({
                "id": idx * 2 + 1,
                "task": f"Study {sname} for 45 minutes today and review Unit lecture notes.",
                "subject": sname,
                "duration": "45 mins",
                "urgency": "High" if s["percentage"] < 65.0 else "Medium"
            })
            ai_tasks.append({
                "id": idx * 2 + 2,
                "task": f"Revise {sname} key concepts and practice 5 textbook problem questions.",
                "subject": sname,
                "duration": "30 mins",
                "urgency": "Medium"
            })

        # Generate weekly goals
        ai_weekly_goals = [
            f"Attend all upcoming {lead_sub} classes to quickly restore attendance momentum.",
            f"Dedicate at least {round(daily_mins * 5 / 60, 1)} hours this week toward prioritized subjects.",
            f"Review and clear doubts with classmates or teacher for missed {lead_sub} topics.",
            "Complete a 15-minute self-test on Friday to measure topic retention."
        ]
    else:
        # Healthy attendance profile
        daily_study_time = "45 minutes"
        ai_recommendation_summary = "Your attendance is strong! Focus on consistent concept mastery and revision."

        top_subjects = [s["subject_name"] for s in subject_attendance[:3]]
        for idx, sname in enumerate(top_subjects):
            ai_priorities.append({
                "subject": sname,
                "code": "",
                "percentage": 90.0,
                "urgency": "Standard",
                "badge_class": "success",
                "needed": 0
            })
            ai_tasks.append({
                "id": idx + 1,
                "task": f"Review advanced topics and practical exercises in {sname}.",
                "subject": sname,
                "duration": "30 mins",
                "urgency": "Normal"
            })

        ai_weekly_goals = [
            "Maintain your 85%+ attendance across all active subjects.",
            "Prepare ahead for next week's upcoming lab practicals and assignments.",
            "Form a peer study group to discuss complex questions."
        ]

    ai_study_plan = {
        "summary": ai_recommendation_summary,
        "daily_study_time": daily_study_time,
        "priorities": ai_priorities,
        "tasks": ai_tasks,
        "weekly_goals": ai_weekly_goals
    }

    # ============================================================
    # 7. WEEKLY & MONTHLY ATTENDANCE REPORTS (DISPATCHED & LIVE)
    # ============================================================
    # 7A. Published Reports from Faculty / attendance_reports
    published_reports = []
    all_teachers = run_query("SELECT teacher_id, name, email FROM teachers ORDER BY name", fetch_all=True) or []
    curr_student_name = student_name or (student.get("name") if student else session.get("name")) or "Student"
    curr_student_usn = usn or (student.get("student_number") if student else session.get("username")) or ""

    try:
        raw_reports = run_query(
            """
            SELECT ar.report_id, ar.teacher_id, ar.report_type, ar.period_label, ar.subject_name,
                   ar.class_id, ar.total_classes, ar.regular_count, ar.shortage_count, ar.student_records_json,
                   ar.remarks, ar.created_at,
                   COALESCE(t.name, 'Course Faculty') AS teacher_name,
                   COALESCE(t.email, '') AS teacher_email,
                   COALESCE(c.branch, sub_c.branch, st_c.branch, 'AIML') AS branch,
                   COALESCE(c.semester, sub_c.semester, st_c.semester, 7) AS semester
            FROM attendance_reports ar
            LEFT JOIN teachers t ON ar.teacher_id = t.teacher_id
            LEFT JOIN classes c ON ar.class_id = c.class_id
            LEFT JOIN subjects sub ON ar.subject_name LIKE CONCAT(sub.subject_name, '%%')
            LEFT JOIN classes sub_c ON sub.class_id = sub_c.class_id
            LEFT JOIN class_students cs ON cs.student_id = %s
            LEFT JOIN classes st_c ON cs.class_id = st_c.class_id
            ORDER BY ar.created_at DESC
            LIMIT 50
            """,
            (student_id,),
            fetch_all=True
        ) or []

        for r in raw_reports:
            rec_map = {}
            if r.get("student_records_json"):
                try:
                    rec_map = json.loads(r["student_records_json"])
                except Exception:
                    rec_map = {}

            if not rec_map:
                continue

            st_data = rec_map.get(str(student_id)) or rec_map.get(str(curr_student_usn))
            if not st_data:
                continue

            st_att = int(st_data.get("attended") or 0)
            st_tot = int(st_data.get("total") or r.get("total_classes") or 1)
            st_pct = float(st_data.get("percentage") or 0.0)
            st_stat = st_data.get("status") or ("Regular" if st_pct >= 75.0 else "Shortage")
            st_needed = int(st_data.get("classes_needed") or (max(1, math.ceil(3 * st_tot - 4 * st_att)) if st_pct < 75.0 else 0))

            c_at = r.get("created_at")
            time_display = c_at.strftime("%d %b %Y, %I:%M %p") if hasattr(c_at, "strftime") else str(c_at or "Recently")
            sem_num = int(r.get("semester") or 7)
            year_num = max(1, (sem_num + 1) // 2)

            published_reports.append({
                "report_id": r.get("report_id"),
                "report_type": r.get("report_type") or "weekly",
                "period_label": r.get("period_label") or "Academic Term",
                "subject_name": r.get("subject_name") or "All Subjects",
                "branch": r.get("branch") or "AIML",
                "semester": sem_num,
                "year": year_num,
                "teacher_id": r.get("teacher_id"),
                "teacher_name": r.get("teacher_name") or "Course Faculty",
                "teacher_email": r.get("teacher_email") or "",
                "student_name": curr_student_name,
                "student_usn": curr_student_usn,
                "attended": st_att,
                "total": st_tot,
                "percentage": st_pct,
                "status": st_stat,
                "classes_needed": st_needed,
                "remarks": r.get("remarks") or "",
                "created_at": time_display,
                "is_verified": True
            })
    except Exception as e:
        logger.error("Could not fetch published attendance reports: %s", e)

    # 7B. Live Real-Time Weekly Snapshot (Dynamically computed from actual DB date range)
    ref_date_row = run_query("SELECT MAX(DATE(check_in_time)) as max_d FROM attendance", fetch_one=True) or {}
    max_d = ref_date_row.get("max_d") or date.today()
    week_start = max_d - timedelta(days=6)

    week_sess_row = run_query(
        "SELECT COUNT(DISTINCT CONCAT(DATE(check_in_time), '_', COALESCE(subject_id, 0))) as total FROM attendance WHERE DATE(check_in_time) >= %s",
        (str(week_start),),
        fetch_one=True
    ) or {}
    week_tot = int(week_sess_row.get("total") or 0)
    week_att_row = run_query(
        """
        SELECT COUNT(DISTINCT CONCAT(DATE(check_in_time), '_', COALESCE(subject_id, 0))) as attended
        FROM attendance
        WHERE student_id = %s AND LOWER(status) = 'present' AND DATE(check_in_time) >= %s
        """,
        (student_id, str(week_start)),
        fetch_one=True
    ) or {}
    week_att = int(week_att_row.get("attended") or 0)
    week_pct = round((week_att / week_tot) * 100, 1) if week_tot > 0 else 100.0
    week_needed = max(1, math.ceil(3 * week_tot - 4 * week_att)) if week_pct < 75.0 and week_tot > 0 else 0

    weekly_summary = {
        "period_label": f"Weekly ({week_start.strftime('%d %b')} - {max_d.strftime('%d %b %Y')})",
        "attended": week_att,
        "total": week_tot,
        "percentage": week_pct,
        "status": "Regular" if week_pct >= 75.0 else "Shortage",
        "classes_needed": week_needed
    }

    # 7C. Live Real-Time Monthly Snapshot
    cur_month_str = max_d.strftime("%Y-%m")
    month_sess_row = run_query(
        "SELECT COUNT(DISTINCT CONCAT(DATE(check_in_time), '_', COALESCE(subject_id, 0))) as total FROM attendance WHERE DATE_FORMAT(check_in_time, '%%Y-%%m') = %s",
        (cur_month_str,),
        fetch_one=True
    ) or {}
    month_tot = int(month_sess_row.get("total") or 0)
    month_att_row = run_query(
        """
        SELECT COUNT(DISTINCT CONCAT(DATE(check_in_time), '_', COALESCE(subject_id, 0))) as attended
        FROM attendance
        WHERE student_id = %s AND LOWER(status) = 'present' AND DATE_FORMAT(check_in_time, '%%Y-%%m') = %s
        """,
        (student_id, cur_month_str),
        fetch_one=True
    ) or {}
    month_att = int(month_att_row.get("attended") or 0)
    month_pct = round((month_att / month_tot) * 100, 1) if month_tot > 0 else 100.0
    month_needed = max(1, math.ceil(3 * month_tot - 4 * month_att)) if month_pct < 75.0 and month_tot > 0 else 0

    monthly_summary = {
        "period_label": max_d.strftime("%B %Y"),
        "attended": month_att,
        "total": month_tot,
        "percentage": month_pct,
        "status": "Regular" if month_pct >= 75.0 else "Shortage",
        "classes_needed": month_needed
    }

    # Build teacher lookup per subject for robust teacher association
    sub_teacher_map = {}
    try:
        st_rows = run_query(
            """
            SELECT v.subject_id, t.teacher_id, t.name AS teacher_name, t.email AS teacher_email
            FROM video_uploads v
            JOIN teachers t ON v.teacher_id = t.teacher_id
            WHERE v.subject_id IS NOT NULL
            ORDER BY v.uploaded_at DESC
            """,
            fetch_all=True
        ) or []
        for row in st_rows:
            sid = row.get("subject_id")
            if sid and sid not in sub_teacher_map:
                sub_teacher_map[sid] = row
    except Exception as e:
        logger.warning(f"Error querying sub_teacher_map: {e}")

    default_t_name = all_teachers[0]["name"] if all_teachers else "Course Faculty"
    default_t_email = all_teachers[0]["email"] if all_teachers else ""

    history_records_clean = []
    for r in all_attendance_records:
        adate = r.get("attendance_date")
        date_str = adate.strftime("%Y-%m-%d") if hasattr(adate, "strftime") else str(adate)[:10] if adate else ""
        display_date = adate.strftime("%d %b %Y") if hasattr(adate, "strftime") else date_str
        t = r.get("check_in_time")
        t_str = t.strftime("%I:%M %p") if hasattr(t, "strftime") else ""

        t_id = r.get("teacher_id")
        t_name = r.get("teacher_name")
        t_email = r.get("teacher_email")
        if not t_name and r.get("subject_id") in sub_teacher_map:
            t_id = sub_teacher_map[r["subject_id"]].get("teacher_id")
            t_name = sub_teacher_map[r["subject_id"]].get("teacher_name")
            t_email = sub_teacher_map[r["subject_id"]].get("teacher_email")
        if not t_name:
            t_name = default_t_name
            t_email = default_t_email

        history_records_clean.append({
            "attendance_id": r.get("attendance_id"),
            "date": date_str,
            "display_date": display_date,
            "time": t_str,
            "status": (r.get("status") or "Present").capitalize(),
            "subject_id": r.get("subject_id"),
            "subject_name": r.get("subject_name") or "Subject",
            "subject_code": r.get("subject_code") or "",
            "branch": r.get("branch") or "CSE (AIML)",
            "year": r.get("year") or 4,
            "semester": r.get("semester") or 7,
            "teacher_id": t_id,
            "teacher_name": t_name,
            "teacher_email": t_email
        })

    # Extract unique months for month-wise reporting
    month_names = {
        "01": "January", "02": "February", "03": "March", "04": "April",
        "05": "May", "06": "June", "07": "July", "08": "August",
        "09": "September", "10": "October", "11": "November", "12": "December"
    }
    raw_months = sorted(list({r["date"][:7] for r in history_records_clean if r.get("date")}), reverse=True)
    available_months = []
    for ym in raw_months:
        parts = ym.split("-")
        if len(parts) == 2:
            m_label = f"{month_names.get(parts[1], parts[1])} {parts[0]}"
            available_months.append({"value": ym, "label": m_label})

    # ============================================================
    # RENDER TEMPLATE
    # ============================================================
    return render_template(
        "student/dashboard.html",
        # Backward compatibility
        name=student_name,
        student=student,
        attendance_records=all_attendance_records,
        total_attendance=len(all_attendance_records),
        present_count=total_attended_overall,
        attendance_percentage=overall_percentage,
        # Enhanced properties
        usn=usn,
        class_display=class_display,
        student_email=student_email,
        student_photo=student_photo,
        initials=initials,
        today_data={
            "status": today_status,
            "percentage": today_percentage,
            "present_count": today_present_count,
            "total_count": today_total_count,
            "marked": today_marked,
            "time_str": today_time_str,
            "records": today_records
        },
        subject_attendance=subject_attendance,
        attendance_warnings=attendance_warnings,
        attendance_warnings_json=json.dumps(attendance_warnings),
        is_healthy=is_healthy,
        calendar_data=calendar_data,
        calendar_json=json.dumps(calendar_data),
        subject_attendance_json=json.dumps(subject_attendance),
        all_subjects_catalog=all_subjects_catalog,
        all_subjects_json=json.dumps(all_subjects_catalog),
        history_records_json=json.dumps(history_records_clean),
        ai_study_plan=ai_study_plan,
        published_reports=published_reports,
        published_reports_json=json.dumps(published_reports),
        all_teachers=all_teachers,
        available_months=available_months,
        weekly_summary=weekly_summary,
        monthly_summary=monthly_summary,
        now=datetime.now()
    )


# ============================================================
# 8. UPDATE STUDENT PROFILE
# ============================================================

@student_bp.route("/update-profile", methods=["POST"])
@login_required(role="student")
def update_profile():
    """
    Allows a student to update their name, email, and password.
    """
    from backend.utilities.security import hash_password
    student_id = session.get("user_id")

    payload = request.get_json(silent=True) or request.form
    new_name = payload.get("name", "").strip()
    new_email = payload.get("email", "").strip()
    new_password = payload.get("password", "").strip()
    confirm_password = payload.get("confirm_password", "").strip()

    if not new_name or not new_email:
        msg = "Full Name and Email Address are required."
        if request.is_json:
            return jsonify({"success": False, "message": msg}), 400
        flash(msg, "danger")
        return redirect(url_for("student.dashboard"))

    if new_password:
        if new_password != confirm_password:
            msg = "New password and confirm password do not match."
            if request.is_json:
                return jsonify({"success": False, "message": msg}), 400
            flash(msg, "danger")
            return redirect(url_for("student.dashboard"))
        if len(new_password) < 4:
            msg = "New password must be at least 4 characters long."
            if request.is_json:
                return jsonify({"success": False, "message": msg}), 400
            flash(msg, "danger")
            return redirect(url_for("student.dashboard"))

    try:
        if new_password:
            pwd_hash = hash_password(new_password)
            run_query(
                "UPDATE students SET name = %s, email = %s, password_hash = %s WHERE student_id = %s",
                (new_name, new_email, pwd_hash, student_id),
                commit=True
            )
        else:
            run_query(
                "UPDATE students SET name = %s, email = %s WHERE student_id = %s",
                (new_name, new_email, student_id),
                commit=True
            )

        session["name"] = new_name
        session["email"] = new_email

        msg = "Profile updated successfully!"
        if request.is_json:
            return jsonify({"success": True, "message": msg, "name": new_name, "email": new_email})
        flash(msg, "success")
        return redirect(url_for("student.dashboard"))
    except Exception as e:
        logger.error("Failed to update student profile: %s", e)
        msg = f"Update failed: {str(e)}"
        if request.is_json:
            return jsonify({"success": False, "message": msg}), 500
        flash(msg, "danger")
        return redirect(url_for("student.dashboard"))