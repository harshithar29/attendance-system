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
import urllib.request
import urllib.error
import re
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

        sub_sem = 7
        m = re.search(r'\d', str(sub_code or ''))
        if m:
            val = int(m.group(0))
            if 1 <= val <= 8:
                sub_sem = val
        sub_year = (sub_sem + 1) // 2

        subject_attendance.append({
            "subject_id": sub_id,
            "subject_name": sub_name,
            "subject_code": sub_code,
            "semester": sub_sem,
            "year": sub_year,
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

            st_data = rec_map.get(str(student_id)) or rec_map.get(str(curr_student_usn))
            if not st_data and curr_student_usn:
                for k, v in rec_map.items():
                    if str(k).strip().upper() == str(curr_student_usn).strip().upper():
                        st_data = v
                        break
                    if isinstance(v, dict) and str(v.get("usn") or "").strip().upper() == str(curr_student_usn).strip().upper():
                        st_data = v
                        break

            if st_data:
                st_att = int(st_data.get("attended") or 0)
                st_tot = int(st_data.get("total") or r.get("total_classes") or 1)
                st_pct = float(st_data.get("percentage") or 0.0)
                st_stat = st_data.get("status") or ("Regular" if st_pct >= 75.0 else "Shortage")
                st_needed = int(st_data.get("classes_needed") or (max(1, math.ceil(3 * st_tot - 4 * st_att)) if st_pct < 75.0 else 0))
            else:
                st_att = int(present_count or 0)
                st_tot = int(r.get("total_classes") or total_conducted or 1)
                st_pct = round((st_att / st_tot * 100), 1) if st_tot > 0 else 100.0
                st_stat = "Regular" if st_pct >= 75.0 else "Shortage"
                st_needed = max(1, math.ceil(3 * st_tot - 4 * st_att)) if st_pct < 75.0 else 0

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
    if isinstance(max_d, str):
        try:
            max_d = datetime.strptime(str(max_d)[:10], "%Y-%m-%d").date()
        except Exception:
            max_d = date.today()
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
        student_branch=br,
        student_semester=sem,
        student_year=(int(sem) + 1) // 2 if str(sem).isdigit() else 4,
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


# ============================================================
# 9. AI STUDY PLANNER & COPILOT CHATBOT (OPENAI POWERED)
# ============================================================

def call_openai_chat(messages, max_tokens=1000, temperature=0.7):
    """
    Sends request to OpenAI Chat Completions API using OPENAI_API_KEY.
    Returns response text string, or None if error/quota exceeded.
    """
    api_key = getattr(Config, "OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        logger.warning("OpenAI API HTTP Error %s: %s", e.code, e.reason)
        return None
    except Exception as e:
        logger.warning("OpenAI API call failed: %s", e)
        return None


def generate_academic_study_plan_fallback(subject, branch=None, semester=None, iteration=0):
    """
    Intelligent subject-specific curriculum generator for ANY subject entered by the student.
    Dynamically crafts clear topics, daily tasks, revision activities, and exam preparation guidance.
    Uses iteration offset to guarantee diverse, non-repeating modules on every regeneration.
    """
    sub_raw = (subject or "General Engineering Course").strip()
    sub_title = sub_raw.title()
    s_lower = sub_raw.lower()
    br = (branch or "Engineering").strip()
    sem = str(semester or "7").strip()
    iter_num = int(iteration or 0)

    # Pre-built academic modules for common subjects & domains
    domain_knowledge = {
        "operating system": {
            "variants": [
                {
                    "topics": [
                        {"title": "Process Management & CPU Scheduling", "description": "Process states, PCB, IPC message passing, and CPU scheduling algorithms (FCFS, SJF, Round Robin, Multi-level queues)."},
                        {"title": "Process Synchronization & Concurrency", "description": "Critical section problem, Peterson's algorithm, counting and binary semaphores, and classic synchronization problems."},
                        {"title": "Memory Management & Paging", "description": "Logical vs physical address translation, paging tables, TLB hits/misses, and segmentation models."},
                        {"title": "Virtual Memory & Page Replacement", "description": "Demand paging, page fault handling, FIFO, LRU, Optimal page replacement, and thrashing prevention."}
                    ],
                    "dailyTasks": [
                        {"day": "Day 1 - 2", "task": "Study Process Lifecycle & CPU Scheduling: Solve 4 numerical problems on FCFS, SJF, and Round Robin Gantt charts."},
                        {"day": "Day 3 - 4", "task": "Master Synchronization: Work out Peterson's solution and semaphore implementations for Producer-Consumer."},
                        {"day": "Day 5 - 6", "task": "Deep Dive into Paging: Calculate effective memory access times and simulate LRU/FIFO page faults on reference strings."},
                        {"day": "Day 7", "task": "Full Revision & Exam Drill: Solve 5 high-weightage 10-mark questions and draw clean OS architecture diagrams."}
                    ]
                },
                {
                    "topics": [
                        {"title": "Deadlock Characterization & Prevention", "description": "Four Coffman deadlock conditions, Resource Allocation Graphs, and Deadlock Prevention strategies."},
                        {"title": "Banker's Algorithm & Deadlock Avoidance", "description": "Safety algorithm, Resource-Request algorithm, safe states vs unsafe states, and recovery strategies."},
                        {"title": "File System Interface & Directory Structure", "description": "File concepts, access methods (sequential, direct), directory structures, and file mounting."},
                        {"title": "Disk Scheduling & Mass-Storage Structure", "description": "Magnetic disks, SSD characteristics, and disk scheduling algorithms (FCFS, SSTF, SCAN, C-SCAN, LOOK)."}
                    ],
                    "dailyTasks": [
                        {"day": "Day 1 - 2", "task": "Deadlock Dynamics: Draw Resource Allocation Graphs and evaluate cycle conditions for single vs multiple resource instances."},
                        {"day": "Day 3 - 4", "task": "Banker's Algorithm Matrix Drills: Work through 3 comprehensive numerical problems with allocation, max, and need matrices."},
                        {"day": "Day 5 - 6", "task": "Storage Allocation: Compare Contiguous, Linked, and Indexed file allocation methods with disk seek times."},
                        {"day": "Day 7", "task": "Full Revision: Calculate total head movements for SSTF and SCAN on a sample disk cylinder sequence."}
                    ]
                }
            ],
            "revisionActivities": [
                [
                    "Active Recall: Draw the 5-state process transition model and explain each state change from memory.",
                    "Gantt Chart Drills: Solve 3 scheduling problems comparing average turnaround time and waiting time.",
                    "Memory Translation: Trace a 2-level paging address translation with TLB from hex address.",
                    "Summary Memory Sheet: Summarize Disk Scheduling vs CPU Scheduling formulas on a single cheat sheet."
                ],
                [
                    "Banker's Algorithm Walkthrough: Trace safety verification on paper with a 5-process, 3-resource type matrix.",
                    "Deadlock Detection Drill: Explain the difference between Deadlock Prevention and Deadlock Avoidance in bullet points.",
                    "Disk Head Movement Chart: Calculate seek distance for FCFS vs SSTF vs SCAN on 200-cylinder disks.",
                    "Inverted Page Table: Explain how hashed and inverted page tables save physical frame memory."
                ]
            ],
            "examGuidance": [
                "Always draw clear, labeled Gantt charts and state tables for scheduling and Banker's algorithm problems.",
                "State conditions for deadlock clearly (Mutual Exclusion, Hold & Wait, No Preemption, Circular Wait) with real-world examples.",
                "Allocate 15-18 minutes per 10-mark question; write crisp points rather than long unbroken paragraphs.",
                "Begin your examination with numerical questions (Scheduling/Banker's) to secure full marks early."
            ]
        },
        "network": {
            "variants": [
                {
                    "topics": [
                        {"title": "Network Models & Physical Layer", "description": "OSI 7-Layer model, TCP/IP architecture, transmission media, framing, and error detection (CRC, Hamming code)."},
                        {"title": "Data Link Layer & MAC Protocols", "description": "Flow control (Stop-and-Wait, Sliding Window, Go-Back-N, Selective Repeat) and multiple access protocols (CSMA/CD, ALOHA)."},
                        {"title": "Network Layer & Routing Protocols", "description": "IPv4/IPv6 addressing, subnetting, CIDR, Distance Vector Routing (Bellman-Ford), and Link State Routing (Dijkstra)."},
                        {"title": "Transport & Application Layers", "description": "TCP 3-way handshake, TCP vs UDP, congestion control (Leaky Bucket, Token Bucket), DNS, HTTP, and socket basics."}
                    ],
                    "dailyTasks": [
                        {"day": "Day 1 - 2", "task": "Master OSI vs TCP/IP: Write down layer-by-layer responsibilities and solve 4 CRC error detection polynomial problems."},
                        {"day": "Day 3 - 4", "task": "Subnetting & Routing: Practice IP address allocation, CIDR slash notations, and Dijkstra's algorithm paths."},
                        {"day": "Day 5 - 6", "task": "Transport Layer Protocols: Draw TCP segment header, explain 3-way handshake, and compare Go-Back-N with Selective Repeat."},
                        {"day": "Day 7", "task": "Final Revision: Review past 3 years university question papers and practice drawing clean protocol timing diagrams."}
                    ]
                },
                {
                    "topics": [
                        {"title": "Data Link Control & Error Correction", "description": "Hamming code 7-bit generation and single-bit error correction, HDLC, and Point-to-Point protocol."},
                        {"title": "Internetworking & Routing Algorithms", "description": "Hierarchical routing, BGP inter-domain routing, OSPF link-state database, and IPv4 packet fragmentation."},
                        {"title": "TCP Congestion Control & Flow Mechanics", "description": "TCP Slow Start, Congestion Avoidance, Fast Retransmit, Fast Recovery, and TCP timer management."},
                        {"title": "Application Layer & Network Security", "description": "Domain Name System (DNS) resolution hierarchy, SMTP/IMAP email architecture, TLS 1.3 handshake, and firewalls."}
                    ],
                    "dailyTasks": [
                        {"day": "Day 1 - 2", "task": "Hamming Code Drill: Calculate parity bit positions (P1, P2, P4) and verify error syndrome detection."},
                        {"day": "Day 3 - 4", "task": "IP Fragmentation: Break down a 4000-byte datagram across MTU=1500 links, computing Offset and More Fragments flags."},
                        {"day": "Day 5 - 6", "task": "Congestion Window Tracing: Plot TCP CWND growth across Slow Start and Congestion Avoidance rounds upon packet loss."},
                        {"day": "Day 7", "task": "DNS & Security Review: Trace recursive vs iterative DNS queries and write steps of TLS certificate validation."}
                    ]
                }
            ],
            "revisionActivities": [
                [
                    "Layer Recall: List the PDU (Packet Data Unit) name and protocols operating at every layer of OSI.",
                    "Subnetting Sprint: Solve 5 quick subnet mask, network address, and broadcast address numericals.",
                    "Protocol Contrast: Prepare a side-by-side comparison chart between TCP and UDP.",
                    "Routing Simulation: Trace Bellman-Ford count-to-infinity problem on a 4-node topology."
                ],
                [
                    "Header Layout Memory: Sketch IPv4 header with all 14 fields and bit lengths from memory.",
                    "Sliding Window Formula: Calculate channel utilization efficiency for Stop-and-Wait vs Go-Back-N.",
                    "Leaky Bucket vs Token Bucket: Draw operational block diagrams and contrast burst handling capacity.",
                    "Port Matching Drill: Test your memory on 10 standard application layer protocol ports."
                ]
            ],
            "examGuidance": [
                "Include neat layer diagrams and header formats with exact bit counts for TCP, UDP, and IPv4.",
                "Show all intermediate steps and calculations for CRC polynomials and subnetting questions.",
                "Highlight key differences between Distance Vector and Link State routing in a neat tabular format.",
                "Review port numbers for common protocols (HTTP 80, DNS 53, SMTP 25, HTTPS 443) for quick 2-mark answers."
            ]
        },
        "database": {
            "variants": [
                {
                    "topics": [
                        {"title": "ER Modeling & Relational Algebra", "description": "Entity-Relationship diagrams, cardinality ratios, weak entities, and relational algebra operations (Selection, Projection, Joins)."},
                        {"title": "SQL & Advanced Queries", "description": "DDL, DML, complex joins, nested subqueries, views, triggers, and integrity constraints."},
                        {"title": "Normalization & Functional Dependencies", "description": "Armstrong's axioms, closure sets, 1NF, 2NF, 3NF, BCNF decomposition, and dependency preservation."},
                        {"title": "Transactions & Concurrency Control", "description": "ACID properties, serializability, two-phase locking (2PL), deadlocks, and recovery techniques (WAL, checkpointing)."}
                    ],
                    "dailyTasks": [
                        {"day": "Day 1 - 2", "task": "ER Diagram to Relational Schema: Draw 2 complete ER diagrams and convert them into normalized tables with foreign keys."},
                        {"day": "Day 3 - 4", "task": "SQL Query Practice: Write 10 complex queries involving GROUP BY, HAVING, INNER/LEFT joins, and correlated subqueries."},
                        {"day": "Day 5 - 6", "task": "Normalization Mastery: Solve 4 normalization decomposition problems up to BCNF and test for lossless join."},
                        {"day": "Day 7", "task": "Transactions & Past Papers: Review conflict serializability precedence graphs and solve 5 university 10-markers."}
                    ]
                },
                {
                    "topics": [
                        {"title": "Extended ER & Relational Calculus", "description": "Generalization, Specialization, Aggregation, Tuple Relational Calculus (TRC), and Domain Relational Calculus (DRC)."},
                        {"title": "Advanced Normal Forms & Minimal Covers", "description": "Canonical minimal cover computation, 4NF (multi-valued dependencies), and 5NF join dependencies."},
                        {"title": "Indexing & Physical Storage", "description": "Primary vs secondary indexes, clustered indexing, B-Trees and B+ Trees insertion and search mechanics."},
                        {"title": "Advanced Concurrency & Recovery Mechanisms", "description": "Strict 2PL, Timestamp ordering protocol, Thomas' write rule, ARIES recovery algorithm, and shadow paging."}
                    ],
                    "dailyTasks": [
                        {"day": "Day 1 - 2", "task": "EER Specialization Drills: Model disjoint vs overlapping constraints and total vs partial participation."},
                        {"day": "Day 3 - 4", "task": "Minimal Cover Algorithm: Compute canonical cover F_min for complex functional dependency sets."},
                        {"day": "Day 5 - 6", "task": "B+ Tree Walkthrough: Trace node splitting and root elevation for a B+ tree of order 4 on 10 inserted keys."},
                        {"day": "Day 7", "task": "ARIES Recovery Trace: Write out Analysis, Redo, and Undo passes for crash recovery log records."}
                    ]
                }
            ],
            "revisionActivities": [
                [
                    "Precedence Graph Drills: Determine if 3 sample transaction schedules are conflict serializable.",
                    "Closure Calculation: Find the candidate keys of relational schemas given a set of functional dependencies.",
                    "SQL Syntax Flashcards: Review syntax for CREATE TABLE, ALTER, TRIGGER, and cursor operations.",
                    "ACID Explanation: Write concise definitions and real-world banking examples for Atomicity and Isolation."
                ],
                [
                    "B+ Tree vs B Tree: Write 4 core differences regarding data storage in leaf vs internal nodes.",
                    "Lossless Join Verification: Check if R1 cap R2 determines R1 or R2 using attribute closure.",
                    "Lock Compatibility Matrix: Draw shared (S) vs exclusive (X) lock matrices and explain deadlock prevention via Wait-Die.",
                    "Relational Algebra Translation: Translate 3 complex natural language requests into relational algebra expressions."
                ]
            ],
            "examGuidance": [
                "Use standard ER modeling notation (rectangles for entities, diamonds for relationships, ovals for attributes).",
                "For normalization questions, always write the candidate keys, functional dependency violations, and decomposed relations clearly.",
                "Write clean, properly indented SQL queries with clear keywords uppercase (SELECT, FROM, WHERE).",
                "Explain ACID properties with practical transaction examples to score full marks in descriptive sections."
            ]
        }
    }

    # Match domain or build dynamic synthesis
    matched = None
    for key, spec in domain_knowledge.items():
        if key in s_lower:
            matched = spec
            break

    if matched:
        variants = matched["variants"]
        var_idx = iter_num % len(variants)
        topics = variants[var_idx]["topics"]
        daily_tasks = variants[var_idx]["dailyTasks"]
        rev_acts = matched["revisionActivities"]
        revision_activities = rev_acts[iter_num % len(rev_acts)]
        exam_guidance = matched["examGuidance"]
    else:
        # Dynamic curriculum synthesis for ANY other subject with variation offset
        kw = [w.capitalize() for w in re.findall(r"[A-Za-z]+", sub_raw) if len(w) > 2]
        lead_kw = " ".join(kw[:3]) if kw else sub_title

        if iter_num % 2 == 1:
            topics = [
                {
                    "title": f"Applied Architecture & Modeling of {lead_kw}",
                    "description": f"Subsystem interaction diagrams, mathematical governing relations, and state transition models of {sub_title}."
                },
                {
                    "title": f"Algorithmic Derivations & Analytical Drills",
                    "description": f"Standard numerical derivations, step-by-step problem sets, variable tracking, and parametric benchmarks in {sub_title}."
                },
                {
                    "title": f"Optimization Strategies & System Trade-offs",
                    "description": f"Performance bottlenecks, comparative assessment against alternative architectures, failure mitigation, and best practices."
                },
                {
                    "title": f"Exam Paper Solutions & Model Answering",
                    "description": f"Timed 10-mark university question walkthroughs, model diagrams, step-by-step point structure, and final revision summary."
                }
            ]
            daily_tasks = [
                {
                    "day": "Day 1 - 2",
                    "task": f"Architectural Modeling: Draw and label the complete core architecture diagram of {sub_title} with component interactions."
                },
                {
                    "day": "Day 3 - 4",
                    "task": f"Analytical Derivations: Practice 4 high-frequency numerical problems or governing equation proofs on paper."
                },
                {
                    "day": "Day 5 - 6",
                    "task": f"Trade-off Analysis: Create a comparison matrix contrasting alternative methodologies and mitigation techniques."
                },
                {
                    "day": "Day 7",
                    "task": f"Timed Mock Test: Complete a 60-minute sprint solving 4 previous semester examination questions with clean diagrams."
                }
            ]
            revision_activities = [
                f"Architecture Trace: Reconstruct the complete structural diagram of {sub_title} entirely from memory without notes.",
                f"Formula Verification: Verify and calculate boundary conditions for the primary operational formulas of {sub_title}.",
                "Teach-Back Review: Explain the 3 biggest design trade-offs of this course to a friend in plain, intuitive English.",
                "Summary Cards: Write down 5 high-yield 2-mark definitions and units on index cards for quick review."
            ]
        else:
            topics = [
                {
                    "title": f"Foundations & Principles of {lead_kw}",
                    "description": f"Core definitions, governing principles, historical development, and prerequisite concepts of {sub_title}."
                },
                {
                    "title": f"Core Mechanisms & Methodologies in {lead_kw}",
                    "description": f"Fundamental mechanisms, step-by-step procedural logic, primary analytical formulas, and structural architectures."
                },
                {
                    "title": f"Advanced Problem Solving & Applications",
                    "description": f"Complex scenario analysis, trade-off evaluation, integration workflows, and industry-standard best practices in {sub_title}."
                },
                {
                    "title": f"High-Yield University Modules & Synthesis",
                    "description": f"Key examination topics, high-frequency numerical derivations, previous year question bank patterns, and final review."
                }
            ]
            daily_tasks = [
                {
                    "day": "Day 1 - 2",
                    "task": f"Foundational Review: Study fundamental terminology and scope of {sub_title}. Create a glossary of key terms and core concepts."
                },
                {
                    "day": "Day 3 - 4",
                    "task": f"Core Methodology Drill: Work through standard procedural workflows and practice 3-5 standard textbook problems step-by-step."
                },
                {
                    "day": "Day 5 - 6",
                    "task": f"Analytical Problem Sets: Solve complex case scenarios and create a 1-page summary sheet of all governing formulas and diagrams."
                },
                {
                    "day": "Day 7",
                    "task": f"Full Revision & Exam Sprint: Attempt previous semester examination questions under a 60-minute timed sprint."
                }
            ]
            revision_activities = [
                f"Active Recall: Write down the primary architecture and working principles of {sub_title} without referring to notes.",
                "Summary Sheet: Prepare a single-page formula and block diagram summary sheet for quick pre-exam review.",
                "Previous Exam Papers: Solve 5 high-frequency questions from previous university exams under timed conditions.",
                f"Teach-Back Technique: Explain the toughest concept in {sub_title} out loud to a peer or in your own words."
            ]

        exam_guidance = [
            "Structure answers with clear headings, neat block diagrams, and bullet points rather than dense paragraphs.",
            "Highlight high-weightage derivations and clearly explain intermediate steps and assumptions.",
            "Manage your time: Allocate 15-18 minutes per 10-mark question during the examination.",
            "Begin with your strongest subject module to build examination confidence and secure early marks smoothly."
        ]

    # Compatible phases for backward compatibility
    phases = [
        {"name": f"Phase {i+1}: {t['title']}", "bullets": [t["description"], f"Practice key problems and analytical diagrams for {t['title']}."]}
        for i, t in enumerate(topics)
    ]
    concepts = [t["title"] for t in topics]

    return {
        "subject": sub_title,
        "summary": f"Personalized study plan for {sub_title} ({br}, Semester {sem}). Designed to systematically master core topics, build strong exam confidence, and achieve top grades.",
        "topics": topics,
        "dailyTasks": daily_tasks,
        "revisionActivities": revision_activities,
        "examGuidance": exam_guidance,
        "phases": phases,
        "importantConcepts": concepts
    }


@student_bp.route("/ai-study-plan", methods=["POST"])
@login_required(role="student")
def generate_ai_study_plan_api():
    """
    Generates a personalized AI study plan using OpenAI API with dynamic fallback.
    Accepts ANY subject name entered by the student.
    Returns clear topics, daily tasks, revision activities, and exam preparation guidance.
    """
    data = request.get_json(silent=True) or {}
    branch = data.get("branch", "").strip()
    semester = data.get("semester", "").strip()
    subject = data.get("subject", "").strip()
    iteration = int(data.get("iteration") or 0)

    # Input validation: check for valid subject name
    if not subject or len(re.findall(r"[A-Za-z0-9]", subject)) < 2:
        return jsonify({
            "success": False,
            "message": "Please enter a valid subject name (e.g., Operating Systems, Python, Machine Learning) to generate your study plan."
        }), 400

    regen_clause = f" This is regeneration attempt #{iteration + 1}. Please provide a FRESH, alternative set of high-yield modules, distinct daily practice exercises, and different active recall drills to avoid repeating earlier plans." if iteration > 0 else ""

    prompt = f"""You are an expert university professor and academic mentor. Create a personalized, student-friendly study plan for the course "{subject}".{regen_clause}
Return ONLY a valid JSON object matching EXACTLY this structure:
{{
  "subject": "{subject}",
  "summary": "Clear, friendly 2-3 sentence overview explaining how to master this subject and achieve top marks.",
  "topics": [
    {{"title": "Specific Module/Topic 1 Name", "description": "Key concepts, core definitions, and formulas to master in this topic."}},
    {{"title": "Specific Module/Topic 2 Name", "description": "Key concepts, algorithms, and practical workflows for this topic."}},
    {{"title": "Specific Module/Topic 3 Name", "description": "Advanced mechanisms, design patterns, and analytical problem-solving."}},
    {{"title": "Specific Module/Topic 4 Name", "description": "High-yield topics, common university questions, and application areas."}}
  ],
  "dailyTasks": [
    {{"day": "Day 1 - 2", "task": "Foundations: Study core terminology, read Chapter 1-2, and practice 3 basic numerical/diagram problems."}},
    {{"day": "Day 3 - 4", "task": "Core Principles: Deep dive into standard algorithms and mechanisms, solve previous question papers."}},
    {{"day": "Day 5 - 6", "task": "Advanced Applications: Work through analytical case studies, compare methods, and create formula summary sheets."}},
    {{"day": "Day 7", "task": "Full Revision: Conduct a 60-minute mock test on high-weightage topics and review mistakes."}}
  ],
  "revisionActivities": [
    "Active Recall: Quiz yourself on key definitions and state diagrams without looking at notes.",
    "Formula & Diagram Sheet: Create a 1-page summary sheet with all essential formulas and architecture diagrams.",
    "Previous Exam Papers: Solve 5 high-frequency questions from previous university exams under timed conditions.",
    "Teach-Back Technique: Explain the toughest concept out loud to a peer or in your own words to solidify intuition."
  ],
  "examGuidance": [
    "Focus on high-weightage derivations and clearly labeled diagrams for 8-10 mark questions.",
    "Write answers with clear step-by-step points and subheadings rather than long continuous paragraphs.",
    "Time Management: Spend approximately 15-18 minutes per 10-mark question during the examination.",
    "Prioritize your strongest module first in the exam to secure early confidence and top marks."
  ]
}}
CRITICAL RULES:
- The topics, daily tasks, revision activities, and exam guidance MUST be tailored specifically to "{subject}".
- Use simple, student-friendly language.
- Do NOT include any placement, campus recruitment, or job interview sections.
- Return ONLY valid JSON with no markdown formatting or backticks."""

    messages = [
        {"role": "system", "content": "You are a university academic study planner that outputs ONLY raw JSON."},
        {"role": "user", "content": prompt}
    ]

    ai_response = call_openai_chat(messages, max_tokens=1200, temperature=0.7)
    plan = None

    if ai_response:
        try:
            cleaned = ai_response.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)
            plan = json.loads(cleaned)
        except Exception as e:
            logger.warning("Failed to parse OpenAI JSON output: %s", e)
            plan = None

    if not plan or not isinstance(plan, dict) or "topics" not in plan:
        plan = generate_academic_study_plan_fallback(subject, branch, semester, iteration)
        source = "academic_engine"
    else:
        source = "openai"
        # Ensure backward-compatible phases exist
        if "phases" not in plan and "topics" in plan:
            plan["phases"] = [
                {"name": f"Phase {i+1}: {t.get('title', '')}", "bullets": [t.get('description', '')]}
                for i, t in enumerate(plan["topics"])
            ]
        if "importantConcepts" not in plan and "topics" in plan:
            plan["importantConcepts"] = [t.get("title", "") for t in plan["topics"]]

    return jsonify({
        "success": True,
        "plan": plan,
        "source": source,
        "subject": subject,
        "branch": branch,
        "semester": semester,
        "iteration": iteration
    })


@student_bp.route("/chat", methods=["POST"])
@login_required(role="student")
def student_chat_api():
    """
    Real-time AI academic assistant chat.
    Answers queries dynamically based on the logged-in student's actual attendance records.
    Answers subject-wise attendance, total classes, attended, absences, and attendance percentage.
    Protects privacy by scoping strictly to the current logged-in student session.
    """
    student_id = session.get("user_id")
    data = request.get_json(silent=True) or {}
    user_query = data.get("message", "").strip()

    if not user_query:
        return jsonify({"success": False, "reply": "Please ask a question about your attendance or academics."}), 400

    # 1. Fetch current student identity
    student = Student.find_by_id(student_id) or {}
    sname = student.get("name") or session.get("name") or "Student"
    susn = student.get("student_number") or student.get("usn") or "N/A"
    sdept = student.get("department") or student.get("branch") or "Engineering"
    ssem = student.get("semester") or "Current"

    # 2. Query overall attendance statistics
    att_stats = run_query(
        """
        SELECT 
            COUNT(*) as total_conducted,
            SUM(CASE WHEN LOWER(status) = 'present' THEN 1 ELSE 0 END) as present_count
        FROM attendance
        WHERE student_id = %s
        """,
        (student_id,),
        fetch_one=True
    ) or {}

    total_conducted = int(att_stats.get("total_conducted") or 0)
    present_count = int(att_stats.get("present_count") or 0)
    absent_count = max(0, total_conducted - present_count)
    pct = round((present_count / total_conducted * 100), 1) if total_conducted > 0 else 100.0

    # 3. Query subject-wise breakdown for THIS student
    subjects_raw = run_query(
        """
        SELECT 
            s.subject_id,
            s.subject_name,
            s.subject_code,
            COUNT(a.attendance_id) as total_classes,
            SUM(CASE WHEN LOWER(a.status) = 'present' THEN 1 ELSE 0 END) as present_classes
        FROM subjects s
        LEFT JOIN attendance a ON s.subject_id = a.subject_id AND a.student_id = %s
        WHERE s.subject_id IN (
            SELECT DISTINCT subject_id FROM attendance WHERE student_id = %s
        ) OR s.semester = %s
        GROUP BY s.subject_id, s.subject_name, s.subject_code
        ORDER BY s.subject_name ASC
        """,
        (student_id, student_id, student.get("semester", 7)),
        fetch_all=True
    ) or []

    subject_list = []
    shortage_subs = []
    healthy_subs = []
    min_sub = None
    max_sub = None

    for row in subjects_raw:
        s_name = row.get("subject_name") or "Subject"
        s_code = row.get("subject_code") or ""
        t_cls = int(row.get("total_classes") or 0)
        p_cls = int(row.get("present_classes") or 0)
        a_cls = max(0, t_cls - p_cls)
        s_pct = round((p_cls / t_cls * 100), 1) if t_cls > 0 else 100.0

        item = {
            "name": s_name,
            "code": s_code,
            "total": t_cls,
            "present": p_cls,
            "absent": a_cls,
            "percentage": s_pct,
            "needed_for_75": max(1, math.ceil(3 * t_cls - 4 * p_cls)) if s_pct < 75.0 else 0,
            "safe_miss": max(0, math.floor((p_cls - 0.75 * t_cls) / 0.75)) if s_pct >= 75.0 and t_cls > 0 else 0
        }
        subject_list.append(item)

        if s_pct < 75.0 and t_cls > 0:
            shortage_subs.append(item)
        elif t_cls > 0:
            healthy_subs.append(item)

        if t_cls > 0:
            if min_sub is None or s_pct < min_sub["percentage"]:
                min_sub = item
            if max_sub is None or s_pct > max_sub["percentage"]:
                max_sub = item

    # Build concise context string
    sub_desc_list = [f"{s['name']}: {s['percentage']}% ({s['present']}/{s['total']} attended, {s['absent']} absent)" for s in subject_list]
    context_str = f"Student: {sname} (USN: {susn}, Branch: {sdept}, Sem: {ssem}).\n"
    context_str += f"Overall Attendance: {pct}% ({present_count} attended, {absent_count} absent out of {total_conducted} conducted classes).\n"
    context_str += "Subjects Breakdown: " + ("; ".join(sub_desc_list) if sub_desc_list else "No subject attendance records found.")

    system_prompt = f"""You are EduBot AI, a helpful, encouraging academic assistant at GSSS Institute of Engineering & Technology.
Logged-in Student Live Records:
{context_str}

Rules:
1. Base your answer strictly on the student's real attendance figures above. Never invent data or mention other students.
2. In VTU regulations, minimum required attendance to appear for semester examinations is 75% (condonation up to 65% with approved medical certificate).
3. If asked about a subject, give total classes, attended, absences, and attendance percentage.
4. Format response in clean HTML with <strong>, <em>, and <ul>/<li> where appropriate (no markdown code blocks).
5. Keep response friendly, motivating, and concise (under 130 words). If the requested information is unavailable, politely explain so."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    ai_reply = call_openai_chat(messages, max_tokens=400, temperature=0.6)

    # Dynamic intelligent fallback matching student question with exact database figures
    if not ai_reply:
        q = user_query.lower().strip()

        # A. Specific Subject Lookup
        matched_sub = None
        for s in subject_list:
            s_name_l = s["name"].lower()
            s_code_l = s["code"].lower() if s["code"] else ""
            words = [w for w in re.findall(r"[a-z0-9]+", s_name_l) if len(w) > 2]
            if (s_code_l and s_code_l in q) or (s_name_l in q):
                matched_sub = s
                break
            for w in words:
                if w in q:
                    matched_sub = s
                    break
            if matched_sub:
                break

        if matched_sub:
            s = matched_sub
            status_text = '<span class="text-success fw-bold">Regular (&ge;75%)</span>' if s["percentage"] >= 75 else '<span class="text-danger fw-bold">Shortage Warning (&lt;75%)</span>'
            advice = (
                f"You need <strong>{s['needed_for_75']}</strong> consecutive classes to reach the mandatory 75% VTU threshold."
                if s["percentage"] < 75.0 else
                f"Your attendance is healthy! You can safely miss up to <strong>{s['safe_miss']}</strong> class(es) while staying &ge; 75%."
            )
            ai_reply = f"""<div><strong>{s['name']}</strong> {f'({s["code"]})' if s['code'] else ''} Attendance:</div>
            <ul class="ps-3 mt-2 mb-1">
                <li>Total Classes Conducted: <strong>{s['total']}</strong></li>
                <li>Classes Attended: <strong>{s['present']}</strong></li>
                <li>Classes Absent: <strong>{s['absent']}</strong></li>
                <li>Attendance Percentage: <strong>{s['percentage']}%</strong> ({status_text})</li>
            </ul>
            <div class="small text-muted mt-1">{advice}</div>"""

        # B. Absence / Missed classes
        elif any(k in q for k in ["miss", "absent", "absences", "leave", "skipped"]):
            if any(k in q for k in ["can i miss", "safe to miss", "bunk", "can i take"]):
                if healthy_subs:
                    safe_list = [f"<strong>{s['name']}</strong> (can miss {s['safe_miss']} class{'es' if s['safe_miss'] != 1 else ''})" for s in healthy_subs if s['safe_miss'] > 0]
                    safe_msg = f"In subjects where you have buffer: {', '.join(safe_list)}." if safe_list else "Your attendance does not have a safe buffer to miss upcoming classes without risking shortage."
                else:
                    safe_msg = "You currently have no attendance buffer. It is strongly recommended to attend all upcoming classes."
                ai_reply = f"""<div><strong>Attendance Buffer Analysis:</strong></div>
                <div class="mt-1">{safe_msg}</div>
                <div class="small text-muted mt-1"><em>Always maintain &ge; 75% across all courses to meet VTU examination rules.</em></div>"""
            else:
                top_absences = sorted(subject_list, key=lambda x: x["absent"], reverse=True)
                abs_breakdown = ", ".join([f"{s['name']}: {s['absent']} absent" for s in top_absences[:4] if s['absent'] > 0])
                ai_reply = f"""You have been absent for <strong>{absent_count} classes</strong> out of <strong>{total_conducted} conducted lectures</strong> (Overall Attendance: <strong>{pct}%</strong>).
                {f'<br><br>Absences by subject: {abs_breakdown}.' if abs_breakdown else '<br><br>Excellent! You have 0 absences recorded.'}"""

        # C. Attended / Present classes
        elif any(k in q for k in ["how many class", "attended", "present count", "classes attended"]):
            ai_reply = f"""You have attended <strong>{present_count} out of {total_conducted} conducted lectures</strong> across all enrolled subjects, giving you an overall attendance of <strong>{pct}%</strong>.
            <div class="small text-muted mt-1">Status: <strong>{'Good Academic Standing (VTU Eligible)' if pct >= 75 else 'Attendance Shortage Alert (Below 75%)'}</strong>.</div>"""

        # D. Shortage / Warning / Detention
        elif any(k in q for k in ["shortage", "warning", "detention", "below 75", "deficit"]):
            if shortage_subs:
                items_html = "".join([f"<li><strong>{s['name']}</strong>: <span class='text-danger fw-bold'>{s['percentage']}%</span> ({s['present']}/{s['total']}) — Need <strong>{s['needed_for_75']}</strong> consecutive classes.</li>" for s in shortage_subs])
                ai_reply = f"""<div class="text-danger fw-bold">⚠️ You have {len(shortage_subs)} subject shortage warning(s):</div>
                <ul class="ps-3 mt-1 mb-1">{items_html}</ul>
                <div class="small text-muted mt-1">Please attend all upcoming classes in these subjects to recover above the 75% threshold.</div>"""
            else:
                ai_reply = f"""Great news, <strong>{sname}</strong>! You currently have <strong>no attendance shortages</strong>. All your enrolled subjects are at or above the mandatory 75% VTU threshold (Overall: <strong>{pct}%</strong>). Keep up the great consistency!"""

        # E. Best / Worst attendance subject
        elif any(k in q for k in ["lowest", "worst", "minimum", "highest", "best", "maximum"]):
            if min_sub and max_sub:
                ai_reply = f"""<div><strong>Attendance Extremes:</strong></div>
                <ul class="ps-3 mt-1 mb-1">
                    <li>Highest Attendance: <strong>{max_sub['name']}</strong> at <span class="text-success fw-bold">{max_sub['percentage']}%</span> ({max_sub['present']}/{max_sub['total']}).</li>
                    <li>Lowest Attendance: <strong>{min_sub['name']}</strong> at <span class="text-danger fw-bold">{min_sub['percentage']}%</span> ({min_sub['present']}/{min_sub['total']}).</li>
                </ul>"""
            else:
                ai_reply = f"Your current overall attendance is <strong>{pct}%</strong>. No individual subject variances recorded yet."

        # F. All subjects list / breakdown
        elif any(k in q for k in ["all subject", "breakdown", "list", "subjects", "courses"]):
            if subject_list:
                rows_html = "".join([f"<li><strong>{s['name']}</strong>: <strong>{s['percentage']}%</strong> ({s['present']}/{s['total']} attended, {s['absent']} absent)</li>" for s in subject_list])
                ai_reply = f"""<div><strong>Your Subject-Wise Attendance Breakdown:</strong></div>
                <ul class="ps-3 mt-2 mb-1">{rows_html}</ul>
                <div class="small text-muted mt-1">Aggregate: <strong>{pct}%</strong> ({present_count}/{total_conducted} total attended).</div>"""
            else:
                ai_reply = f"No subject-wise attendance logs are currently recorded for your profile. Your overall attendance is <strong>{pct}%</strong>."

        # G. VTU Rules / Exam eligibility
        elif any(k in q for k in ["vtu", "rule", "exam", "eligibility", "condonation", "medical", "allow"]):
            status_text = "<span class='text-success fw-bold'>Eligible</span>" if pct >= 75 else "<span class='text-danger fw-bold'>Detention Risk</span>"
            ai_reply = f"""<div><strong>VTU Attendance Guidelines (Clause 18OB):</strong></div>
            <ul class="ps-3 mt-1 mb-1">
                <li><strong>&ge; 75%</strong>: Fully eligible for Semester End Examinations (SEE).</li>
                <li><strong>65% - 74.9%</strong>: Condonation permitted only on genuine medical grounds with Principal approval.</li>
                <li><strong>&lt; 65%</strong>: Mandatory Course Detention. The student is not permitted to write exams.</li>
            </ul>
            <div class="small mt-1">Your overall aggregate is <strong>{pct}%</strong> ({status_text}).</div>"""

        # H. Student profile / identity
        elif any(k in q for k in ["who am i", "my name", "my usn", "my profile", "my branch", "my sem"]):
            ai_reply = f"""<div><strong>Your Student Profile:</strong></div>
            <ul class="ps-3 mt-1 mb-1">
                <li>Name: <strong>{sname}</strong></li>
                <li>USN: <strong>{susn}</strong></li>
                <li>Branch &amp; Semester: <strong>{sdept}</strong>, Semester {ssem}</li>
                <li>Conducted Lectures: <strong>{total_conducted}</strong> (Attended: <strong>{present_count}</strong>)</li>
                <li>Current Aggregate: <strong>{pct}%</strong></li>
            </ul>"""

        # I. Study Plan
        elif any(k in q for k in ["study", "plan", "how to study", "prepare", "syllabus"]):
            ai_reply = f"""You can generate a personalized roadmap by navigating to the <strong>AI Study Plan</strong> tab in the sidebar! 
            Simply type any subject name to get structured topics, a day-by-day task routine, revision activities, and exam preparation guidance."""

        # J. General polite greeting / fallback
        elif any(k in q for k in ["hi", "hello", "hey", "good morning", "good afternoon", "help"]):
            status_summary = "in good academic standing" if pct >= 75 else "facing attendance shortages"
            ai_reply = f"""Hello <strong>{sname}</strong>! I am your EduBot academic copilot at GSSS. 
            <br>Your overall attendance is currently <strong>{pct}%</strong> ({present_count}/{total_conducted} classes), and you are {status_summary}.
            <div class="mt-2 small text-muted">You can ask me questions like:
                <br>&bull; <em>"What is my attendance in Operating Systems?"</em>
                <br>&bull; <em>"How many classes did I miss?"</em>
                <br>&bull; <em>"Do I have any attendance shortages?"</em>
                <br>&bull; <em>"Can I safely miss a class?"</em>
            </div>"""

        else:
            ai_reply = f"""I checked your attendance records, <strong>{sname}</strong>. Your overall attendance is <strong>{pct}%</strong> ({present_count} of {total_conducted} classes attended). 
            <br>If you're asking about a specific subject, please mention its name (e.g. <em>"What is my attendance in {subject_list[0]['name'] if subject_list else 'Machine Learning'}?"</em>) or ask about shortages, missed classes, or VTU exam eligibility."""

    return jsonify({
        "success": True,
        "reply": ai_reply
    })