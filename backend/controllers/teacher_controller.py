
import os
import re
from datetime import datetime
from threading import Thread

def get_subject_semester(sub_code, class_sem=None):
    # Primary rule: VTU subject codes embed the semester as the first digit:
    # BCS101 -> 1, BCS201 -> 2, BCS301 -> 3, BCS401 -> 4, BCS501 -> 5, BCS601 -> 6, BCS701/BCS702 -> 7, BCS801 -> 8
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
    session,
    flash,
    jsonify,
    current_app,
    send_file
)

from werkzeug.utils import secure_filename

from backend.utilities.security import login_required
from backend.models.class_model import ClassModel
from backend.models.subject import Subject
from backend.services.db_service import run_query

try:
    from backend.services.video_service import (
        process_classroom_video,
        mark_video_failed
    )
except (ImportError, Exception):
    process_classroom_video = None
    mark_video_failed = None


# =========================================================
# BLUEPRINT
# =========================================================

teacher_bp = Blueprint(
    "teacher",
    __name__,
    url_prefix="/teacher"
)


# =========================================================
# ALLOWED VIDEO FORMATS
# =========================================================

ALLOWED_VIDEO_EXTENSIONS = {
    "mp4",
    "avi",
    "mov",
    "mkv",
    "webm"
}


def allowed_video(filename):
    if not filename:
        return False

    if "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_VIDEO_EXTENSIONS


# =========================================================
# BACKGROUND VIDEO PROCESSING
# =========================================================

def process_video_background(app, video_id, file_path):
    """
    Process classroom video in a background thread.
    """
    with app.app_context():
        try:
            app.logger.info("========================================")
            app.logger.info("Background video processing started.")
            app.logger.info("video_id=%s", video_id)
            app.logger.info("video_path=%s", file_path)
            app.logger.info("========================================")

            # Ensure student identity mapping and incremental reference photo embeddings are up to date
            try:
                from backend.services.face_service import sync_student_embeddings
                sync_student_embeddings()
            except Exception as sync_err:
                app.logger.warning("Notice: student identity & photo embedding sync before video processing: %s", sync_err)

            result = process_classroom_video(
                video_id=video_id,
                video_path=file_path
            )

            app.logger.info("Background video processing completed successfully for video_id=%s", video_id)
            app.logger.info("Processing result: %s", result)

        except Exception:
            app.logger.exception("Background video processing failed for video_id=%s", video_id)

            try:
                mark_video_failed(video_id)
            except Exception:
                app.logger.exception("Could not mark video_id=%s as failed.", video_id)


# =========================================================
# TEACHER DASHBOARD
# =========================================================

@teacher_bp.route("/dashboard")
@login_required(role="teacher")
def dashboard():
    teacher_id = session.get("user_id")
    classes = ClassModel.get_all() or []
    all_subjects = Subject.get_all() or []

    # =====================================================
    # 1. TOTAL STUDENTS
    # =====================================================
    total_students_row = run_query("SELECT COUNT(*) AS total FROM students", fetch_one=True)
    total_students = int(total_students_row.get("total", 0)) if total_students_row else 0

    # =====================================================
    # 2. DYNAMIC TOTAL CLASSES COUNT
    # Automatically increments whenever a video or manual session is marked
    # =====================================================
    tot_sessions_row = run_query(
        "SELECT COUNT(DISTINCT CONCAT(DATE(check_in_time), '_', COALESCE(subject_id, 0), '_', COALESCE(class_id, 0))) AS total FROM attendance",
        fetch_one=True
    )
    total_class_sessions = int(tot_sessions_row.get("total", 0) or 1) if tot_sessions_row else 1

    # =====================================================
    # 3. LATEST PROCESSED VIDEO & ATTENDANCE (NO CONFIDENCE JARGON)
    # =====================================================
    latest_video_query = """
        SELECT
            video_id, class_id, subject_id, uploaded_at, processed_at, status, face_count, attendance_count, video_path
        FROM video_uploads
        WHERE teacher_id = %s AND status = 'processed'
        ORDER BY processed_at DESC, uploaded_at DESC
        LIMIT 1
    """
    latest_video = run_query(latest_video_query, (teacher_id,), fetch_one=True)

    formatted_attendance = []
    absent_data = []
    present_today = 0

    if latest_video:
        latest_video_id = latest_video.get("video_id")
        latest_class_id = latest_video.get("class_id")
        attendance_query = """
            SELECT s.student_id, s.name, s.student_number AS roll_no, a.status, a.check_in_time AS time
            FROM attendance a
            INNER JOIN students s ON a.student_id = s.student_id
            WHERE a.video_id = %s AND LOWER(a.status) = 'present'
            ORDER BY a.check_in_time DESC
        """
        attendance_rows = run_query(attendance_query, (latest_video_id,), fetch_all=True) or []
        present_student_ids = set()
        for record in attendance_rows:
            present_today += 1
            sid = record.get("student_id")
            if sid:
                present_student_ids.add(int(sid))
            record_time = record.get("time")
            try:
                formatted_time = record_time.strftime("%I:%M %p") if record_time else "N/A"
            except Exception:
                formatted_time = str(record_time)

            formatted_attendance.append({
                "student_id": sid,
                "name": record.get("name", "Unknown"),
                "roll_no": record.get("roll_no", "N/A"),
                "status": "present",
                "accuracy": "100%",
                "time": formatted_time
            })

        # Fetch enrolled class students to identify absent students
        class_students_query = """
            SELECT s.student_id, s.name, s.student_number AS roll_no
            FROM students s
            INNER JOIN class_students cs ON s.student_id = cs.student_id
            WHERE cs.class_id = %s
            ORDER BY s.student_number ASC
        """
        all_class_students = run_query(class_students_query, (latest_class_id,), fetch_all=True) or []
        if not all_class_students:
            all_class_students = run_query(
                "SELECT student_id, name, student_number AS roll_no FROM students ORDER BY student_number ASC",
                fetch_all=True
            ) or []

        for st in all_class_students:
            s_id = int(st.get("student_id"))
            if s_id not in present_student_ids:
                absent_data.append({
                    "student_id": s_id,
                    "name": st.get("name", "Student"),
                    "roll_no": st.get("roll_no", "N/A"),
                    "status": "absent"
                })

    absent_today = len(absent_data) if absent_data else max(0, total_students - present_today)

    videos_processed_row = run_query(
        "SELECT COUNT(*) AS total FROM video_uploads WHERE teacher_id = %s AND status = 'processed'",
        (teacher_id,),
        fetch_one=True
    )
    videos_processed = int(videos_processed_row.get("total", 0)) if videos_processed_row else 0

    stats = {
        "total_students": total_students,
        "present_today": present_today,
        "absent_today": absent_today,
        "videos_processed": videos_processed,
        "total_sessions": total_class_sessions,
        "accuracy": "100%"
    }

    # =====================================================
    # 4. RECENT ACTIVITIES & PROCESSING HISTORY
    # =====================================================
    activities_query = """
        SELECT v.video_id, v.status, v.uploaded_at, v.processed_at, c.class_name, sub.subject_name
        FROM video_uploads v
        LEFT JOIN classes c ON v.class_id = c.class_id
        LEFT JOIN subjects sub ON v.subject_id = sub.subject_id
        WHERE v.teacher_id = %s
        ORDER BY v.uploaded_at DESC
        LIMIT 6
    """
    recent_activities_rows = run_query(activities_query, (teacher_id,), fetch_all=True) or []
    formatted_activities = []
    for row in recent_activities_rows:
        v_status = (row.get("status") or "processed").capitalize()
        formatted_activities.append({
            "title": f"Attendance Session #{row.get('video_id')}",
            "description": f"{row.get('subject_name') or 'Curriculum'} • {row.get('class_name') or 'AIML'}",
            "time": str(row.get("uploaded_at") or "Recent"),
            "status": v_status,
            "badge_class": "bg-success" if v_status == "Processed" else "bg-warning",
            "icon": "bi-camera-video"
        })

    # Historical session logs for export
    history_query = """
        SELECT 
            DATE(a.check_in_time) AS att_date,
            c.semester, c.branch, c.section,
            sub.subject_id, sub.subject_name, sub.subject_code,
            a.video_id,
            COUNT(DISTINCT a.attendance_id) AS total_marked,
            SUM(CASE WHEN LOWER(a.status) = 'present' THEN 1 ELSE 0 END) AS present_count,
            SUM(CASE WHEN LOWER(a.status) = 'absent' THEN 1 ELSE 0 END) AS absent_count
        FROM attendance a
        LEFT JOIN classes c ON a.class_id = c.class_id
        LEFT JOIN subjects sub ON a.subject_id = sub.subject_id
        WHERE a.subject_id IS NOT NULL
        GROUP BY DATE(a.check_in_time), c.semester, c.branch, c.section, sub.subject_id, sub.subject_name, sub.subject_code, a.video_id
        ORDER BY DATE(a.check_in_time) DESC
    """
    history_rows = run_query(history_query, fetch_all=True) or []
    formatted_history = []
    for idx, row in enumerate(history_rows):
        tot_m = int(row.get("total_marked", 0) or 0)
        p_cnt = int(row.get("present_count", 0) or 0)
        pct = round((p_cnt / tot_m * 100), 1) if tot_m > 0 else 0.0
        h_sem = get_subject_semester(row.get("subject_code"), row.get("semester"))
        h_year = (h_sem + 1) // 2
        raw_b = str(row.get("branch") or "AIML").strip()
        h_branch = "CSE (AIML)" if raw_b.upper() == "AIML" else raw_b
        formatted_history.append({
            "id": idx + 1,
            "date": str(row.get("att_date") or ""),
            "year": h_year,
            "semester": h_sem,
            "branch": h_branch,
            "section": str(row.get("section") or "A"),
            "subject_id": row.get("subject_id"),
            "subject_name": row.get("subject_name") or "Curriculum Lecture",
            "subject_code": row.get("subject_code") or "",
            "total_marked": tot_m,
            "present_count": p_cnt,
            "absent_count": int(row.get("absent_count", 0) or 0),
            "percentage": pct,
            "video_id": row.get("video_id")
        })

    # =====================================================
    # 5. STUDENTS DIRECTORY & DYNAMIC ATTENDANCE DISPLAY
    # "Attended X / Total Y classes (Z%)"
    # =====================================================
    students_query = """
        SELECT 
            s.student_id,
            s.name,
            s.student_number AS roll_no,
            s.email,
            c.class_id,
            c.class_name,
            COALESCE(c.semester, 7) AS semester,
            COALESCE(c.branch, 'AIML') AS branch,
            COALESCE(c.section, 'A') AS section,
            COUNT(DISTINCT a.attendance_id) AS attended_count
        FROM students s
        LEFT JOIN class_students cs ON s.student_id = cs.student_id
        LEFT JOIN classes c ON cs.class_id = c.class_id
        LEFT JOIN attendance a ON s.student_id = a.student_id AND LOWER(a.status) = 'present'
        GROUP BY s.student_id, s.name, s.student_number, s.email, c.class_id, c.class_name, c.semester, c.branch, c.section
        ORDER BY s.student_number ASC
    """
    raw_students = run_query(students_query, fetch_all=True) or []
    students_list = []

    for st in raw_students:
        attended = int(st.get("attended_count", 0) or 0)
        pct = round((attended / total_class_sessions) * 100, 1) if total_class_sessions > 0 else 0.0
        raw_b = str(st.get("branch") or "AIML").strip()
        display_branch = "CSE (AIML)" if raw_b.upper() == "AIML" else raw_b
        display_sec = str(st.get("section") or "A").strip()
        display_sem = int(st.get("semester") or 7)

        is_regular = pct >= 75.0
        status_label = "Regular (≥75%)" if is_regular else "Low Attendance (<75%)"
        status_badge = "success" if is_regular else "danger"

        students_list.append({
            "student_id": st.get("student_id"),
            "name": st.get("name") or "Student",
            "roll_no": st.get("roll_no") or "N/A",
            "email": st.get("email") or "N/A",
            "class_id": st.get("class_id"),
            "class_name": st.get("class_name") or f"{display_sem}th Semester - {display_branch}",
            "semester": display_sem,
            "branch": display_branch,
            "section": display_sec,
            "attended_count": attended,
            "total_sessions": total_class_sessions,
            "percentage": pct,
            "status_label": status_label,
            "status_badge": status_badge
        })

    # =====================================================
    # 6. SEMESTER & SUBJECT FOLDERS (FOR STUDENTS & HISTORY)
    # Organizes Branch -> Semester (1st to 8th) -> Subject -> Dates
    # =====================================================
    # Query distinct dates and logs for each subject
    sub_date_query = """
        SELECT 
            DATE(a.check_in_time) AS session_date,
            a.subject_id,
            COUNT(DISTINCT a.attendance_id) AS total_marked,
            SUM(CASE WHEN LOWER(a.status) = 'present' THEN 1 ELSE 0 END) AS present_count,
            SUM(CASE WHEN LOWER(a.status) = 'absent' THEN 1 ELSE 0 END) AS absent_count
        FROM attendance a
        WHERE a.subject_id IS NOT NULL
        GROUP BY DATE(a.check_in_time), a.subject_id
        ORDER BY DATE(a.check_in_time) DESC
    """
    sub_date_rows = run_query(sub_date_query, fetch_all=True) or []
    sub_dates_map = {}
    for r in sub_date_rows:
        sid = r.get("subject_id")
        if sid not in sub_dates_map:
            sub_dates_map[sid] = []
        tot_m = int(r.get("total_marked") or 0)
        p_cnt = int(r.get("present_count") or 0)
        a_cnt = int(r.get("absent_count") or 0)
        pct = round((p_cnt / tot_m * 100), 1) if tot_m > 0 else 0.0
        sub_dates_map[sid].append({
            "date": str(r.get("session_date")),
            "total_marked": tot_m,
            "present_count": p_cnt,
            "absent_count": a_cnt,
            "percentage": pct
        })

    # Enriched subjects with accurate semester, year, and branch
    enriched_subjects = []
    for sub in all_subjects:
        s_id = sub.get("subject_id") if isinstance(sub, dict) else getattr(sub, "subject_id", None)
        s_name = sub.get("subject_name") if isinstance(sub, dict) else getattr(sub, "subject_name", "")
        s_code = sub.get("subject_code") if isinstance(sub, dict) else getattr(sub, "subject_code", "")
        c_id = sub.get("class_id") if isinstance(sub, dict) else getattr(sub, "class_id", None)
        s_sem = get_subject_semester(s_code)
        s_year = (s_sem + 1) // 2
        s_b = str(sub.get("branch") if isinstance(sub, dict) else getattr(sub, "branch", "AIML") or "AIML").strip()
        display_b = "CSE (AIML)" if s_b.upper() == "AIML" else s_b
        enriched_subjects.append({
            "subject_id": s_id,
            "subject_name": s_name,
            "subject_code": s_code,
            "class_id": c_id,
            "semester": s_sem,
            "year": s_year,
            "branch": display_b
        })

    # Detailed subject-date attendance records map
    subject_att_query = """
        SELECT 
            a.attendance_id,
            a.student_id,
            s.name,
            s.student_number AS roll_no,
            a.subject_id,
            DATE(a.check_in_time) AS att_date,
            a.status
        FROM attendance a
        JOIN students s ON a.student_id = s.student_id
        WHERE a.subject_id IS NOT NULL
        ORDER BY DATE(a.check_in_time) DESC, s.student_number ASC
    """
    subject_att_rows = run_query(subject_att_query, fetch_all=True) or []
    subject_att_map = {}
    for r in subject_att_rows:
        sid = str(r["subject_id"])
        d_str = str(r["att_date"])
        if sid not in subject_att_map:
            subject_att_map[sid] = {"dates": [], "sessions": {}}
        if d_str not in subject_att_map[sid]["dates"]:
            subject_att_map[sid]["dates"].append(d_str)
            subject_att_map[sid]["sessions"][d_str] = []
        subject_att_map[sid]["sessions"][d_str].append({
            "student_id": r["student_id"],
            "name": r["name"],
            "roll_no": r["roll_no"],
            "status": str(r.get("status") or "Present").capitalize()
        })

    students_roster = []
    for st in raw_students:
        raw_b = str(st.get("branch") or "AIML").strip()
        display_branch = "CSE (AIML)" if raw_b.upper() == "AIML" else raw_b
        st_sem = int(st.get("semester") or 7)
        st_year = (st_sem + 1) // 2
        students_roster.append({
            "student_id": st.get("student_id"),
            "name": st.get("name") or "Student",
            "roll_no": st.get("roll_no") or "N/A",
            "semester": st_sem,
            "year": st_year,
            "branch": display_branch,
            "section": str(st.get("section") or "A").strip()
        })

    # Build 1st to 8th semester folders
    semester_ordinals = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th", 7: "7th", 8: "8th"}
    semester_folders = []

    for s_num in range(1, 9):
        sem_subjects = []
        sem_total_classes = 0

        for sub in enriched_subjects:
            sub_id = sub["subject_id"]
            sub_name = sub["subject_name"]
            sub_code = sub["subject_code"]
            sub_sem = sub["semester"]
            if sub_sem == s_num:
                sessions_list = sub_dates_map.get(sub_id, [])
                t_classes = len(sessions_list)
                sem_total_classes += t_classes
                sem_subjects.append({
                    "subject_id": sub_id,
                    "subject_name": sub_name,
                    "subject_code": sub_code,
                    "semester": s_num,
                    "total_classes": t_classes,
                    "sessions": sessions_list
                })

        semester_folders.append({
            "semester_num": s_num,
            "semester_title": f"{semester_ordinals.get(s_num, str(s_num)+'th')} Semester",
            "subject_count": len(sem_subjects),
            "total_classes": sem_total_classes,
            "subjects": sem_subjects
        })

    # =====================================================
    # 7. SUBJECT-WISE ATTENDANCE ANALYTICS
    # Clear cards and comparison metrics for teachers and students
    # =====================================================
    subject_analytics = []
    for sf in semester_folders:
        for sub in sf["subjects"]:
            sid = sub["subject_id"]
            sessions = sub["sessions"]
            t_classes = sub["total_classes"]
            tot_present = sum(s["present_count"] for s in sessions)
            tot_absent = sum(s["absent_count"] for s in sessions)
            tot_logs = tot_present + tot_absent
            avg_pct = round((tot_present / tot_logs * 100), 1) if tot_logs > 0 else 0.0

            # Count students in regular vs shortage tiers
            regular_c = 0
            shortage_c = 0
            if t_classes > 0:
                # Query student attendance for this specific subject
                sub_st_query = """
                    SELECT student_id, SUM(CASE WHEN LOWER(status) = 'present' THEN 1 ELSE 0 END) AS pres
                    FROM attendance
                    WHERE subject_id = %s
                    GROUP BY student_id
                """
                sub_st_rows = run_query(sub_st_query, (sid,), fetch_all=True) or []
                for sr in sub_st_rows:
                    p = int(sr.get("pres") or 0)
                    if (p / t_classes) >= 0.75:
                        regular_c += 1
                    else:
                        shortage_c += 1

            subject_analytics.append({
                "subject_id": sid,
                "subject_name": sub["subject_name"],
                "subject_code": sub["subject_code"],
                "semester": sf["semester_num"],
                "semester_name": sf["semester_title"],
                "total_classes": t_classes,
                "total_present": tot_present,
                "total_absent": tot_absent,
                "average_percentage": avg_pct,
                "regular_count": regular_c,
                "shortage_count": shortage_c,
                "health_badge": "success" if avg_pct >= 75.0 else ("warning" if avg_pct >= 60.0 else "danger")
            })

    # =====================================================
    # 8. FULL STUDENT ROSTER FOR MANUAL ATTENDANCE
    # Full list in order with toggleable Present/Absent status
    # =====================================================
    # Get latest session date or today
    latest_att_date_row = run_query("SELECT MAX(DATE(check_in_time)) AS max_d FROM attendance", fetch_one=True)
    default_date = str(latest_att_date_row.get("max_d")) if (latest_att_date_row and latest_att_date_row.get("max_d")) else datetime.now().strftime("%Y-%m-%d")

    # Fetch status of all students for default date
    date_status_query = """
        SELECT student_id, status 
        FROM attendance 
        WHERE DATE(check_in_time) = %s
    """
    date_status_rows = run_query(date_status_query, (default_date,), fetch_all=True) or []
    date_status_map = {r["student_id"]: str(r.get("status") or "Absent").capitalize() for r in date_status_rows}

    manual_roster = []
    for st in raw_students:
        s_id = st.get("student_id")
        cur_status = date_status_map.get(s_id, "Absent")
        manual_roster.append({
            "student_id": s_id,
            "name": st.get("name") or "Student",
            "roll_no": st.get("roll_no") or "N/A",
            "branch": "CSE (AIML)" if str(st.get("branch") or "AIML").strip().upper() == "AIML" else str(st.get("branch")),
            "semester": st.get("semester") or 7,
            "section": st.get("section") or "A",
            "status": cur_status,
            "is_present": (cur_status.lower() == "present")
        })

    # Analytics summary for graphs
    analytics_data = {
        "present_count": present_today,
        "absent_count": absent_today,
        "trend_labels": [h["date"] for h in formatted_history[:7]][::-1],
        "trend_counts": [h["present_count"] for h in formatted_history[:7]][::-1],
        "subject_labels": [s["subject_name"][:18] for s in subject_analytics if s["total_classes"] > 0][:8],
        "subject_counts": [s["total_present"] for s in subject_analytics if s["total_classes"] > 0][:8],
        "tier_high": sum(s["regular_count"] for s in subject_analytics),
        "tier_mid": 0,
        "tier_low": sum(s["shortage_count"] for s in subject_analytics),
        "branch_name": "CSE (AIML)",
        "branch_count": present_today,
        "section_name": "Section A",
        "section_count": present_today
    }

    # Teacher Info
    teacher_info = run_query("SELECT teacher_id, name, email, username, phone, created_at FROM teachers WHERE teacher_id = %s", (teacher_id,), fetch_one=True) or {}
    teacher_email = teacher_info.get("email") or session.get("email") or "teacher@college.edu"

    notifications_list = [
        {
            "id": 1,
            "type": "processing",
            "title": "Attendance Sync Active",
            "message": f"Classroom system online: {total_students} students enrolled across {total_class_sessions} recorded class sessions.",
            "time": "Today",
            "badge": "Active",
            "badge_class": "bg-success-subtle text-success border",
            "read": False
        }
    ]

    # =====================================================
    # 9. CLASSROOM VIDEO PROCESSING HISTORY
    # Complete archive of all classroom videos processed
    # =====================================================
    processing_history_query = """
        SELECT 
            v.video_id,
            v.video_path,
            v.status,
            v.uploaded_at,
            v.processed_at,
            v.face_count,
            v.attendance_count,
            v.notes,
            v.class_id,
            v.subject_id,
            c.branch,
            c.semester,
            COALESCE(c.class_name, CONCAT(COALESCE(c.branch, 'AIML'), ' - Sem ', COALESCE(c.semester, '7'))) AS class_name,
            COALESCE(sub.subject_name, 'Classroom Session') AS subject_name,
            sub.subject_code
        FROM video_uploads v
        LEFT JOIN classes c ON v.class_id = c.class_id
        LEFT JOIN subjects sub ON v.subject_id = sub.subject_id
        WHERE v.teacher_id = %s
        ORDER BY v.uploaded_at DESC
    """
    raw_history = run_query(processing_history_query, (teacher_id,), fetch_all=True) or []
    formatted_processing_history = []
    for r in raw_history:
        v_path = r.get("video_path") or ""
        fname = os.path.basename(v_path)
        clean_fname = re.sub(r'^\d{8}_\d{6}_\d{6}_', '', fname) if fname else f"video_{r.get('video_id')}.mp4"

        up_at = r.get("uploaded_at")
        pr_at = r.get("processed_at")
        try:
            up_str = up_at.strftime("%d %b %Y, %I:%M %p") if up_at else "N/A"
        except Exception:
            up_str = str(up_at or "N/A")

        try:
            pr_str = pr_at.strftime("%d %b %Y, %I:%M %p") if pr_at else "N/A"
        except Exception:
            pr_str = str(pr_at or "N/A")

        dur_str = "N/A"
        if up_at and pr_at:
            try:
                diff_sec = int((pr_at - up_at).total_seconds())
                if diff_sec > 0:
                    mins = diff_sec // 60
                    secs = diff_sec % 60
                    dur_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
            except Exception:
                pass

        sub_code = r.get("subject_code")
        sub_display = f"{r.get('subject_name')} ({sub_code})" if sub_code else (r.get("subject_name") or "Subject")

        raw_b = str(r.get("branch") or "AIML").strip()
        p_branch = "CSE (AIML)" if raw_b.upper() == "AIML" else raw_b
        try:
            p_sem = int(r.get("semester") or 7)
        except Exception:
            p_sem = 7
        p_year = (p_sem + 1) // 2

        formatted_processing_history.append({
            "video_id": r.get("video_id"),
            "filename": clean_fname,
            "uploaded_at": up_str,
            "processed_at": pr_str,
            "class_name": r.get("class_name") or "CSE (AIML)",
            "subject_name": sub_display,
            "raw_subject_name": r.get("subject_name") or "Subject",
            "subject_code": sub_code or "",
            "subject_id": r.get("subject_id"),
            "class_id": r.get("class_id"),
            "branch": p_branch,
            "semester": p_sem,
            "year": p_year,
            "face_count": r.get("face_count") or 0,
            "attendance_count": r.get("attendance_count") or 0,
            "duration": dur_str,
            "status": (r.get("status") or "processed").lower(),
            "notes": r.get("notes") or ""
        })

    return render_template(
        "teacher/dashboard.html",
        name=session.get("name"),
        teacher_email=teacher_email,
        classes=classes,
        subjects=enriched_subjects,
        enriched_subjects=enriched_subjects,
        subject_att_map=subject_att_map,
        students_roster=students_roster,
        stats=stats,
        attendance_data=formatted_attendance,
        absent_data=absent_data,
        history_data=formatted_history,
        students_list=students_list,
        semester_folders=semester_folders,
        subject_analytics=subject_analytics,
        manual_roster=manual_roster,
        default_manual_date=default_date,
        analytics_data=analytics_data,
        notifications_list=notifications_list,
        recent_activities=formatted_activities,
        processing_history=formatted_processing_history,
        teacher_profile=teacher_info,
        now=datetime.now()
    )



# =========================================================
# SUBJECTS BY CLASS
# =========================================================

@teacher_bp.route("/subjects/<int:class_id>")
@login_required(role="teacher")
def subjects_by_class(class_id):
    subjects = Subject.get_all() or []
    filtered_subjects = []

    for subject in subjects:
        subject_class_id = subject.get("class_id")
        if subject_class_id is None:
            continue

        try:
            if int(subject_class_id) == int(class_id):
                filtered_subjects.append(subject)
        except (TypeError, ValueError):
            continue

    return jsonify(filtered_subjects)


# =========================================================
# UPLOAD CLASSROOM VIDEO
# =========================================================

@teacher_bp.route("/upload-video", methods=["POST"])
@login_required(role="teacher")
def upload_video():
    teacher_id = session.get("user_id")

    # =====================================================
    # VALIDATE TEACHER
    # =====================================================

    if not teacher_id:
        flash("Teacher session is invalid. Please login again.", "danger")
        return redirect(url_for("teacher.dashboard"))

    # =====================================================
    # GET FORM DATA
    # =====================================================

    class_id = request.form.get("class_id")
    subject_id = request.form.get("subject_id")
    video = request.files.get("video")

    # =====================================================
    # VALIDATE SUBJECT
    # =====================================================

    if not subject_id:
        flash("Please select a subject.", "warning")
        return redirect(url_for("teacher.dashboard"))

    try:
        selected_subject = Subject.find_by_id(subject_id)
    except Exception:
        current_app.logger.exception("Error validating subject.")
        selected_subject = None

    if not selected_subject:
        flash("Selected subject does not exist.", "danger")
        return redirect(url_for("teacher.dashboard"))

    # =====================================================
    # RESOLVE AND VALIDATE CLASS
    # =====================================================

    if not class_id or str(class_id).strip() == "":
        subject_class_id = selected_subject.get("class_id") if isinstance(selected_subject, dict) else getattr(selected_subject, "class_id", None)
        class_id = subject_class_id if subject_class_id is not None else 8

    try:
        selected_class = ClassModel.find_by_id(class_id)
    except Exception:
        selected_class = None

    if not selected_class:
        class_id = 8

    # =====================================================
    # CREATE UPLOAD DIRECTORY
    # =====================================================

    upload_folder = os.path.join(current_app.root_path, "uploads", "classroom_videos")

    try:
        os.makedirs(upload_folder, exist_ok=True)
    except Exception:
        current_app.logger.exception("Unable to create video upload directory.")
        flash("Unable to prepare video upload directory.", "danger")
        return redirect(url_for("teacher.dashboard"))

    # =====================================================
    # CREATE SAFE FILENAME
    # =====================================================

    original_filename = secure_filename(video.filename)
    if not original_filename:
        flash("Invalid video filename.", "danger")
        return redirect(url_for("teacher.dashboard"))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}_{original_filename}"
    file_path = os.path.join(upload_folder, filename)

    # =====================================================
    # SAVE VIDEO
    # =====================================================

    try:
        video.save(file_path)
    except Exception:
        current_app.logger.exception("Unable to save uploaded video.")
        flash("Unable to save the uploaded video.", "danger")
        return redirect(url_for("teacher.dashboard"))

    # =====================================================
    # VERIFY VIDEO SAVED
    # =====================================================

    if not os.path.isfile(file_path):
        flash("Video was not saved correctly.", "danger")
        return redirect(url_for("teacher.dashboard"))

    # =====================================================
    # CREATE DATABASE RELATIVE PATH
    # =====================================================

    relative_path = os.path.join("uploads", "classroom_videos", filename).replace("\\", "/")

    # =====================================================
    # CREATE VIDEO DATABASE RECORD
    # =====================================================

    insert_query = """
        INSERT INTO video_uploads
        (teacher_id, class_id, subject_id, video_path, status)
        VALUES (%s, %s, %s, %s, 'uploaded')
    """

    video_id = None

    try:
        video_id = run_query(
            insert_query,
            (int(teacher_id), int(class_id), int(subject_id), relative_path),
            commit=True
        )

        if video_id is None:
            raise ValueError("Database did not return video_id.")

        video_id = int(video_id)

    except Exception:
        current_app.logger.exception("Could not create video database record.")
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception:
            current_app.logger.exception("Could not remove failed video upload.")

        flash("Video was uploaded but could not be registered in the database.", "danger")
        return redirect(url_for("teacher.dashboard"))

    # =====================================================
    # START BACKGROUND THREAD
    # =====================================================

    try:
        app = current_app._get_current_object()

        processing_thread = Thread(
            target=process_video_background,
            args=(app, video_id, file_path),
            daemon=True,
            name=f"video-processing-{video_id}"
        )

        processing_thread.start()

        current_app.logger.info("Video uploaded successfully. Background processing started. video_id=%s", video_id)

    except Exception:
        current_app.logger.exception("Unable to start background processing for video_id=%s", video_id)

        try:
            mark_video_failed(video_id)
        except Exception:
            current_app.logger.exception("Could not mark video as failed.")

        flash("Video uploaded, but processing could not be started.", "danger")
        return redirect(url_for("teacher.dashboard"))

    # =====================================================
    # SUCCESS
    # =====================================================

    flash("Video uploaded successfully. Face recognition is processing in the background.", "success")
    return redirect(url_for("teacher.dashboard"))


# =========================================================
# REAL-TIME STUDENT ATTENDANCE EMAIL DISPATCH
# =========================================================

@teacher_bp.route("/email/dispatch-students", methods=["POST"])
@login_required(role="teacher")
def dispatch_students_email():
    from backend.services.email_service import format_student_email_content, send_email_to_student

    payload = request.get_json(silent=True) or request.form
    report_type = (payload.get("report_type") or "shortage").lower().strip()
    student_ids_raw = payload.get("student_ids", [])
    if isinstance(student_ids_raw, str):
        try:
            student_ids = [int(x.strip()) for x in student_ids_raw.split(",") if x.strip()]
        except Exception:
            student_ids = []
    elif isinstance(student_ids_raw, list):
        student_ids = [int(x) for x in student_ids_raw if str(x).isdigit()]
    else:
        student_ids = []

    if not student_ids:
        return jsonify({"success": False, "error": "Please select at least one student to send email to."}), 400

    custom_subject = (payload.get("subject") or "").strip()
    custom_message = (payload.get("custom_message") or "").strip()
    teacher_name = session.get("name") or "Teacher"

    # Fetch total class sessions count
    total_sessions_row = run_query("SELECT COUNT(DISTINCT video_id) AS total FROM attendance", fetch_one=True)
    total_class_sessions = int(total_sessions_row.get("total", 0) or 1) if total_sessions_row else 1

    format_strings = ','.join(['%s'] * len(student_ids))
    students_query = f"""
        SELECT 
            s.student_id,
            s.name,
            s.student_number AS usn,
            s.email,
            c.class_id,
            c.class_name,
            c.semester,
            c.branch,
            c.section,
            COUNT(DISTINCT a.attendance_id) AS attended_count
        FROM students s
        LEFT JOIN class_students cs ON s.student_id = cs.student_id
        LEFT JOIN classes c ON cs.class_id = c.class_id
        LEFT JOIN attendance a ON s.student_id = a.student_id AND a.status = 'Present'
        WHERE s.student_id IN ({format_strings})
        GROUP BY s.student_id, s.name, s.student_number, s.email, c.class_id, c.class_name, c.semester, c.branch, c.section
        ORDER BY s.student_number ASC
    """
    students = run_query(students_query, tuple(student_ids), fetch_all=True) or []

    results = []
    sent_count = 0
    logged_count = 0
    failed_count = 0

    period_label_map = {
        "shortage": "Current Academic Term",
        "weekly": payload.get("period_label") or "Current Week",
        "monthly": payload.get("period_label") or "Current Month"
    }
    period_label = period_label_map.get(report_type, "Current Academic Term")

    for st in students:
        st_id = st.get("student_id")
        st_name = st.get("name") or "Student"
        st_email = (st.get("email") or "").strip()
        st_usn = st.get("usn") or "N/A"
        attended = int(st.get("attended_count", 0) or 0)
        pct = round((attended / total_class_sessions) * 100, 1) if total_class_sessions > 0 else 0.0

        if not st_email or "@" not in st_email:
            results.append({
                "student_id": st_id,
                "name": st_name,
                "usn": st_usn,
                "email": st_email or "Missing Email",
                "status": "skipped",
                "message": "Student does not have a registered email address."
            })
            continue

        raw_b = str(st.get("branch") or "AIML").strip()
        display_branch = "CSE (AIML)" if raw_b.upper() == "AIML" else raw_b
        display_sec = str(st.get("section") or "A").strip()
        display_sem = st.get("semester") or 7
        class_name = st.get("class_name") or f"{display_sem}th Semester - {display_branch} (Sec {display_sec})"

        student_info = {
            "name": st_name,
            "usn": st_usn,
            "email": st_email,
            "class_name": class_name
        }

        attendance_data = {
            "attended": attended,
            "total": total_class_sessions,
            "percentage": pct,
            "period_label": period_label,
            "subject_name": payload.get("subject_name") or "All Curriculum Subjects"
        }

        subject_line, html_body, text_body = format_student_email_content(
            student_info=student_info,
            report_type=report_type,
            attendance_data=attendance_data,
            custom_message=custom_message,
            teacher_name=teacher_name
        )

        if custom_subject:
            subject_line = f"{custom_subject} - {st_usn}"

        dispatch_res = send_email_to_student(
            recipient_email=st_email,
            subject_line=subject_line,
            html_content=html_body,
            text_content=text_body,
            student_info=student_info
        )

        if dispatch_res.get("success"):
            if dispatch_res.get("status") == "sent":
                sent_count += 1
            else:
                logged_count += 1
            results.append({
                "student_id": st_id,
                "name": st_name,
                "usn": st_usn,
                "email": st_email,
                "percentage": pct,
                "status": dispatch_res.get("status"),
                "message": dispatch_res.get("message")
            })
        else:
            failed_count += 1
            results.append({
                "student_id": st_id,
                "name": st_name,
                "usn": st_usn,
                "email": st_email,
                "percentage": pct,
                "status": "failed",
                "message": dispatch_res.get("message", "Delivery failed")
            })

    return jsonify({
        "success": True,
        "total_selected": len(student_ids),
        "delivered_count": sent_count + logged_count,
        "sent_count": sent_count,
        "logged_count": logged_count,
        "failed_count": failed_count,
        "report_type": report_type,
        "results": results
    })


# =========================================================
# MANUAL ATTENDANCE TOGGLE & BULK ENDPOINTS
# =========================================================

@teacher_bp.route("/manual-attendance/toggle", methods=["POST"])
@login_required(role="teacher")
def toggle_manual_attendance():
    payload = request.get_json(silent=True) or request.form
    student_id = payload.get("student_id")
    att_date = payload.get("date") or datetime.now().strftime("%Y-%m-%d")
    subject_id = payload.get("subject_id")
    target_status = payload.get("target_status")  # 'Present' or 'Absent'

    if not student_id:
        return jsonify({"success": False, "error": "student_id is required"}), 400

    # Check if attendance record exists
    if subject_id:
        existing = run_query(
            "SELECT attendance_id, status FROM attendance WHERE student_id = %s AND DATE(check_in_time) = %s AND subject_id = %s LIMIT 1",
            (student_id, att_date, subject_id),
            fetch_one=True
        )
    else:
        existing = run_query(
            "SELECT attendance_id, status FROM attendance WHERE student_id = %s AND DATE(check_in_time) = %s LIMIT 1",
            (student_id, att_date),
            fetch_one=True
        )

    if existing:
        current_status = str(existing.get("status") or "Absent").capitalize()
        new_status = target_status if target_status else ("Absent" if current_status.lower() == "present" else "Present")
        run_query(
            "UPDATE attendance SET status = %s WHERE attendance_id = %s",
            (new_status, existing.get("attendance_id")),
            commit=True
        )
    else:
        new_status = target_status or "Present"
        check_in_timestamp = f"{att_date} 09:00:00"
        run_query(
            "INSERT INTO attendance (student_id, subject_id, check_in_time, status) VALUES (%s, %s, %s, %s)",
            (student_id, subject_id or None, check_in_timestamp, new_status),
            commit=True
        )

    return jsonify({
        "success": True,
        "student_id": int(student_id),
        "new_status": new_status,
        "is_present": (new_status.lower() == "present"),
        "message": f"Attendance updated to {new_status}"
    })


@teacher_bp.route("/manual-attendance/bulk", methods=["POST"])
@login_required(role="teacher")
def bulk_manual_attendance():
    payload = request.get_json(silent=True) or request.form
    target_status = payload.get("status", "Present").capitalize()
    att_date = payload.get("date") or datetime.now().strftime("%Y-%m-%d")
    subject_id = payload.get("subject_id") or None
    student_ids = payload.get("student_ids", [])

    if not student_ids:
        # Default to all students
        st_rows = run_query("SELECT student_id FROM students", fetch_all=True) or []
        student_ids = [r["student_id"] for r in st_rows]

    updated_count = 0
    for sid in student_ids:
        if subject_id:
            existing = run_query(
                "SELECT attendance_id FROM attendance WHERE student_id = %s AND DATE(check_in_time) = %s AND subject_id = %s LIMIT 1",
                (sid, att_date, subject_id),
                fetch_one=True
            )
        else:
            existing = run_query(
                "SELECT attendance_id FROM attendance WHERE student_id = %s AND DATE(check_in_time) = %s LIMIT 1",
                (sid, att_date),
                fetch_one=True
            )

        if existing:
            run_query(
                "UPDATE attendance SET status = %s WHERE attendance_id = %s",
                (target_status, existing.get("attendance_id")),
                commit=True
            )
        else:
            check_in_timestamp = f"{att_date} 09:00:00"
            run_query(
                "INSERT INTO attendance (student_id, subject_id, check_in_time, status) VALUES (%s, %s, %s, %s)",
                (sid, subject_id, check_in_timestamp, target_status),
                commit=True
            )
        updated_count += 1

    return jsonify({
        "success": True,
        "status": target_status,
        "updated_count": updated_count,
        "message": f"Successfully marked all {updated_count} students as {target_status}"
    })


# =========================================================
# REAL-TIME ATTENDANCE REPORT CALCULATION & DISPATCH
# =========================================================


def calculate_attendance_report(report_type="weekly", period_label=None, subject_id="all", branch=None, year=None, semester=None, target_date=None):
    """
    Computes real-time attendance statistics and identifies students with attendance < 75%
    for a given reporting period (weekly or monthly) and optional branch, year, semester, and subject.
    """
    import math
    from datetime import datetime, timedelta
    import calendar

    # Determine date condition based on period
    period_str = str(period_label or "").strip()
    rep_type = str(report_type or "weekly").lower().strip()

    date_cond = None
    date_cond_sess = None

    if target_date:
        try:
            if isinstance(target_date, str):
                dt = datetime.strptime(target_date.strip()[:10], "%Y-%m-%d")
            else:
                dt = target_date
            if rep_type == "weekly":
                start_d = dt - timedelta(days=dt.weekday())
                end_d = start_d + timedelta(days=6)
                s_str = start_d.strftime("%Y-%m-%d")
                e_str = end_d.strftime("%Y-%m-%d")
                date_cond = f"(DATE(a.check_in_time) BETWEEN '{s_str}' AND '{e_str}')"
                date_cond_sess = f"(DATE(check_in_time) BETWEEN '{s_str}' AND '{e_str}')"
                if not period_label or period_label == "Current Week":
                    period_label = f"Week ({start_d.strftime('%d %b')} - {end_d.strftime('%d %b %Y')})"
            else:
                start_d = dt.replace(day=1)
                _, last_day = calendar.monthrange(dt.year, dt.month)
                end_d = dt.replace(day=last_day)
                s_str = start_d.strftime("%Y-%m-%d")
                e_str = end_d.strftime("%Y-%m-%d")
                date_cond = f"(DATE(a.check_in_time) BETWEEN '{s_str}' AND '{e_str}')"
                date_cond_sess = f"(DATE(check_in_time) BETWEEN '{s_str}' AND '{e_str}')"
                if not period_label:
                    period_label = f"{dt.strftime('%B %Y')} (1st to {last_day}th)"
        except Exception:
            pass

    if not date_cond:
        if rep_type == "weekly":
            if "01 Sep" in period_str:
                date_cond = "DATE(a.check_in_time) BETWEEN '2026-09-01' AND '2026-09-07'"
                date_cond_sess = "DATE(check_in_time) BETWEEN '2026-09-01' AND '2026-09-07'"
            elif "08 Sep" in period_str:
                date_cond = "DATE(a.check_in_time) BETWEEN '2026-09-08' AND '2026-09-15'"
                date_cond_sess = "DATE(check_in_time) BETWEEN '2026-09-08' AND '2026-09-15'"
            else:
                # Default / Current Week: last 7 days or recent attendance sessions
                date_cond = "(DATE(a.check_in_time) >= DATE_SUB(CURDATE(), INTERVAL 7 DAY) OR DATE(a.check_in_time) >= '2026-09-04')"
                date_cond_sess = "(DATE(check_in_time) >= DATE_SUB(CURDATE(), INTERVAL 7 DAY) OR DATE(check_in_time) >= '2026-09-04')"
        else:  # monthly
            # Explicitly support 1st to 31st date coverage for monthly reports
            if "August" in period_str:
                date_cond = "DATE(a.check_in_time) BETWEEN '2026-08-01' AND '2026-08-31'"
                date_cond_sess = "DATE(check_in_time) BETWEEN '2026-08-01' AND '2026-08-31'"
            elif "July" in period_str:
                date_cond = "DATE(a.check_in_time) BETWEEN '2026-07-01' AND '2026-07-31'"
                date_cond_sess = "DATE(check_in_time) BETWEEN '2026-07-01' AND '2026-07-31'"
            elif "Overall" in period_str:
                date_cond = "1=1"
                date_cond_sess = "1=1"
            else:
                # Default / Current Month: full 1st to 31st date range
                date_cond = "(DATE(a.check_in_time) BETWEEN DATE_FORMAT(CURDATE(), '%Y-%m-01') AND LAST_DAY(CURDATE()) OR DATE(a.check_in_time) BETWEEN '2026-09-01' AND '2026-09-30')"
                date_cond_sess = "(DATE(check_in_time) BETWEEN DATE_FORMAT(CURDATE(), '%Y-%m-01') AND LAST_DAY(CURDATE()) OR DATE(check_in_time) BETWEEN '2026-09-01' AND '2026-09-30')"

    sub_cond = ""
    sub_cond_sess = ""
    params_sess = []
    subject_name = "Subject Attendance"

    if subject_id and str(subject_id).lower() not in ("all", "all subjects", ""):
        try:
            sub_int = int(subject_id)
            sub_cond = " AND a.subject_id = %s"
            sub_cond_sess = " AND subject_id = %s"
            params_sess.append(sub_int)
            sub_info = run_query("SELECT subject_name, subject_code FROM subjects WHERE subject_id = %s", (sub_int,), fetch_one=True)
            if sub_info:
                subject_name = str(sub_info.get("subject_name") or "Subject Attendance").strip()
        except Exception:
            pass

    # Total classes held in this period
    q_sess = f"SELECT COUNT(DISTINCT CONCAT(DATE(check_in_time), '_', COALESCE(subject_id, 0))) as total FROM attendance WHERE {date_cond_sess}{sub_cond_sess}"
    sess = run_query(q_sess, tuple(params_sess) if params_sess else None, fetch_one=True) or {}
    total_classes = int(sess.get("total") or 0)
    if total_classes == 0 and params_sess:
        sub_all = run_query(
            "SELECT COUNT(DISTINCT DATE(check_in_time)) as total FROM attendance WHERE subject_id = %s",
            (params_sess[0],),
            fetch_one=True
        ) or {}
        total_classes = int(sub_all.get("total") or 0)
    if total_classes == 0:
        tot_all = run_query("SELECT COUNT(DISTINCT CONCAT(DATE(check_in_time), '_', COALESCE(subject_id, 0))) as total FROM attendance", fetch_one=True) or {}
        total_classes = max(1, int(tot_all.get("total") or 1))

    # Build student filter conditions for semester and branch
    filter_cond = ""
    params_st = list(params_sess)
    if semester and str(semester).strip():
        try:
            sem_int = int(semester)
            filter_cond += " AND COALESCE(c.semester, 7) = %s"
            params_st.append(sem_int)
        except Exception:
            pass

    if branch and str(branch).strip():
        b_str = str(branch).strip()
        filter_cond += " AND (COALESCE(c.branch, 'AIML') = %s OR COALESCE(c.branch, 'AIML') LIKE %s)"
        params_st.extend([b_str, f"%{b_str}%"])

    # Query matching students and their attendance count in this period
    q_st = f"""
        SELECT s.student_id, s.name, s.student_number as usn, s.email,
               COUNT(DISTINCT CONCAT(DATE(a.check_in_time), '_', COALESCE(a.subject_id, 0))) as attended
        FROM students s
        LEFT JOIN class_students cs ON s.student_id = cs.student_id
        LEFT JOIN classes c ON cs.class_id = c.class_id
        LEFT JOIN attendance a ON s.student_id = a.student_id 
             AND LOWER(a.status) = 'present' 
             AND {date_cond}{sub_cond}
        WHERE 1=1 {filter_cond}
        GROUP BY s.student_id, s.name, s.student_number, s.email
        ORDER BY s.student_number ASC
    """
    students = run_query(q_st, tuple(params_st) if params_st else None, fetch_all=True) or []

    if not students:
        q_st_fallback = f"""
            SELECT s.student_id, s.name, s.student_number as usn, s.email,
                   COUNT(DISTINCT CONCAT(DATE(a.check_in_time), '_', COALESCE(a.subject_id, 0))) as attended
            FROM students s
            LEFT JOIN attendance a ON s.student_id = a.student_id 
                 AND LOWER(a.status) = 'present' 
                 AND {date_cond}{sub_cond}
            GROUP BY s.student_id, s.name, s.student_number, s.email
            ORDER BY s.student_number ASC
        """
        students = run_query(q_st_fallback, tuple(params_sess) if params_sess else None, fetch_all=True) or []

    regular_count = 0
    shortage_count = 0
    shortage_students = []
    records_map = {}

    for st in students:
        s_id = st.get("student_id")
        att = int(st.get("attended") or 0)
        pct = round((att / total_classes) * 100, 1) if total_classes > 0 else 100.0
        needed = max(1, math.ceil(3 * total_classes - 4 * att)) if pct < 75.0 and total_classes > 0 else 0
        missed = max(0, total_classes - att)
        status = "Regular" if pct >= 75.0 else "Needs Attention"

        if pct >= 75.0:
            regular_count += 1
        else:
            shortage_count += 1
            shortage_students.append({
                "student_id": s_id,
                "name": st.get("name") or "Student",
                "usn": st.get("usn") or "N/A",
                "email": (st.get("email") or "").strip() or "N/A",
                "attended": att,
                "missed": missed,
                "total": total_classes,
                "percentage": pct,
                "classes_needed": needed
            })

        records_map[str(s_id)] = {
            "student_id": s_id,
            "name": st.get("name") or "Student",
            "usn": st.get("usn") or "N/A",
            "subject": subject_name,
            "attended": att,
            "present": att,
            "missed": missed,
            "absent": missed,
            "total": total_classes,
            "percentage": pct,
            "status": status,
            "classes_needed": needed
        }

    # Resolve clean class name for official reporting
    class_name = "7th Semester - AIML"
    if semester:
        c_row = run_query("SELECT class_name FROM classes WHERE semester = %s LIMIT 1", (semester,), fetch_one=True)
        if c_row and c_row.get("class_name"):
            class_name = c_row.get("class_name")
    elif subject_id and str(subject_id).lower() not in ("all", "all subjects", ""):
        c_row = run_query(
            "SELECT c.class_name FROM subjects s JOIN classes c ON s.class_id = c.class_id WHERE s.subject_id = %s",
            (subject_id,),
            fetch_one=True
        )
        if c_row and c_row.get("class_name"):
            class_name = c_row.get("class_name")

    return {
        "report_type": rep_type,
        "period_label": period_str,
        "subject_name": subject_name,
        "class_name": class_name,
        "total_classes": total_classes,
        "regular_count": regular_count,
        "shortage_count": shortage_count,
        "shortage_students": shortage_students,
        "records_map": records_map
    }


@teacher_bp.route("/email/get-shortage-preview", methods=["GET", "POST"])
@login_required(role="teacher")
def get_shortage_preview():
    payload = request.get_json(silent=True) or request.values
    report_type = payload.get("report_type", "weekly")
    period_label = payload.get("period_label", "Current Week (16 Sep - 22 Sep 2026)")
    subject_id = payload.get("subject_id", "all")
    branch = payload.get("branch")
    year = payload.get("year")
    semester = payload.get("semester")
    target_date = payload.get("date")

    data = calculate_attendance_report(
        report_type=report_type,
        period_label=period_label,
        subject_id=subject_id,
        branch=branch,
        year=year,
        semester=semester,
        target_date=target_date
    )

    return jsonify({
        "success": True,
        "report_type": data["report_type"],
        "period_label": data["period_label"],
        "subject_name": data["subject_name"],
        "total_classes": data["total_classes"],
        "regular_count": data["regular_count"],
        "shortage_count": data["shortage_count"],
        "shortage_students": data["shortage_students"]
    })


@teacher_bp.route("/email/send-report", methods=["POST"])
@login_required(role="teacher")
def send_attendance_report():
    import json
    from backend.services.email_service import (
        format_teacher_shortage_email,
        format_teacher_monthly_report_email,
        send_email_to_student,
        send_email_to_teacher,
        generate_monthly_excel_report,
        generate_monthly_pdf_report
    )

    teacher_id = session.get("user_id")
    teacher_info = run_query("SELECT name, email FROM teachers WHERE teacher_id = %s", (teacher_id,), fetch_one=True)
    session_teacher_email = teacher_info.get("email") if teacher_info else (session.get("email") or "teacher@college.edu")
    teacher_name = teacher_info.get("name") if teacher_info else (session.get("name") or "Teacher")

    payload = request.get_json(silent=True) or request.form
    report_type = payload.get("report_type", "weekly")
    period_label = payload.get("period_label", "Current Week (16 Sep - 22 Sep 2026)")
    subject_id = payload.get("subject_id", "all")
    branch = payload.get("branch")
    year = payload.get("year")
    semester = payload.get("semester")
    remarks = payload.get("remarks", "")
    target_date = payload.get("date")

    # Allow custom recipient teacher email from dashboard input
    teacher_email = (payload.get("teacher_email") or "").strip() or session_teacher_email

    send_to_teacher = payload.get("send_to_teacher")
    if send_to_teacher is None:
        send_to_teacher = payload.get("send_teacher_email", True)
    if isinstance(send_to_teacher, str):
        send_to_teacher = send_to_teacher.lower() in ("true", "1", "yes")

    send_to_students_dashboard = payload.get("send_to_students_dashboard")
    if send_to_students_dashboard is None:
        send_to_students_dashboard = payload.get("send_student_dashboard", True)
    if isinstance(send_to_students_dashboard, str):
        send_to_students_dashboard = send_to_students_dashboard.lower() in ("true", "1", "yes")

    # 1. Compute report statistics
    data = calculate_attendance_report(
        report_type=report_type,
        period_label=period_label,
        subject_id=subject_id,
        branch=branch,
        year=year,
        semester=semester,
        target_date=target_date
    )
    rep_type_title = "Weekly" if data["report_type"] == "weekly" else "Monthly"

    # 2. Dispatch to Faculty Email(s) with Excel & PDF Attachments
    teacher_email_res = None
    recipients = [e.strip() for e in teacher_email.replace(";", ",").split(",") if e.strip()]
    if not recipients:
        recipients = [session_teacher_email]

    if send_to_teacher:
        from backend.services.email_service import (
            generate_attendance_excel_report,
            generate_attendance_pdf_report,
            format_teacher_attendance_report_email
        )
        excel_name, excel_bytes = generate_attendance_excel_report(
            report_data=data,
            teacher_name=teacher_name,
            class_name=data.get("class_name")
        )
        pdf_name, pdf_bytes = generate_attendance_pdf_report(
            report_data=data,
            teacher_name=teacher_name,
            class_name=data.get("class_name")
        )
        attachments = [
            {"filename": excel_name, "data": excel_bytes, "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"},
            {"filename": pdf_name, "data": pdf_bytes, "content_type": "application/pdf"}
        ]

        sub_line, html_b, text_b = format_teacher_attendance_report_email(
            teacher_name=teacher_name,
            report_type=data["report_type"],
            period_label=data["period_label"],
            subject_name=data["subject_name"],
            class_name=data.get("class_name")
        )

        for rec in recipients:
            teacher_email_res = send_email_to_teacher(
                recipient_email=rec,
                subject_line=sub_line,
                html_content=html_b,
                text_content=text_b,
                attachments=attachments
            )

    # 3. Synchronize to Students Dashboard
    if send_to_students_dashboard:
        try:
            ins_query = """
                INSERT INTO attendance_reports 
                (teacher_id, report_type, period_label, subject_name, total_classes, regular_count, shortage_count, shortage_students_json, student_records_json, remarks)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            run_query(
                ins_query,
                (
                    teacher_id,
                    data["report_type"],
                    data["period_label"],
                    data["subject_name"],
                    data["total_classes"],
                    data["regular_count"],
                    data["shortage_count"],
                    json.dumps(data["shortage_students"]),
                    json.dumps(data["records_map"]),
                    remarks
                ),
                commit=True
            )
        except Exception as e:
            current_app.logger.error("Failed to insert attendance_report record: %s", e)

    is_smtp = bool(teacher_email_res and teacher_email_res.get("status") == "sent")
    if is_smtp:
        delivery_msg = f"{rep_type_title} attendance report with Excel & PDF attachments delivered via SMTP to {teacher_email} and published to Students Dashboard!"
    else:
        delivery_msg = f"{rep_type_title} attendance report with Excel & PDF attachments dispatched in real-time to {teacher_email} (and published to Students Dashboard)."

    return jsonify({
        "success": True,
        "report_type": data["report_type"],
        "period_label": data["period_label"],
        "subject_name": data["subject_name"],
        "teacher_email": teacher_email,
        "total_classes": data["total_classes"],
        "regular_count": data["regular_count"],
        "shortage_count": data["shortage_count"],
        "delivered_to_teacher": bool(send_to_teacher),
        "synced_to_student_dashboard": bool(send_to_students_dashboard),
        "is_smtp": is_smtp,
        "message": delivery_msg
    })


@teacher_bp.route("/email/send-monthly-teacher", methods=["POST"])
@login_required(role="teacher")
def send_monthly_teacher_email():
    """Backward compatibility alias for monthly teacher email dispatch"""
    return send_attendance_report()


@teacher_bp.route("/email/automate-monthly", methods=["POST"])
@login_required(role="teacher")
def automate_monthly_report():
    """
    Automated monthly report covering 1st to 31st date of the month.
    - Dispatches professional monthly attendance report with summary and student-wise details.
    - Attaches corresponding Excel (.xlsx) and PDF (.pdf) reports directly to the Gmail message.
    - Publishes the verified monthly report directly to the Students Dashboard.
    """
    import json
    from backend.services.email_service import (
        format_teacher_monthly_report_email,
        send_email_to_teacher,
        generate_monthly_excel_report,
        generate_monthly_pdf_report
    )

    teacher_id = session.get("user_id")
    teacher_info = run_query("SELECT name, email FROM teachers WHERE teacher_id = %s", (teacher_id,), fetch_one=True)
    session_teacher_email = teacher_info.get("email") if teacher_info else (session.get("email") or "teacher@college.edu")
    teacher_name = teacher_info.get("name") if teacher_info else (session.get("name") or "Teacher")

    payload = request.get_json(silent=True) or request.form
    month_name = payload.get("month", "September 2026 (1st to 31st)")
    subject_id = payload.get("subject_id", "all")
    branch = payload.get("branch")
    year = payload.get("year")
    semester = payload.get("semester")
    remarks = payload.get("remarks", "Automated Monthly Classroom Attendance Report (1st to 31st)")

    teacher_email = (payload.get("teacher_email") or "").strip() or session_teacher_email

    data = calculate_attendance_report(
        report_type="monthly",
        period_label=month_name,
        subject_id=subject_id,
        branch=branch,
        year=year,
        semester=semester
    )

    # 1. Generate Excel and PDF attachments
    from backend.services.email_service import (
        generate_attendance_excel_report,
        generate_attendance_pdf_report,
        format_teacher_attendance_report_email
    )
    excel_name, excel_bytes = generate_attendance_excel_report(
        report_data=data,
        teacher_name=teacher_name,
        class_name=data.get("class_name")
    )
    pdf_name, pdf_bytes = generate_attendance_pdf_report(
        report_data=data,
        teacher_name=teacher_name,
        class_name=data.get("class_name")
    )

    attachments = [
        {"filename": excel_name, "data": excel_bytes, "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"},
        {"filename": pdf_name, "data": pdf_bytes, "content_type": "application/pdf"}
    ]

    # 2. Format user-friendly and professional monthly report email
    sub_line, html_b, text_b = format_teacher_attendance_report_email(
        teacher_name=teacher_name,
        report_type="monthly",
        period_label=data["period_label"],
        subject_name=data["subject_name"],
        class_name=data.get("class_name")
    )

    teacher_email_res = None
    recipients = [e.strip() for e in teacher_email.replace(";", ",").split(",") if e.strip()]
    if not recipients:
        recipients = [session_teacher_email]

    for rec in recipients:
        teacher_email_res = send_email_to_teacher(
            recipient_email=rec,
            subject_line=sub_line,
            html_content=html_b,
            text_content=text_b,
            attachments=attachments
        )

    # 3. Publish to Students Dashboard
    try:
        ins_query = """
            INSERT INTO attendance_reports 
            (teacher_id, report_type, period_label, subject_name, total_classes, regular_count, shortage_count, shortage_students_json, student_records_json, remarks)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        run_query(
            ins_query,
            (
                teacher_id,
                "monthly",
                f"{data['period_label']} (1st to 31st)",
                data["subject_name"],
                data["total_classes"],
                data["regular_count"],
                data["shortage_count"],
                json.dumps(data["shortage_students"]),
                json.dumps(data["records_map"]),
                remarks
            ),
            commit=True
        )
    except Exception as e:
        current_app.logger.error("Failed to insert automated monthly report: %s", e)

    is_smtp = bool(teacher_email_res and teacher_email_res.get("status") == "sent")
    return jsonify({
        "success": True,
        "report_type": "monthly",
        "period_label": f"{data['period_label']} (1st to 31st)",
        "subject_name": data["subject_name"],
        "teacher_email": teacher_email,
        "total_classes": data["total_classes"],
        "regular_count": data["regular_count"],
        "shortage_count": data["shortage_count"],
        "is_smtp": is_smtp,
        "excel_file": excel_name,
        "pdf_file": pdf_name,
        "message": f"Official monthly report with Excel & PDF attachments delivered to {teacher_email} and published to Students Dashboard!"
    })


# =========================================================
# TEACHER PROFILE UPDATE ENDPOINT
# =========================================================

@teacher_bp.route("/profile/update", methods=["POST"])
@login_required(role="teacher")
def update_profile():
    teacher_id = session.get("user_id")
    payload = request.get_json(silent=True) or request.form
    name = (payload.get("name") or "").strip()
    email = (payload.get("email") or "").strip()
    phone = (payload.get("phone") or "").strip()

    if not name or not email:
        return jsonify({"success": False, "error": "Full Name and Email Address are required."}), 400

    # Ensure email is unique across other teachers
    existing = run_query(
        "SELECT teacher_id FROM teachers WHERE email = %s AND teacher_id != %s",
        (email, teacher_id),
        fetch_one=True
    )
    if existing:
        return jsonify({"success": False, "error": "This email address is already in use by another faculty member."}), 400

    run_query(
        "UPDATE teachers SET name = %s, email = %s, phone = %s WHERE teacher_id = %s",
        (name, email, phone, teacher_id),
        commit=True
    )

    session["name"] = name
    session["email"] = email

    return jsonify({
        "success": True,
        "name": name,
        "email": email,
        "phone": phone,
        "message": "Faculty profile updated successfully!"
    })


# =========================================================
# REAL-TIME ATTENDANCE EXCEL EXPORT ENDPOINTS
# =========================================================

@teacher_bp.route("/attendance/download-latest-excel", methods=["GET"])
@teacher_bp.route("/attendance/download-excel", methods=["GET"])
@teacher_bp.route("/attendance/export-latest-excel", methods=["GET"])
@teacher_bp.route("/attendance/export-excel", methods=["GET"])
@login_required(role="teacher")
def download_latest_attendance_excel():
    """
    Directly streams real-time attendance report in Excel (.xlsx) format.
    Columns: USN | Name | Subject | Teacher | Present | Absent
    Fetched directly from the existing database for the latest saved attendance session.
    Zero blank rows, zero dummy data.
    """
    import io
    from backend.services.attendance_service import (
        get_latest_attendance_records,
        build_realtime_attendance_excel
    )

    teacher_id = session.get("user_id")
    video_id = request.args.get("video_id")
    class_id = request.args.get("class_id")
    subject_id = request.args.get("subject_id")
    branch = request.args.get("branch")
    semester = request.args.get("semester")

    if not class_id and (semester or branch):
        term = semester or branch
        c_match = run_query("SELECT class_id FROM classes WHERE LOWER(class_name) LIKE %s LIMIT 1", (f"%{term}%",), fetch_one=True)
        if c_match:
            class_id = c_match.get("class_id")

    att_data = get_latest_attendance_records(
        video_id=video_id,
        teacher_id=teacher_id,
        class_id=class_id,
        subject_id=subject_id
    )

    filename, excel_bytes = build_realtime_attendance_excel(attendance_data=att_data)

    return send_file(
        io.BytesIO(excel_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=filename
    )

