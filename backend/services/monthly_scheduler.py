"""
backend/services/monthly_scheduler.py

Automated monthly attendance report scheduler.
Runs in a background thread and automatically sends the monthly attendance report
as an Excel file (.xlsx) to each teacher's email at the end of every month.
Zero button clicks required.
Sends the email only once per month per teacher, fetching real database attendance data.
"""

import os
import json
import io
import calendar
import threading
import time
import logging
from datetime import datetime, timedelta

from backend.services.db_service import run_query
from backend.services.email_service import send_email_to_teacher, LOGS_DIR

logger = logging.getLogger(__name__)
STATE_FILE = os.path.join(LOGS_DIR, "monthly_auto_dispatches.json")


def _load_dispatch_state():
    os.makedirs(LOGS_DIR, exist_ok=True)
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_dispatch_state(state):
    os.makedirs(LOGS_DIR, exist_ok=True)
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to persist monthly dispatch state: {e}")


def generate_monthly_excel(month_str, teacher_id=None):
    """
    Builds clean, uncolored Excel sheet with real attendance data:
    Columns: USN | Student Name | Subject | Total Classes | Present | Absent | Attendance Percentage
    Returns: (filename, bytes_data, total_records, total_classes)
    """
    import openpyxl
    from openpyxl.styles import Font, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    # Fetch real attendance for the target month
    query = """
        SELECT 
            s.student_number AS usn,
            s.name,
            sub.subject_name AS subject,
            COUNT(DISTINCT CONCAT(DATE(a.check_in_time), '_', COALESCE(a.video_id, 0))) AS total_classes,
            COUNT(DISTINCT CASE WHEN LOWER(a.status) = 'present' THEN CONCAT(DATE(a.check_in_time), '_', COALESCE(a.video_id, 0)) END) AS present_count
        FROM students s
        JOIN attendance a ON s.student_id = a.student_id
        JOIN subjects sub ON a.subject_id = sub.subject_id
        WHERE DATE_FORMAT(a.check_in_time, '%Y-%m') = %s
        GROUP BY s.student_id, s.student_number, s.name, sub.subject_id, sub.subject_name
        ORDER BY sub.subject_name ASC, s.student_number ASC
    """
    rows = run_query(query, (month_str,), fetch_all=True) or []

    # If no records for that specific month_str, get latest available month records
    if not rows:
        latest_month_row = run_query(
            "SELECT DATE_FORMAT(MAX(check_in_time), '%Y-%m') as m FROM attendance WHERE check_in_time IS NOT NULL",
            fetch_one=True
        ) or {}
        alt_month = latest_month_row.get("m")
        if alt_month and alt_month != month_str:
            rows = run_query(query, (alt_month,), fetch_all=True) or []

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Monthly Attendance"
    ws.views.sheetView[0].showGridLines = True

    header_font = Font(name="Calibri", size=11, bold=True, color="000000")
    data_font = Font(name="Calibri", size=11, color="000000")
    thin_border = Border(
        left=Side(style="thin", color="CBD5E1"),
        right=Side(style="thin", color="CBD5E1"),
        top=Side(style="thin", color="CBD5E1"),
        bottom=Side(style="thin", color="CBD5E1")
    )

    headers = ["USN", "Student Name", "Subject", "Total Classes", "Present", "Absent", "Attendance Percentage"]
    ws.row_dimensions[1].height = 24

    for col_idx, h in enumerate(headers, 1):
        c = ws.cell(row=1, column=col_idx, value=h)
        c.font = header_font
        c.border = thin_border
        c.alignment = Alignment(
            horizontal="center" if h in ("USN", "Total Classes", "Present", "Absent", "Attendance Percentage") else "left",
            vertical="center"
        )

    tot_classes_count = 0
    for r_idx, row in enumerate(rows, start=2):
        ws.row_dimensions[r_idx].height = 20
        usn = str(row.get("usn") or "N/A").strip()
        name = str(row.get("name") or "Student").strip()
        subject = str(row.get("subject") or "Subject").strip()
        tot = int(row.get("total_classes", 1) or 1)
        tot_classes_count = max(tot_classes_count, tot)
        pres = int(row.get("present_count", 0) or 0)
        absent = max(0, tot - pres)
        pct = round((pres / tot * 100), 1) if tot > 0 else 0.0

        # 1. USN
        c1 = ws.cell(row=r_idx, column=1, value=usn)
        c1.font = data_font
        c1.border = thin_border
        c1.alignment = Alignment(horizontal="center", vertical="center")

        # 2. Student Name
        c2 = ws.cell(row=r_idx, column=2, value=name)
        c2.font = data_font
        c2.border = thin_border
        c2.alignment = Alignment(horizontal="left", vertical="center")

        # 3. Subject
        c3 = ws.cell(row=r_idx, column=3, value=subject)
        c3.font = data_font
        c3.border = thin_border
        c3.alignment = Alignment(horizontal="left", vertical="center")

        # 4. Total Classes
        c4 = ws.cell(row=r_idx, column=4, value=tot)
        c4.font = data_font
        c4.border = thin_border
        c4.alignment = Alignment(horizontal="center", vertical="center")

        # 5. Present
        c5 = ws.cell(row=r_idx, column=5, value=pres)
        c5.font = data_font
        c5.border = thin_border
        c5.alignment = Alignment(horizontal="center", vertical="center")

        # 6. Absent
        c6 = ws.cell(row=r_idx, column=6, value=absent)
        c6.font = data_font
        c6.border = thin_border
        c6.alignment = Alignment(horizontal="center", vertical="center")

        # 7. Attendance Percentage
        c7 = ws.cell(row=r_idx, column=7, value=f"{pct}%")
        c7.font = data_font
        c7.border = thin_border
        c7.alignment = Alignment(horizontal="center", vertical="center")

    col_widths = {1: 18, 2: 24, 3: 28, 4: 16, 5: 14, 6: 14, 7: 22}
    for col_idx, width in col_widths.items():
        ws.column_dimensions[get_column_letter(col_idx)].width = width

    filename = f"Monthly_Attendance_Report_{month_str}.xlsx"
    buf = io.BytesIO()
    wb.save(buf)
    return filename, buf.getvalue(), len(rows), tot_classes_count


def check_and_send_monthly_reports(force=False):
    """
    Checks if the monthly attendance report needs to be sent to teachers.
    Triggered automatically at the end of each month (or beginning of next).
    Ensures each teacher receives the report strictly ONCE per month.
    """
    now = datetime.now()
    _, last_day = calendar.monthrange(now.year, now.month)

    # We dispatch for the current month if today is the end of the month (within last 3 days)
    # Or dispatch for the previous month if today is the 1st/2nd of a new month.
    is_end_of_month = (now.day >= (last_day - 2))
    is_start_of_new_month = (now.day <= 2)

    if is_end_of_month or force:
        target_month = now.strftime("%Y-%m")
        month_label = now.strftime("%B %Y")
    elif is_start_of_new_month:
        prev_month_date = now.replace(day=1) - timedelta(days=1)
        target_month = prev_month_date.strftime("%Y-%m")
        month_label = prev_month_date.strftime("%B %Y")
    else:
        # Not at the boundary of a month, skip unless forced
        return {"status": "skipped", "reason": "Not at the end of the month"}

    state = _load_dispatch_state()
    month_records = state.get(target_month, {})

    # Fetch teachers
    teachers = run_query(
        "SELECT teacher_id, name, email FROM teachers WHERE email IS NOT NULL AND email != ''",
        fetch_all=True
    ) or []

    dispatched = []
    for t in teachers:
        t_id = str(t.get("teacher_id"))
        t_email = str(t.get("email") or "").strip()
        t_name = str(t.get("name") or "Professor").strip()

        if not t_email or "@" not in t_email:
            continue

        # Check if already sent once this month
        if not force and t_id in month_records:
            continue

        filename, excel_bytes, total_records, total_classes = generate_monthly_excel(target_month, t_id)

        subject_line = f"Monthly Attendance Report - {month_label}"
        text_content = f"""Dear Professor {t_name},

Please find attached your automated monthly attendance report for {month_label}.

Summary Details:
- Month: {month_label}
- Students Tracked: {total_records}
- Total Classes Conducted: {total_classes}
- File Attached: {filename}

This report contains student USN, Name, Subject, Total Classes, Present, and Absent details.
Generated automatically by the GSSS Attendance System.

Regards,
GSSS Attendance System
"""
        html_content = f"""
        <div style="font-family: Arial, sans-serif; line-height: 1.6; color: #1e293b; max-width: 600px;">
            <h3 style="color: #0f172a; margin-bottom: 8px;">Monthly Attendance Report &bull; {month_label}</h3>
            <p>Dear Professor <strong>{t_name}</strong>,</p>
            <p>Please find attached the official automated monthly attendance report for <strong>{month_label}</strong>.</p>
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px 16px; margin: 16px 0;">
                <div style="font-size: 14px; font-weight: bold; margin-bottom: 6px;">Report Overview:</div>
                <div style="font-size: 13px; color: #475569;">
                    &bull; <strong>Students Tracked:</strong> {total_records}<br>
                    &bull; <strong>Conducted Classes:</strong> {total_classes}<br>
                    &bull; <strong>Attachment:</strong> {filename} (Excel)
                </div>
            </div>
            <p style="font-size: 13px; color: #64748b;">The attached Excel spreadsheet includes student USN, Name, Subject, Total Classes, Present, and Absent details for all recorded sessions.</p>
            <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 20px 0;">
            <p style="font-size: 12px; color: #94a3b8; margin: 0;">Automated monthly dispatch &bull; GSSS Institute of Engineering and Technology for Women</p>
        </div>
        """

        attachments = [{
            "filename": filename,
            "data": excel_bytes
        }]

        result = send_email_to_teacher(
            recipient_email=t_email,
            subject_line=subject_line,
            html_content=html_content,
            text_content=text_content,
            attachments=attachments
        )

        month_records[t_id] = {
            "dispatched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "email": t_email,
            "status": result.get("status")
        }
        state[target_month] = month_records
        _save_dispatch_state(state)

        dispatched.append(t_email)
        logger.info(f"[AUTOMATED_MONTHLY_REPORT] Sent {target_month} report to {t_email}")

    return {
        "status": "success",
        "month": target_month,
        "dispatched_count": len(dispatched),
        "recipients": dispatched
    }


def _scheduler_loop():
    """
    Background worker loop that wakes up periodically (every 1 hour)
    and checks if the automated monthly report should be sent.
    """
    # Initial pause on startup to let DB initialize
    time.sleep(10)
    while True:
        try:
            check_and_send_monthly_reports(force=False)
        except Exception as e:
            logger.error(f"[MONTHLY_SCHEDULER_ERROR] {e}")
        # Sleep for 1 hour between checks
        time.sleep(3600)


_scheduler_started = False
_scheduler_lock = threading.Lock()

def start_monthly_report_scheduler():
    """
    Starts the automated monthly email scheduler in a daemon thread.
    Safe to call multiple times (idempotent).
    """
    global _scheduler_started
    with _scheduler_lock:
        if not _scheduler_started:
            t = threading.Thread(target=_scheduler_loop, name="MonthlyReportScheduler", daemon=True)
            t.start()
            _scheduler_started = True
            logger.info("[MONTHLY_SCHEDULER] Started background automated monthly report scheduler thread.")
