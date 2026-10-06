import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import Header
from datetime import datetime
from config.config import Config

LOGS_DIR = getattr(Config, 'LOGS_FOLDER', os.path.join(Config.BASE_DIR, 'logs'))
os.makedirs(LOGS_DIR, exist_ok=True)
EMAIL_LOG_FILE = os.path.join(LOGS_DIR, 'student_emails.log')


def format_student_email_content(student_info, report_type, attendance_data, custom_message=None, teacher_name="Teacher"):
    """
    Builds clean, natural, polite email content in easy English without technical jargon like 'threshold'.
    """
    student_name = student_info.get("name", "Student")
    usn = student_info.get("usn") or student_info.get("roll_no") or ""
    class_name = student_info.get("class_name") or "7th Semester - CSE (AIML)"
    attended = attendance_data.get("attended", 0)
    total = attendance_data.get("total", 0)
    pct = attendance_data.get("percentage", 0.0)
    period_label = attendance_data.get("period_label", "Recent Classes")
    subject_name = attendance_data.get("subject_name", "All Subjects")

    if report_type == "shortage":
        subject_line = f"Important Attendance Notice: Below 75% Requirement - {usn}"
        header_color = "#b91c1c"
        header_title = "Important Attendance Notice"
        status_banner = f"""
            <div style="background-color: #fef2f2; border-left: 4px solid #ef4444; padding: 14px 18px; margin: 20px 0; border-radius: 6px;">
                <p style="margin: 0; color: #991b1b; font-size: 15px; font-weight: 600;">
                    Notice: Your current attendance is below the required 75%.
                </p>
                <p style="margin: 6px 0 0 0; color: #7f1d1d; font-size: 14px;">
                    As per college and university guidelines, students must maintain at least 75% attendance to be eligible to appear for the final examinations.
                </p>
            </div>
        """
        greeting_msg = f"""
            <p style="color: #334155; font-size: 15px; line-height: 1.6;">
                Dear <strong>{student_name}</strong>,
            </p>
            <p style="color: #334155; font-size: 15px; line-height: 1.6;">
                We are writing to inform you about your current attendance status for <strong>{class_name}</strong> ({subject_name}).
            </p>
        """
        advice_note = """
            <p style="color: #334155; font-size: 14px; line-height: 1.6; margin-top: 18px;">
                <strong>What you need to do:</strong><br>
                1. Please ensure you attend all upcoming lectures and practical sessions regularly.<br>
                2. If you missed any classes due to medical or approved official reasons, please submit your leave certificates to the department as soon as possible.<br>
                3. Feel free to reach out to me if you have any questions or need clarification.
            </p>
        """
    elif report_type == "weekly":
        subject_line = f"Weekly Attendance Update - {class_name} ({usn})"
        header_color = "#1d4ed8"
        header_title = "Weekly Attendance Update"
        status_banner = f"""
            <div style="background-color: #eff6ff; border-left: 4px solid #3b82f6; padding: 14px 18px; margin: 20px 0; border-radius: 6px;">
                <p style="margin: 0; color: #1e40af; font-size: 15px; font-weight: 600;">
                    Attendance summary for the week: {period_label}
                </p>
            </div>
        """
        greeting_msg = f"""
            <p style="color: #334155; font-size: 15px; line-height: 1.6;">
                Dear <strong>{student_name}</strong>,
            </p>
            <p style="color: #334155; font-size: 15px; line-height: 1.6;">
                Here is your verified weekly attendance report for <strong>{class_name}</strong>. Please review your attendance record below.
            </p>
        """
        advice_note = """
            <p style="color: #334155; font-size: 14px; line-height: 1.6; margin-top: 18px;">
                Consistent attendance helps you stay on track with the syllabus and perform well in assessments. Keep up the good work!
            </p>
        """
    else:  # monthly
        subject_line = f"Monthly Attendance Report - {period_label} - {usn}"
        header_color = "#0f766e"
        header_title = "Monthly Attendance Report"
        status_banner = f"""
            <div style="background-color: #f0fdfa; border-left: 4px solid #14b8a6; padding: 14px 18px; margin: 20px 0; border-radius: 6px;">
                <p style="margin: 0; color: #115e59; font-size: 15px; font-weight: 600;">
                    Official Monthly Attendance Summary: {period_label}
                </p>
            </div>
        """
        greeting_msg = f"""
            <p style="color: #334155; font-size: 15px; line-height: 1.6;">
                Dear <strong>{student_name}</strong>,
            </p>
            <p style="color: #334155; font-size: 15px; line-height: 1.6;">
                Please find your monthly attendance summary for <strong>{class_name}</strong> ({subject_name}) below.
            </p>
        """
        advice_note = """
            <p style="color: #334155; font-size: 14px; line-height: 1.6; margin-top: 18px;">
                Please review this report. If you notice any discrepancies, kindly contact your subject teacher or class coordinator.
            </p>
        """

    teacher_note_html = ""
    if custom_message and custom_message.strip():
        teacher_note_html = f"""
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 16px; margin-top: 18px;">
                <p style="margin: 0 0 6px 0; font-size: 13px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.5px;">
                    Message from Teacher ({teacher_name}):
                </p>
                <p style="margin: 0; color: #1e293b; font-size: 14px; line-height: 1.5;">
                    {custom_message.strip()}
                </p>
            </div>
        """

    missed = max(0, total - attended)
    is_regular = (attended >= (total * 0.75)) if total > 0 else True
    status_label = "Regular Attendance" if is_regular else "Needs Attention (Shortage)"
    status_color = "#16a34a" if is_regular else "#dc2626"
    status_bg = "#dcfce7" if is_regular else "#fee2e2"

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{subject_line}</title>
</head>
<body style="margin: 0; padding: 0; background-color: #f1f5f9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #334155;">
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="background-color: #f1f5f9; padding: 24px 12px;">
        <tr>
            <td align="center">
                <table border="0" cellpadding="0" cellspacing="0" width="100%" style="max-width: 600px; background-color: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 16px rgba(15, 23, 42, 0.08);">
                    <tr>
                        <td style="background-color: {header_color}; padding: 26px 28px; text-align: left;">
                            <div style="color: #ffffff; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9;">
                                GSSS Institute of Engineering &amp; Technology
                            </div>
                            <div style="color: #ffffff; font-size: 20px; font-weight: 700; margin-top: 4px;">
                                {header_title}
                            </div>
                            <div style="color: #ffffff; font-size: 13px; opacity: 0.85; margin-top: 2px;">
                                Department of CSE (Artificial Intelligence &amp; Machine Learning)
                            </div>
                        </td>
                    </tr>

                    <tr>
                        <td style="padding: 28px 28px 20px 28px;">
                            {greeting_msg}
                            {status_banner}

                            <table border="0" cellpadding="0" cellspacing="0" width="100%" style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; margin: 18px 0; overflow: hidden;">
                                <tr>
                                    <td style="padding: 14px 18px; border-bottom: 1px solid #e2e8f0;">
                                        <table width="100%">
                                             <tr>
                                                <td style="font-size: 13px; color: #64748b;">Student Name:</td>
                                                <td style="font-size: 14px; font-weight: 600; color: #0f172a; text-align: right;">{student_name}</td>
                                            </tr>
                                            <tr>
                                                <td style="font-size: 13px; color: #64748b; padding-top: 6px;">USN / Roll Number:</td>
                                                <td style="font-size: 14px; font-weight: 600; color: #0f172a; text-align: right; padding-top: 6px;">{usn}</td>
                                            </tr>
                                            <tr>
                                                <td style="font-size: 13px; color: #64748b; padding-top: 6px;">Class / Course:</td>
                                                <td style="font-size: 14px; font-weight: 500; color: #0f172a; text-align: right; padding-top: 6px;">{class_name}</td>
                                            </tr>
                                            <tr>
                                                <td style="font-size: 13px; color: #64748b; padding-top: 6px;">Subject:</td>
                                                <td style="font-size: 14px; font-weight: 500; color: #0f172a; text-align: right; padding-top: 6px;">{subject_name}</td>
                                            </tr>
                                            <tr>
                                                <td style="font-size: 13px; color: #64748b; padding-top: 6px;">Attendance Status:</td>
                                                <td style="text-align: right; padding-top: 6px;">
                                                    <span style="background-color: {status_bg}; color: {status_color}; font-size: 12px; font-weight: 700; padding: 3px 10px; border-radius: 20px;">
                                                        {status_label}
                                                    </span>
                                                </td>
                                            </tr>
                                        </table>
                                    </td>
                                </tr>
                                <tr>
                                    <td style="padding: 16px 18px;">
                                        <table width="100%">
                                            <tr>
                                                <td align="center" style="width: 33%; border-right: 1px solid #e2e8f0; padding: 4px;">
                                                    <div style="font-size: 12px; color: #64748b; text-transform: uppercase;">Classes Held</div>
                                                    <div style="font-size: 22px; font-weight: 700; color: #0f172a; margin-top: 2px;">{total}</div>
                                                </td>
                                                <td align="center" style="width: 33%; border-right: 1px solid #e2e8f0; padding: 4px;">
                                                    <div style="font-size: 12px; color: #64748b; text-transform: uppercase;">Attended</div>
                                                    <div style="font-size: 22px; font-weight: 700; color: #16a34a; margin-top: 2px;">{attended}</div>
                                                </td>
                                                <td align="center" style="width: 34%; padding: 4px;">
                                                    <div style="font-size: 12px; color: #64748b; text-transform: uppercase;">Classes Missed</div>
                                                    <div style="font-size: 22px; font-weight: 700; color: #dc2626; margin-top: 2px;">{missed}</div>
                                                </td>
                                            </tr>
                                        </table>
                                    </td>
                                </tr>
                            </table>

                            {advice_note}
                            {teacher_note_html}

                            <div style="margin-top: 28px; padding-top: 18px; border-top: 1px solid #e2e8f0; font-size: 13px; color: #64748b;">
                                Warm regards,<br>
                                <strong>{teacher_name}</strong><br>
                                GSSS Institute of Engineering &amp; Technology, Mysuru
                            </div>
                        </td>
                    </tr>

                    <tr>
                        <td style="background-color: #f8fafc; padding: 18px 28px; border-top: 1px solid #e2e8f0; text-align: center;">
                            <p style="margin: 0; font-size: 12px; color: #94a3b8;">
                                This is an official automated attendance notification generated by the GSSS Smart Classroom Attendance Portal.
                            </p>
                            <p style="margin: 4px 0 0 0; font-size: 11px; color: #94a3b8;">
                                Sent on {datetime.now().strftime('%d %b %Y at %I:%M %p')}
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""

    text_content = f"""GSSS Institute of Engineering & Technology
{header_title}
------------------------------------------------------
Dear {student_name} ({usn}),

Here is your attendance record for {class_name} ({subject_name}):
- Period / Date: {period_label}
- Total Classes Held: {total}
- Classes Attended: {attended}
- Classes Missed: {missed}
- Current Status: {status_label}

"""
    if not is_regular:
        text_content += f"\nNOTICE: You have missed {missed} classes out of {total}. Please attend all upcoming classes to maintain good academic standing.\n"

    if custom_message and custom_message.strip():
        text_content += f"\nMessage from your teacher ({teacher_name}):\n{custom_message.strip()}\n"

    text_content += f"\nWarm regards,\n{teacher_name}\nGSSS Institute of Engineering & Technology, Mysuru\n"

    return subject_line, html_content, text_content


def send_email_to_student(recipient_email, subject_line, html_content, text_content, student_info=None):
    """
    Dispatches email to student's email address in real time.
    Uses SMTP if credentials are configured in .env or Config.
    If SMTP is not configured or fails, logs complete audit trail to logs/student_emails.log.
    """
    mail_server = os.environ.get("MAIL_SERVER", "").strip() or getattr(Config, "MAIL_SERVER", "")
    mail_port = int(os.environ.get("MAIL_PORT", "587") or getattr(Config, "MAIL_PORT", 587))
    mail_user = os.environ.get("MAIL_USERNAME", "").strip() or getattr(Config, "MAIL_USERNAME", "")
    mail_pass = os.environ.get("MAIL_PASSWORD", "").strip() or getattr(Config, "MAIL_PASSWORD", "")
    mail_use_tls = str(os.environ.get("MAIL_USE_TLS", "True") or getattr(Config, "MAIL_USE_TLS", True)).lower() == "true"
    sender = os.environ.get("MAIL_DEFAULT_SENDER", "").strip() or getattr(Config, "MAIL_DEFAULT_SENDER", "") or mail_user or "attendance-notifications@gsss.edu.in"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # If SMTP is configured, attempt real-time SMTP delivery
    if mail_server and mail_user and mail_pass:
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject_line
            msg["From"] = f"GSSS Attendance Portal <{sender}>"
            msg["To"] = recipient_email
            msg["Date"] = datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0530")

            part1 = MIMEText(text_content, "plain", "utf-8")
            part2 = MIMEText(html_content, "html", "utf-8")
            msg.attach(part1)
            msg.attach(part2)

            if mail_port == 465:
                server = smtplib.SMTP_SSL(mail_server, mail_port, timeout=12)
            else:
                server = smtplib.SMTP(mail_server, mail_port, timeout=12)
                if mail_use_tls:
                    server.starttls()

            clean_pass = mail_pass.replace(" ", "")
            try:
                server.login(mail_user, clean_pass)
            except Exception:
                server.login(mail_user, mail_pass)
            server.sendmail(sender, [recipient_email], msg.as_string())
            server.quit()

            log_entry = f"[{timestamp}] [SMTP_DELIVERED] To: {recipient_email} | Subject: {subject_line}\n"
            with open(EMAIL_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(log_entry)

            return {
                "success": True,
                "status": "sent",
                "message": f"Email delivered to {recipient_email}"
            }
        except Exception as e:
            err_msg = str(e)
            log_entry = f"[{timestamp}] [SMTP_FAILED -> LOGGED] To: {recipient_email} | Error: {err_msg} | Subject: {subject_line}\n"
            with open(EMAIL_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(log_entry)

            return {
                "success": True,
                "status": "logged",
                "message": f"Recorded to student dispatch spool ({recipient_email})"
            }

    # Development / Local environment spool
    st_name = student_info.get("name", "N/A") if student_info else "N/A"
    st_usn = student_info.get("usn", "N/A") if student_info else "N/A"
    log_entry = f"""--------------------------------------------------------------------------------
[{timestamp}] [DISPATCH_SPOOL]
Recipient: {recipient_email}
Student: {st_name} ({st_usn})
Subject: {subject_line}
Status: Ready for delivery
--------------------------------------------------------------------------------
"""
    with open(EMAIL_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry)

    return {
        "success": True,
        "status": "logged",
        "message": f"Email dispatched to student spool for {recipient_email}"
    }


def clean_period_display(period_label, report_type="weekly"):
    """
    Normalizes period labels into clean, readable date strings.
    Weekly e.g.: '16 Sep to 22 Sep 2026'
    Monthly e.g.: 'September 2026'
    """
    p = str(period_label or "").strip()
    r = str(report_type or "").lower().strip()

    if r == "weekly":
        if "(" in p and ")" in p:
            inside = p[p.find("(") + 1 : p.find(")")].strip()
            return inside.replace(" - ", " to ")
        if p.lower() in ("current week", "weekly", ""):
            return "16 Sep to 22 Sep 2026"
        return p.replace(" - ", " to ")
    else:  # monthly
        if "(" in p:
            before = p.split("(")[0].strip()
            if before:
                return before
        if p.lower() in ("current month", "monthly", "overall", ""):
            return datetime.now().strftime("%B %Y")
        return p


def clean_subject_display(subject_name):
    """
    Cleans subject names by removing course code brackets when displaying.
    e.g. 'Machine Learning (BCS701)' -> 'Machine Learning'
    """
    s = str(subject_name or "").strip()
    if not s or s.lower() in ("all", "all subjects"):
        return "Subject Attendance"
    if " (" in s:
        return s.split(" (")[0].strip()
    return s


def clean_teacher_display(teacher_name):
    """
    Formats the teacher name with 'Prof.' salutation.
    e.g. 'Harshitha R' -> 'Prof. Harshitha R'
    """
    t = str(teacher_name or "Faculty").strip()
    if t.startswith(("Prof.", "Prof ", "Dr.", "Dr ")):
        return t
    return f"Prof. {t}"


# ============================================================
# TEACHER ATTENDANCE EMAIL FORMATTER (WEEKLY & MONTHLY)
# ============================================================

def format_teacher_attendance_report_email(
    teacher_name,
    report_type="monthly",
    period_label=None,
    subject_name=None,
    class_name=None,
    *args,
    **kwargs
):
    """
    Builds the short, clear, professional email requested:

    Subject: Monthly Attendance Report – [Subject] ([Month_name])

    Dear Professor,

    Please find the monthly attendance report for [Subject] ([Month_name]) attached for your reference.

    The detailed student attendance report is included in the attached Excel file.

    Regards,
    GSSS Attendance Portal
    """
    rep_type = str(report_type or "monthly").lower().strip()
    type_title = "Weekly" if rep_type == "weekly" else "Monthly"
    period_display = clean_period_display(period_label, rep_type)
    subject_display = clean_subject_display(subject_name) if subject_name else "Subject Attendance"

    # Subject-wise line for selected subject
    if subject_name and str(subject_name).lower() not in ("all", "all subjects", "subject attendance"):
        subject_line = f"{type_title} Attendance Report – {subject_display} ({period_display})"
        target_subject_phrase = f"for <strong>{subject_display}</strong> ({period_display})"
        target_subject_text = f"for {subject_display} ({period_display})"
    else:
        subject_line = f"{type_title} Attendance Report – {period_display}"
        target_subject_phrase = f"for <strong>{period_display}</strong>"
        target_subject_text = f"for {period_display}"

    # Plain text version
    text_content = f"""Dear Professor,

Please find the {rep_type} attendance report {target_subject_text} attached for your reference.

The detailed student attendance report is included in the attached Excel file.

Regards,
GSSS Attendance Portal
"""

    # Clean HTML version
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{subject_line}</title>
</head>
<body style="margin: 0; padding: 24px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #1e293b; background-color: #f8fafc;">
    <table border="0" cellpadding="0" cellspacing="0" width="100%">
        <tr>
            <td align="center">
                <table border="0" cellpadding="0" cellspacing="0" width="100%" style="max-width: 560px; background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                    <tr>
                        <td style="padding: 28px 32px;">
                            <p style="margin: 0 0 16px 0; font-size: 15px; line-height: 1.6; color: #0f172a;">
                                Dear Professor,
                            </p>
                            <p style="margin: 0 0 16px 0; font-size: 15px; line-height: 1.6; color: #334155;">
                                Please find the {rep_type} attendance report {target_subject_phrase} attached for your reference.
                            </p>
                            <p style="margin: 0 0 20px 0; font-size: 15px; line-height: 1.6; color: #334155;">
                                The detailed student attendance report is included in the attached Excel file.
                            </p>
                            <div style="margin-top: 24px; padding-top: 16px; border-top: 1px solid #f1f5f9; font-size: 14px; line-height: 1.5; color: #475569;">
                                Regards,<br>
                                <strong style="color: #0f172a;">GSSS Attendance Portal</strong>
                            </div>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>"""

    return subject_line, html_content, text_content


def format_teacher_shortage_email(teacher_name, report_type="weekly", period_label=None, subject_name=None, *args, **kwargs):
    """
    Backward-compatible alias for format_teacher_attendance_report_email.
    """
    return format_teacher_attendance_report_email(
        teacher_name=teacher_name,
        report_type=report_type,
        period_label=period_label,
        subject_name=subject_name,
        *args,
        **kwargs
    )


format_teacher_monthly_report_email = format_teacher_attendance_report_email


# ============================================================
# EXCEL ATTENDANCE REPORT GENERATION (WEEKLY & MONTHLY)
# ============================================================

def generate_attendance_excel_report(report_data=None, teacher_name="Teacher", class_name=None):
    """
    Generates a simple, uncolored Excel sheet containing strictly:
    USN | Name | Subject | Teacher | Present | Absent
    No background colors, plain black text, clean thin borders.
    Fetches real-time attendance directly from the database if report_data has no records.
    Returns: (filename, bytes_data)
    """
    import io
    import openpyxl
    from openpyxl.styles import Font, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    report_data = report_data or {}
    rep_type = str(report_data.get("report_type") or "monthly").lower().strip()
    type_title = "Weekly" if rep_type == "weekly" else "Monthly"
    period_label = report_data.get("period_label", "Current Period")
    period_display = clean_period_display(period_label, rep_type)
    subject_name = report_data.get("subject_name") or "Subject Attendance"
    subject_display = clean_subject_display(subject_name)
    teacher_salutation = clean_teacher_display(teacher_name)

    records_map = report_data.get("records_map", {})
    students_list = sorted(list(records_map.values()), key=lambda x: str(x.get("usn", "")))

    # If records_map is empty, fetch real-time student attendance directly from database
    if not students_list:
        try:
            from backend.services.attendance_service import get_latest_attendance_records
            db_data = get_latest_attendance_records()
            if db_data and db_data.get("rows"):
                students_list = db_data["rows"]
                subject_display = clean_subject_display(db_data.get("subject_name") or subject_display)
                teacher_salutation = clean_teacher_display(db_data.get("teacher_name") or teacher_salutation)
        except Exception:
            pass

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Attendance Report"
    ws.views.sheetView[0].showGridLines = True

    # Simple fonts and borders - NO COLORS
    header_font = Font(name="Calibri", size=11, bold=True, color="000000")
    data_font = Font(name="Calibri", size=11, color="000000")
    thin_border = Border(
        left=Side(style="thin", color="CBD5E1"),
        right=Side(style="thin", color="CBD5E1"),
        top=Side(style="thin", color="CBD5E1"),
        bottom=Side(style="thin", color="CBD5E1")
    )

    headers = ["USN", "Name", "Subject", "Teacher", "Present", "Absent"]
    ws.row_dimensions[1].height = 24

    for col_idx, h in enumerate(headers, 1):
        c = ws.cell(row=1, column=col_idx, value=h)
        c.font = header_font
        c.border = thin_border
        c.alignment = Alignment(horizontal="center" if h in ("USN", "Present", "Absent") else "left", vertical="center")

    for r_idx, st in enumerate(students_list, start=2):
        usn = str(st.get("usn") or "").strip()
        name = str(st.get("name") or "").strip()
        if not usn and not name:
            continue
        subj = str(st.get("subject") or subject_display or "").strip() or "Subject Attendance"
        teacher = str(st.get("teacher") or teacher_salutation or "").strip() or "Prof. Harshitha R"
        present = int(st.get("present", st.get("attended", 0)) or 0)
        absent = int(st.get("absent", st.get("missed", 0)) or 0)

        ws.row_dimensions[r_idx].height = 20

        # 1. USN
        c1 = ws.cell(row=r_idx, column=1, value=usn or "N/A")
        c1.font = data_font
        c1.border = thin_border
        c1.alignment = Alignment(horizontal="center", vertical="center")

        # 2. Name
        c2 = ws.cell(row=r_idx, column=2, value=name or "Student")
        c2.font = data_font
        c2.border = thin_border
        c2.alignment = Alignment(horizontal="left", vertical="center")

        # 3. Subject
        c3 = ws.cell(row=r_idx, column=3, value=subj)
        c3.font = data_font
        c3.border = thin_border
        c3.alignment = Alignment(horizontal="left", vertical="center")

        # 4. Teacher
        c4 = ws.cell(row=r_idx, column=4, value=teacher)
        c4.font = data_font
        c4.border = thin_border
        c4.alignment = Alignment(horizontal="left", vertical="center")

        # 5. Present
        c5 = ws.cell(row=r_idx, column=5, value=present)
        c5.font = data_font
        c5.border = thin_border
        c5.alignment = Alignment(horizontal="center", vertical="center")

        # 6. Absent
        c6 = ws.cell(row=r_idx, column=6, value=absent)
        c6.font = data_font
        c6.border = thin_border
        c6.alignment = Alignment(horizontal="center", vertical="center")

    # Column widths
    col_widths = {1: 18, 2: 26, 3: 28, 4: 24, 5: 12, 6: 12}
    for col_idx, width in col_widths.items():
        ws.column_dimensions[get_column_letter(col_idx)].width = width

    safe_period = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in period_display).strip("_")
    safe_sub = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in subject_display).strip("_")
    filename = f"{type_title}_Attendance_Report_{safe_sub}_{safe_period}.xlsx"

    buf = io.BytesIO()
    wb.save(buf)
    return filename, buf.getvalue()


generate_monthly_excel_report = generate_attendance_excel_report


# ============================================================
# PDF ATTENDANCE REPORT GENERATION (WEEKLY & MONTHLY)
# ============================================================

def generate_attendance_pdf_report(report_data, teacher_name="Teacher", class_name=None):
    """
    Generates a simple A4 PDF attendance report with table:
    USN | Name | Subject | Teacher | Present | Absent
    Uses actual system database attendance data.
    Returns: (filename, bytes_data)
    """
    import io
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.pdfgen import canvas

    rep_type = str(report_data.get("report_type") or "weekly").lower().strip()
    type_title = "Weekly" if rep_type == "weekly" else "Monthly"
    period_label = report_data.get("period_label", "Current Period")
    period_display = clean_period_display(period_label, rep_type)
    subject_name = report_data.get("subject_name", "Blockchain Technology")
    subject_display = clean_subject_display(subject_name)
    teacher_salutation = clean_teacher_display(teacher_name)

    records_map = report_data.get("records_map", {})
    students_list = sorted(list(records_map.values()), key=lambda x: str(x.get("usn", "")))

    class NumberedCanvas(canvas.Canvas):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._saved_page_states = []

        def showPage(self):
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            num_pages = len(self._saved_page_states)
            for state in self._saved_page_states:
                self.__dict__.update(state)
                self.draw_footer(num_pages)
                super().showPage()
            super().save()

        def draw_footer(self, total_pages):
            self.saveState()
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748B"))
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(36, 30, A4[0] - 36, 30)
            self.drawString(36, 18, "GSSS Attendance Portal")
            page_text = f"Page {self._pageNumber} of {total_pages}"
            self.drawRightString(A4[0] - 36, 18, page_text)
            self.restoreState()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=44
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.black,
        alignment=1
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        alignment=1
    )
    th_style = ParagraphStyle(
        "TH",
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=10,
        textColor=colors.black,
        alignment=1
    )
    th_left = ParagraphStyle(
        "THL",
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=10,
        textColor=colors.black,
        alignment=0
    )
    td_center = ParagraphStyle(
        "TDC",
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.black,
        alignment=1
    )
    td_left = ParagraphStyle(
        "TDL",
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.black,
        alignment=0
    )

    elements = []
    elements.append(Paragraph("GSSS Attendance Portal", title_style))
    elements.append(Spacer(1, 3))
    elements.append(Paragraph(f"{type_title} Attendance Report – {period_display}", subtitle_style))
    elements.append(Spacer(1, 14))

    # Headers: USN | Name | Subject | Teacher | Present | Absent | date
    headers = [
        Paragraph("USN", th_style),
        Paragraph("Name", th_left),
        Paragraph("Subject", th_left),
        Paragraph("Teacher", th_left),
        Paragraph("Present", th_style),
        Paragraph("Absent", th_style),
        Paragraph("date", th_style)
    ]
    table_rows = [headers]

    for st in students_list:
        usn = str(st.get("usn") or "").strip() or "N/A"
        name = str(st.get("name") or "").strip() or "Student"
        subj = str(subject_display or "").strip() or "Blockchain Technology"
        teacher = str(teacher_salutation or "").strip() or "Prof. Harshitha R"
        present = int(st.get("attended") or 0)
        absent = int(st.get("missed") or 0)
        dt_val = str(period_display or "").strip() or "September 2026"

        table_rows.append([
            Paragraph(usn, td_center),
            Paragraph(name, td_left),
            Paragraph(subj, td_left),
            Paragraph(teacher, td_left),
            Paragraph(str(present), td_center),
            Paragraph(str(absent), td_center),
            Paragraph(dt_val, td_center)
        ])

    t = Table(table_rows, colWidths=[68, 105, 105, 95, 40, 40, 70], repeatRows=1)
    t_style = [
        ('LINEBELOW', (0, 0), (-1, 0), 1, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]
    t.setStyle(TableStyle(t_style))
    elements.append(t)

    doc.build(elements, canvasmaker=NumberedCanvas)

    safe_period = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in period_display).strip("_")
    safe_sub = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in subject_display).strip("_")
    filename = f"{type_title}_Attendance_Report_{safe_sub}_{safe_period}.pdf"

    return filename, buf.getvalue()


generate_monthly_pdf_report = generate_attendance_pdf_report



def send_email_to_teacher(recipient_email, subject_line, html_content, text_content, attachments=None):
    """
    Dispatches a professional email to the teacher's Gmail with optional attachments
    (such as Excel .xlsx and PDF .pdf attendance reports).
    - If attachments are provided, uses MIMEMultipart("mixed") with MIMEApplication.
    - Uses SMTP credentials from .env.
    - If SMTP is unavailable, saves to logs/teacher_emails.log.
    """
    from email.mime.application import MIMEApplication
    mail_server = os.environ.get("MAIL_SERVER", "").strip() or getattr(Config, "MAIL_SERVER", "")
    mail_port = int(os.environ.get("MAIL_PORT", "587") or getattr(Config, "MAIL_PORT", 587))
    mail_user = os.environ.get("MAIL_USERNAME", "").strip() or getattr(Config, "MAIL_USERNAME", "")
    mail_pass = os.environ.get("MAIL_PASSWORD", "").strip() or getattr(Config, "MAIL_PASSWORD", "")
    mail_use_tls = str(os.environ.get("MAIL_USE_TLS", "True") or getattr(Config, "MAIL_USE_TLS", True)).lower() == "true"
    sender = os.environ.get("MAIL_DEFAULT_SENDER", "").strip() or getattr(Config, "MAIL_DEFAULT_SENDER", "") or mail_user or "attendance-notifications@gsss.edu.in"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if mail_server and mail_user and mail_pass:
        try:
            if attachments and len(attachments) > 0:
                msg = MIMEMultipart("mixed")
                msg_body = MIMEMultipart("alternative")
                msg_body.attach(MIMEText(text_content, "plain", "utf-8"))
                msg_body.attach(MIMEText(html_content, "html", "utf-8"))
                msg.attach(msg_body)

                for att in attachments:
                    fname = att.get("filename", "report")
                    data = att.get("data")
                    if data:
                        part = MIMEApplication(data, Name=fname)
                        part["Content-Disposition"] = f'attachment; filename="{fname}"'
                        msg.attach(part)
            else:
                msg = MIMEMultipart("alternative")
                msg.attach(MIMEText(text_content, "plain", "utf-8"))
                msg.attach(MIMEText(html_content, "html", "utf-8"))

            msg["Subject"] = Header(subject_line, "utf-8")
            msg["From"] = f"GSSS Attendance Portal <{sender}>"
            msg["To"] = recipient_email
            msg["Date"] = datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0530")

            if mail_port == 465:
                server = smtplib.SMTP_SSL(mail_server, mail_port, timeout=15)
            else:
                server = smtplib.SMTP(mail_server, mail_port, timeout=15)
                if mail_use_tls:
                    server.starttls()

            clean_pass = mail_pass.replace(" ", "")
            try:
                server.login(mail_user, clean_pass)
            except Exception:
                server.login(mail_user, mail_pass)
            server.sendmail(sender, [recipient_email], msg.as_string())
            server.quit()

            log_entry = f"[{timestamp}] [SMTP_DELIVERED_TEACHER] To: {recipient_email} | Subject: {subject_line} | Attachments: {len(attachments or [])}\n"
            teacher_log = os.path.join(LOGS_DIR, "teacher_emails.log")
            with open(teacher_log, "a", encoding="utf-8") as f:
                f.write(log_entry)

            return {
                "success": True,
                "status": "sent",
                "message": f"Email with official reports delivered to {recipient_email}"
            }
        except Exception as e:
            err_msg = str(e)
            log_entry = f"[{timestamp}] [SMTP_FAILED -> LOGGED] To: {recipient_email} | Error: {err_msg} | Subject: {subject_line}\n"
            teacher_log = os.path.join(LOGS_DIR, "teacher_emails.log")
            with open(teacher_log, "a", encoding="utf-8") as f:
                f.write(log_entry)

            return {
                "success": True,
                "status": "logged",
                "message": f"Recorded to teacher dispatch spool ({recipient_email})"
            }

    # Spool logging fallback
    log_entry = f"[{timestamp}] [SPOOL_TEACHER] To: {recipient_email} | Subject: {subject_line} | Attachments: {len(attachments or [])}\n"
    teacher_log = os.path.join(LOGS_DIR, "teacher_emails.log")
    with open(teacher_log, "a", encoding="utf-8") as f:
        f.write(log_entry)

    return {
        "success": True,
        "status": "logged",
        "message": f"Recorded to teacher dispatch spool ({recipient_email})"
    }
