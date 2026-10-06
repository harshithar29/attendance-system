"""
backend/controllers/auth_controller.py

Handles:
    - Direct login page
    - Role-based authentication
    - Admin / Teacher / Student login
    - Session creation
    - Logout

Authentication flow:

Login page
    ↓
Select role
    ↓
Username + Password
    ↓
Check selected role's MySQL table
    ↓
Verify password
    ↓
Create session
    ↓
Redirect to appropriate dashboard
"""

from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash
)

from backend.models.admin import Admin
from backend.models.teacher import Teacher
from backend.models.student import Student
from backend.utilities.security import verify_password, hash_password
from backend.services.db_service import run_query


# ============================================================
# BLUEPRINT
# ============================================================

auth_bp = Blueprint("auth", __name__)


# ============================================================
# DIRECT LOGIN PAGE
# ============================================================

@auth_bp.route("/")
def landing():
    """
    The application starts directly at the login page.

    If the user is already logged in, send them directly
    to their appropriate dashboard.
    """

    if session.get("user_id") and session.get("role"):

        role = session.get("role")

        if role in ("admin", "teacher", "student"):
            return redirect(url_for(f"{role}.dashboard"))

        # Invalid session role
        session.clear()

    return redirect(url_for("auth.login"))


# ============================================================
# LOGIN
# ============================================================

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    """
    Handles Admin, Teacher and Student login.

    POST data expected from login.html:

        username
        password
        role

    Example:

        role=admin
        role=teacher
        role=student
    """

    # --------------------------------------------------------
    # GET REQUEST
    # --------------------------------------------------------

    if request.method == "GET":
        return render_template("login.html")


    # --------------------------------------------------------
    # READ FORM DATA
    # --------------------------------------------------------

    username = request.form.get("username", "").strip()

    password = request.form.get("password", "")

    selected_role = request.form.get("role", "").strip().lower()


    # --------------------------------------------------------
    # VALIDATE ROLE
    # --------------------------------------------------------

    allowed_roles = {
        "admin",
        "teacher",
        "student"
    }

    if selected_role not in allowed_roles:

        flash(
            "Please select Admin, Teacher, or Student.",
            "danger"
        )

        return redirect(url_for("auth.login"))


    # --------------------------------------------------------
    # VALIDATE USERNAME AND PASSWORD
    # --------------------------------------------------------

    if not username:

        flash(
            "Please enter your username.",
            "danger"
        )

        return redirect(url_for("auth.login"))


    if not password:

        flash(
            "Please enter your password.",
            "danger"
        )

        return redirect(url_for("auth.login"))


    # ========================================================
    # FIND USER ACCORDING TO SELECTED ROLE
    # ========================================================

    user = None


    # --------------------------------------------------------
    # ADMIN
    # --------------------------------------------------------

    if selected_role == "admin":

        user = Admin.find_by_username(username)


    # --------------------------------------------------------
    # TEACHER
    # --------------------------------------------------------

    elif selected_role == "teacher":

        user = Teacher.find_by_username(username)


    # --------------------------------------------------------
    # STUDENT
    # --------------------------------------------------------

    elif selected_role == "student":

        user = Student.find_by_username(username)


    # ========================================================
    # USER NOT FOUND
    # ========================================================

    if not user:

        flash(
            f"No {selected_role} account was found with "
            f"the username '{username}'.",
            "danger"
        )

        return redirect(url_for("auth.login"))


    # ========================================================
    # PASSWORD VERIFICATION
    # ========================================================

    password_hash = user.get("password_hash")


    if not password_hash:

        flash(
            "This account does not have a valid password configured.",
            "danger"
        )

        return redirect(url_for("auth.login"))


    try:

        password_valid = verify_password(
            password,
            password_hash
        )

    except Exception:

        password_valid = False


    if not password_valid:

        flash(
            "Incorrect password.",
            "danger"
        )

        return redirect(url_for("auth.login"))


    # ========================================================
    # GET USER ID
    # ========================================================

    user_id_key = f"{selected_role}_id"

    user_id = user.get(user_id_key)


    if user_id is None:

        flash(
            "Unable to identify this account. "
            "Please contact the administrator.",
            "danger"
        )

        return redirect(url_for("auth.login"))


    # ========================================================
    # CREATE SECURE SESSION
    # ========================================================

    session.clear()

    session.permanent = True

    session["user_id"] = user_id

    session["name"] = user.get("name", username)

    session["role"] = selected_role

    session["username"] = username


    # ========================================================
    # SUCCESS MESSAGE
    # ========================================================

    flash(
        f"Welcome back, {user.get('name', username)}!",
        "success"
    )


    # ========================================================
    # REDIRECT TO ROLE DASHBOARD
    # ========================================================

    return redirect(
        url_for(f"{selected_role}.dashboard")
    )


# ============================================================
# LOGOUT
# ============================================================

@auth_bp.route("/logout")
def logout():
    """
    Clears the complete login session.
    """

    session.clear()

    flash(
        "You have been logged out successfully.",
        "info"
    )

    return redirect(
        url_for("auth.login")
    )


# ============================================================
# REGISTRATION (CREATE NEW USER: TEACHER / STUDENT)
# ============================================================

@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    """
    Allows registering a new Teacher or Student account.
    """
    if request.method == "GET":
        return render_template("register.html")

    role = request.form.get("role", "").strip().lower()
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "").strip()
    confirm_password = request.form.get("confirm_password", "").strip()

    if not role or role not in ("teacher", "student"):
        flash("Please select your role (Teacher or Student).", "danger")
        return redirect(url_for("auth.register"))

    if not name or not email or not password:
        flash("Please fill in all required fields.", "danger")
        return redirect(url_for("auth.register"))

    if password != confirm_password:
        flash("Passwords do not match.", "danger")
        return redirect(url_for("auth.register"))

    if len(password) < 4:
        flash("Password should be at least 4 characters long.", "danger")
        return redirect(url_for("auth.register"))

    pwd_hash = hash_password(password)

    if role == "teacher":
        username = request.form.get("username", "").strip()
        phone = request.form.get("phone", "").strip() or None
        if not username:
            username = email.split("@")[0]

        existing = run_query(
            "SELECT teacher_id FROM teachers WHERE username = %s OR email = %s LIMIT 1",
            (username, email),
            fetch_one=True
        )
        if existing:
            flash("A teacher account with this username or email already exists.", "warning")
            return redirect(url_for("auth.register"))

        try:
            Teacher.create(name=name, email=email, username=username, password_hash=pwd_hash, phone=phone)
            flash("Teacher registration successful! You can now log in.", "success")
            return redirect(url_for("auth.login"))
        except Exception as e:
            flash(f"Registration failed: {str(e)}", "danger")
            return redirect(url_for("auth.register"))

    elif role == "student":
        usn = request.form.get("student_number", "").strip().upper()
        username = request.form.get("username", "").strip() or usn
        if not usn:
            flash("Please enter your USN / Student Number.", "danger")
            return redirect(url_for("auth.register"))

        existing = run_query(
            "SELECT student_id FROM students WHERE UPPER(student_number) = UPPER(%s) OR username = %s OR email = %s LIMIT 1",
            (usn, username, email),
            fetch_one=True
        )
        if existing:
            student_id = existing["student_id"]
            run_query(
                "UPDATE students SET name = %s, username = %s, password_hash = %s, email = %s WHERE student_id = %s",
                (name, username, pwd_hash, email, student_id),
                commit=True
            )
            flash("Student account updated / activated successfully! You can now log in.", "success")
            return redirect(url_for("auth.login"))

        try:
            Student.create(name=name, usn=usn, email=email, username=username, password_hash=pwd_hash)
            flash("Student registration successful! You can now log in.", "success")
            return redirect(url_for("auth.login"))
        except Exception as e:
            flash(f"Registration failed: {str(e)}", "danger")
            return redirect(url_for("auth.register"))