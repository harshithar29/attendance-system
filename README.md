# AI-Based Student Attendance Management System (Video-Upload Workflow)

**Phase 1 — Foundation** (working code, no AI yet)

---

## Updated project objective

A teacher records a classroom video on a mobile phone and uploads it
through the Teacher Dashboard. The system will then (in later phases)
process that video with OpenCV, RetinaFace, and FaceNet to recognize
~50–60 students and mark attendance automatically — **no live webcam
stream is used anywhere in this project.**

### Updated workflow (for reference — not built yet)
```
Teacher Login → Select Class → Select Subject
  → Record classroom video on phone → Upload video
  → OpenCV reads the video → Extract frames → Preprocess images
  → RetinaFace detects faces → FaceNet generates embeddings
  → Compare with stored student embeddings → Recognize students
  → Remove duplicate detections → Mark attendance once
  → Store attendance in MySQL
  → Teacher reviews attendance → Student Dashboard auto-updates
```

---

## What Phase 1 actually contains

Per your instructions, Phase 1 is **only**:

- Complete project folder structure
- Python virtual environment setup
- Required library installation (only the lightweight web/DB stack — no AI libraries yet)
- Flask project setup
- MySQL database connection
- Database schema (including the tables Phase 2 will need, so nothing has to be redesigned later)
- Landing page
- Admin login
- Teacher login
- Student login
- Basic dashboard UI for all three roles

**Explicitly NOT in Phase 1:** OpenCV, RetinaFace, FaceNet, video upload handling, frame extraction, face recognition, attendance processing, or report generation. The Teacher Dashboard shows the future "Upload Classroom Video" form, but every control on it is disabled — it's a preview of Phase 2, not working functionality.

I'm stopping here, as instructed, for your confirmation before starting Phase 2.

---

## 1. Prerequisites

- Python 3.11
- MySQL Server 8.x running locally (or reachable over network)
- pip

```bash
python3 --version
mysql --version
```

---

## 2. Project setup

### 2.1 Unzip and enter the project
```bash
unzip attendance_system.zip
cd attendance_system
```

### 2.2 Create and activate a virtual environment

A virtual environment keeps this project's Python packages isolated
from anything else installed on your machine.

```bash
python3 -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

You'll know it worked because your terminal prompt will show `(venv)`
at the start of the line.

### 2.3 Install dependencies

```bash
pip install -r requirements.txt
```

Only 4 lightweight packages install right now: `Flask`, `python-dotenv`,
`mysql-connector-python`, `Werkzeug`. Everything else (OpenCV,
RetinaFace, FaceNet, TensorFlow, openpyxl, reportlab) is listed in
`requirements.txt` but **commented out** — we'll uncomment and install
those only when Phase 2 begins, so Phase 1 setup stays fast and simple.

### 2.4 Configure environment variables

```bash
cp .env.example .env
```

Open `.env` in a text editor and set:
- `DB_PASSWORD` → your real MySQL root (or app-user) password
- `SECRET_KEY` → any long random string (used to sign Flask session cookies)

Everything else can stay at its default for local development.

---

## 3. Database setup

### 3.1 Create the schema

```bash
mysql -u root -p < database/schema.sql
```

This single command creates the `attendance_system` database and all
9 tables: `admin`, `teachers`, `students`, `classes`, `subjects`,
`video_uploads`, `face_embeddings`, `attendance`, `unknown_faces`.

Only `admin`, `teachers`, and `students` are actually used by Phase 1's
login system — the rest exist now so Phase 2 can start writing to them
immediately without an extra migration step.

### 3.2 Create the first Admin account

Passwords are hashed in Python (never stored in plain text), so the
very first admin account is created by a script rather than a raw
`INSERT` statement:

```bash
python create_admin.py
```

This reads the `DEFAULT_ADMIN_*` values from your `.env` file. With
the defaults untouched, it creates:
- Username: `admin`
- Password: `Admin@123`

### 3.3 (Optional) Seed a demo Teacher and Student

So you can test all three logins immediately, without building the
Admin's "Add Teacher / Add Student" screens (those come in Phase 2):

```bash
python seed_demo_data.py
```

This creates:
- Teacher → username: `teacher1` / password: `Teacher@123`
- Student → username: `student1` / password: `Student@123`

---

## 4. Run the application

```bash
python app.py
```

You should see log output ending with something like:
```
* Running on http://0.0.0.0:5000
```

Open your browser at **http://127.0.0.1:5000**

---

## 5. Try it out

1. Landing page → click **Login**.
2. Log in as Admin (`admin` / `Admin@123`) → Admin dashboard with live
   teacher/student counts pulled from MySQL, and a roadmap of what's
   coming in later phases.
3. Log out, log in as Teacher (`teacher1` / `Teacher@123`) → Teacher
   dashboard showing the (disabled, preview-only) class/subject/video
   upload form.
4. Log out, log in as Student (`student1` / `Student@123`) → Student
   profile dashboard.

---

## 6. Full project structure, explained

```
attendance_system/
├── app.py                     # Flask entrypoint — creates the app, loads config,
│                               # initializes the MySQL pool, registers all blueprints
├── create_admin.py             # One-time script: creates the first Admin account
├── seed_demo_data.py           # Optional script: creates a demo Teacher + Student
├── requirements.txt            # Python dependencies (Phase 1 active, later phases commented out)
├── .env.example                 # Template for environment variables (copy to .env)
├── .gitignore
│
├── config/
│   └── config.py               # Every setting (secret key, DB credentials, folder
│                                # paths, video upload limits) loaded from .env —
│                                # nothing is hard-coded in the application logic
│
├── backend/
│   ├── models/                 # One file per DB table, each responsible only for
│   │   │                       # talking to its own table (no SQL anywhere else)
│   │   ├── admin.py            # admin table: find_by_username, create, count_all
│   │   ├── teacher.py          # teachers table: same pattern
│   │   └── student.py          # students table: same pattern
│   │
│   ├── controllers/            # Flask blueprints — one per role, each owns its
│   │   │                       # own URL prefix so routes never collide or overlap
│   │   ├── auth_controller.py  # "/" landing page, "/login", "/logout"
│   │   ├── admin_controller.py # "/admin/dashboard"
│   │   ├── teacher_controller.py # "/teacher/dashboard"
│   │   └── student_controller.py # "/student/dashboard"
│   │
│   ├── services/
│   │   └── db_service.py       # MySQL connection pool + a run_query() helper —
│   │                           # the ONLY place that opens a raw DB connection
│   │
│   ├── ai_modules/             # Empty in Phase 1. Phase 2 will add here:
│   │                           # frame_extractor.py, face_detector.py (RetinaFace),
│   │                           # face_recognizer.py (FaceNet), preprocessing.py
│   │
│   └── utilities/
│       └── security.py         # hash_password / verify_password (Werkzeug PBKDF2)
│                                # + the login_required(role=...) decorator used
│                                # to protect every dashboard route
│
├── templates/                  # Jinja2 + Bootstrap 5 HTML
│   ├── base.html                # Shared layout: navbar, flash messages, footer scripts
│   ├── landing.html             # Public homepage
│   ├── login.html               # Single login form for all 3 roles
│   ├── admin/dashboard.html     # Admin dashboard + "coming in later phases" roadmap
│   ├── teacher/dashboard.html   # Teacher dashboard with disabled video-upload preview
│   └── student/dashboard.html   # Student profile dashboard
│
├── static/
│   ├── css/style.css            # Small stylesheet on top of Bootstrap 5
│   └── javascript/main.js       # Placeholder — Phase 2 will add upload-progress JS here
│
├── database/
│   └── schema.sql               # All 9 tables (see section 7 below)
│
├── uploads/
│   └── videos/                  # Phase 2: raw classroom videos uploaded by teachers
├── extracted_frames/            # Phase 2: frames pulled out of a video during processing
├── dataset/                     # Phase 2: registered student face images (for embeddings)
├── face_embeddings/             # Phase 2: serialized FaceNet embedding vectors
├── reports/                     # Phase 3: generated Excel/PDF attendance reports
└── logs/
    └── app.log                  # Application log file (rotated per run)
```

---

## 7. Database schema, explained

| Table | Purpose | Used starting |
|---|---|---|
| `admin` | Admin accounts (login) | Phase 1 |
| `teachers` | Teacher accounts (login) | Phase 1 |
| `students` | Student accounts (login) | Phase 1 |
| `classes` | Class definitions (e.g. "CSE-3A") | Phase 2 |
| `subjects` | Subjects, linked to a class | Phase 2 |
| `video_uploads` | One row per uploaded classroom video, with a `status` column (`uploaded` → `processing` → `completed`/`failed`) so the Teacher Dashboard can show processing progress | Phase 2 |
| `face_embeddings` | One row per registered face image + its FaceNet embedding vector, linked to a student | Phase 2 |
| `attendance` | One row per recognized student per session; linked back to the `video_uploads` row it came from for traceability; a unique constraint on `(student_id, subject_id, session_date)` is what enforces "mark attendance only once" | Phase 2 |
| `unknown_faces` | Cropped faces detected in a video but not matched to any student, linked to the source video, for teacher review | Phase 2 |

Why build all 9 tables now, in Phase 1, if only 3 are used? So that
Phase 2 never has to modify or re-migrate a table that's already in
production use — it only ever adds new code on top of a schema that's
already correct and complete.

---

## 8. Why the code is organized this way

- **Blueprints per role** keep routing isolated. Phase 2's "Upload
  Video" and "Process Attendance" routes will only touch
  `teacher_controller.py` — nothing else needs to change.
- **Models only talk to their own table.** No SQL lives in controllers
  or templates, so the DB layer can be tested or swapped independently.
- **`db_service.py` centralizes MySQL access** through a connection
  pool, which is standard production practice (reusing connections
  instead of opening a new one per request).
- **`security.py`** centralizes password hashing and the
  `login_required(role=...)` decorator, so every protected route gets
  the same, auditable access control in a single line.
- **`.env` / `config.py`** keep every secret and environment-specific
  value out of source code entirely — including the new video upload
  size limit and folder paths, defined now even though nothing uses
  them yet.

---

## 9. Dependency reference (Phase 1)

| Package | Why it's needed |
|---|---|
| `Flask` | The web framework — routing, templating, sessions |
| `python-dotenv` | Loads `.env` values into `os.environ` at startup |
| `mysql-connector-python` | Official MySQL driver, used via a connection pool |
| `Werkzeug` | Ships with Flask; used here specifically for `generate_password_hash` / `check_password_hash` |

Everything else in `requirements.txt` (OpenCV, RetinaFace, FaceNet,
TensorFlow, openpyxl, reportlab) is commented out and will be
uncommented at the start of Phase 2, when we actually write the video
processing code that needs them.

---

## 10. Stopping point

This completes Phase 1 as scoped: folder structure, environment setup,
Flask + MySQL wiring, full schema, and working login + basic dashboards
for all three roles. **No AI/OpenCV/FaceNet/RetinaFace code has been
written.** Let me know when you'd like to proceed to Phase 2 (video
upload handling, frame extraction, and face registration for Admin).
