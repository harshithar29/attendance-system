"""
config/config.py

Central configuration for the whole application.
All values are pulled from environment variables (loaded from a .env file
by python-dotenv in app.py) so that secrets never live in source code.
"""

import os


class Config:
    """Base configuration shared by the whole app."""

    # --- Flask ---
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")
    DEBUG = os.environ.get("FLASK_DEBUG", "True") == "True"

    # --- Session ---
    # Sessions are the mechanism we use for authentication (Admin/Teacher/Student).
    SESSION_TYPE = "filesystem"
    PERMANENT_SESSION_LIFETIME = 60 * 60 * 8  # 8 hours

    # --- MySQL Database ---
    DB_HOST = os.environ.get("DB_HOST", "localhost")
    DB_PORT = int(os.environ.get("DB_PORT", 3306))
    DB_USER = os.environ.get("DB_USER", "root")
    DB_PASSWORD = os.environ.get("DB_PASSWORD", "Aiml@123#456")
    DB_NAME = os.environ.get("DB_NAME", "attendance_system")

    # --- File storage locations (used from Phase 2 onward) ---
    # None of these are read from or written to in Phase 1 -- routes for
    # video upload / processing don't exist yet. The folders are created
    # now so the folder structure requirement is satisfied and Phase 2
    # can start writing to them immediately without any setup step.
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")                     # generic uploads (e.g. student profile photos)
    VIDEO_UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads", "videos")     # classroom videos uploaded by teachers
    FRAMES_FOLDER = os.path.join(BASE_DIR, "extracted_frames")           # frames extracted from a video during processing
    DATASET_FOLDER = os.path.join(BASE_DIR, "dataset")                    # registered student face images (for embeddings)
    EMBEDDINGS_FOLDER = os.path.join(BASE_DIR, "face_embeddings")        # serialized FaceNet embeddings
    REPORTS_FOLDER = os.path.join(BASE_DIR, "reports")                    # generated Excel/PDF attendance reports
    LOGS_FOLDER = os.path.join(BASE_DIR, "logs")

    # Video upload constraints (enforced starting Phase 2, defined here now
    # so the limits are configurable from day one instead of hard-coded later).
    ALLOWED_VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv"}
    MAX_VIDEO_SIZE_MB = int(os.environ.get("MAX_VIDEO_SIZE_MB", 500))

    # --- SMTP Mail Configuration ---
    MAIL_SERVER = os.environ.get("MAIL_SERVER", "smtp.gmail.com")
    MAIL_PORT = int(os.environ.get("MAIL_PORT", 587))
    MAIL_USE_TLS = os.environ.get("MAIL_USE_TLS", "True") == "True"
    MAIL_USERNAME = os.environ.get("MAIL_USERNAME", "")
    MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD", "")
    MAIL_DEFAULT_SENDER = os.environ.get("MAIL_DEFAULT_SENDER", "")
