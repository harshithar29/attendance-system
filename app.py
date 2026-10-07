"""
app.py

Application entrypoint. Creates the Flask app, loads configuration,
initializes the MySQL connection pool, and registers all blueprints
(controllers).

Run with:
    python app.py
"""

import logging
import os

from dotenv import load_dotenv

# Load environment variables from .env BEFORE importing anything that reads them.
load_dotenv()

from flask import Flask

from config.config import Config
from backend.services.db_service import init_db_pool
from backend.controllers.auth_controller import auth_bp
from backend.controllers.admin_controller import admin_bp
from backend.controllers.teacher_controller import teacher_bp
from backend.controllers.student_controller import student_bp


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Make sure runtime folders exist (uploads, dataset, embeddings, reports, logs).
    for folder in (
        Config.UPLOAD_FOLDER,
        Config.VIDEO_UPLOAD_FOLDER,
        Config.FRAMES_FOLDER,
        Config.DATASET_FOLDER,
        Config.EMBEDDINGS_FOLDER,
        Config.REPORTS_FOLDER,
        Config.LOGS_FOLDER,
    ):
        os.makedirs(folder, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(Config.LOGS_FOLDER, "app.log")),
            logging.StreamHandler(),
        ],
    )

    # Set up the MySQL connection pool once at startup.
    init_db_pool()

    # Register blueprints (each module owns its own URL prefix).
    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(teacher_bp)
    app.register_blueprint(student_bp)

    @app.route("/download-apk")
    def download_apk():
        from flask import send_from_directory
        return send_from_directory("static/apk", "GSSS_Attendance.apk", as_attachment=True)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=Config.DEBUG, host="0.0.0.0", port=5000)
