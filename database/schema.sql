-- ==========================================================
-- AI-Based Student Attendance Management System
-- (Video-upload workflow: teacher uploads a classroom video,
--  the system processes it offline with OpenCV + RetinaFace + FaceNet)
--
-- Database Schema
--
-- Run this once against a fresh database, e.g.:
--   mysql -u root -p < database/schema.sql
--
-- Phase 1 actively uses: admin, teachers, students
-- Phase 2+ will populate: classes, subjects, video_uploads,
--                          face_embeddings, attendance, unknown_faces
-- ==========================================================

CREATE DATABASE IF NOT EXISTS attendance_system
    CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE attendance_system;

-- ----------------------------------------------------------
-- Admin
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS admin (
    admin_id       INT AUTO_INCREMENT PRIMARY KEY,
    name           VARCHAR(100) NOT NULL,
    email          VARCHAR(150) NOT NULL UNIQUE,
    username       VARCHAR(50)  NOT NULL UNIQUE,
    password_hash  VARCHAR(255) NOT NULL,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ----------------------------------------------------------
-- Classes  (referenced by Students & Subjects)
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS classes (
    class_id     INT AUTO_INCREMENT PRIMARY KEY,
    class_name   VARCHAR(100) NOT NULL,
    section      VARCHAR(20),
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ----------------------------------------------------------
-- Teachers
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS teachers (
    teacher_id     INT AUTO_INCREMENT PRIMARY KEY,
    name           VARCHAR(100) NOT NULL,
    email          VARCHAR(150) NOT NULL UNIQUE,
    username       VARCHAR(50)  NOT NULL UNIQUE,
    password_hash  VARCHAR(255) NOT NULL,
    phone          VARCHAR(20),
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ----------------------------------------------------------
-- Students
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS students (
    student_id     INT AUTO_INCREMENT PRIMARY KEY,
    name           VARCHAR(100) NOT NULL,
    email          VARCHAR(150) NOT NULL UNIQUE,
    username       VARCHAR(50)  NOT NULL UNIQUE,
    password_hash  VARCHAR(255) NOT NULL,
    roll_no        VARCHAR(50)  NOT NULL UNIQUE,
    class_id       INT,
    phone          VARCHAR(20),
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (class_id) REFERENCES classes(class_id) ON DELETE SET NULL
);

-- ----------------------------------------------------------
-- Subjects
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS subjects (
    subject_id     INT AUTO_INCREMENT PRIMARY KEY,
    subject_name   VARCHAR(150) NOT NULL,
    subject_code   VARCHAR(30)  NOT NULL UNIQUE,
    class_id       INT,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (class_id) REFERENCES classes(class_id) ON DELETE SET NULL
);

-- ----------------------------------------------------------
-- Video Uploads (Phase 2)
-- One row per classroom video a teacher uploads for processing.
-- `status` lets the Teacher Dashboard show a processing progress state.
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS video_uploads (
    video_id       INT AUTO_INCREMENT PRIMARY KEY,
    teacher_id     INT NOT NULL,
    class_id       INT NOT NULL,
    subject_id     INT NOT NULL,
    video_path     VARCHAR(255) NOT NULL,
    status         ENUM('uploaded', 'processing', 'completed', 'failed') NOT NULL DEFAULT 'uploaded',
    total_students_detected INT DEFAULT 0,
    total_recognized         INT DEFAULT 0,
    total_unknown             INT DEFAULT 0,
    uploaded_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at   TIMESTAMP NULL,
    FOREIGN KEY (teacher_id) REFERENCES teachers(teacher_id) ON DELETE CASCADE,
    FOREIGN KEY (class_id) REFERENCES classes(class_id) ON DELETE CASCADE,
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id) ON DELETE CASCADE
);

-- ----------------------------------------------------------
-- Face Embeddings (Phase 2)
-- Stores one row per registered face image / embedding vector.
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS face_embeddings (
    embedding_id   INT AUTO_INCREMENT PRIMARY KEY,
    student_id     INT NOT NULL,
    image_path     VARCHAR(255) NOT NULL,
    embedding      LONGBLOB NOT NULL,   -- serialized FaceNet embedding vector
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
);

-- ----------------------------------------------------------
-- Attendance (Phase 2)
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS attendance (
    attendance_id  INT AUTO_INCREMENT PRIMARY KEY,
    student_id     INT NOT NULL,
    subject_id     INT NOT NULL,
    class_id       INT NOT NULL,
    teacher_id     INT NOT NULL,
    video_id       INT NULL,          -- which uploaded video this attendance came from (NULL for manual entries)
    session_date   DATE NOT NULL,
    session_time   TIME NOT NULL,
    status         ENUM('present', 'absent') NOT NULL DEFAULT 'present',
    marked_by      ENUM('system', 'manual') NOT NULL DEFAULT 'system',
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id) ON DELETE CASCADE,
    FOREIGN KEY (class_id) REFERENCES classes(class_id) ON DELETE CASCADE,
    FOREIGN KEY (teacher_id) REFERENCES teachers(teacher_id) ON DELETE CASCADE,
    FOREIGN KEY (video_id) REFERENCES video_uploads(video_id) ON DELETE SET NULL,
    UNIQUE KEY unique_attendance_per_session (student_id, subject_id, session_date)
);

-- ----------------------------------------------------------
-- Unknown Faces (Phase 2)
-- Faces detected in an uploaded video but not matched to any
-- registered student (cropped face image saved for teacher review).
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS unknown_faces (
    unknown_id     INT AUTO_INCREMENT PRIMARY KEY,
    video_id       INT,
    image_path     VARCHAR(255) NOT NULL,
    class_id       INT,
    detected_date  DATE NOT NULL,
    detected_time  TIME NOT NULL,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES video_uploads(video_id) ON DELETE CASCADE,
    FOREIGN KEY (class_id) REFERENCES classes(class_id) ON DELETE SET NULL
);
